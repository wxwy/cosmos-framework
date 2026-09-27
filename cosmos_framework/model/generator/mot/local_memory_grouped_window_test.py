"""H3-C 候选 fast state、GA2 权重与 optimizer 后单次发布 CPU 合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_test import make_segment
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import RankLocalGroupedPlanner
from cosmos_framework.model.generator.mot.robocasa_grouped_segment_test import _rank_catalog


def _setup(*, frames: int = 64) -> tuple[nn.Module, GroupedLocalMemoryWindow]:
    torch.manual_seed(17)
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_memory_runtime = LocalMemoryRuntime()
    model.net.moe_gen = nn.Parameter(torch.tensor(2.0))
    planner = RankLocalGroupedPlanner(_rank_catalog(16, frames=frames), rank=0)
    return model, GroupedLocalMemoryWindow(model, planner)


def _segments(runner: GroupedLocalMemoryWindow, member: int):
    assert runner.plan is not None
    segments = []
    for request in runner.plan.members[member]:
        segment = make_segment(
            cursor=request.identity.cursor, slot=request.identity.slot_id, episode=request.identity.episode_id
        )
        valid = torch.arange(16)[None] < request.valid_count
        evidence = valid & segment.consumer_step.gt(0)
        segments.append(
            replace(
                segment,
                consumer_payload=(
                    segment.consumer_payload[0][: request.valid_count] + (None,) * (16 - request.valid_count),
                ),
                consumer_valid=valid,
                evidence_valid=evidence,
                evidence_source_step=torch.where(evidence, segment.consumer_step - 1, -1),
                category=(request.identity.category,),
                segment_provenance=SegmentProvenance(
                    "manifest", "config", request.identity.source_digest, request.identity.segment_id
                ),
            )
        )
    return tuple(segments)


def _backward(loss: torch.Tensor, retain_graph: bool) -> None:
    loss.backward(retain_graph=retain_graph)


def test_ga2_full_window_uses_exact_256_consumer_weight_and_publishes_after_step() -> None:
    model, runner = _setup()
    before = runner.live
    plan = runner.begin()
    assert plan.member_counts == (128, 128) and plan.n_window == 256
    seen: list[tuple[int, int, bool]] = []

    def native(payloads, prefixes, index):
        seen.append((index, len(payloads), all(prefix is None for prefix in prefixes)))
        return model.net.moe_gen

    first = runner.run_member(_segments(runner, 0), native, _backward)
    assert runner.live is before and len(runner._candidate.sidecar.snapshot()) == 8
    second = runner.run_member(_segments(runner, 1), native, _backward)
    assert runner.live is before and len(runner._candidate.sidecar.snapshot()) == 0
    assert float(first + second) == 2.0
    torch.testing.assert_close(model.net.moe_gen.grad, torch.tensor(1.0))
    assert len(seen) == 32
    assert seen[0] == (0, 8, True)
    assert all(size == 8 for _, size, _ in seen)
    assert not seen[16][2]  # continuation 在下一 member 消费上一段 candidate fast state

    steps = []
    runner.finish(lambda: steps.append("optimizer_step") or True)
    assert steps == ["optimizer_step"] and runner.live is not before
    assert runner.live.frontier == plan.candidate_frontier
    assert len(runner.live.scheduler._committed) == 8
    assert runner.live.sidecar.snapshot() == ()  # terminal 完成后清除 fast state


def test_joint_outer_gradient_reaches_host_and_model_owned_local_slow_params() -> None:
    model, runner = _setup()
    runner.begin()

    def native(payloads, prefixes, index):
        present = [prefix for prefix in prefixes if prefix is not None]
        signal = torch.stack([prefix.square().mean() for prefix in present]).mean() if present else 0.0
        return model.net.moe_gen * (signal + 1.0)

    runner.run_member(_segments(runner, 0), native, _backward)
    runner.run_member(_segments(runner, 1), native, _backward)
    named = dict(model.net.named_parameters())
    assert model.net.moe_gen.grad is not None and torch.isfinite(model.net.moe_gen.grad)
    for name in ("local_memory_runtime.encoder.visual_proj.weight", "local_memory_runtime.core.slot_queries"):
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
    assert runner.live.sidecar.snapshot() == ()
    runner.abort()


def test_terminal_remainder_uses_actual_window_count_without_ga_rescale() -> None:
    model, runner = _setup(frames=49)
    plan = runner.begin()
    assert plan.member_counts == (128, 8) and plan.n_window == 136
    calls = []

    def native(payloads, prefixes, index):
        calls.append((index, len(payloads)))
        return model.net.moe_gen

    for member in range(2):
        runner.run_member(_segments(runner, member), native, _backward)
    assert len(calls) == 17 and calls[-1] == (0, 8)
    torch.testing.assert_close(model.net.moe_gen.grad, torch.tensor(1.0))
    runner.abort()


def test_terminal_rebind_in_second_member_has_new_s0_without_local_prefix() -> None:
    model, runner = _setup(frames=48)
    plan = runner.begin()
    assert plan.member_counts == (128, 128) and all(
        first.episode.uid != second.episode.uid for first, second in zip(*plan.members, strict=True)
    )
    seen = []

    def native(payloads, prefixes, index):
        seen.append(all(prefix is None for prefix in prefixes))
        return model.net.moe_gen

    runner.run_member(_segments(runner, 0), native, _backward)
    runner.run_member(_segments(runner, 1), native, _backward)
    assert seen[0] and seen[16]
    assert runner.live.sidecar.snapshot() == ()
    runner.abort()


def test_malformed_member_aborts_pending_window_without_live_change() -> None:
    _, runner = _setup()
    before = runner.live
    runner.begin()
    bad = tuple(reversed(_segments(runner, 0)))
    with pytest.raises(ValueError, match="slot 顺序"):
        runner.run_member(bad, lambda *_: torch.tensor(0.0, requires_grad=True), _backward)
    assert runner.live is before and runner.plan is None


@pytest.mark.parametrize("failure", ["native", "nonfinite", "backward", "second_member"])
def test_member_failure_leaves_live_state_unchanged(failure: str) -> None:
    model, runner = _setup()
    before = runner.live
    runner.begin()
    if failure == "second_member":
        runner.run_member(_segments(runner, 0), lambda *_: model.net.moe_gen, _backward)
        member = 1
    else:
        member = 0

    def native(payloads, prefixes, index):
        if index == 3 and failure in ("native", "second_member"):
            raise RuntimeError("native failure")
        if index == 3 and failure == "nonfinite":
            return model.net.moe_gen * torch.tensor(float("nan"))
        return model.net.moe_gen

    def backward(loss, retain_graph):
        if failure == "backward":
            raise RuntimeError("backward failure")
        _backward(loss, retain_graph)

    with pytest.raises((RuntimeError, ValueError)):
        runner.run_member(_segments(runner, member), native, backward)
    assert runner.live is before and runner.live.sidecar.snapshot() == ()
    assert runner.live.scheduler._committed == {} and runner.plan is None


@pytest.mark.parametrize("mode", ["skip", "exception"])
def test_optimizer_skip_or_exception_never_publishes_candidate(mode: str) -> None:
    model, runner = _setup()
    before = runner.live
    runner.begin()
    for member in range(2):
        runner.run_member(_segments(runner, member), lambda *_: model.net.moe_gen, _backward)

    def step():
        if mode == "exception":
            raise RuntimeError("optimizer failure")
        return False

    with pytest.raises(RuntimeError):
        runner.finish(step)
    assert runner.live is before and runner.live.frontier == before.frontier
    assert runner.live.sidecar.snapshot() == () and runner.plan is None

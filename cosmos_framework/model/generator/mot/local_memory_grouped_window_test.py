"""H3-C 候选 fast state、GA2 权重与 optimizer 后单次发布 CPU 合同。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import RoboCasaExactWindowCacheCatalog
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    RoboCasaExactWindowCachedDataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import RoboCasaExactWindowSourceReader
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _setup as _exact_setup


class _Net(nn.Module):
    def __init__(self, t: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(local_memory_enabled=True)
        self.local_memory_runtime = LocalMemoryRuntime(
            evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=t, k_local=1
        )
        self.moe_gen = nn.Parameter(torch.tensor(2.0))
        self.scan_calls = 0

    def scan_local_memory(self, visual, action, valid, state, *, continuation_mask=None):
        self.scan_calls += 1
        return Cosmos3VFMNetwork.scan_local_memory(
            self, visual, action, valid, state, continuation_mask=continuation_mask
        )


def _setup(
    *, frames: int = 64, t: int = 16, b_stream: int = 8, active_ga: int = 2
) -> tuple[nn.Module, GroupedLocalMemoryWindow]:
    torch.manual_seed(17)
    _, _, catalog, producer = _exact_setup((frames - 32,) * 16, t=t)
    model = nn.Module()
    model.net = _Net(t)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=b_stream, active_ga=active_ga)
    runner = GroupedLocalMemoryWindow(model, planner)
    runner._test_producer = producer
    return model, runner


def _segments(runner: GroupedLocalMemoryWindow, member: int):
    assert runner.plan is not None
    return tuple(runner._test_producer.produce(request) for request in runner.plan.members[member])


def _backward(loss: torch.Tensor, retain_graph: bool) -> None:
    loss.backward(retain_graph=retain_graph)


def test_ga2_full_window_uses_exact_256_consumer_weight_and_publishes_after_step() -> None:
    model, runner = _setup()
    before = runner.live
    plan = runner.begin()
    assert plan.member_counts == (128, 128) and plan.n_window == 256
    seen: list[tuple[tuple[tuple[int, int], ...], int, int]] = []

    def native(payloads, prefixes, positions):
        seen.append((positions, len(payloads), sum(prefix is None for prefix in prefixes)))
        return model.net.moe_gen

    first = runner.run_member(_segments(runner, 0), native, _backward)
    assert runner.live is before and len(runner._candidate.sidecar.snapshot()) == 8
    second = runner.run_member(_segments(runner, 1), native, _backward)
    assert model.net.scan_calls == 2
    assert runner.live is before and len(runner._candidate.sidecar.snapshot()) == 0
    assert float(first + second) == 2.0
    torch.testing.assert_close(model.net.moe_gen.grad, torch.tensor(1.0))
    assert len(seen) == 2
    assert seen[0][1:] == (128, 8)  # each slot has S0 exactly once
    assert seen[1][1:] == (128, 0)  # continuing segment has all Local prefixes
    assert seen[0][0][:17] == tuple((0, t) for t in range(16)) + ((1, 0),)
    assert seen[1][0][:2] == ((0, 0), (0, 1))

    steps = []
    runner.finish(lambda: steps.append("optimizer_step") or True)
    assert steps == ["optimizer_step"] and runner.live is not before
    assert runner.live.frontier == plan.candidate_frontier
    assert len(runner.live.scheduler._committed) == 8
    assert runner.live.sidecar.snapshot() == ()  # terminal 完成后清除 fast state



def test_packed128_fp32_loss_and_slow_gradients_match_index_weighted_reference() -> None:
    """CPU oracle uses identical scan graph and evaluates each time-index loss separately."""
    model, runner = _setup(frames=64, t=16, b_stream=8, active_ga=2)
    plan = runner.begin()
    assert plan.member_counts == (128, 128) and plan.n_window == 256
    selected = (
        model.net.moe_gen,
        model.net.local_memory_runtime.encoder.visual_proj.weight,
        model.net.local_memory_runtime.core.slot_queries,
    )
    total_reference = [torch.zeros_like(p) for p in selected]
    counts = []

    for member in range(2):
        def native(payloads, prefixes, positions):
            assert len(payloads) == len(prefixes) == len(positions) == 128
            counts.append(len(payloads))
            # Identical per-consumer function, varying Local prefixes and S0.
            values = torch.stack([
                model.net.moe_gen
                * (prefix.square().mean() + 1.0 if prefix is not None else model.net.moe_gen.new_tensor(1.0))
                for prefix in prefixes
            ])
            packed_mean = values.mean()
            # Independent old V3 Index8 oracle: each index mean weighted
            # by its actual valid count, not by a flat GA constant.
            indexed = torch.zeros_like(packed_mean)
            for index in range(16):
                active = [value for value, pos in zip(values, positions, strict=True) if pos[1] == index]
                indexed = indexed + torch.stack(active).mean() * (len(active) / len(values))
            torch.testing.assert_close(packed_mean, indexed, rtol=1e-5, atol=1e-6)
            reference_grad = torch.autograd.grad(indexed, selected, retain_graph=True)
            for i, grad in enumerate(reference_grad):
                total_reference[i] = total_reference[i] + grad * (len(values) / plan.n_window)
            return packed_mean

        runner.run_member(_segments(runner, member), native, _backward)

    assert counts == [128, 128]
    for param, expected in zip(selected, total_reference, strict=True):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad, expected, rtol=1e-4, atol=1e-5)
    runner.abort()



def test_joint_outer_gradient_reaches_host_and_model_owned_local_slow_params() -> None:
    model, runner = _setup()
    runner.begin()

    def native(payloads, prefixes, positions):
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

    def native(payloads, prefixes, positions):
        calls.append((positions, len(payloads)))
        return model.net.moe_gen

    for member in range(2):
        runner.run_member(_segments(runner, member), native, _backward)
    assert len(calls) == 2 and calls[0][1] == 128 and calls[1][1] == 8
    torch.testing.assert_close(model.net.moe_gen.grad, torch.tensor(1.0))
    runner.abort()


def test_terminal_rebind_in_second_member_has_new_s0_without_local_prefix() -> None:
    model, runner = _setup(frames=48)
    plan = runner.begin()
    assert plan.member_counts == (128, 128) and all(
        first.episode.uid != second.episode.uid for first, second in zip(*plan.members, strict=True)
    )
    seen = []

    def native(payloads, prefixes, positions):
        seen.append(sum(prefix is None for prefix in prefixes))
        return model.net.moe_gen

    runner.run_member(_segments(runner, 0), native, _backward)
    runner.run_member(_segments(runner, 1), native, _backward)
    assert seen == [8, 8]
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

    def native(payloads, prefixes, positions):
        if failure in ("native", "second_member"):
            raise RuntimeError("native failure")
        if failure == "nonfinite":
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


def test_t32_b3_ga3_uses_one_scan_and_one_packed_native_per_member() -> None:
    model, runner = _setup(frames=96, t=32, b_stream=3, active_ga=3)
    before = runner.live
    plan = runner.begin()
    assert len(plan.members) == 3 and all(len(member) == 3 for member in plan.members)
    assert plan.member_counts == (96, 96, 96) and plan.n_window == 288
    seen = []

    def native(payloads, prefixes, positions):
        seen.append((positions, len(payloads), sum(prefix is None for prefix in prefixes)))
        assert all(item["video_latent"].shape == (5, 48, 2, 2) for item in payloads)
        return model.net.moe_gen

    for member in range(3):
        runner.run_member(_segments(runner, member), native, _backward)
        assert runner.live is before
        assert model.net.scan_calls == member + 1
    assert len(seen) == 3
    assert [count for _, count, _ in seen] == [96, 96, 96]
    assert [none_count for _, _, none_count in seen] == [3, 0, 3]
    assert seen[0][0][:3] == ((0, 0), (0, 1), (0, 2))
    torch.testing.assert_close(model.net.moe_gen.grad, torch.tensor(1.0))
    runner.finish(lambda: True)
    assert runner.live is not before and runner.live.frontier == plan.candidate_frontier


@pytest.mark.skipif(
    not (os.environ.get("PSM_PHASE4A_CACHE_ROOT") and os.environ.get("PSM_PHASE4A_SOURCE_ROOT")),
    reason="strict real debug cache/source paths were not supplied",
)
def test_optional_strict_real_data_grouped_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = RoboCasaExactWindowCacheCatalog(Path(os.environ["PSM_PHASE4A_CACHE_ROOT"]))
    source = RoboCasaExactWindowSourceReader(cache, Path(os.environ["PSM_PHASE4A_SOURCE_ROOT"]))
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(cache)
    raw = RoboCasaExactWindowCachedDataset(cache, source, contract)
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        max_action_dim=contract.max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
    )
    catalog = ExactWindowLocalCatalog(ActionSFTDataset(raw, transform, None), ttt_tbptt_steps=16)
    producer = ExactWindowSegmentProducer(catalog, config_digest="phase4b-debug")
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=2, active_ga=1)
    model = nn.Module()
    model.net = _Net(16)
    runner = GroupedLocalMemoryWindow(model, planner)
    before = runner.live
    plan = runner.begin()
    segments = tuple(producer.produce(request) for request in plan.members[0])
    assert len(segments) == 2 and all(segment.consumer_valid.shape == (1, 16) for segment in segments)

    def native(payloads, prefixes, positions):
        assert all(payload["video_latent"].shape == cache.latent_shape for payload in payloads)
        signal = (
            torch.stack([prefix.square().mean() for prefix in prefixes if prefix is not None]).mean()
            if any(prefix is not None for prefix in prefixes)
            else 0.0
        )
        return model.net.moe_gen * (1.0 + signal)

    runner.run_member(segments, native, _backward)
    assert model.net.scan_calls == 1 and runner.live is before
    runner.finish(lambda: True)
    assert runner.live is not before and runner.live.frontier == plan.candidate_frontier

"""H3-D grouped fast/frontier DCP 状态严格恢复的 CPU 合同。"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.trainer.local_memory_grouped as grouped_module
from cosmos_framework.checkpoint.dcp import _DataloaderWrapper
from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.local_memory_grouped_window_test import _backward, _segments, _setup
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowRankPlanner
from cosmos_framework.trainer.local_memory_grouped_resume import (
    GroupedLocalMemoryStateCallback,
    require_dcp_grouped_resume_component,
    restore_grouped_local_state,
    snapshot_grouped_local_state,
)
from cosmos_framework.trainer.local_memory_grouped_test import _Model, _trainer


def _owned_segments(window, member):
    return tuple(
        replace(
            segment,
            segment_provenance=replace(
                segment.segment_provenance, manifest_digest=window.planner.catalog.manifest_digest
            ),
        )
        for segment in _segments(window, member)
    )


@pytest.fixture(scope="module")
def committed_runtime():
    model, window = _setup(frames=80)
    plan = window.begin()
    assert plan.member_counts == (128, 128)
    for member in range(2):
        window.run_member(_owned_segments(window, member), lambda *_: model.net.moe_gen, _backward)
    window.finish(lambda: True)
    state = snapshot_grouped_local_state(window, iteration=1, config_digest="config")
    assert len(state["sidecar"]) == 8
    return model, window, state


@pytest.fixture(scope="module")
def committed_state(committed_runtime):
    return committed_runtime[2]


def test_restore_next_identity_and_all_fast_values(committed_state) -> None:
    _, resumed = _setup(frames=80)
    restore_grouped_local_state(resumed, committed_state, iteration=1, config_digest="config")
    assert resumed.live.frontier == committed_state["frontier"]
    assert resumed.live.scheduler._committed == committed_state["scheduler"]
    assert resumed.plan is None
    expected = resumed.planner.plan_window(committed_state["frontier"])
    actual = resumed.begin()
    assert actual.members == expected.members and actual.candidate_frontier == expected.candidate_frontier
    resumed.abort()
    for identity, provenance, fast in resumed.live.sidecar.snapshot():
        stored_identity, stored_provenance, stored_fast = committed_state["sidecar"][identity.slot_id]
        assert (identity, provenance) == (stored_identity, stored_provenance)
        for value, saved in zip(fast, stored_fast, strict=True):
            assert value.dtype == torch.float32 and value.grad_fn is None
            torch.testing.assert_close(value.cpu(), saved)


def test_interrupted_and_uninterrupted_next_loss_and_gradient_match(committed_runtime) -> None:
    source_model, source, state = committed_runtime
    resumed_model, resumed = _setup(frames=80)
    resumed_model.load_state_dict(source_model.state_dict())
    restore_grouped_local_state(resumed, state, iteration=1, config_digest="config")
    source_model.zero_grad(set_to_none=True)
    resumed_model.zero_grad(set_to_none=True)
    assert source.begin().members == resumed.begin().members
    rng = torch.get_rng_state()
    source_segment = _owned_segments(source, 0)
    torch.set_rng_state(rng)
    resumed_segment = _owned_segments(resumed, 0)
    for left, right in zip(source_segment, resumed_segment, strict=True):
        torch.testing.assert_close(left.evidence_visual_summary_prev, right.evidence_visual_summary_prev)
        torch.testing.assert_close(left.evidence_executed_action_prev, right.evidence_executed_action_prev)

    def native(model):
        def callback(payloads, prefixes, index):
            present = [prefix for prefix in prefixes if prefix is not None]
            return model.net.moe_gen * (torch.stack([value.square().mean() for value in present]).mean() + 1)

        return callback

    source_loss = source.run_member(source_segment, native(source_model), _backward)
    resumed_loss = resumed.run_member(resumed_segment, native(resumed_model), _backward)
    torch.testing.assert_close(source_loss, resumed_loss, rtol=1e-5, atol=1e-6)
    for (name, left), (other_name, right) in zip(
        source_model.named_parameters(), resumed_model.named_parameters(), strict=True
    ):
        assert name == other_name and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
    source.abort()
    resumed.abort()


@pytest.mark.parametrize(
    "damage",
    [
        "iteration",
        "manifest",
        "corpus",
        "source",
        "config",
        "profile",
        "uid",
        "scheduler",
        "missing_fast",
        "fp16",
        "nan",
        "parameter",
        "grad_fn",
        "requires_grad",
        "v1",
    ],
)
def test_corrupt_or_mismatched_state_fails_without_live_mutation(committed_state, damage: str) -> None:
    _, resumed = _setup(frames=80)
    before = resumed.live
    state = copy.deepcopy(committed_state)
    if damage == "iteration":
        state["iteration"] = 2
    elif damage == "manifest":
        state["cache_manifest_sha256"] = "other"
    elif damage == "corpus":
        state["cache_corpus_digest"] = "other"
    elif damage == "source":
        state["source_binding_digest"] = "other"
    elif damage == "config":
        state["config_digest"] = "other"
    elif damage == "v1":
        state["format"] = "psm_v3_h3d_grouped_local_v1"
    elif damage == "profile":
        state["profile"] = (-1, *state["profile"][1:])
    elif damage == "uid":
        frontier = state["frontier"]
        state["frontier"] = replace(frontier, slots=(replace(frontier.slots[0], uid="foreign"), *frontier.slots[1:]))
    elif damage == "scheduler":
        state["scheduler"].pop(0)
    else:
        record = state["sidecar"][0]
        if damage == "missing_fast":
            state["sidecar"].pop(0)
        elif damage == "fp16":
            state["sidecar"][0] = (record[0], record[1], (record[2][0].half(), *record[2][1:]))
        elif damage == "parameter":
            state["sidecar"][0] = (record[0], record[1], (torch.nn.Parameter(record[2][0]), *record[2][1:]))
        elif damage == "grad_fn":
            state["sidecar"][0] = (record[0], record[1], (record[2][0].requires_grad_() * 1, *record[2][1:]))
        elif damage == "requires_grad":
            state["sidecar"][0] = (record[0], record[1], (record[2][0].requires_grad_(), *record[2][1:]))
        else:
            bad = record[2][0].clone()
            bad.flatten()[0] = float("nan")
            state["sidecar"][0] = (record[0], record[1], (bad, *record[2][1:]))
    with pytest.raises((ValueError, KeyError)):
        restore_grouped_local_state(resumed, state, iteration=1, config_digest="config")
    assert resumed.live is before and resumed.live.sidecar.snapshot() == ()


def test_terminal_restoration_requires_exact_last_segment_source(committed_state) -> None:
    state = copy.deepcopy(committed_state)
    frontier = state["frontier"]
    state["frontier"] = replace(
        frontier,
        slots=tuple(
            replace(slot, uid=None, binding_epoch=None, cursor=0, next_segment_id=3) for slot in frontier.slots
        ),
    )
    state["scheduler"] = {
        slot: replace(identity, cursor=2, segment_id=2, training_stream_end=True)
        for slot, identity in state["scheduler"].items()
    }
    state["sidecar"] = {}
    state["iteration"] = 2
    _, valid = _setup(frames=80)
    restore_grouped_local_state(valid, state, iteration=2, config_digest="config")
    assert valid.live.sidecar.snapshot() == ()
    state["scheduler"][0] = replace(state["scheduler"][0], source_digest="foreign")
    _, invalid = _setup(frames=80)
    before = invalid.live
    with pytest.raises(ValueError, match="terminal"):
        restore_grouped_local_state(invalid, state, iteration=2, config_digest="config")
    assert invalid.live is before


def test_corrected_format_and_dynamic_profile(committed_state) -> None:
    _, window = _setup(frames=80)
    profile = committed_state["profile"]
    assert committed_state["format"] == "psm_v3_corrected_grouped_local_v2"
    assert profile[:5] == (0, 1, 0, 8, 2)
    assert profile[-3:] == (64, 16, 17)
    assert set(committed_state) == {
        "format",
        "iteration",
        "cache_manifest_sha256",
        "cache_corpus_digest",
        "source_binding_digest",
        "config_digest",
        "profile",
        "frontier",
        "scheduler",
        "sidecar",
    }
    assert profile[12] == window.model.net.local_memory_runtime.core.ttt_tbptt_steps


@pytest.mark.parametrize("b_stream,active_ga,t", [(1, 1, 2), (3, 2, 3), (8, 3, 4)])
def test_dynamic_slots_and_profile(b_stream: int, active_ga: int, t: int) -> None:
    model, window = _setup(frames=40, t=t, b_stream=b_stream, active_ga=active_ga)
    window.begin()
    for member in range(active_ga):
        window.run_member(_owned_segments(window, member), lambda *_: model.net.moe_gen, _backward)
    window.finish(lambda: True)
    state = snapshot_grouped_local_state(window, iteration=1, config_digest="config")
    assert state["profile"][3:5] == (b_stream, active_ga)
    assert state["profile"][12] == t
    assert set(state["scheduler"]) == set(range(b_stream))
    _, resumed = _setup(frames=40, t=t, b_stream=b_stream, active_ga=active_ga)
    restore_grouped_local_state(resumed, state, iteration=1, config_digest="config")
    assert resumed.live.frontier == window.live.frontier


def test_rank_offset_slots_and_k_profile() -> None:
    model, original = _setup(frames=40, t=2, b_stream=3, active_ga=1)
    model.net.local_memory_runtime = LocalMemoryRuntime(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=2, k_local=4
    )
    planner = ExactWindowRankPlanner(original.planner.catalog, rank=1, world_size=2, b_stream=3, active_ga=1)
    window = GroupedLocalMemoryWindow(model, planner)
    window._test_producer = original._test_producer
    window.begin()
    window.run_member(_owned_segments(window, 0), lambda *_: model.net.moe_gen, _backward)
    window.finish(lambda: True)
    state = snapshot_grouped_local_state(window, iteration=1, config_digest="config")
    assert state["profile"][:5] == (1, 2, 0, 3, 1)
    assert state["profile"][11:13] == (4, 2)
    assert set(state["scheduler"]) == {3, 4, 5}
    resumed = GroupedLocalMemoryWindow(model, planner)
    restore_grouped_local_state(resumed, state, iteration=1, config_digest="config")
    assert resumed.live.frontier == window.live.frontier


def test_public_frontier_validator_is_non_mutating() -> None:
    _, window = _setup(frames=40, b_stream=1)
    frontier = window.planner.initial_frontier()
    before = copy.deepcopy(frontier)
    window.planner.validate_frontier(frontier)
    assert frontier == before
    with pytest.raises(ValueError, match="类型"):
        window.planner.validate_frontier(replace(frontier, slots=list(frontier.slots)))
    with pytest.raises(ValueError, match="类型"):
        window.planner.validate_frontier(replace(frontier, slots=(replace(frontier.slots[0], next_segment_id=1.0),)))


def test_zero_step_snapshot_and_restore_rejected() -> None:
    model, window = _setup(frames=40, b_stream=1)
    with pytest.raises(RuntimeError, match="成功 optimizer step"):
        snapshot_grouped_local_state(window, iteration=0, config_digest="config")
    with pytest.raises(RuntimeError, match="成功 optimizer step"):
        snapshot_grouped_local_state(window, iteration=1, config_digest="config")
    window.begin()
    for member in range(2):
        window.run_member(_owned_segments(window, member), lambda *_: model.net.moe_gen, _backward)
    window.finish(lambda: True)
    state = snapshot_grouped_local_state(window, iteration=1, config_digest="config")
    with pytest.raises(ValueError, match="schema"):
        restore_grouped_local_state(window, state, iteration=0, config_digest="config")


def test_dcp_dataloader_wrapper_selects_grouped_callback_and_stages_restore(committed_state) -> None:
    trainer = _trainer()
    assert isinstance(trainer.callbacks._callbacks[0], GroupedLocalMemoryStateCallback)
    _, trainer._grouped_window = _setup(frames=80)
    restore_grouped_local_state(trainer._grouped_window, committed_state, iteration=1, config_digest="config")
    trainer._grouped_completed_iteration = 1
    wrapper = _DataloaderWrapper(trainer.callbacks)
    assert wrapper.has_state()
    saved = wrapper.state_dict()
    assert saved["iteration"] == 1 and len(saved["sidecar"]) == 8

    other = _trainer()
    receiver = _DataloaderWrapper(other.callbacks)
    receiver.load_state_dict(saved)
    assert other._pending_grouped_resume["iteration"] == 1
    _, other._grouped_window = _setup(frames=80)
    restore_grouped_local_state(
        other._grouped_window, other._pending_grouped_resume, iteration=1, config_digest="config"
    )
    assert other._grouped_window.live.frontier == trainer._grouped_window.live.frontier


def test_resume_restores_completed_iteration_before_first_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    trainer._resume_required = True
    trainer._pending_grouped_resume = {"iteration": 100}
    model = _Model()
    observed: dict[str, int] = {}

    def fake_restore(window, state, *, iteration, config_digest):
        assert state["iteration"] == iteration == 100
        assert config_digest == "config"
        observed["iteration"] = iteration

    def stop_after_restore(self):
        raise RuntimeError("stop after resume restore")

    monkeypatch.setattr(grouped_module, "restore_grouped_local_state", fake_restore)
    monkeypatch.setattr(grouped_module.GroupedLocalMemoryWindow, "begin", stop_after_restore)

    with pytest.raises(RuntimeError, match="stop after resume restore"):
        trainer.training_step(model, None, None, None, {}, 100, 0)

    assert observed["iteration"] == 100
    assert trainer._grouped_completed_iteration == 100
    assert trainer._pending_grouped_resume is None


def test_missing_callback_state_blocks_first_resumed_training_step() -> None:
    trainer = _trainer()
    trainer._resume_required = True
    model = _Model()
    with pytest.raises(RuntimeError, match="未向 grouped callback 恢复"):
        trainer.training_step(model, None, None, None, {}, 1, 0)
    assert trainer._grouped_window.live.sidecar.snapshot() == ()
    with pytest.raises(RuntimeError, match="不得继续使用"):
        trainer.training_step(model, None, None, None, {}, 1, 0)


def test_corrupt_pending_resume_is_terminal_for_trainer_instance() -> None:
    trainer = _trainer()
    trainer._resume_required = True
    trainer._pending_grouped_resume = {"format": "invalid"}
    model = _Model()
    with pytest.raises(ValueError, match="schema"):
        trainer.training_step(model, None, None, None, {}, 1, 0)
    with pytest.raises(RuntimeError, match="不得继续使用"):
        trainer.training_step(model, None, None, None, {}, 1, 0)


def test_callback_refuses_checkpoint_before_first_successful_window() -> None:
    trainer = _trainer()
    wrapper = _DataloaderWrapper(trainer.callbacks)
    assert wrapper.has_state()
    with pytest.raises(RuntimeError, match="禁止零步 checkpoint"):
        wrapper.state_dict()


def test_same_job_requires_this_rank_component_and_warmstart_does_not(tmp_path) -> None:
    source = SimpleNamespace(path=str(tmp_path), backend_key=None, uses_object_store=False, warm_start=False)
    checkpointer = SimpleNamespace(
        keys_to_resume_during_load=lambda: ({"model", "optim", "scheduler", "trainer", "dataloader"}, source),
        load_training_state=True,
    )
    with pytest.raises(FileNotFoundError, match="rank"):
        require_dcp_grouped_resume_component(checkpointer)
    checkpointer.keys_to_resume_during_load = lambda: ({"model", "optim", "scheduler", "trainer"}, source)
    with pytest.raises(FileNotFoundError, match="dataloader"):
        require_dcp_grouped_resume_component(checkpointer)
    checkpointer.keys_to_resume_during_load = lambda: ({"model", "optim", "scheduler", "trainer", "dataloader"}, source)
    folder = tmp_path / "dataloader"
    folder.mkdir()
    (folder / "rank_0.pkl").write_bytes(b"fixture")
    assert require_dcp_grouped_resume_component(checkpointer)
    source.warm_start = True
    checkpointer.keys_to_resume_during_load = lambda: ({"model"}, source)
    checkpointer.load_training_state = False
    assert not require_dcp_grouped_resume_component(checkpointer)
    checkpointer.load_training_state = True
    with pytest.raises(ValueError, match="warm-start"):
        require_dcp_grouped_resume_component(checkpointer)

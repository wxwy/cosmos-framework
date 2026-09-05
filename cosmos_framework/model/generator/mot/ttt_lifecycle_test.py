# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU/static acceptance tests for the R09-B TTT v0.3.2 active-wiring lifecycle.

Drives ``TTTLifecycle`` directly with synthetic evidence rows plus the trainer
seams it owns (mark/commit on the external backward, transaction resolution),
the ``OmniMoTModel._inject_local_history`` TTT seam, the recipe env wiring, and
the model-config fields. No GPU, dataset, or checkpoint I/O.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import attrs
import pytest
import torch
from torch import nn

import cosmos_framework.model.generator.omni_mot_model as omni_mot_model
from cosmos_framework.configs.base.defaults.model_config import (
    OmniMoTModelConfig,
    _require_ttt_bool,
    _require_ttt_finite_positive_scalar,
    _require_ttt_k_local,
    _require_ttt_positive_int,
    _require_ttt_runtime_evidence_steps,
)
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.config_checkpoint_contract import SELECTORS
from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
    LocalHistoryRuntime,
    StatelessLocalReplayReadout,
)
from cosmos_framework.model.generator.mot.production_runtime_adapter import ProductionLocalMemoryRuntime
from cosmos_framework.model.generator.mot.runtime_authority import AdmissionAuthority
from cosmos_framework.model.generator.mot.ttt_lifecycle import (
    RESOLUTION_SCALER_SKIP,
    RESOLUTION_SUCCESS,
    TTTLifecycle,
    TTTLifecycleCallback,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata

OWNER = "libero_spatial#0"


def _source(value: float = 0.0, horizon: int = 2) -> dict[str, torch.Tensor]:
    return {
        "history_visual_summary": torch.full((1, horizon, 4), value),
        "local_history_action": torch.full((1, horizon, 3), value),
        "history_age_steps": torch.zeros(1, horizon, dtype=torch.long),
        "history_dt_s": torch.ones(1, horizon, 1),
        "history_mask": torch.ones(1, horizon, dtype=torch.bool),
    }


def _make_lifecycle(
    segment_steps: int = 16, *, k_local: int = 1, seed: int = 0
) -> tuple[TTTLifecycle, LocalEvidenceEncoder, ContinualTTTLocalMemoryCore, nn.Linear, nn.Parameter]:
    torch.manual_seed(seed)
    encoder = LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10, ttt_tbptt_steps=segment_steps, k_local=k_local
    )
    projector = nn.Linear(4, 5)
    modality = nn.Parameter(torch.zeros(5))
    lifecycle = TTTLifecycle(
        ProductionLocalMemoryRuntime(AdmissionAuthority(), encoder, core),
        encoder=encoder,
        core=core,
        local_memory2llm=projector,
        modality_embed=modality,
    )
    return lifecycle, encoder, core, projector, modality


def _authority(lifecycle: TTTLifecycle):
    return lifecycle.runtime.runtime_authority


def _micro_batch(lifecycle: TTTLifecycle, loss: torch.Tensor, *, backward: bool = True) -> None:
    """Simulate the trainer seam for one micro-batch: observe -> backward -> after."""
    lifecycle.observe_loss(loss)
    if backward:
        loss.backward()
    lifecycle.on_after_backward()


def _token_loss(projector: nn.Linear, modality: nn.Parameter, token: torch.Tensor) -> torch.Tensor:
    return projector(token).square().mean() + modality.square().mean()


def _committed_snapshot(lifecycle: TTTLifecycle, owner: str = OWNER) -> tuple:
    authority = _authority(lifecycle)
    state = authority._state_by_owner.get(owner)
    state_copy = None if state is None else tuple(value.clone() for value in state)
    committed = {key: record.value.clone() for key, record in authority._committed.items()}
    return state_copy, dict(authority._last_timestep), committed, dict(authority._epoch_by_owner)


def _assert_snapshot_equal(before: tuple, after: tuple) -> None:
    assert before[1] == after[1] and before[3] == after[3] and set(before[2]) == set(after[2])
    assert all(torch.equal(before[2][key], after[2][key]) for key in before[2])
    assert (before[0] is None) == (after[0] is None)
    if before[0] is not None:
        assert all(torch.equal(left, right) for left, right in zip(before[0], after[0], strict=True))


def _commit_segment(lifecycle: TTTLifecycle, projector: nn.Linear, modality: nn.Parameter, steps: int) -> None:
    """Drive one full segment through the trainer seam (steps = core.ttt_tbptt_steps)."""
    for window in range(steps):
        token = lifecycle.process_sample(
            owner_key=OWNER, terminal=False, valid=True, source=_source(0.05 * (window + 1))
        )
        assert token is not None
        _micro_batch(lifecycle, _token_loss(projector, modality, token))


# ---------------------------------------------------------------------------
# N=1/3/16 timing: detached intermediate reads, closing pre-write witness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 3, 16])
def test_n_matrix_closing_timing_and_prewrite_witness(steps: int) -> None:
    lifecycle, encoder, core, projector, modality = _make_lifecycle(segment_steps=steps)
    authority = _authority(lifecycle)
    sources = [_source(0.05 * (window + 1)) for window in range(steps)]
    closing_token = None
    for window in range(steps):
        token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=sources[window])
        assert token is not None and tuple(token.shape) == (core.k_local, 4)
        if window < steps - 1:
            # Intermediate window: detached candidate read, no graph, no arm.
            assert token.grad_fn is None
            assert lifecycle._segments[OWNER].armed is False
            assert authority._pending_by_owner[OWNER].witness is None
            _micro_batch(lifecycle, _token_loss(projector, modality, token))
            # No commit outside the closing seam.
            assert OWNER in lifecycle._segments
            assert OWNER not in authority._state_by_owner
            assert not authority._committed
        else:
            # Closing window: pre-write witness token (read(S_{t-1})) with graph.
            closing_token = token
            assert token.grad_fn is not None
            assert lifecycle._segments[OWNER].armed is True
            pending = authority._pending_by_owner[OWNER]
            assert pending.phase == "MATERIALIZED_PENDING"
            assert pending.witness is not None and pending.witness_leaf is not None
            _micro_batch(lifecycle, _token_loss(projector, modality, token))
            assert OWNER not in lifecycle._segments
            assert OWNER not in authority._pending_by_owner
            assert authority._last_timestep[OWNER] == steps - 1
    # Exactly one detached write per admitted window.
    assert authority.c5_write_count == steps
    # The committed state equals a fresh reference scan over the same raw rows.
    # The scan requires ordinary grad mode (create_graph=False already avoids a
    # higher-order graph), so no torch.no_grad() here.
    evidence = torch.stack([encoder(**source)[:, -1] for source in sources], dim=1)
    ref_tokens, ref_state, _ = core.scan_segment_many(
        evidence, torch.ones(1, steps, dtype=torch.bool), None, create_graph=False, emit_prewrite_tokens=True
    )
    for member, reference in zip(authority._state_by_owner[OWNER], ref_state, strict=True):
        torch.testing.assert_close(member[0], reference[0])
    # The closing window's witness token equals the reference scan's last
    # pre-write token (same values as the detached candidate read, with graph).
    assert closing_token is not None
    torch.testing.assert_close(closing_token.detach(), ref_tokens[0, -1])


def test_terminal_remainder_closes_commits_and_resets() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=16)
    authority = _authority(lifecycle)
    for window in range(2):
        token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.1 + window))
        _micro_batch(lifecycle, _token_loss(projector, modality, token))
    token = lifecycle.process_sample(owner_key=OWNER, terminal=True, valid=True, source=_source(0.3))
    assert token is not None and token.grad_fn is not None
    _micro_batch(lifecycle, _token_loss(projector, modality, token))
    # Terminal: commit then reset; the owner starts a new epoch with no state.
    assert OWNER not in authority._state_by_owner and OWNER not in authority._last_timestep
    assert authority._epoch_by_owner[OWNER] == 1
    owner = lifecycle._owners[OWNER]
    assert owner.epoch == 1 and owner.committed_state is None and owner.committed_timestep == -1
    # New epoch: the same owner key admits again from timestep 0.
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.4))
    assert token is not None and token.grad_fn is None
    assert len(authority._pending_by_owner[OWNER].rows) == 1


def test_zero_valid_window_no_admit_no_prefix() -> None:
    lifecycle, _, _, _, _ = _make_lifecycle(segment_steps=4)
    authority = _authority(lifecycle)
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=False, source=_source())
    assert token is None
    assert OWNER not in lifecycle._segments
    assert OWNER not in authority._pending_by_owner
    assert authority.c5_write_count == 0


def test_terminal_zero_valid_aborts_open_segment_and_resets() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=16)
    authority = _authority(lifecycle)
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.1))
    _micro_batch(lifecycle, _token_loss(projector, modality, token))
    assert OWNER in lifecycle._segments
    token = lifecycle.process_sample(owner_key=OWNER, terminal=True, valid=False, source=_source())
    assert token is None
    # Segment aborted (never committed), owner reset into a fresh epoch.
    assert OWNER not in lifecycle._segments and OWNER not in authority._pending_by_owner
    assert OWNER not in authority._state_by_owner
    assert authority._epoch_by_owner[OWNER] == 1


# ---------------------------------------------------------------------------
# T>=17 cross-segment causality: segment 2 resumes from the committed state
# ---------------------------------------------------------------------------


def test_second_segment_resumes_from_committed_state_not_w0() -> None:
    steps = 16
    lifecycle, encoder, core, projector, modality = _make_lifecycle(segment_steps=steps)
    sources = [_source(0.05 * (window + 1)) for window in range(steps)]
    for window in range(steps):
        token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=sources[window])
        _micro_batch(lifecycle, _token_loss(projector, modality, token))
    committed = _committed_snapshot(lifecycle)
    assert committed[0] is not None
    source17 = _source(0.9)
    token17 = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=source17)
    assert token17 is not None and token17.grad_fn is None
    segment = lifecycle._segments[OWNER]
    # The open segment's base is the committed state, not w0.
    assert segment.base_state is not None
    for member, reference in zip(segment.base_state, committed[0], strict=True):
        assert torch.equal(member, reference)
    # Window 17 reads the committed state (read-after-16), not a fresh w0 read.
    evidence = torch.stack([encoder(**source)[:, -1] for source in sources], dim=1)
    _, ref_state, _ = core.scan_segment_many(
        evidence, torch.ones(1, steps, dtype=torch.bool), None, create_graph=False, emit_prewrite_tokens=True
    )
    _, query_base17, _ = core.project_evidence(encoder(**source17)[:, -1])
    reference17 = core.read_many(core.project_queries(query_base17), ref_state)[0]
    w0_read17 = core.read_many(core.project_queries(query_base17), core.initial_state(1))[0]
    torch.testing.assert_close(token17, reference17)
    assert not torch.allclose(token17, w0_read17)


# ---------------------------------------------------------------------------
# Dual-evidence fail-closed negatives (finite loss + witness traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_fails_closed_even_when_witness_traversed(bad: float) -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=1)
    authority = _authority(lifecycle)
    _commit_segment(lifecycle, projector, modality, 1)
    baseline = _committed_snapshot(lifecycle)
    # Open and arm a second segment; really backward so witness_leaf.grad is set.
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.2))
    _token_loss(projector, modality, token).backward()  # traversal evidence satisfied
    lifecycle.observe_loss(torch.tensor(bad))
    with pytest.raises(ValueError, match="finite"):
        lifecycle.on_after_backward()
    # Fail-closed: abort, no BACKWARD_OK, no commit, committed snapshot intact.
    assert OWNER not in lifecycle._segments and OWNER not in authority._pending_by_owner
    _assert_snapshot_equal(baseline, _committed_snapshot(lifecycle))


def test_finite_loss_without_witness_traversal_fails_closed() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=1)
    authority = _authority(lifecycle)
    _commit_segment(lifecycle, projector, modality, 1)
    baseline = _committed_snapshot(lifecycle)
    lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.2))
    # Backward an unrelated finite graph: witness_leaf.grad stays None.
    (projector.weight.square().mean() + modality.square().mean()).backward()
    lifecycle.observe_loss(torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="did not traverse"):
        lifecycle.on_after_backward()
    assert OWNER not in lifecycle._segments and OWNER not in authority._pending_by_owner
    _assert_snapshot_equal(baseline, _committed_snapshot(lifecycle))


def test_missing_observed_loss_fails_closed() -> None:
    lifecycle, _, _, _, _ = _make_lifecycle(segment_steps=1)
    authority = _authority(lifecycle)
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.2))
    assert token is not None and lifecycle._segments[OWNER].armed
    # No observe_loss: the mark sees loss=None and must reject.
    with pytest.raises(ValueError, match="finite"):
        lifecycle.on_after_backward()
    assert OWNER not in lifecycle._segments and OWNER not in authority._pending_by_owner


# ---------------------------------------------------------------------------
# Backward-failure abort route and Option B skip semantics
# ---------------------------------------------------------------------------


def test_backward_error_abort_preserves_committed_snapshot() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=1)
    authority = _authority(lifecycle)
    _commit_segment(lifecycle, projector, modality, 1)
    baseline = _committed_snapshot(lifecycle)
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.2))
    loss = _token_loss(projector, modality, token)
    loss.register_hook(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    lifecycle.observe_loss(loss)
    with pytest.raises(RuntimeError, match="boom"):
        loss.backward()
    # The trainer seam aborts every open segment and re-raises unchanged.
    lifecycle.abort_open_segments()
    assert OWNER not in lifecycle._segments and OWNER not in authority._pending_by_owner
    _assert_snapshot_equal(baseline, _committed_snapshot(lifecycle))


def test_scaler_skip_keeps_commit_and_clears_slow_grads() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=1)
    _commit_segment(lifecycle, projector, modality, 1)
    baseline = _committed_snapshot(lifecycle)
    assert any(parameter.grad is not None for parameter in lifecycle.slow_parameters)
    lifecycle.resolve_transaction(RESOLUTION_SCALER_SKIP)
    # Option B: the fast commit stays valid; the slow side drops its .grad.
    assert all(parameter.grad is None for parameter in lifecycle.slow_parameters)
    _assert_snapshot_equal(baseline, _committed_snapshot(lifecycle))


def test_resolve_transaction_rejects_unknown_and_armed_survivor() -> None:
    lifecycle, _, _, _, _ = _make_lifecycle(segment_steps=1)
    with pytest.raises(ValueError, match="unknown transaction resolution"):
        lifecycle.resolve_transaction("BOGUS")
    # An armed segment that never reached on_after_backward must fail closed.
    lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.1))
    assert lifecycle._segments[OWNER].armed
    with pytest.raises(RuntimeError, match="armed segment survived"):
        lifecycle.resolve_transaction(RESOLUTION_SUCCESS)


def test_scaler_found_inf_stub_states() -> None:
    lifecycle, _, _, _, _ = _make_lifecycle(segment_steps=1)
    optimizer = torch.optim.SGD(lifecycle.slow_parameters, lr=0.1)
    found = SimpleNamespace(
        _per_optimizer_states={id(optimizer): {"found_inf_per_device": {torch.device("cpu"): torch.tensor(1.0)}}}
    )
    assert lifecycle.scaler_found_inf(found, optimizer) is True
    clean = SimpleNamespace(
        _per_optimizer_states={id(optimizer): {"found_inf_per_device": {torch.device("cpu"): torch.tensor(0.0)}}}
    )
    assert lifecycle.scaler_found_inf(clean, optimizer) is False
    # Fail-closed when GradScaler never recorded state for this optimizer.
    with pytest.raises(RuntimeError, match="no state"):
        lifecycle.scaler_found_inf(SimpleNamespace(_per_optimizer_states={}), optimizer)


# ---------------------------------------------------------------------------
# Reverse assertions: no commit path outside BACKWARD_OK
# ---------------------------------------------------------------------------


def test_no_commit_without_closing_and_authority_rejects_direct_commit() -> None:
    lifecycle, _, _, projector, modality = _make_lifecycle(segment_steps=4)
    authority = _authority(lifecycle)
    token = lifecycle.process_sample(owner_key=OWNER, terminal=False, valid=True, source=_source(0.1))
    _micro_batch(lifecycle, _token_loss(projector, modality, token))
    # Intermediate micro-batch seam ran, but nothing was armed: no commit.
    assert OWNER in lifecycle._segments
    assert not authority._committed and OWNER not in authority._state_by_owner
    # The authority rejects commit without the BACKWARD_OK phase.
    with pytest.raises(RuntimeError, match="commit requires"):
        authority.commit(OWNER)


# ---------------------------------------------------------------------------
# ContinualTTTLocalMemoryCore.reset_parameters (meta materialization path)
# ---------------------------------------------------------------------------


def test_continual_core_meta_reset_is_finite_and_deterministic() -> None:
    def _materialize() -> list[torch.Tensor]:
        with torch.device("meta"):
            core = ContinualTTTLocalMemoryCore(evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10)
        core.to_empty(device="cpu")
        core.reset_parameters()
        values = [param.detach().clone() for param in core.parameters()]
        assert all(torch.isfinite(value).all() for value in values)
        return values

    torch.manual_seed(7)
    first = _materialize()
    torch.manual_seed(7)
    second = _materialize()
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))


def test_continual_core_reset_reachable_via_runtime_reset_parameters() -> None:
    torch.manual_seed(0)
    core = ContinualTTTLocalMemoryCore(evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10)
    runtime = LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=4, hidden_dim=8),
        core,
    )
    before = [param.detach().clone() for param in core.parameters()]
    runtime.reset_parameters()
    after = [param.detach().clone() for param in core.parameters()]
    assert all(torch.isfinite(value).all() for value in after)
    # k_local=1: slot_queries are re-initialized to zero, matching __init__.
    assert torch.equal(core.slot_queries.detach(), torch.zeros_like(core.slot_queries))
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))


# ---------------------------------------------------------------------------
# OmniMoTModel._inject_local_history TTT seam
# ---------------------------------------------------------------------------


def _make_ttt_model(monkeypatch: pytest.MonkeyPatch, segment_steps: int = 2, *, local_ttt_enabled: bool = True):
    monkeypatch.setattr(omni_mot_model, "DEVICE", torch.device("cpu"))
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        local_history_horizon=2,
        local_history_state_enabled=False,
        local_ttt_enabled=local_ttt_enabled,
        local_history_evidence_dim=8,
        local_memory_dim=4,
        ttt_inner_lr=0.1,
        ttt_tbptt_steps=segment_steps,
        k_local=1,
    )
    torch.manual_seed(0)
    encoder = LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10, ttt_tbptt_steps=segment_steps
    )
    model.net = nn.Module()
    model.net.local_history_runtime = LocalHistoryRuntime(
        encoder, StatelessLocalReplayReadout(evidence_dim=8, local_dim=4, hidden_dim=8), core
    )
    model.net.local_memory2llm = nn.Linear(4, 5)
    model.net.local_memory_modality_embed = nn.Parameter(torch.zeros(5))
    return model


def _ttt_data_batch(
    batch: int = 2,
    *,
    masks: torch.Tensor | None = None,
    ends: list[bool] | None = None,
    episodes: list[int] | None = None,
) -> dict:
    return {
        "history_visual_summary": torch.randn(batch, 2, 4),
        "local_history_action": torch.randn(batch, 2, 3),
        "history_age_steps": torch.arange(2).repeat(batch, 1),
        "history_dt_s": torch.ones(batch, 2, 1),
        "history_mask": torch.ones(batch, 2, dtype=torch.bool) if masks is None else masks,
        "dataset_name": "libero_spatial",
        "episode_index": torch.tensor([0] * batch if episodes is None else episodes),
        "is_episode_end": torch.tensor([False] * batch if ends is None else ends),
    }


def test_model_ttt_seam_closes_segment_and_commits_via_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ttt_model(monkeypatch, segment_steps=2)
    plans = [SequencePlan(has_text=True), SequencePlan(has_text=True)]
    data_batch = _ttt_data_batch()
    model._inject_local_history(data_batch, plans)
    local_memory = data_batch["local_memory"]
    # Sample 0 is the intermediate window (detached), sample 1 closes the segment.
    assert local_memory[0] is not None and local_memory[0].grad_fn is None
    assert local_memory[1] is not None and local_memory[1].grad_fn is not None
    assert tuple(local_memory[0].shape) == (1, 4) and tuple(local_memory[1].shape) == (1, 4)
    assert [plan.has_local_memory for plan in plans] == [True, True]
    lifecycle = model._ttt_lifecycle
    assert isinstance(lifecycle, TTTLifecycle)
    # Drive the real callback seam: the closing loss reaches the witness graph.
    callback = TTTLifecycleCallback()
    loss = (
        model.net.local_memory2llm(local_memory[1]).square().mean()
        + model.net.local_memory_modality_embed.square().mean()
    )
    callback.on_before_backward(model, loss, iteration=0)
    loss.backward()
    callback.on_after_backward(model, iteration=0)
    authority = _authority(lifecycle)
    assert "libero_spatial#0" not in lifecycle._segments
    assert authority._last_timestep["libero_spatial#0"] == 1
    # Meta-gradient reached the encoder/core groups through the witness graph.
    assert all(param.grad is not None for param in authority.encoder.parameters())
    assert all(param.grad is not None for param in authority.core.parameters())


def test_model_ttt_seam_reuses_one_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ttt_model(monkeypatch, segment_steps=4)
    plans = [SequencePlan(has_text=True)]
    model._inject_local_history(_ttt_data_batch(batch=1), plans)
    first = model._ttt_lifecycle
    model._inject_local_history(_ttt_data_batch(batch=1), plans)
    assert model._ttt_lifecycle is first


def test_model_ttt_seam_requires_terminal_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ttt_model(monkeypatch, segment_steps=2)
    data_batch = _ttt_data_batch()
    del data_batch["is_episode_end"]
    with pytest.raises(ValueError, match="is_episode_end"):
        model._inject_local_history(data_batch, [SequencePlan(has_text=True), SequencePlan(has_text=True)])


def test_model_ttt_seam_zero_valid_sample_gets_no_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ttt_model(monkeypatch, segment_steps=4)
    masks = torch.tensor([[True, True], [False, False]])
    plans = [SequencePlan(has_text=True), SequencePlan(has_text=True)]
    data_batch = _ttt_data_batch(masks=masks)
    model._inject_local_history(data_batch, plans)
    local_memory = data_batch["local_memory"]
    assert local_memory[0] is not None and tuple(local_memory[0].shape) == (1, 4)
    assert local_memory[1] is None
    assert [plan.has_local_memory for plan in plans] == [True, False]


def test_model_disabled_parity_no_ttt_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """local_ttt_enabled=False: no lifecycle, no provenance requirement, runtime path."""
    model = _make_ttt_model(monkeypatch, segment_steps=2, local_ttt_enabled=False)
    # Replace the runtime with the plain R08 readout path (no TTT backend).
    torch.manual_seed(0)
    runtime = LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=4, hidden_dim=8),
    )
    model.net.local_history_runtime = runtime

    def _forbidden(*args, **kwargs):
        raise AssertionError("TTT lifecycle must not be built when local_ttt_enabled=False")

    monkeypatch.setattr(TTTLifecycle, "from_registered_modules", staticmethod(_forbidden))
    plans = [SequencePlan(has_text=True), SequencePlan(has_text=True)]
    data_batch = _ttt_data_batch()
    # Disabled path must not require terminal provenance keys.
    del data_batch["is_episode_end"]
    del data_batch["episode_index"]
    del data_batch["dataset_name"]
    model._inject_local_history(data_batch, plans)
    assert getattr(model, "_ttt_lifecycle", None) is None
    # Tokens are bitwise the plain runtime output.
    expected_tokens, expected_present, _ = runtime(
        history_visual_summary=data_batch["history_visual_summary"],
        local_history_action=data_batch["local_history_action"],
        history_age_steps=data_batch["history_age_steps"],
        history_dt_s=data_batch["history_dt_s"],
        history_mask=data_batch["history_mask"],
        history_state=None,
    )
    for index in range(2):
        if expected_present[index]:
            assert torch.equal(data_batch["local_memory"][index], expected_tokens[index])
        else:
            assert data_batch["local_memory"][index] is None
    assert [plan.has_local_memory for plan in plans] == [bool(value) for value in expected_present.tolist()]


# ---------------------------------------------------------------------------
# Recipe env wiring: PSM_R09_B_TTT_ENABLED
# ---------------------------------------------------------------------------


def _reload_recipe(monkeypatch: pytest.MonkeyPatch, envs: dict[str, str | None]):
    for name, value in envs.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    from cosmos_framework.configs.base.experiment.action.posttrain_config import action_policy_libero_edge_all as recipe

    return importlib.reload(recipe)


_BASE_ENVS = {
    "PSM_R08_LOCAL_HISTORY_ENABLED": "1",
    "PSM_LOCAL_DUMMY_ENABLED": "0",
    "PSM_R09_B1_TTT_ENABLED": None,
    "PSM_R09_A1_ENABLED": None,
    "PSM_R09_A1_PROBE_OUTPUT": None,
}


def test_recipe_r09_b_ttt_selects_four_slow_groups_and_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "1"})
    cfg = recipe._action_policy_libero_edge_model_config()
    assert cfg["local_ttt_enabled"] is True
    assert cfg["local_history_backend"] == "ttt_fast_weight"
    # Exact replacement with the contract selectors (four frozen slow groups).
    assert list(SELECTORS) == [
        "local_history_runtime.encoder",
        "local_history_runtime.recurrent_backend",
        "local_memory2llm",
        "local_memory_modality_embed",
    ]
    assert recipe.action_policy_libero_edge_all["optimizer"]["keys_to_select"] == list(SELECTORS)
    callbacks = recipe.action_policy_libero_edge_all["trainer"]["callbacks"]
    assert "r09_b_ttt_lifecycle" in callbacks
    target = callbacks["r09_b_ttt_lifecycle"]["_target_"]
    assert target is TTTLifecycleCallback or "TTTLifecycleCallback" in str(target)


def test_recipe_r09_b_ttt_selector_covers_continual_core_not_readout(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "1"})
    selected = recipe.action_policy_libero_edge_all["optimizer"]["keys_to_select"]
    torch.manual_seed(0)
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_history_runtime = LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=4, hidden_dim=8),
        ContinualTTTLocalMemoryCore(evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10),
    )
    model.net.local_memory2llm = nn.Linear(4, 5)
    model.net.local_memory_modality_embed = nn.Parameter(torch.zeros(5))
    model.unrelated_outer_module = nn.Linear(3, 3)
    selected_params = _build_params_with_metadata(
        model,
        keys_to_select=selected,
        lr_multipliers={},
        base_lr=1.0,
        disable_weight_decay_for_1d_params=False,
    )
    selected_ids = {id(param) for param, _ in selected_params}
    runtime = model.net.local_history_runtime
    expected_ids = {id(param) for param in runtime.encoder.parameters()}
    expected_ids |= {id(param) for param in runtime.recurrent_backend.parameters()}
    expected_ids |= {id(param) for param in model.net.local_memory2llm.parameters()}
    expected_ids.add(id(model.net.local_memory_modality_embed))
    assert expected_ids.issubset(selected_ids)
    assert not any(id(param) in selected_ids for param in runtime.readout.parameters())
    assert id(model.unrelated_outer_module.weight) not in selected_ids


def test_recipe_r09_b_ttt_requires_r08_history(monkeypatch: pytest.MonkeyPatch) -> None:
    # The recipe composes the model config at module level, so the reload itself raises.
    with pytest.raises(ValueError, match="requires PSM_R08_LOCAL_HISTORY_ENABLED=1"):
        _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R08_LOCAL_HISTORY_ENABLED": "0", "PSM_R09_B_TTT_ENABLED": "1"})


def test_recipe_r09_b_ttt_mutual_exclusion_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "1"})
    monkeypatch.setenv("PSM_R09_B1_TTT_ENABLED", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        recipe._action_policy_libero_edge_model_config()
    monkeypatch.delenv("PSM_R09_B1_TTT_ENABLED")
    monkeypatch.setenv("PSM_R09_A1_ENABLED", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        recipe._action_policy_libero_edge_model_config()
    monkeypatch.delenv("PSM_R09_A1_ENABLED")
    monkeypatch.setenv("PSM_R09_A1_PROBE_OUTPUT", "/tmp/a1.json")
    with pytest.raises(ValueError, match="mutually exclusive"):
        recipe._action_policy_libero_edge_model_config()


def test_recipe_r09_b_ttt_env_strict_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    # The recipe composes the model config at module level, so the reload itself raises.
    with pytest.raises(ValueError, match="PSM_R09_B_TTT_ENABLED must be 0 or 1"):
        _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "2"})


def test_recipe_disabled_parity_selectors_and_callbacks_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "0"})
    cfg = recipe._action_policy_libero_edge_model_config()
    assert cfg["local_ttt_enabled"] is False
    assert cfg["local_history_backend"] == "recurrent"
    selected = recipe.action_policy_libero_edge_all["optimizer"]["keys_to_select"]
    # Baseline R08 selection: the whole local_history_runtime group is appended,
    # never the exact four-group TTT replacement.
    assert "local_history_runtime" in selected
    assert selected != list(SELECTORS)
    assert "r09_b_ttt_lifecycle" not in recipe.action_policy_libero_edge_all["trainer"]["callbacks"]


def test_recipe_composed_config_diff_is_only_ttt_flag_and_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe_off = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "0"})
    cfg_off = recipe_off._action_policy_libero_edge_model_config()
    recipe_on = _reload_recipe(monkeypatch, {**_BASE_ENVS, "PSM_R09_B_TTT_ENABLED": "1"})
    cfg_on = recipe_on._action_policy_libero_edge_model_config()
    changed = {key for key in set(cfg_off) | set(cfg_on) if cfg_off.get(key) != cfg_on.get(key)}
    assert changed == {"local_ttt_enabled", "local_history_backend"}
    # The four TTT hyperparameters keep their frozen defaults via attrs; the
    # recipe never overrides them.
    for name in ("ttt_tbptt_steps", "ttt_inner_lr", "k_local", "runtime_evidence_steps"):
        assert (
            name not in cfg_on
            or cfg_on[name]
            == {
                "ttt_tbptt_steps": 16,
                "ttt_inner_lr": 0.1,
                "k_local": 1,
                "runtime_evidence_steps": 1,
            }[name]
        )


# ---------------------------------------------------------------------------
# Model config: five TTT fields, defaults, validators, mutual exclusion
# ---------------------------------------------------------------------------


def test_model_config_ttt_field_defaults() -> None:
    fields = {field.name: field for field in attrs.fields(OmniMoTModelConfig)}
    expected = {
        "local_ttt_enabled": False,
        "ttt_tbptt_steps": 16,
        "ttt_inner_lr": 0.1,
        "k_local": 1,
        "runtime_evidence_steps": 1,
    }
    for name, default in expected.items():
        assert name in fields
        assert fields[name].default == default


def test_model_config_ttt_validators() -> None:
    # attrs calls field validators as validator(instance, attribute, value).
    def _attr(name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name)

    _require_ttt_bool(None, _attr("local_ttt_enabled"), False)
    with pytest.raises(ValueError):
        _require_ttt_bool(None, _attr("local_ttt_enabled"), 0)
    _require_ttt_positive_int(None, _attr("ttt_tbptt_steps"), 16)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            _require_ttt_positive_int(None, _attr("ttt_tbptt_steps"), bad)
    _require_ttt_finite_positive_scalar(None, _attr("ttt_inner_lr"), 0.1)
    for bad in (0.0, -0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            _require_ttt_finite_positive_scalar(None, _attr("ttt_inner_lr"), bad)
    _require_ttt_k_local(None, _attr("k_local"), 1)
    for bad in (0, 2, True):
        with pytest.raises(ValueError):
            _require_ttt_k_local(None, _attr("k_local"), bad)
    _require_ttt_runtime_evidence_steps(None, _attr("runtime_evidence_steps"), 1)
    for bad in (0, 2, True):
        with pytest.raises(ValueError):
            _require_ttt_runtime_evidence_steps(None, _attr("runtime_evidence_steps"), bad)


def test_model_config_ttt_post_init_mutual_exclusion() -> None:
    OmniMoTModelConfig()
    with pytest.raises(ValueError, match="local_ttt_enabled requires"):
        OmniMoTModelConfig(local_ttt_enabled=True)
    with pytest.raises(ValueError, match="local_ttt_enabled requires"):
        OmniMoTModelConfig(local_ttt_enabled=True, local_history_enabled=True)
    config = OmniMoTModelConfig(
        local_ttt_enabled=True, local_history_enabled=True, local_history_backend="ttt_fast_weight"
    )
    assert config.local_ttt_enabled is True

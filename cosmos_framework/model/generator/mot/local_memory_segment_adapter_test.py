from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
    SegmentScanResult,
)
from cosmos_framework.trainer import ImaginaireTrainer


def _state() -> ContinualTTTFastState:
    return ContinualTTTFastState(*(torch.ones(1, 1) for _ in range(4)))


def test_sidecar_uses_canonical_cursor_and_terminal_reset() -> None:
    sidecar = LocalMemorySegmentSidecar()
    first = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    sidecar.commit(first, _state())
    carried = sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))
    assert carried is not None and not carried[0].requires_grad
    with pytest.raises(ValueError, match="canonical continuation"):
        sidecar.read(SegmentIdentity(2, "episode", "suite", 2, 2, "source"))
    with pytest.raises(ValueError, match="canonical continuation"):
        sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "other-source"))
    terminal = SegmentIdentity(2, "episode", "suite", 1, 1, "source", training_stream_end=True)
    sidecar.commit(terminal, _state())
    assert sidecar.read(SegmentIdentity(2, "replacement", "suite", 0, 2, "source")) is None


def test_adapter_scans_masked_segment_and_preserves_gather_identity() -> None:
    payload0, payload1, payload2 = object(), object(), object()
    segment = SegmentBatch(
        consumer_visual_summary=torch.randn(2, 3, 96),
        consumer_payload=((payload0, payload1, None), (payload2, None, None)),
        consumer_valid=torch.tensor([[True, True, False], [True, False, False]]),
        consumer_step=torch.tensor([[0, 1, -1], [0, -1, -1]]),
        evidence_visual_summary_prev=torch.randn(2, 3, 96),
        evidence_executed_action_prev=torch.randn(2, 3, 10),
        evidence_valid=torch.tensor([[False, True, False], [False, False, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1], [-1, -1, -1]]),
        slot_id=torch.tensor([2, 3]), episode_id=("episode", "other"), category=("suite", "suite"),
        segment_provenance=SegmentProvenance("m", "c", "source", 0),
    )
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(ttt_tbptt_steps=3)
    adapter = CanonicalLocalMemorySegmentAdapter(encoder, core, LocalMemorySegmentSidecar())
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    result = adapter.scan(segment, identity=identity)
    assert result.payloads == (payload0, payload1, payload2)
    assert result.locals[0] is None and result.locals[1] is not None
    assert result.identities == ((2, "episode", 0), (2, "episode", 1), (3, "other", 0))
    assert result.state_out.fast_in_weight.requires_grad
    with pytest.raises(TypeError):
        adapter.commit(identity, result)
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    assert scheduler.admit((identity,)) == identity
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 0),), (3,)), scheduler)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_local_memory_segment_backward(
        transaction.plan, 0, result.locals[1].sum(), torch.zeros((), requires_grad=True), 3,
        transaction=transaction, identity=identity, clear_slow_grads=lambda: None,
    )
    adapter.commit(identity, result, transaction=transaction)
    carried = adapter.sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))
    assert carried is not None and not carried.fast_in_weight.requires_grad


def test_adapter_rejects_grad_scaler_skip_sidecar_write() -> None:
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((identity,))
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 0),), (1,)), scheduler)
    transaction.grad_scaler_skip()
    state = _state()
    result = SegmentScanResult(torch.zeros(1, 1, 1, 32), torch.zeros(1, 1, dtype=torch.bool), state, (), (), ())
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar(),
    )
    with pytest.raises(RuntimeError, match="successful trainer transaction"):
        adapter.commit(identity, result, transaction=transaction)
    assert adapter.sidecar.read(SegmentIdentity(2, "replacement", "suite", 0, 1, "source")) is None


def test_adapter_rejects_terminal_failure_without_replacing_committed_carry() -> None:
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    failed = SegmentIdentity(2, "episode", "suite", 1, 1, "source")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((identity,))
    scheduler.admit((failed,))
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 1),), (1,)), scheduler)
    transaction.terminal_failure("LOCAL_MEM_OUTER_FAILURE")
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar(),
    )
    adapter.sidecar.commit(identity, _state())
    result = SegmentScanResult(torch.zeros(1, 1, 1, 32), torch.zeros(1, 1, dtype=torch.bool), _state(), (), (), ())
    with pytest.raises(RuntimeError, match="successful trainer transaction"):
        adapter.commit(failed, result, transaction=transaction)
    assert adapter.sidecar.read(failed) is not None
    with pytest.raises(ValueError, match="canonical continuation"):
        adapter.sidecar.read(SegmentIdentity(2, "episode", "suite", 2, 2, "source"))


def test_adapter_terminal_success_deletes_carry_after_trainer_seam() -> None:
    previous = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    terminal = SegmentIdentity(2, "episode", "suite", 1, 1, "source", training_stream_end=True)
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((previous,))
    assert scheduler.admit((terminal,)) == terminal
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 1),), (1,)), scheduler)
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar(),
    )
    adapter.sidecar.commit(previous, _state())
    state = _state()
    result = SegmentScanResult(torch.zeros(1, 1, 1, 32), torch.zeros(1, 1, dtype=torch.bool), state, (), (), ())
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_local_memory_segment_backward(
        transaction.plan, 0, torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True), 1,
        transaction=transaction, identity=terminal, clear_slow_grads=lambda: None,
    )
    adapter.commit(terminal, result, transaction=transaction)
    assert adapter.sidecar.read(SegmentIdentity(2, "replacement", "suite", 0, 2, "source")) is None

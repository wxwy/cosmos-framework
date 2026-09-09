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
from cosmos_framework.model.generator.mot.production_runtime_adapter import ProductionLocalMemoryRuntime
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


def test_committed_snapshot_is_detached_fp32_and_isolated() -> None:
    sidecar = LocalMemorySegmentSidecar()
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), sidecar,
    )
    sidecar.commit(identity, ContinualTTTFastState(*(torch.ones(1, 1, dtype=torch.float16) for _ in range(4))))
    _, snapshot = adapter.committed_snapshot()[0]
    assert snapshot[0].dtype is torch.float32
    snapshot[0].zero_()
    assert sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))[0].item() == 1


def test_adapter_scans_masked_segment_and_preserves_gather_identity() -> None:
    payload0, payload1, payload2 = object(), object(), object()
    segment = SegmentBatch(
        consumer_visual_summary=torch.randn(2, 3, 96),
        consumer_payload=((payload0, payload1, None), (payload2, None, None)),
        consumer_valid=torch.tensor([[True, True, False], [True, False, False]]),
        consumer_step=torch.tensor([[0, 1, -1], [0, -1, -1]]),
        evidence_visual_summary_prev=torch.tensor(float("nan")).expand(2, 3, 96).clone(),
        evidence_executed_action_prev=torch.tensor(float("nan")).expand(2, 3, 10).clone(),
        evidence_valid=torch.tensor([[False, True, False], [False, False, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1], [-1, -1, -1]]),
        slot_id=torch.tensor([2, 3]), episode_id=("episode", "other"), category=("suite", "suite"),
        segment_provenance=SegmentProvenance("m", "c", "source", 0),
    )
    segment.evidence_visual_summary_prev[0, 1] = torch.randn(96)
    segment.evidence_executed_action_prev[0, 1] = torch.randn(10)
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(ttt_tbptt_steps=3)
    adapter = CanonicalLocalMemorySegmentAdapter(encoder, core, LocalMemorySegmentSidecar())
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    assert scheduler.admit((identity,)) == identity
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 0),), (3,)), scheduler)
    with pytest.raises(TypeError):
        adapter.scan(segment, identity=identity)
    stale_result = adapter.scan(segment, identity=identity, transaction=transaction)
    result = adapter.scan(segment, identity=identity, transaction=transaction)
    assert result.payloads == (payload0, payload1, payload2)
    assert result.locals[0] is None and result.locals[1] is not None
    assert result.identities == ((2, "episode", 0), (2, "episode", 1), (3, "other", 0))
    calls: list[tuple[object, torch.Tensor | None]] = []

    def consumer_spy(payload: object, local: torch.Tensor | None) -> None:
        calls.append((payload, local))

    for payload, local in zip(result.payloads, result.locals, strict=True):
        consumer_spy(payload, local)
    assert calls == [(payload0, None), (payload1, result.locals[1]), (payload2, None)]
    assert result.state_out.fast_in_weight.requires_grad
    with pytest.raises(TypeError):
        adapter.commit(identity, result)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_local_memory_segment_backward(
        transaction.plan, 0, result.locals[1].sum(), torch.zeros((), requires_grad=True), 3,
        transaction=transaction, identity=identity, clear_slow_grads=lambda: None,
    )
    other_scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    assert other_scheduler.admit((identity,)) == identity
    other_transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 0),), (3,)), other_scheduler)
    other_transaction.successful_backward(0, identity, 3)
    with pytest.raises(RuntimeError, match="successful trainer transaction"):
        adapter.commit(identity, result, transaction=other_transaction)
    with pytest.raises(RuntimeError, match="successful trainer transaction"):
        adapter.commit(identity, stale_result, transaction=transaction)
    adapter.commit(identity, result, transaction=transaction)
    carried = adapter.sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))
    assert carried is not None and not carried.fast_in_weight.requires_grad


def test_adapter_rejects_grad_scaler_skip_sidecar_write() -> None:
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((identity,))
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 0),), (1,)), scheduler)
    core = ContinualTTTLocalMemoryCore()
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core, LocalMemorySegmentSidecar(),
    )
    result = adapter.scan(
        SegmentBatch(
            consumer_visual_summary=torch.zeros(1, 1, 96), consumer_payload=((object(),),),
            consumer_valid=torch.tensor([[True]]), consumer_step=torch.tensor([[0]]),
            evidence_visual_summary_prev=torch.full((1, 1, 96), float("nan")),
            evidence_executed_action_prev=torch.full((1, 1, 10), float("nan")),
            evidence_valid=torch.tensor([[False]]), evidence_source_step=torch.tensor([[-1]]),
            slot_id=torch.tensor([2]), episode_id=("episode",), category=("suite",),
            segment_provenance=SegmentProvenance("m", "c", "source", 0),
        ),
        identity=identity, transaction=transaction,
    )
    trainer = object.__new__(ImaginaireTrainer)
    with pytest.raises(RuntimeError, match="LOCAL_MEM_GRAD_SCALER_SKIP"):
        trainer._run_local_memory_segment_backward(
            transaction.plan, 0, torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True), 1,
            transaction=transaction, identity=identity, clear_slow_grads=lambda: None, grad_scaler_skip=True,
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
    core = ContinualTTTLocalMemoryCore()
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        core, LocalMemorySegmentSidecar(),
    )
    adapter.sidecar.commit(identity, core.initial_state(1, device=torch.device("cpu")))
    result = adapter.scan(
        SegmentBatch(torch.zeros(1, 1, 96), ((object(),),), torch.tensor([[True]]), torch.tensor([[0]]),
                     torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")),
                     torch.tensor([[False]]), torch.tensor([[-1]]), torch.tensor([2]), ("episode",), ("suite",),
                     SegmentProvenance("m", "c", "source", 1)),
        identity=failed, transaction=transaction,
    )
    with pytest.raises(RuntimeError, match="LOCAL_MEM_OUTER_FAILURE"):
        object.__new__(ImaginaireTrainer)._run_local_memory_segment_backward(
            transaction.plan, 0, torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True), 1,
            transaction=transaction, identity=failed, clear_slow_grads=lambda: None, failure_kind="OUTER",
        )
    with pytest.raises(RuntimeError, match="successful trainer transaction"):
        adapter.commit(failed, result, transaction=transaction)
    assert adapter.sidecar.read(failed) is not None
    with pytest.raises(ValueError, match="canonical continuation"):
        adapter.sidecar.read(SegmentIdentity(2, "episode", "suite", 2, 2, "source"))


def test_disabled_path_preserves_native_payload_loss_and_gradient() -> None:
    sample = torch.tensor([2.0], requires_grad=True)
    legacy_packed, legacy_loss = ProductionLocalMemoryRuntime.disabled_path(sample)
    native = sample.clone()
    native_loss = native.square().mean()
    torch.testing.assert_close(legacy_packed[0], native)
    assert legacy_packed[1] is None
    torch.testing.assert_close(legacy_loss, native_loss)
    legacy_loss.backward()
    torch.testing.assert_close(sample.grad, torch.tensor([4.0]))


def test_adapter_terminal_success_deletes_carry_after_trainer_seam() -> None:
    previous = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    terminal = SegmentIdentity(2, "episode", "suite", 1, 1, "source", training_stream_end=True)
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((previous,))
    assert scheduler.admit((terminal,)) == terminal
    transaction = LocalMemoryTransaction(GAWindowPlan(((2, "episode", 1),), (1,)), scheduler)
    core = ContinualTTTLocalMemoryCore()
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core, LocalMemorySegmentSidecar(),
    )
    adapter.sidecar.commit(previous, core.initial_state(1, device=torch.device("cpu")))
    result = adapter.scan(
        SegmentBatch(
            consumer_visual_summary=torch.zeros(1, 1, 96), consumer_payload=((object(),),),
            consumer_valid=torch.tensor([[True]]), consumer_step=torch.tensor([[0]]),
            evidence_visual_summary_prev=torch.zeros(1, 1, 96), evidence_executed_action_prev=torch.zeros(1, 1, 10),
            evidence_valid=torch.tensor([[False]]), evidence_source_step=torch.tensor([[-1]]),
            slot_id=torch.tensor([2]), episode_id=("episode",), category=("suite",),
            segment_provenance=SegmentProvenance("m", "c", "source", 1),
        ),
        identity=terminal, transaction=transaction,
    )
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_local_memory_segment_backward(
        transaction.plan, 0, torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True), 1,
        transaction=transaction, identity=terminal, clear_slow_grads=lambda: None,
    )
    adapter.commit(terminal, result, transaction=transaction)
    assert adapter.sidecar.read(SegmentIdentity(2, "replacement", "suite", 0, 2, "source")) is None

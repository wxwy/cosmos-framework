from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring


def _owner() -> CanonicalSegmentRuntimeOwner:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    adapter = CanonicalLocalMemorySegmentAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar())
    return CanonicalSegmentRuntimeOwner(scheduler, CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),)))


def _identity(slot: int = 0, cursor: int = 0, *, terminal: bool = False) -> SegmentIdentity:
    return SegmentIdentity(slot, f"episode-{slot}", "suite", cursor, cursor, "source", terminal)


def _segment(identity: SegmentIdentity) -> SegmentBatch:
    return SegmentBatch(
        torch.zeros(1, 1, 96), ((object(),),), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
        torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")),
        torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1), -1, dtype=torch.long), torch.tensor([identity.slot_id]),
        (identity.episode_id,), (identity.category,), SegmentProvenance("manifest", "config", identity.source_digest, identity.segment_id),
    )


def _prepare(owner: CanonicalSegmentRuntimeOwner, identity: SegmentIdentity, plan: GAWindowPlan):
    owner.admit((identity,))
    transaction = owner.begin(plan)
    return transaction, owner.prepare(_segment(identity))


def _commit(owner: CanonicalSegmentRuntimeOwner, transaction, forward, identity: SegmentIdentity, index: int = 0) -> None:
    transaction.successful_backward(index, identity, transaction.plan.planned_n_valid[index])
    owner.commit(transaction, forward)


def _finish(owner: CanonicalSegmentRuntimeOwner, transaction, *, scaler_skipped: bool = False) -> None:
    owner.resolve_local_memory_slow_window(owner.finish_window(transaction), scaler_skipped=scaler_skipped)


def test_public_commit_finish_and_snapshot_are_consistent() -> None:
    owner, identity = _owner(), _identity()
    plan = GAWindowPlan(((identity.slot_id, identity.episode_id, identity.cursor),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    _commit(owner, transaction, forward, identity)
    _finish(owner, transaction)
    snapshot = owner.snapshot()
    assert snapshot.scheduler["stable_slots"][identity.slot_id] is identity
    assert snapshot.committed[0][0] is identity


def test_public_two_member_continuation_rejects_wrong_candidate_without_mutation() -> None:
    owner, first, second = _owner(), _identity(), _identity(cursor=1)
    plan = GAWindowPlan(((0, first.episode_id, 0), (0, first.episode_id, 1)), (1, 1))
    transaction, forward = _prepare(owner, first, plan)
    _commit(owner, transaction, forward, first)
    before = owner.scheduler.snapshot()
    with pytest.raises(RuntimeError, match="frozen plan"):
        owner.admit_next((_identity(slot=1),))
    assert owner.scheduler.snapshot() == before
    assert owner.admit_next((second,)) is second
    forward = owner.prepare(_segment(second))
    _commit(owner, transaction, forward, second, 1)
    _finish(owner, transaction)
    assert len(owner.snapshot().committed) == 1


def test_public_scaler_skip_resume_keeps_exact_admission() -> None:
    owner, identity = _owner(), _identity()
    plan = GAWindowPlan(((0, identity.episode_id, 0),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    owner.abort_scaler_skip(transaction, forward)
    resumed = owner.resume_skipped()
    assert owner.identity is identity and resumed.plan is plan
    forward = owner.prepare(_segment(identity))
    _commit(owner, resumed, forward, identity)
    _finish(owner, resumed)


def test_public_retry_rejects_same_projection_replacement_identity() -> None:
    owner, identity = _owner(), _identity()
    plan = GAWindowPlan(((0, identity.episode_id, 0),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    retry = owner.abort_retry(transaction, forward)
    owner.scheduler.stable_slots[identity.slot_id] = SegmentIdentity(0, identity.episode_id, "suite", 0, 99, "other")
    with pytest.raises(RuntimeError, match="retained admitted identity"):
        owner.begin_retry(retry)


def test_public_begin_rejects_fabricated_attempt_one_without_mutation() -> None:
    owner, identity = _owner(), _identity()
    owner.admit((identity,))
    before = owner.scheduler.snapshot()
    fabricated = GAWindowPlan(((0, identity.episode_id, 0),), (1,), attempt=1)
    with pytest.raises(RuntimeError, match="attempt-0"):
        owner.begin(fabricated)
    assert owner.scheduler.snapshot() == before
    assert owner.identity is identity


def test_public_scaler_skip_rejects_attempt_one_and_later_member_without_mutation() -> None:
    owner, identity = _owner(), _identity()
    plan = GAWindowPlan(((0, identity.episode_id, 0),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    retry = owner.abort_retry(transaction, forward)
    transaction = owner.begin_retry(retry)
    forward = owner.prepare(_segment(identity))
    before_phase, before_identity = owner.phase, owner.identity
    before_scheduler, before_transaction = owner.scheduler.snapshot(), transaction.snapshot()
    before_pending, before_committed = owner.adapter.pending(), owner.adapter.committed_snapshot()
    with pytest.raises(RuntimeError, match="scaler skip"):
        owner.abort_scaler_skip(transaction, forward)
    assert owner.phase is before_phase and owner.identity is before_identity
    assert owner.scheduler.snapshot() == before_scheduler and transaction.snapshot() == before_transaction
    assert owner.adapter.pending() is before_pending
    assert tuple(identity for identity, _ in owner.adapter.committed_snapshot()) == tuple(identity for identity, _ in before_committed)

    owner, first, second = _owner(), _identity(), _identity(cursor=1)
    plan = GAWindowPlan(((0, first.episode_id, 0), (0, first.episode_id, 1)), (1, 1))
    transaction, forward = _prepare(owner, first, plan)
    _commit(owner, transaction, forward, first)
    owner.admit_next((second,))
    forward = owner.prepare(_segment(second))
    before_phase, before_identity = owner.phase, owner.identity
    before_scheduler, before_transaction = owner.scheduler.snapshot(), transaction.snapshot()
    before_pending, before_committed = owner.adapter.pending(), owner.adapter.committed_snapshot()
    with pytest.raises(RuntimeError, match="scaler skip"):
        owner.abort_scaler_skip(transaction, forward)
    assert owner.phase is before_phase and owner.identity is before_identity
    assert owner.scheduler.snapshot() == before_scheduler and transaction.snapshot() == before_transaction
    assert owner.adapter.pending() is before_pending
    assert tuple(identity for identity, _ in owner.adapter.committed_snapshot()) == tuple(identity for identity, _ in before_committed)


def test_public_retry_then_terminal_commit_has_no_sidecar_frontier() -> None:
    owner, identity = _owner(), _identity(terminal=True)
    plan = GAWindowPlan(((0, identity.episode_id, 0),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    retry = owner.abort_retry(transaction, forward)
    transaction = owner.begin_retry(retry)
    forward = owner.prepare(_segment(identity))
    _commit(owner, transaction, forward, identity)
    _finish(owner, transaction)
    snapshot = owner.snapshot()
    assert snapshot.committed == ()
    assert identity.slot_id in snapshot.scheduler["terminal_slots"]


def test_public_terminal_abort_discards_pending_without_commit() -> None:
    owner, identity = _owner(), _identity()
    plan = GAWindowPlan(((0, identity.episode_id, 0),), (1,))
    transaction, forward = _prepare(owner, identity, plan)
    owner.abort_terminal(transaction, forward, "synthetic-terminal")
    assert owner.adapter.pending() is None
    assert owner.adapter.committed_snapshot() == ()
    with pytest.raises(RuntimeError, match="idle committed frontier"):
        owner.snapshot()


def test_snapshot_rejects_admitted_uncommitted_residue_and_second_owner() -> None:
    owner = _owner()
    with pytest.raises(RuntimeError, match="exactly one runtime owner"):
        CanonicalSegmentRuntimeOwner(owner.scheduler, owner.wiring)
    owner.scheduler.admit((_identity(),))
    with pytest.raises(RuntimeError, match="admitted-but-uncommitted"):
        owner.snapshot()


def test_final_member_requires_exact_one_shot_slow_resolution() -> None:
    owner, identity = _owner(), _identity()
    transaction, forward = _prepare(owner, identity, GAWindowPlan(((0, identity.episode_id, 0),), (1,)))
    _commit(owner, transaction, forward, identity)
    capability = owner.finish_window(transaction)
    assert owner.phase.name == "SLOW_RESOLUTION_PENDING"
    for operation in (
        owner.snapshot,
        lambda: owner.admit((identity,)),
        lambda: owner.begin(transaction.plan),
        lambda: owner.prepare(_segment(identity)),
    ):
        with pytest.raises(RuntimeError):
            operation()
    with pytest.raises(RuntimeError, match="exact unconsumed"):
        owner.resolve_local_memory_slow_window(type(capability)(owner, transaction), scaler_skipped=False)
    owner.resolve_local_memory_slow_window(capability, scaler_skipped=False)
    assert transaction.snapshot().slow_optimizer_steps == transaction.snapshot().slow_lr_scheduler_steps == 1
    with pytest.raises(RuntimeError, match="exact unconsumed"):
        owner.resolve_local_memory_slow_window(capability, scaler_skipped=False)
    assert owner.snapshot().committed[0][0] is identity


def test_final_member_scaler_skip_clears_real_grads_and_preserves_fast_frontier() -> None:
    owner, identity = _owner(), _identity()
    transaction, forward = _prepare(owner, identity, GAWindowPlan(((0, identity.episode_id, 0),), (1,)))
    _commit(owner, transaction, forward, identity)
    parameter = owner.wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    owner.resolve_local_memory_slow_window(owner.finish_window(transaction), scaler_skipped=True)
    assert parameter.grad is None
    snapshot = transaction.snapshot()
    assert snapshot.slow_grads_cleared and snapshot.slow_optimizer_steps == snapshot.slow_lr_scheduler_steps == 0
    assert owner.snapshot().committed[0][0] is identity

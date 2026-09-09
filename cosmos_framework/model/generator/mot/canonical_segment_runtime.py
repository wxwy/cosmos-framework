"""CPU/static owner for the exact Local-Memory segment capability."""
from __future__ import annotations

from enum import Enum, auto

from .local_memory_segment import GAWindowPlan, LocalMemoryTransaction, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity
from .local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, SegmentScanResult
from .production_segment_wiring import CanonicalSegmentForward, CanonicalSegmentWiring


class RuntimePhase(Enum):
    IDLE = auto(); ADMITTED = auto(); MEMBER_READY = auto(); PREPARED = auto(); MEMBER_COMMITTED = auto(); RETRY_READY = auto(); SKIP_READY = auto(); ABORTED = auto()


class CanonicalSegmentRuntimeOwner:
    def __init__(self, scheduler: RankLocalSegmentScheduler, wiring: CanonicalSegmentWiring) -> None:
        self.scheduler, self.wiring = scheduler, wiring
        self.adapter: CanonicalLocalMemorySegmentAdapter = wiring.adapter
        self.phase = RuntimePhase.IDLE
        self.identity: SegmentIdentity | None = None
        self.transaction: LocalMemoryTransaction | None = None
        self.forward: CanonicalSegmentForward | None = None
        self._skipped_plan: GAWindowPlan | None = None
        self._skipped_identity: SegmentIdentity | None = None

    def admit(self, candidates: tuple[SegmentIdentity, ...]) -> SegmentIdentity:
        if self.phase is not RuntimePhase.IDLE: raise RuntimeError("runtime owner is not idle")
        self.identity = self.scheduler.admit(candidates); self.phase = RuntimePhase.ADMITTED
        return self.identity

    def begin(self, plan: GAWindowPlan) -> LocalMemoryTransaction:
        if self.phase is not RuntimePhase.ADMITTED or self.identity is None or plan.members[0] != (self.identity.slot_id, self.identity.episode_id, self.identity.cursor): raise RuntimeError("begin requires exact admitted first member")
        self.transaction = LocalMemoryTransaction(plan, self.scheduler); self.phase = RuntimePhase.MEMBER_READY
        return self.transaction

    def prepare(self, segment: SegmentBatch) -> CanonicalSegmentForward:
        if self.phase is not RuntimePhase.MEMBER_READY or self.identity is None or self.transaction is None: raise RuntimeError("prepare requires member-ready transaction")
        self.forward = self.wiring.prepare(segment, self.identity, self.transaction); self.phase = RuntimePhase.PREPARED
        return self.forward

    def abort_scaler_skip(self, transaction: LocalMemoryTransaction, forward: CanonicalSegmentForward) -> None:
        pending = self.adapter.pending()
        if (self.phase is not RuntimePhase.PREPARED or transaction is not self.transaction or forward is not self.forward or pending is None or pending[0] is not self.identity or pending[1] is not transaction or pending[2] is not forward.result or transaction.plan.attempt != 0 or transaction.completed_members):
            raise RuntimeError("scaler skip requires exact first-member prepared capability")
        self._skipped_identity, self._skipped_plan = self.identity, transaction.plan
        transaction.grad_scaler_skip(); self.adapter.discard_pending(self.identity, transaction, forward.result)
        self.transaction = None; self.forward = None; self.identity = None; self.phase = RuntimePhase.SKIP_READY

    def resume_skipped(self) -> LocalMemoryTransaction:
        identity, plan = self._skipped_identity, self._skipped_plan
        if (self.phase is not RuntimePhase.SKIP_READY or identity is None or plan is None or plan.attempt != 0 or plan.members[0] != (identity.slot_id, identity.episode_id, identity.cursor) or self.scheduler.stable_slots.get(identity.slot_id) is not identity or identity not in self.scheduler.admission_order or identity in self.scheduler.committed_identities or self.adapter.pending() is not None): raise RuntimeError("skip resume requires retained exact authority")
        self.identity, self.transaction = identity, LocalMemoryTransaction(plan, self.scheduler)
        self._skipped_identity = self._skipped_plan = None; self.phase = RuntimePhase.MEMBER_READY
        return self.transaction

"""CPU/static owner for the exact Local-Memory segment capability."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .local_memory_segment import GAWindowPlan, LocalMemoryTransaction, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity
from .local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, SegmentScanResult
from .production_segment_wiring import CanonicalSegmentForward, CanonicalSegmentWiring


class RuntimePhase(Enum):
    IDLE = auto(); ADMITTED = auto(); MEMBER_READY = auto(); PREPARED = auto(); MEMBER_COMMITTED = auto(); RETRY_READY = auto(); SKIP_READY = auto(); ABORTED = auto()

@dataclass(frozen=True)
class CanonicalRuntimeSnapshot:
    generation: int
    scheduler: dict[str, object]
    committed: tuple


class CanonicalSegmentRuntimeOwner:
    def __init__(self, scheduler: RankLocalSegmentScheduler, wiring: CanonicalSegmentWiring) -> None:
        if getattr(scheduler, "_canonical_runtime_owner", None) is not None or getattr(wiring, "_canonical_runtime_owner", None) is not None:
            raise RuntimeError("scheduler and wiring may bind exactly one runtime owner")
        self.scheduler, self.wiring = scheduler, wiring
        scheduler._canonical_runtime_owner = self
        wiring._canonical_runtime_owner = self
        self.adapter: CanonicalLocalMemorySegmentAdapter = wiring.adapter
        self.phase = RuntimePhase.IDLE
        self.identity: SegmentIdentity | None = None
        self.transaction: LocalMemoryTransaction | None = None
        self.forward: CanonicalSegmentForward | None = None
        self._skipped_plan: GAWindowPlan | None = None
        self._skipped_identity: SegmentIdentity | None = None
        self._retry_plan: GAWindowPlan | None = None
        self._retry_identity: SegmentIdentity | None = None
        self.generation = 0

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

    def commit(self, transaction: LocalMemoryTransaction, forward: CanonicalSegmentForward) -> None:
        if self.phase is not RuntimePhase.PREPARED or transaction is not self.transaction or forward is not self.forward or self.identity is None:
            raise RuntimeError("commit requires exact prepared capability")
        self.adapter.commit(self.identity, forward.result, transaction=transaction)
        self.forward = None
        self.phase = RuntimePhase.MEMBER_COMMITTED

    def admit_next(self, candidates: tuple[SegmentIdentity, ...]) -> SegmentIdentity:
        if self.phase is not RuntimePhase.MEMBER_COMMITTED or self.transaction is None:
            raise RuntimeError("next admission requires committed member")
        index = len(self.transaction.completed_members)
        if index >= self.transaction.plan.ga_effective:
            raise RuntimeError("all plan members are already committed")
        member = self.transaction.plan.members[index]
        eligible = tuple(item for item in candidates if (item.slot_id, item.episode_id, item.cursor) == member)
        if len(eligible) != 1:
            raise RuntimeError("next admission must follow the frozen plan")
        identity = self.scheduler.admit(eligible)
        self.identity = identity
        self.phase = RuntimePhase.MEMBER_READY
        return identity

    def finish_window(self, transaction: LocalMemoryTransaction) -> None:
        if self.phase is not RuntimePhase.MEMBER_COMMITTED or transaction is not self.transaction or self.adapter.pending() is not None:
            raise RuntimeError("finish requires committed transaction without pending scan")
        if len(transaction.completed_members) != transaction.plan.ga_effective:
            raise RuntimeError("finish requires all plan members")
        self.identity = self.transaction = self.forward = None
        self.phase = RuntimePhase.IDLE

    def snapshot(self) -> CanonicalRuntimeSnapshot:
        if (self.phase is not RuntimePhase.IDLE or self.adapter.pending() is not None or self.identity is not None
                or self.transaction is not None or self.forward is not None or self._skipped_plan is not None
                or self._skipped_identity is not None or self._retry_plan is not None or self._retry_identity is not None):
            raise RuntimeError("snapshot requires idle committed frontier")
        committed = self.adapter.committed_snapshot()
        committed_by_slot = {identity.slot_id: identity for identity in self.scheduler.committed_identities}
        if any(identity not in self.scheduler.committed_identities for identity in self.scheduler.admission_order):
            raise RuntimeError("snapshot has admitted-but-uncommitted authority")
        for identity, _ in committed:
            if (committed_by_slot.get(identity.slot_id) is not identity
                    or self.scheduler.stable_slots.get(identity.slot_id) is not identity):
                raise RuntimeError("snapshot committed frontier mismatch")
        if any(slot in {identity.slot_id for identity, _ in committed} for slot in self.scheduler.terminal_slots):
            raise RuntimeError("snapshot terminal slot retains sidecar state")
        return CanonicalRuntimeSnapshot(self.generation, self.scheduler.snapshot(), committed)

    def abort_terminal(self, transaction: LocalMemoryTransaction, forward: CanonicalSegmentForward, code: str) -> None:
        if self.phase is not RuntimePhase.PREPARED or transaction is not self.transaction or forward is not self.forward or self.identity is None:
            raise RuntimeError("terminal abort requires exact prepared capability")
        pending = self.adapter.pending()
        if pending is None or pending[0] is not self.identity or pending[1] is not transaction or pending[2] is not forward.result:
            raise RuntimeError("terminal abort requires exact pending capability")
        transaction.terminal_failure(code)
        self.adapter.discard_pending(self.identity, transaction, forward.result)
        self.forward = None
        self.phase = RuntimePhase.ABORTED

    def abort_retry(self, transaction: LocalMemoryTransaction, forward: CanonicalSegmentForward) -> GAWindowPlan:
        if self.phase is not RuntimePhase.PREPARED or transaction is not self.transaction or forward is not self.forward or self.identity is None:
            raise RuntimeError("retry abort requires exact prepared capability")
        pending = self.adapter.pending()
        if pending is None or pending[0] is not self.identity or pending[1] is not transaction or pending[2] is not forward.result:
            raise RuntimeError("retry abort requires exact pending capability")
        plan = transaction.recover_transient(len(transaction.completed_members))
        self.adapter.discard_pending(self.identity, transaction, forward.result)
        self._retry_identity = self.identity
        self.forward = self.transaction = self.identity = None
        self._retry_plan = plan
        self.phase = RuntimePhase.RETRY_READY
        return plan

    def begin_retry(self, plan: GAWindowPlan) -> LocalMemoryTransaction:
        retained = getattr(self, "_retry_plan", None)
        if self.phase is not RuntimePhase.RETRY_READY or plan is not retained:
            raise RuntimeError("retry requires the exact retained suffix plan")
        identity = getattr(self, "_retry_identity", None)
        if identity is None or plan.members[0] != (identity.slot_id, identity.episode_id, identity.cursor) or self.scheduler.stable_slots.get(identity.slot_id) is not identity or identity not in self.scheduler.admission_order or identity in self.scheduler.committed_identities:
            raise RuntimeError("retry requires retained admitted identity")
        self.identity, self.transaction = identity, LocalMemoryTransaction(plan, self.scheduler)
        self._retry_plan = None
        self._retry_identity = None
        self.phase = RuntimePhase.MEMBER_READY
        return self.transaction

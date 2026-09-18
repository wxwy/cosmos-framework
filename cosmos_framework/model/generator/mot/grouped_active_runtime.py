"""A2 active runtime: batched scan, one backward, atomic per-slot publication."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .canonical_segment_runtime import CanonicalSegmentRuntimeOwner, RuntimePhase
from .grouped_active_contract import (
    GroupedGAWindowPlan,
    GroupedLocalMemoryTransaction,
    GroupedPlanMember,
    dependency_waves,
    segment_fingerprint,
    take_rows,
)
from .local_evidence import ContinualTTTFastState
from .local_memory_segment import RankLocalSegmentScheduler
from .local_memory_segment_adapter import SegmentScanResult
from .production_active_wiring import (
    ActiveNativeBatchInputs,
    PreparedActiveMemberCapability,
    ProductionActiveWiringRegistry,
)
from .production_segment_wiring import CanonicalSegmentForward

_SCHEDULER_FIELDS = (
    "target_distribution",
    "cumulative_valid_consumer_exposure",
    "admission_order",
    "committed_identities",
    "stable_slots",
    "terminal_slots",
    "queue_seed",
    "queue_epoch",
    "queue_permutation",
    "segment_provenance",
)


@dataclass(frozen=True)
class PreparedActiveGroupCapability(PreparedActiveMemberCapability):
    identity: GroupedPlanMember


def _candidate_scheduler(current: RankLocalSegmentScheduler) -> RankLocalSegmentScheduler:
    candidate = RankLocalSegmentScheduler(rank=current.rank, target_distribution=current.target_distribution)
    for name in _SCHEDULER_FIELDS:
        value = getattr(current, name)
        if isinstance(value, (dict, list)):
            value = value.copy()
        setattr(candidate, name, value)
    return candidate


def _detached(state: ContinualTTTFastState) -> ContinualTTTFastState:
    return ContinualTTTFastState(*(value.detach().float().clone() for value in state))


class GroupedSegmentRuntimeOwner(CanonicalSegmentRuntimeOwner):
    """Scalar scheduler remains authoritative; a group stages a sequence of commits."""

    def __init__(self, scheduler, wiring) -> None:
        super().__init__(scheduler, wiring)
        self._group_pending = None
        self.last_dependency_wave_count = 0

    def admit(self, candidates):
        if self.phase is not RuntimePhase.IDLE or len(candidates) != 1:
            raise RuntimeError("A2 admission requires idle owner and one exact group")
        if not isinstance(candidates[0], GroupedPlanMember):
            raise TypeError("A2 admission requires grouped row authority")
        self.identity = candidates[0]
        self.phase = RuntimePhase.ADMITTED
        return self.identity

    def begin(self, plan):
        if (
            self.phase is not RuntimePhase.ADMITTED
            or not isinstance(plan, GroupedGAWindowPlan)
            or plan.attempt != 0
            or plan.members[0] is not self.identity
        ):
            raise RuntimeError("A2 begin requires the exact initial plan/group")
        self.transaction = GroupedLocalMemoryTransaction(plan, self.scheduler)
        self.phase = RuntimePhase.MEMBER_READY
        return self.transaction

    def admit_next(self, candidates):
        if self.phase is not RuntimePhase.MEMBER_COMMITTED or self.transaction is None:
            raise RuntimeError("A2 continuation requires committed predecessor")
        index = len(self.transaction.completed_members)
        if (
            len(candidates) != 1
            or index >= self.transaction.plan.ga_effective
            or candidates[0] is not self.transaction.plan.members[index]
        ):
            raise RuntimeError("A2 continuation differs from frozen group order")
        self.identity = candidates[0]
        self.phase = RuntimePhase.MEMBER_READY
        return self.identity

    def _stage_scheduler(self, group):
        candidate = _candidate_scheduler(self.scheduler)
        for identity, count in zip(group.row_identities, group.row_planned_n_valid, strict=True):
            terminal = candidate.terminal_slots.get(identity.slot_id)
            if terminal is not None and identity.cursor == 0:
                candidate.terminal_rebind(terminal)
            candidate.admit((identity,))
            candidate.commit(identity, count)
        return candidate

    def _scan_group(self, segment, group):
        encoder, core = self.adapter.encoder, self.adapter.core
        device = next(encoder.parameters()).device
        records = self.adapter.sidecar._records.copy()
        size = len(group.row_identities)
        tokens, presents, states = [None] * size, [None] * size, [None] * size
        waves = dependency_waves(group.row_identities)
        for indices in waves:
            wave = take_rows(segment, indices)
            fresh = core.initial_state(len(indices), device=device, dtype=torch.float32)
            priors = []
            for index in indices:
                identity = group.row_identities[index]
                record = records.get(identity.slot_id)
                if identity.cursor == 0:
                    if record is not None:
                        raise ValueError("fresh A2 row retains previous episode fast state")
                    priors.append(None)
                else:
                    if record is None:
                        raise ValueError("continued A2 row lacks committed fast state")
                    previous, state = record
                    if (
                        previous.episode_id != identity.episode_id
                        or previous.category != identity.category
                        or previous.source_digest != identity.source_digest
                        or previous.cursor + 1 != identity.cursor
                    ):
                        raise ValueError("A2 row is not an exact sidecar continuation")
                    priors.append(state)
            state_in = ContinualTTTFastState(
                *(
                    torch.stack(
                        [
                            base[row] if prior is None else prior[field][0].to(device=device, dtype=torch.float32)
                            for row, prior in enumerate(priors)
                        ]
                    )
                    for field, base in enumerate(fresh)
                )
            )
            local, state_out, present = core.scan_segment_masked_encoded_many(
                encoder,
                wave.evidence_visual_summary_prev.to(device),
                wave.evidence_executed_action_prev.to(device),
                wave.evidence_valid.to(device),
                state_in,
                create_graph=True,
            )
            for row, index in enumerate(indices):
                identity = group.row_identities[index]
                row_state = ContinualTTTFastState(*(value[row : row + 1] for value in state_out))
                tokens[index], presents[index], states[index] = local[row], present[row], row_state
                if identity.training_stream_end:
                    records.pop(identity.slot_id, None)
                else:
                    records[identity.slot_id] = (identity, _detached(row_state))
        local_tokens, local_present = torch.stack(tokens), torch.stack(presents)
        state_out = ContinualTTTFastState(*(torch.cat([state[field] for state in states]) for field in range(4)))
        payloads, locals_, identities = segment.gather_consumers(local_tokens, local_present)
        result = SegmentScanResult(
            local_tokens, local_present, state_out, tuple(payloads), tuple(locals_), tuple(identities)
        )
        self.last_dependency_wave_count = len(waves)
        return result, records

    def prepare(self, segment):
        if self.phase is not RuntimePhase.MEMBER_READY or self._group_pending is not None:
            raise RuntimeError("A2 prepare requires one unprepared admitted group")
        group = self.identity
        try:
            group.validate_batch(segment)
            candidate = self._stage_scheduler(group)
            result, records = self._scan_group(segment, group)
            if any(not bool(torch.isfinite(value).all()) for value in result.state_out):
                raise RuntimeError("LOCAL_MEM_NUMERICAL_FAILURE")
            forward = CanonicalSegmentForward(self.wiring, result, result.payloads, result.locals, result.identities)
        except Exception:
            self.wiring.clear_local_slow_grads()
            self.transaction.terminal_failure("LOCAL_MEM_GROUP_PREPARE_FAILURE")
            self.phase = RuntimePhase.ABORTED
            raise
        self._group_pending = (group, self.transaction, forward, candidate, records)
        self.forward, self.phase = forward, RuntimePhase.PREPARED
        return forward

    def commit(self, transaction, forward) -> None:
        pending = self._group_pending
        if (
            self.phase is not RuntimePhase.PREPARED
            or pending is None
            or pending[0] is not self.identity
            or pending[1] is not transaction
            or pending[2] is not forward
            or transaction is not self.transaction
            or not transaction.completed_members
            or transaction.completed_members[-1] is not self.identity
        ):
            raise RuntimeError("A2 commit requires exact successful group backward")
        candidate, records = pending[3:]
        for name in _SCHEDULER_FIELDS:
            setattr(self.scheduler, name, getattr(candidate, name))
        self.adapter.sidecar._records = records
        self._group_pending = self.forward = None
        self.phase = RuntimePhase.MEMBER_COMMITTED

    def abort_terminal(self, transaction, forward, code) -> None:
        if (
            self.phase is not RuntimePhase.PREPARED
            or transaction is not self.transaction
            or forward is not self.forward
        ):
            raise RuntimeError("A2 abort requires exact prepared group")
        self.wiring.clear_local_slow_grads()
        transaction.terminal_failure(code)
        self._group_pending = self.forward = None
        self.phase = RuntimePhase.ABORTED

    def abort_retry(self, transaction, forward):
        if (
            self.phase is not RuntimePhase.PREPARED
            or transaction is not self.transaction
            or forward is not self.forward
            or transaction.completed_members
            or transaction.plan.attempt
        ):
            raise RuntimeError("A2 retry requires exact first attempt first group")
        self.wiring.clear_local_slow_grads()
        plan = transaction.recover_transient(0)
        self._retry_plan, self._retry_identity = plan, self.identity
        self._group_pending = self.forward = self.transaction = self.identity = None
        self.phase = RuntimePhase.RETRY_READY
        return plan

    def begin_retry(self, plan):
        if self.phase is not RuntimePhase.RETRY_READY or plan is not self._retry_plan:
            raise RuntimeError("A2 retry plan is stale or foreign")
        if plan.members[0] is not self._retry_identity:
            raise RuntimeError("A2 retry lost its exact first-group identity")
        self.identity = self._retry_identity
        self.transaction = GroupedLocalMemoryTransaction(plan, self.scheduler)
        self._retry_plan = self._retry_identity = None
        self.phase = RuntimePhase.MEMBER_READY
        return self.transaction


class GroupedActiveWiringRegistry(ProductionActiveWiringRegistry):
    """Reuse the existing one-shot trainer/model capability protocol for A2."""

    def prepare_initial(self, identity, segment, plan, *, trainer_grad_accum_iter):
        if (
            trainer_grad_accum_iter != 0
            or not isinstance(plan, GroupedGAWindowPlan)
            or plan.attempt != 0
            or plan.members[0] is not identity
        ):
            raise RuntimeError("A2 initial member requires exact group and native GA counter zero")
        if self.owner.phase is not RuntimePhase.IDLE or self._prepared is not None or self._published is not None:
            raise RuntimeError("A2 registry is not idle")
        self.owner.admit((identity,))
        transaction = self.owner.begin(plan)
        return self._prepare(identity, segment, transaction, 0)

    def _prepare(self, identity, segment, transaction, member_index):
        if member_index >= transaction.plan.ga_effective or transaction.plan.members[member_index] is not identity:
            raise RuntimeError("A2 prepared member differs from frozen plan")
        if member_index == 0:
            fingerprint = segment_fingerprint(segment)
            if transaction.plan.attempt:
                if segment is not getattr(self, "_first_segment", None) or fingerprint != getattr(
                    self, "_first_fingerprint", None
                ):
                    self.owner.wiring.clear_local_slow_grads()
                    transaction.terminal_failure("LOCAL_MEM_RETRY_SOURCE_CHANGED")
                    self.owner.phase = RuntimePhase.ABORTED
                    raise RuntimeError("LOCAL_MEM_RETRY_SOURCE_CHANGED")
            else:
                self._first_segment, self._first_fingerprint = segment, fingerprint
        forward = self.owner.prepare(segment)
        try:
            inputs = ActiveNativeBatchInputs(forward.payloads, forward.locals, forward.identities)
            if len(inputs.payloads) != transaction.plan.planned_n_valid[member_index]:
                raise RuntimeError("A2 native payload count differs from plan")
        except Exception:
            self.owner.abort_terminal(transaction, forward, "LOCAL_MEM_IDENTITY_CONTRACT_FAILURE")
            raise
        if self._ga_window_token is None:
            self._ga_window_token = object()
        prepared = PreparedActiveGroupCapability(
            self,
            self.owner,
            transaction,
            identity,
            segment,
            member_index,
            forward,
            inputs,
            len(inputs.payloads),
            self._ga_window_token,
        )
        self._prepared = prepared
        return prepared

"""CPU/static production-ABI contracts for canonical Local-Memory segments."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    CanonicalSegmentContractError,
    MicrobatchPlanMember,
    NativeConsumerBatch,
    PreparedCanonicalReconcile,
)
from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .local_memory_segment import SegmentBatch


@dataclass(frozen=True)
class CanonicalProductionSegmentRequest:
    scheduler: CanonicalBatchScheduler
    plan: CanonicalGAWindowPlan
    transaction: CanonicalBatchWindowTransaction
    member: MicrobatchPlanMember
    member_index: int
    segment_batch: SegmentBatch


@dataclass(frozen=True)
class CanonicalProductionScanResult:
    local_tokens: torch.Tensor
    local_present: torch.Tensor
    candidate_state_out: ContinualTTTFastState
    gathered: NativeConsumerBatch


@dataclass(frozen=True)
class CanonicalProductionCommitCapability:
    request: CanonicalProductionSegmentRequest
    result: CanonicalProductionScanResult
    prepared_reconcile: PreparedCanonicalReconcile


def _fp32_clone(state: ContinualTTTFastState, *, detach: bool) -> ContinualTTTFastState:
    values = (value.detach() if detach else value for value in state)
    return ContinualTTTFastState(*(value.to(dtype=torch.float32).clone() for value in values))


class CanonicalProductionFastStateFrontier:
    """In-memory, slot-bound fp32 fast states; intentionally no runtime sidecar."""

    def __init__(self, core: ContinualTTTLocalMemoryCore) -> None:
        self._core = core
        self._states: dict[tuple[int, str, str, int], ContinualTTTFastState] = {}

    def state_for(self, member: MicrobatchPlanMember) -> ContinualTTTFastState:
        rows = []
        for identity in member.row_identities:
            key = (identity.slot_id, identity.episode_id, identity.source_digest, identity.cursor)
            if identity.cursor == 0:
                rows.append(None)
            else:
                previous = self._states.get((identity.slot_id, identity.episode_id, identity.source_digest, identity.cursor - 1))
                if previous is None:
                    raise CanonicalSegmentContractError("continuation lacks exact committed fast state")
                rows.append(previous)
        fresh = self._core.initial_state(len(rows), dtype=torch.float32)
        values = []
        for index in range(4):
            source = getattr(fresh, fresh._fields[index])
            selected = []
            for row, prior in enumerate(rows):
                selected.append(source[row] if prior is None else getattr(prior, prior._fields[index])[0])
            values.append(torch.stack(selected))
        return ContinualTTTFastState(*values)

    def commit(self, member: MicrobatchPlanMember, state: ContinualTTTFastState) -> None:
        for value in state:
            if value.dtype is not torch.float32:
                raise CanonicalSegmentContractError("fast-state commit requires fp32")
        for row, identity in enumerate(member.row_identities):
            key = (identity.slot_id, identity.episode_id, identity.source_digest, identity.cursor)
            if identity.training_stream_end:
                self._states.pop(key, None)
            else:
                self._states[key] = ContinualTTTFastState(*(value[row : row + 1].detach().clone() for value in state))


class CanonicalProductionAdapter:
    """Own the canonical encoded scan and derive gather/count from its result."""

    def __init__(self, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        self.encoder = encoder
        self.core = core
        self.frontier = CanonicalProductionFastStateFrontier(core)

    def scan(self, request: CanonicalProductionSegmentRequest) -> CanonicalProductionScanResult:
        if (
            request.transaction.plan is not request.plan
            or request.member_index < 0
            or request.member_index >= len(request.plan.members)
            or request.plan.members[request.member_index] is not request.member
        ):
            raise CanonicalSegmentContractError("canonical request identity is invalid")
        request.member.validate_batch(request.segment_batch)
        state_in = self.frontier.state_for(request.member)
        tokens, state_out, present = self.core.scan_segment_masked_encoded_many(
            self.encoder,
            request.segment_batch.evidence_visual_summary_prev,
            request.segment_batch.evidence_executed_action_prev,
            request.segment_batch.evidence_valid,
            state_in,
            create_graph=True,
        )
        gathered = NativeConsumerBatch.from_segment(request.segment_batch, request.member, tokens, present)
        return CanonicalProductionScanResult(tokens, present, state_out, gathered)

    def prepare_commit(
        self, request: CanonicalProductionSegmentRequest, result: CanonicalProductionScanResult
    ) -> CanonicalProductionCommitCapability:
        if result.gathered.item_count != request.member.planned_n_valid:
            raise CanonicalSegmentContractError("adapter gathered count differs from frozen member")
        prepared = request.scheduler.prepare_reconcile_after_backward(request.member, result.gathered.item_count)
        return CanonicalProductionCommitCapability(request, result, prepared)

    def commit_success(self, capability: CanonicalProductionCommitCapability) -> None:
        request, result = capability.request, capability.result
        if capability.prepared_reconcile.scheduler is not request.scheduler:
            raise CanonicalSegmentContractError("commit capability scheduler is foreign")
        if any(value.dtype is not torch.float32 for value in result.candidate_state_out):
            raise CanonicalSegmentContractError("commit candidate fast state is not fp32")
        self.frontier.commit(request.member, result.candidate_state_out)
        request.scheduler.consume_prepared_reconcile(capability.prepared_reconcile)
        request.transaction.mark_reconciled(request.member_index)

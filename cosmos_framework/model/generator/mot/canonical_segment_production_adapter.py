"""CPU/static production-ABI contracts for canonical Local-Memory segments."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

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
from .local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
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
class CanonicalExpectedTraversal:
    identities: tuple[tuple[int, str, int], ...]
    logical_indexes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CanonicalRawRowCarrier:
    """Immutable nested raw source; flat views are derived, never stored."""

    request: CanonicalProductionSegmentRequest
    member: MicrobatchPlanMember
    segment_batch: SegmentBatch
    row_identities: tuple[Any, ...]
    row_chronology: tuple[Any, ...]
    raw_rows: tuple[tuple[Mapping[str, Any] | None, ...], ...]
    row_model_samples: tuple[tuple[Mapping[str, Any] | None, ...], ...]
    model_data_batch: Mapping[str, Any]
    stacked_model_batch_sources: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    raw_row_source_identities: tuple[tuple[tuple[int, str, str, int] | None, ...], ...] = ()
    row_model_source_rows: tuple[tuple[Mapping[str, Any] | None, ...], ...] = ()

    _MODEL_BATCH_KEYS = frozenset(
        {
            "text_token_ids", "video_latent", "verify_cached_latent", "image_size",
            "enable_per_camera_vae_encoding", "sample_n_views", "num_video_frames_per_view", "action",
            "domain_id", "raw_action_dim", "sound", "conditioning_fps", "conditioning_fps_action",
            "control_weights", "num_vision_items_per_sample", "is_preprocessed", "sequence_plan",
        }
    )

    def expected_for(self, request: CanonicalProductionSegmentRequest) -> CanonicalExpectedTraversal:
        if (
            self.request is not request
            or self.member is not request.member
            or self.segment_batch is not request.segment_batch
            or self.row_identities is not request.member.row_identities
            or self.row_chronology is not request.member.row_chronology
        ):
            raise CanonicalSegmentContractError("canonical carrier authority is foreign")
        if (
            request.transaction.plan is not request.plan
            or request.member_index < 0
            or request.member_index >= len(request.plan.members)
            or request.plan.members[request.member_index] is not request.member
        ):
            raise CanonicalSegmentContractError("canonical request identity is invalid")
        request.member.validate_batch(request.segment_batch)
        valid = request.segment_batch.consumer_valid
        batch, steps = valid.shape
        if len(self.raw_rows) != batch or len(self.row_model_samples) != batch:
            raise CanonicalSegmentContractError("canonical carrier outer batch differs from segment")
        identities: list[tuple[int, str, int]] = []
        indexes: list[tuple[int, int]] = []
        for row in range(batch):
            if len(self.raw_rows[row]) != steps or len(self.row_model_samples[row]) != steps:
                raise CanonicalSegmentContractError("canonical carrier inner width differs from segment")
            for index in range(steps):
                raw, sample = self.raw_rows[row][index], self.row_model_samples[row][index]
                if not bool(valid[row, index]):
                    if raw is not None or sample is not None:
                        raise CanonicalSegmentContractError("canonical carrier PAD entries must be absent")
                    continue
                if raw is None or sample is None:
                    raise CanonicalSegmentContractError("canonical carrier valid entries must be present")
                if raw.get("canonical_identity") != (
                    int(request.segment_batch.slot_id[row]),
                    request.segment_batch.episode_id[row],
                    int(request.segment_batch.consumer_step[row, index]),
                ):
                    raise CanonicalSegmentContractError("canonical carrier raw identity is foreign")
                if sample.get("canonical_identity") != raw.get("canonical_identity"):
                    raise CanonicalSegmentContractError("canonical carrier model sample identity is foreign")
                identities.append(raw["canonical_identity"])
                indexes.append((row, index))
        expected = CanonicalExpectedTraversal(tuple(identities), tuple(indexes))
        if len(expected.identities) != request.member.planned_n_valid:
            raise CanonicalSegmentContractError("canonical carrier expected count differs from frozen member")
        return expected

    def validate_model_data_batch(
        self, expected: CanonicalExpectedTraversal, *, input_image_key: str, input_video_key: str
    ) -> None:
        allowed = self._MODEL_BATCH_KEYS | {input_image_key, input_video_key}
        unexpected = set(self.model_data_batch) - allowed
        if unexpected:
            raise CanonicalSegmentContractError("canonical carrier model batch has an unexpected key")
        image_present = input_image_key in self.model_data_batch
        video_present = input_video_key in self.model_data_batch
        if image_present == video_present:
            raise CanonicalSegmentContractError("canonical carrier model batch requires exactly one vision input key")
        if set(self.stacked_model_batch_sources) - set(self.model_data_batch):
            raise CanonicalSegmentContractError("canonical carrier stacked source key is foreign")
        if (
            len(self.raw_row_source_identities) != len(self.raw_rows)
            or len(self.row_model_source_rows) != len(self.raw_rows)
        ):
            raise CanonicalSegmentContractError("canonical carrier raw source metadata is foreign")
        for key, value in self.model_data_batch.items():
            sources = tuple(
                self.row_model_samples[row][index] for row, index in expected.logical_indexes
            )
            for source, (row, index) in zip(sources, expected.logical_indexes, strict=True):
                raw = self.raw_rows[row][index]
                source_identity = self.raw_row_source_identities[row][index]
                source_raw = self.row_model_source_rows[row][index]
                expected_identity = (
                    int(self.segment_batch.slot_id[row]),
                    self.segment_batch.episode_id[row],
                    self.member.row_identities[row].source_digest,
                    int(self.segment_batch.consumer_step[row, index]),
                )
                if (
                    raw is None
                    or source is None
                    or source_raw is not raw
                    or source_identity != expected_identity
                ):
                    raise CanonicalSegmentContractError("canonical carrier model sample source is foreign")
            if isinstance(value, (list, tuple)):
                if len(value) != len(sources):
                    raise CanonicalSegmentContractError("canonical carrier model batch cardinality is foreign")
                for value_item, source in zip(value, sources, strict=True):
                    if source is None or source.get(key) is not value_item:
                        raise CanonicalSegmentContractError("canonical carrier model batch source is foreign")
                continue
            if not isinstance(value, torch.Tensor):
                raise CanonicalSegmentContractError("canonical carrier model batch has an unsupported source form")
            source_items = self.stacked_model_batch_sources.get(key)
            if source_items is None or len(source_items) != len(sources):
                raise CanonicalSegmentContractError("canonical carrier stacked source cardinality is foreign")
            for source_item, source in zip(source_items, sources, strict=True):
                if source is None or source.get(key) is not source_item or not isinstance(source_item, torch.Tensor):
                    raise CanonicalSegmentContractError("canonical carrier stacked source is foreign")
            expected_tensor = torch.stack(source_items)
            if (
                value.ndim == 0
                or value.shape != expected_tensor.shape
                or value.dtype != expected_tensor.dtype
                or value.device != expected_tensor.device
                or value.shape[0] != len(expected.logical_indexes)
                or not torch.equal(value, expected_tensor)
            ):
                raise CanonicalSegmentContractError("canonical carrier stacked model batch is foreign")

    def preflight(
        self, request: CanonicalProductionSegmentRequest, *, input_image_key: str, input_video_key: str
    ) -> CanonicalExpectedTraversal:
        expected = self.expected_for(request)
        self.validate_model_data_batch(
            expected, input_image_key=input_image_key, input_video_key=input_video_key
        )
        return expected


@dataclass(frozen=True)
class CanonicalProductionScanResult:
    local_tokens: torch.Tensor
    local_present: torch.Tensor
    candidate_state_out: ContinualTTTFastState
    gathered: NativeConsumerBatch
    slot_chain: tuple[tuple[int, str, str, int], ...]


@dataclass(frozen=True)
class CanonicalProductionCommitCapability:
    request: CanonicalProductionSegmentRequest
    result: CanonicalProductionScanResult
    prepared_reconcile: PreparedCanonicalReconcile


@dataclass(frozen=True)
class CanonicalProductionRetryCapability:
    """One-shot typed lineage from the exact attempt-0 request to attempt-1."""

    original_request: CanonicalProductionSegmentRequest
    original_plan: CanonicalGAWindowPlan
    original_transaction: CanonicalBatchWindowTransaction
    retry_request: CanonicalProductionSegmentRequest
    retry_plan: CanonicalGAWindowPlan
    retry_transaction: CanonicalBatchWindowTransaction


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
                for prior_key in tuple(self._states):
                    if prior_key[:3] == key[:3]:
                        self._states.pop(prior_key)
            else:
                self._states[key] = ContinualTTTFastState(*(value[row : row + 1].detach().clone() for value in state))


class CanonicalProductionAdapter:
    """Own the canonical encoded scan and derive gather/count from its result."""

    def __init__(self, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        if encoder.feature_config is not CANONICAL_EVIDENCE_FEATURE_CONFIG:
            raise CanonicalSegmentContractError("canonical adapter requires the canonical evidence feature config")
        self.encoder = encoder
        self.core = core
        self.frontier = CanonicalProductionFastStateFrontier(core)
        self._scan_requests: set[int] = set()
        self._scan_results: dict[int, CanonicalProductionSegmentRequest] = {}
        self._commit_capabilities: set[int] = set()
        self._retry_capabilities: set[int] = set()

    def retry_first_member_pre_backward(
        self, request: CanonicalProductionSegmentRequest
    ) -> CanonicalProductionRetryCapability:
        """Mint the only permitted attempt-1 request without admitting again."""
        if (
            request.plan.attempt != 0
            or request.member_index != 0
            or request.member is not request.plan.members[0]
            or request.transaction.plan is not request.plan
            or id(request) in self._scan_requests
        ):
            raise CanonicalSegmentContractError("retry requires the exact unscanned attempt-0 first member")
        request.scheduler.prepare_reconcile_after_backward(request.member, request.member.planned_n_valid)
        retry_plan = request.transaction.retry_first_member_pre_backward()
        retry_transaction = CanonicalBatchWindowTransaction(retry_plan)
        retry_request = replace(request, plan=retry_plan, transaction=retry_transaction)
        capability = CanonicalProductionRetryCapability(
            request,
            request.plan,
            request.transaction,
            retry_request,
            retry_plan,
            retry_transaction,
        )
        self._retry_capabilities.add(id(capability))
        return capability

    def consume_retry(self, capability: CanonicalProductionRetryCapability) -> CanonicalProductionSegmentRequest:
        """Return the exact attempt-1 request once, preserving original admission."""
        if id(capability) not in self._retry_capabilities:
            raise CanonicalSegmentContractError("retry capability is foreign or already consumed")
        original = capability.original_request
        retry = capability.retry_request
        if (
            original.plan is not capability.original_plan
            or original.transaction is not capability.original_transaction
            or original.plan.attempt != 0
            or original.member_index != 0
            or original.member is not original.plan.members[0]
            or retry.plan is not capability.retry_plan
            or retry.transaction is not capability.retry_transaction
            or retry.plan.attempt != 1
            or retry.plan.members != original.plan.members
            or retry.plan.members[0] is not original.member
            or retry.plan.original_n_valid_window != original.plan.original_n_valid_window
            or retry.plan.original_ga_effective != original.plan.original_ga_effective
            or retry.plan.plan_chain_id != original.plan.plan_chain_id
            or retry.scheduler is not original.scheduler
            or retry.member is not original.member
            or retry.member_index != 0
            or retry.transaction.plan is not retry.plan
        ):
            raise CanonicalSegmentContractError("retry capability lineage is foreign or stale")
        self._retry_capabilities.remove(id(capability))
        return retry

    def scan(self, request: CanonicalProductionSegmentRequest) -> CanonicalProductionScanResult:
        if (
            request.transaction.plan is not request.plan
            or request.member_index < 0
            or request.member_index >= len(request.plan.members)
            or request.plan.members[request.member_index] is not request.member
        ):
            raise CanonicalSegmentContractError("canonical request identity is invalid")
        if id(request) in self._scan_requests:
            raise CanonicalSegmentContractError("canonical request already has a scan capability")
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
        result = CanonicalProductionScanResult(
            tokens,
            present,
            state_out,
            gathered,
            tuple(
                (identity.slot_id, identity.episode_id, identity.source_digest, identity.cursor)
                for identity in request.member.row_identities
            ),
        )
        self._scan_requests.add(id(request))
        self._scan_results[id(result)] = request
        return result

    def prepare_commit(
        self, request: CanonicalProductionSegmentRequest, result: CanonicalProductionScanResult
    ) -> CanonicalProductionCommitCapability:
        if self._scan_results.get(id(result)) is not request:
            raise CanonicalSegmentContractError("commit preparation requires this adapter scan result")
        if result.gathered.item_count != request.member.planned_n_valid:
            raise CanonicalSegmentContractError("adapter gathered count differs from frozen member")
        if any(value.dtype is not torch.float32 for value in result.candidate_state_out):
            raise CanonicalSegmentContractError("commit candidate fast state is not fp32")
        prepared = request.scheduler.prepare_reconcile_after_backward(request.member, result.gathered.item_count)
        capability = CanonicalProductionCommitCapability(request, result, prepared)
        self._commit_capabilities.add(id(capability))
        return capability

    def abort_scan(self, request: CanonicalProductionSegmentRequest, result: CanonicalProductionScanResult) -> None:
        """Dispose one uncommitted scan capability without mutating the frontier."""
        if id(request) not in self._scan_requests or self._scan_results.get(id(result)) is not request:
            raise CanonicalSegmentContractError("scan abort requires this exact pending request/result pair")
        self._scan_requests.remove(id(request))
        self._scan_results.pop(id(result))

    def abort_commit(self, capability: CanonicalProductionCommitCapability) -> None:
        """Consume one pre-mutation commit capability and its exact pending scan."""
        request, result = capability.request, capability.result
        if id(capability) not in self._commit_capabilities:
            raise CanonicalSegmentContractError("commit abort requires an exact pending capability")
        if self._scan_results.get(id(result)) is not request or id(request) not in self._scan_requests:
            raise CanonicalSegmentContractError("commit abort requires this exact pending request/result pair")
        if capability.prepared_reconcile.scheduler is not request.scheduler:
            raise CanonicalSegmentContractError("commit abort capability scheduler is foreign")
        request.scheduler.validate_prepared_reconcile(capability.prepared_reconcile)
        request.transaction.validate_reconcile(request.member_index)
        self._commit_capabilities.remove(id(capability))
        self.abort_scan(request, result)

    def commit_success(self, capability: CanonicalProductionCommitCapability) -> None:
        request, result = capability.request, capability.result
        if id(capability) not in self._commit_capabilities:
            raise CanonicalSegmentContractError("commit capability is foreign or already consumed")
        if self._scan_results.get(id(result)) is not request:
            raise CanonicalSegmentContractError("commit capability scan result is foreign")
        if capability.prepared_reconcile.scheduler is not request.scheduler:
            raise CanonicalSegmentContractError("commit capability scheduler is foreign")
        if any(value.dtype is not torch.float32 for value in result.candidate_state_out):
            raise CanonicalSegmentContractError("commit candidate fast state is not fp32")
        request.scheduler.validate_prepared_reconcile(capability.prepared_reconcile)
        request.transaction.validate_reconcile(request.member_index)
        self.frontier.commit(request.member, result.candidate_state_out)
        request.scheduler.consume_prepared_reconcile(capability.prepared_reconcile)
        request.transaction.mark_reconciled(request.member_index)
        self._commit_capabilities.remove(id(capability))

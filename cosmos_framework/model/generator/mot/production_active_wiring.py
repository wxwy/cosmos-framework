"""CPU/static authority objects for the Local-Memory active trainer path.

This module intentionally owns no producer, model, optimizer, or I/O work.  Its
objects are process-local identity capabilities consumed by the trainer/model
seams described by the active-wiring design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .canonical_segment_runtime import CanonicalSegmentRuntimeOwner, RuntimePhase
from .local_memory_segment import GAWindowPlan, LocalMemoryTransaction, SegmentBatch, SegmentIdentity
from .production_segment_bridge import NativeBatchResult
from .production_segment_wiring import CanonicalSegmentForward


@dataclass(frozen=True)
class ActiveNativeBatchInputs:
    payloads: tuple[Mapping[str, object], ...]
    locals: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]

    def __post_init__(self) -> None:
        if not self.payloads or len(self.payloads) != len(self.locals) or len(self.payloads) != len(self.identities):
            raise ValueError("active native inputs must be non-empty and cardinality-aligned")
        if any(not isinstance(payload, Mapping) or callable(payload) for payload in self.payloads):
            raise TypeError("active native payloads must be non-callable mappings")


@dataclass(frozen=True)
class PreparedActiveMemberCapability:
    registry: "ProductionActiveWiringRegistry"
    owner: CanonicalSegmentRuntimeOwner
    transaction: LocalMemoryTransaction
    identity: SegmentIdentity
    member_index: int
    forward: CanonicalSegmentForward
    inputs: ActiveNativeBatchInputs
    actual_n_valid: int
    ga_window_token: object


@dataclass(frozen=True)
class ActiveForwardCapability:
    registry: "ProductionActiveWiringRegistry"
    prepared: PreparedActiveMemberCapability
    result: NativeBatchResult


class ActiveSourceTransientError(RuntimeError):
    """Explicit producer-tagged transient; arbitrary forward errors are not retryable."""


class ProductionActiveWiringRegistry:
    """One trainer/model-bound, identity-only active-window registry."""

    def __init__(self, owner: CanonicalSegmentRuntimeOwner) -> None:
        self.owner = owner
        self._prepared: PreparedActiveMemberCapability | None = None
        self._model_consumed: PreparedActiveMemberCapability | None = None
        self._published: ActiveForwardCapability | None = None
        self._ga_window_token: object | None = None

    def prepare_initial(
        self, identity: SegmentIdentity, segment: SegmentBatch, plan: GAWindowPlan, *, trainer_grad_accum_iter: int
    ) -> PreparedActiveMemberCapability:
        if trainer_grad_accum_iter != 0 or plan.ga_effective <= 0:
            raise RuntimeError("active initial member requires trainer counter zero")
        if plan.attempt != 0 or plan.members[0] != (identity.slot_id, identity.episode_id, identity.cursor):
            raise RuntimeError("active initial plan must match the exact first member")
        if self.owner.phase is not RuntimePhase.IDLE or self._prepared is not None or self._published is not None:
            raise RuntimeError("active registry is not idle")
        self.owner.admit((identity,))
        transaction = self.owner.begin(plan)
        return self._prepare(identity, segment, transaction, 0)

    def prepare_continuation(
        self, identity: SegmentIdentity, segment: SegmentBatch, transaction: LocalMemoryTransaction, *, trainer_grad_accum_iter: int
    ) -> PreparedActiveMemberCapability:
        if self._ga_window_token is None or trainer_grad_accum_iter != len(transaction.completed_members):
            raise RuntimeError("active continuation counter/token mismatch")
        if self.owner.phase is not RuntimePhase.MEMBER_COMMITTED or transaction is not self.owner.transaction:
            raise RuntimeError("active continuation requires exact open transaction")
        self.owner.admit_next((identity,))
        return self._prepare(identity, segment, transaction, len(transaction.completed_members))

    def _prepare(
        self, identity: SegmentIdentity, segment: SegmentBatch, transaction: LocalMemoryTransaction, member_index: int
    ) -> PreparedActiveMemberCapability:
        if member_index >= transaction.plan.ga_effective:
            raise RuntimeError("active member is outside frozen plan")
        forward = self.owner.prepare(segment)
        try:
            inputs = ActiveNativeBatchInputs(
                tuple(forward.payloads), tuple(forward.locals), tuple(forward.identities)
            )
            if len(inputs.payloads) != transaction.plan.planned_n_valid[member_index]:
                raise RuntimeError("active gathered count does not match frozen plan")
        except (TypeError, ValueError, RuntimeError):
            self.owner.abort_terminal(transaction, forward, "LOCAL_MEM_IDENTITY_CONTRACT_FAILURE")
            self._prepared = self._model_consumed = self._published = None
            raise
        if self._ga_window_token is None:
            self._ga_window_token = object()
        prepared = PreparedActiveMemberCapability(
            self, self.owner, transaction, identity, member_index, forward, inputs, len(inputs.payloads), self._ga_window_token
        )
        self._prepared = prepared
        return prepared

    def consume_prepared_for_model(self, prepared: PreparedActiveMemberCapability) -> PreparedActiveMemberCapability:
        if prepared.registry is not self or prepared is not self._prepared or self._model_consumed is not None:
            raise RuntimeError("active prepared capability is stale or foreign")
        self._model_consumed = prepared
        return prepared

    def publish_active_forward(
        self, prepared: PreparedActiveMemberCapability, result: NativeBatchResult) -> ActiveForwardCapability:
        if prepared.registry is not self or prepared is not self._model_consumed or self._published is not None:
            raise RuntimeError("active forward capability is stale or foreign")
        active = ActiveForwardCapability(self, prepared, result)
        self._published = active
        return active

    def consume_active_forward(self, active: ActiveForwardCapability) -> ActiveForwardCapability:
        if active.registry is not self or active is not self._published:
            raise RuntimeError("active forward capability is stale or foreign")
        self._prepared = self._model_consumed = self._published = None
        return active

    def abort_source_transient(self, prepared: PreparedActiveMemberCapability) -> GAWindowPlan:
        """Return the owner-retained retry plan only for the exact first active member."""
        if prepared.registry is not self or prepared is not self._prepared:
            raise RuntimeError("active transient capability is stale or foreign")
        if prepared.member_index != 0 or prepared.transaction.completed_members:
            self.owner.abort_terminal(prepared.transaction, prepared.forward, "LOCAL_MEM_RETRY_AFTER_MEMBER")
            self._prepared = self._model_consumed = self._published = None
            raise RuntimeError("LOCAL_MEM_RETRY_AFTER_MEMBER")
        plan = self.owner.abort_retry(prepared.transaction, prepared.forward)
        self._prepared = self._model_consumed = self._published = None
        return plan

    def prepare_retry(
        self, segment: SegmentBatch, plan: GAWindowPlan, *, trainer_grad_accum_iter: int
    ) -> PreparedActiveMemberCapability:
        if trainer_grad_accum_iter != 0:
            raise RuntimeError("active retry requires trainer counter zero")
        transaction = self.owner.begin_retry(plan)
        identity = self.owner.identity
        if identity is None:
            raise RuntimeError("active retry requires retained owner identity")
        return self._prepare(identity, segment, transaction, 0)

    def assert_trainer_bound_model(self, model: object) -> None:
        if getattr(model, "_psm_active_wiring_registry", None) is not self:
            raise RuntimeError("active registry must be trainer/model object-identically bound")

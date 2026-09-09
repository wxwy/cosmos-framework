"""CPU/static one-member bridge for the canonical Local-Memory owner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .canonical_segment_runtime import CanonicalSegmentRuntimeOwner, CompletedWindowCapability, RuntimePhase
from .local_memory_segment import GAWindowPlan, LocalMemoryTransaction, SegmentBatch, SegmentIdentity


@dataclass(frozen=True)
class NativeBatchResult:
    primary_consumer_mean: torch.Tensor
    auxiliary_loss: torch.Tensor


@dataclass(frozen=True)
class NativeBatchOutcome:
    result: NativeBatchResult | None = None
    source_failure: str | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.source_failure is None):
            raise ValueError("native batch outcome must contain exactly one tagged result")


@dataclass(frozen=True)
class OpenMemberCapability:
    owner: CanonicalSegmentRuntimeOwner
    transaction: LocalMemoryTransaction


@dataclass(frozen=True)
class RetryMemberCapability:
    owner: CanonicalSegmentRuntimeOwner
    transaction: LocalMemoryTransaction


@dataclass(frozen=True)
class TerminalMemberResult:
    code: str


MemberBridgeResult = OpenMemberCapability | CompletedWindowCapability | RetryMemberCapability | TerminalMemberResult


def _pure_backward(
    member_index: int, result: NativeBatchResult, actual_n_valid: int, transaction: LocalMemoryTransaction, identity: SegmentIdentity
):
    from cosmos_framework.trainer import ImaginaireTrainer

    return ImaginaireTrainer._run_local_memory_bridge_backward(
        object.__new__(ImaginaireTrainer), member_index, result.primary_consumer_mean, result.auxiliary_loss,
        actual_n_valid, transaction=transaction, identity=identity,
    )


def run_member(
    owner: CanonicalSegmentRuntimeOwner,
    identity: SegmentIdentity,
    segment: SegmentBatch,
    native_batch: Callable[[tuple[Any, ...], tuple[torch.Tensor | None, ...]], NativeBatchOutcome],
    *,
    initial_plan: GAWindowPlan | None = None,
    prior: OpenMemberCapability | None = None,
    retry: RetryMemberCapability | None = None,
) -> MemberBridgeResult:
    """Attempt one native callback and leave every disposition with the owner."""
    if sum(item is not None for item in (initial_plan, prior, retry)) != 1:
        raise RuntimeError("run_member requires exactly one initial, continuation, or retry authority")
    if initial_plan is not None:
        if owner.phase is not RuntimePhase.IDLE:
            raise RuntimeError("initial member requires idle owner")
        owner.admit((identity,))
        transaction = owner.begin(initial_plan)
    elif prior is not None:
        if prior.owner is not owner or prior.transaction is not owner.transaction or owner.phase is not RuntimePhase.MEMBER_COMMITTED:
            raise RuntimeError("continuation requires exact open capability")
        transaction = prior.transaction
        owner.admit_next((identity,))
    else:
        assert retry is not None
        if retry.owner is not owner or retry.transaction is not owner.transaction or owner.phase is not RuntimePhase.MEMBER_READY:
            raise RuntimeError("retry requires exact retained capability")
        transaction = retry.transaction
    member_index = len(transaction.completed_members)
    if owner.identity is not identity or transaction.plan.members[member_index] != (identity.slot_id, identity.episode_id, identity.cursor):
        raise RuntimeError("member identity does not match exact transaction plan")
    forward = owner.prepare(segment)
    actual_n_valid = len(forward.payloads)
    if actual_n_valid != transaction.plan.planned_n_valid[member_index]:
        raise RuntimeError("gathered consumer count does not match frozen plan")
    try:
        outcome = native_batch(forward.payloads, forward.locals)
    except Exception:
        owner.abort_terminal(transaction, forward, "LOCAL_MEM_OUTER_FAILURE")
        return TerminalMemberResult("LOCAL_MEM_OUTER_FAILURE")
    if not isinstance(outcome, NativeBatchOutcome):
        owner.abort_terminal(transaction, forward, "LOCAL_MEM_OUTER_FAILURE")
        return TerminalMemberResult("LOCAL_MEM_OUTER_FAILURE")
    if outcome.source_failure is not None:
        if outcome.source_failure != "LOAD_DECODE_TRANSIENT":
            owner.abort_terminal(transaction, forward, "LOCAL_MEM_OUTER_FAILURE")
            return TerminalMemberResult("LOCAL_MEM_OUTER_FAILURE")
        if transaction.plan.attempt != 0:
            owner.abort_terminal(transaction, forward, "LOCAL_MEM_RETRY_EXHAUSTED")
            return TerminalMemberResult("LOCAL_MEM_RETRY_EXHAUSTED")
        suffix = owner.abort_retry(transaction, forward)
        return RetryMemberCapability(owner, owner.begin_retry(suffix))
    assert outcome.result is not None
    backward = _pure_backward(member_index, outcome.result, actual_n_valid, transaction, identity)
    if backward.terminal_code is not None:
        owner.abort_terminal(transaction, forward, backward.terminal_code)
        return TerminalMemberResult(backward.terminal_code)
    transaction.successful_backward(member_index, identity, actual_n_valid)
    owner.commit(transaction, forward)
    if len(transaction.completed_members) == transaction.plan.ga_effective:
        return owner.finish_window(transaction)
    return OpenMemberCapability(owner, transaction)


def run_disabled(payloads: tuple[Any, ...], native_batch: Callable[[tuple[Any, ...]], torch.Tensor]) -> torch.Tensor:
    """No-Local parity path: no owner, plan, scan, or Local mutation."""
    return native_batch(payloads)

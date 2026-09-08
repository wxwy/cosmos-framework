# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU contracts for canonical Local-Memory segments.

This module deliberately has no dataset, trainer, checkpoint, or model-forward
dependencies. Production wiring is a separate Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class SegmentProvenance:
    manifest_digest: str
    config_digest: str
    source_digest: str
    segment_id: int


@dataclass(frozen=True)
class SegmentBatch:
    consumer_visual_summary: torch.Tensor
    consumer_payload: tuple[tuple[Any | None, ...], ...]
    consumer_valid: torch.Tensor
    consumer_step: torch.Tensor
    evidence_visual_summary_prev: torch.Tensor
    evidence_executed_action_prev: torch.Tensor
    evidence_valid: torch.Tensor
    evidence_source_step: torch.Tensor
    slot_id: torch.Tensor
    episode_id: tuple[str, ...]
    category: tuple[str, ...]
    segment_provenance: SegmentProvenance

    def validate(self, ttt_tbptt_steps: int) -> None:
        if self.consumer_visual_summary.ndim != 3 or self.consumer_visual_summary.shape[-1] != 96:
            raise ValueError("consumer_visual_summary must have shape [B,T,96].")
        batch, steps, _ = self.consumer_visual_summary.shape
        if not 1 <= steps <= ttt_tbptt_steps:
            raise ValueError("segment length must be in [1, ttt_tbptt_steps].")
        expected = (batch, steps)
        for name, value in (
            ("consumer_valid", self.consumer_valid),
            ("evidence_valid", self.evidence_valid),
            ("consumer_step", self.consumer_step),
            ("evidence_source_step", self.evidence_source_step),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape [B,T].")
        if tuple(self.evidence_visual_summary_prev.shape) != (batch, steps, 96):
            raise ValueError("evidence_visual_summary_prev must have shape [B,T,96].")
        if tuple(self.evidence_executed_action_prev.shape) != (batch, steps, 10):
            raise ValueError("evidence_executed_action_prev must have shape [B,T,10].")
        if tuple(self.slot_id.shape) != (batch,) or len(self.episode_id) != batch or len(self.category) != batch:
            raise ValueError("slot/episode/category batch identities are incompatible.")
        if len(self.consumer_payload) != batch or any(len(row) != steps for row in self.consumer_payload):
            raise ValueError("consumer_payload must have shape [B,T].")
        valid = self.consumer_valid.bool()
        evidence_valid = self.evidence_valid.bool()
        if not torch.equal(self.evidence_source_step.eq(-1), ~evidence_valid):
            raise ValueError("evidence_source_step must be -1 iff evidence is invalid.")
        if torch.any(evidence_valid & ~valid):
            raise ValueError("evidence cannot be valid for PAD consumers.")
        if torch.any(valid & self.consumer_step.eq(0) & evidence_valid):
            raise ValueError("consumer step0 must have absent evidence.")
        if torch.any(valid & self.consumer_step.lt(0)):
            raise ValueError("valid consumers must have non-negative consumer_step.")
        non_s0 = valid & self.consumer_step.gt(0)
        if torch.any(non_s0 & ~evidence_valid):
            raise ValueError("valid non-S0 consumers require previous evidence.")
        if torch.any(evidence_valid & self.evidence_source_step.ne(self.consumer_step - 1)):
            raise ValueError("evidence_source_step must equal consumer_step - 1.")
        for row in range(batch):
            for index in range(steps):
                if bool(valid[row, index]) and self.consumer_payload[row][index] is None:
                    raise ValueError("valid consumers require an opaque payload.")
                if not bool(valid[row, index]) and self.consumer_payload[row][index] is not None:
                    raise ValueError("PAD consumer payload must be absent.")

    def gather_consumers(self, local_tokens: torch.Tensor, local_present: torch.Tensor) -> tuple[list[Any], list[torch.Tensor | None], list[tuple[int, str, int]]]:
        batch, steps = self.consumer_valid.shape
        if local_tokens.ndim != 4 or tuple(local_tokens.shape[:2]) != (batch, steps):
            raise ValueError("Local tokens must have shape [B,T,K,D].")
        if tuple(local_present.shape) != (batch, steps):
            raise ValueError("Local token/presence shapes are incompatible with SegmentBatch.")
        payloads: list[Any] = []
        local: list[torch.Tensor | None] = []
        identities: list[tuple[int, str, int]] = []
        for row in range(batch):
            for index in range(steps):
                if not bool(self.consumer_valid[row, index]):
                    continue
                step = int(self.consumer_step[row, index])
                is_s0 = step == 0
                if bool(local_present[row, index]) != (not is_s0):
                    raise ValueError("only valid S0 consumers may have an absent Local payload.")
                payloads.append(self.consumer_payload[row][index])
                local.append(local_tokens[row, index] if bool(local_present[row, index]) else None)
                identities.append((int(self.slot_id[row]), self.episode_id[row], step))
        return payloads, local, identities


@dataclass(frozen=True)
class GAWindowPlan:
    members: tuple[tuple[int, str, int], ...]
    planned_n_valid: tuple[int, ...]
    attempt: int = 0
    plan_chain_id: str = "local-memory-plan"
    suffix_snapshot: tuple[tuple[int, str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.members or len(self.members) != len(self.planned_n_valid) or any(value <= 0 for value in self.planned_n_valid):
            raise ValueError("GAWindowPlan requires non-empty positive planned counts.")
        if self.attempt not in (0, 1) or not self.plan_chain_id:
            raise ValueError("GAWindowPlan attempt/plan_chain_id are invalid.")
        if self.suffix_snapshot and self.suffix_snapshot != self.members:
            raise ValueError("suffix_snapshot must describe this plan's members.")

    @property
    def n_window(self) -> int:
        return sum(self.planned_n_valid)

    @property
    def ga_effective(self) -> int:
        return len(self.members)

    def objective(self, index: int, consumer_loss: torch.Tensor, auxiliary_loss: torch.Tensor, actual_n_valid: int) -> torch.Tensor:
        if actual_n_valid != self.planned_n_valid[index]:
            raise ValueError("actual gathered count must equal planned count.")
        return (self.planned_n_valid[index] / self.n_window) * consumer_loss + auxiliary_loss / self.ga_effective

    def suffix_after_failure(self, failed_index: int) -> "GAWindowPlan":
        """Return the one permitted immutable suffix retry plan."""
        if failed_index < 0 or failed_index >= self.ga_effective:
            raise ValueError("failed_index is outside GAWindowPlan.")
        if self.attempt == 1:
            raise RuntimeError("LOCAL_MEM_RETRY_EXHAUSTED")
        members = self.members[failed_index:]
        return GAWindowPlan(
            members=members,
            planned_n_valid=self.planned_n_valid[failed_index:],
            attempt=1,
            plan_chain_id=self.plan_chain_id,
            suffix_snapshot=members,
        )


@dataclass(frozen=True)
class SegmentIdentity:
    slot_id: int
    episode_id: str
    category: str
    cursor: int
    segment_id: int
    source_digest: str
    training_stream_end: bool = False


@dataclass(frozen=True)
class LocalMemoryTransactionSnapshot:
    """Pure-Python record for one planned GA chain; it owns no trainer state."""

    plan_chain_id: str
    attempt: int
    completed_members: tuple[SegmentIdentity, ...]
    slow_grads_cleared: bool
    slow_optimizer_steps: int
    slow_lr_scheduler_steps: int
    terminal_failure_code: str | None
    remaining_members_suppressed: bool
    suffix_recovery: GAWindowPlan | None


class LocalMemoryTransaction:
    """Static model of GA failure/retry and GradScaler separation.

    Each successful backward commits episode chronology through the rank-local
    scheduler.  Slow optimizer/LR state is deliberately separate: a scaler skip
    preserves those commits but clears partial slow gradients and takes no slow
    step.  This class is a CPU contract double, never a trainer integration.
    """

    def __init__(self, plan: GAWindowPlan, scheduler: "RankLocalSegmentScheduler") -> None:
        self.plan = plan
        self.scheduler = scheduler
        self.completed_members: list[SegmentIdentity] = []
        self.slow_grads_cleared = False
        self.slow_optimizer_steps = 0
        self.slow_lr_scheduler_steps = 0
        self.terminal_failure_code: str | None = None
        self.remaining_members_suppressed = False
        self.suffix_recovery: GAWindowPlan | None = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("local memory transaction is closed.")

    def validate_success(self, index: int, identity: SegmentIdentity, actual_n_valid: int) -> None:
        """Validate frozen GA identity/count before the member backward runs."""
        self._require_open()
        if index != len(self.completed_members) or self.plan.members[index] != (
            identity.slot_id,
            identity.episode_id,
            identity.cursor,
        ):
            raise ValueError("successful backward must follow the frozen GA plan order.")
        if actual_n_valid != self.plan.planned_n_valid[index]:
            raise ValueError("actual gathered count must equal planned count.")

    def successful_backward(self, index: int, identity: SegmentIdentity, actual_n_valid: int) -> None:
        self.validate_success(index, identity, actual_n_valid)
        self.scheduler.commit(identity, actual_n_valid)
        self.completed_members.append(identity)

    def fail_transient(self, failed_index: int) -> GAWindowPlan:
        """Discard only partial slow grads and expose exactly one suffix retry."""
        self._require_open()
        if self.suffix_recovery is not None:
            raise RuntimeError("local memory suffix recovery already exists.")
        if failed_index != len(self.completed_members):
            raise ValueError("failure index must follow completed members.")
        self.slow_grads_cleared = True
        return self.plan.suffix_after_failure(failed_index)

    def recover_transient(self, failed_index: int) -> GAWindowPlan:
        self._require_open()
        if self.suffix_recovery is not None:
            raise RuntimeError("local memory suffix recovery already exists.")
        self.suffix_recovery = self.fail_transient(failed_index)
        self._closed = True
        return self.suffix_recovery

    def terminal_failure(self, code: str) -> None:
        """Suppress the unexecuted suffix without rolling back prior fast commits."""
        self.slow_grads_cleared = True
        self.terminal_failure_code = code
        self.remaining_members_suppressed = True
        self._closed = True

    def grad_scaler_skip(self) -> None:
        self.slow_grads_cleared = True
        self._closed = True

    def slow_optimizer_step_succeeded(self) -> None:
        self._require_open()
        self.slow_optimizer_steps += 1
        self.slow_lr_scheduler_steps += 1
        self.slow_grads_cleared = False

    def snapshot(self) -> LocalMemoryTransactionSnapshot:
        return LocalMemoryTransactionSnapshot(
            plan_chain_id=self.plan.plan_chain_id,
            attempt=self.plan.attempt,
            completed_members=tuple(self.completed_members),
            slow_grads_cleared=self.slow_grads_cleared,
            slow_optimizer_steps=self.slow_optimizer_steps,
            slow_lr_scheduler_steps=self.slow_lr_scheduler_steps,
            terminal_failure_code=self.terminal_failure_code,
            remaining_members_suppressed=self.remaining_members_suppressed,
            suffix_recovery=self.suffix_recovery,
        )


class RankLocalSegmentScheduler:
    """Deterministic, main-process-only synthetic metadata scheduler."""

    def __init__(self, *, rank: int, target_distribution: dict[str, float], num_workers: int = 0) -> None:
        if num_workers != 0:
            raise ValueError("RankLocalSegmentScheduler requires num_workers=0.")
        if rank < 0 or not target_distribution or any(value <= 0 for value in target_distribution.values()):
            raise ValueError("rank and target_distribution are invalid.")
        total = sum(target_distribution.values())
        self.rank = rank
        self.num_workers = num_workers
        self.target_distribution = {key: value / total for key, value in target_distribution.items()}
        self.cumulative_valid_consumer_exposure = {key: 0 for key in self.target_distribution}
        self.admission_order: list[SegmentIdentity] = []
        self.committed_identities: list[SegmentIdentity] = []
        self.stable_slots: dict[int, SegmentIdentity] = {}
        self.terminal_slots: dict[int, SegmentIdentity] = {}
        self.queue_seed: int | None = None
        self.queue_epoch: int | None = None
        self.queue_permutation: tuple[int, ...] = ()
        self.segment_provenance: SegmentProvenance | None = None

    def admit(self, candidates: Iterable[SegmentIdentity]) -> SegmentIdentity:
        eligible = [item for item in candidates if self._is_admissible(item)]
        if not eligible:
            raise ValueError("no candidate is admissible for its stable stream slot.")
        total = sum(self.cumulative_valid_consumer_exposure.values())
        def deficit(item: SegmentIdentity) -> tuple[float, str, int, str, int]:
            observed = self.cumulative_valid_consumer_exposure[item.category] / max(total, 1)
            return (self.target_distribution[item.category] - observed, item.category, item.slot_id, item.episode_id, item.cursor)
        chosen = max(eligible, key=deficit)
        self.admission_order.append(chosen)
        self.stable_slots[chosen.slot_id] = chosen
        return chosen

    def _is_admissible(self, identity: SegmentIdentity) -> bool:
        if identity.category not in self.target_distribution or identity.slot_id in self.terminal_slots:
            return False
        previous = self.stable_slots.get(identity.slot_id)
        if previous is None:
            return identity.cursor == 0
        return (
            identity.category == previous.category
            and identity.episode_id == previous.episode_id
            and identity.source_digest == previous.source_digest
            and identity.cursor == previous.cursor + 1
        )

    def commit(self, identity: SegmentIdentity, valid_consumers: int) -> None:
        if (
            valid_consumers <= 0
            or identity not in self.admission_order
            or identity in self.committed_identities
        ):
            raise ValueError("only one admitted identity may commit a positive valid count.")
        self.cumulative_valid_consumer_exposure[identity.category] += valid_consumers
        self.committed_identities.append(identity)
        if identity.training_stream_end:
            self.training_stream_end(identity)

    def configure_queue(self, *, seed: int, epoch: int, permutation: Iterable[int], provenance: SegmentProvenance) -> None:
        if seed < 0 or epoch < 0:
            raise ValueError("queue seed/epoch must be non-negative.")
        self.queue_seed = seed
        self.queue_epoch = epoch
        self.queue_permutation = tuple(permutation)
        self.segment_provenance = provenance

    def terminal_rebind(self, identity: SegmentIdentity) -> None:
        if self.terminal_slots.get(identity.slot_id) != identity:
            raise ValueError("terminal rebind requires the terminal stable slot.")
        del self.terminal_slots[identity.slot_id]
        del self.stable_slots[identity.slot_id]

    def training_stream_end(self, identity: SegmentIdentity) -> None:
        if not identity.training_stream_end or self.stable_slots.get(identity.slot_id) != identity:
            raise ValueError("training_stream_end requires terminal identity.")
        self.terminal_slots[identity.slot_id] = identity

    @classmethod
    def rebuild(cls, snapshot: dict[str, object]) -> "RankLocalSegmentScheduler":
        scheduler = cls(
            rank=int(snapshot["rank"]),
            target_distribution=dict(snapshot["target_distribution"]),
            num_workers=int(snapshot["num_workers"]),
        )
        scheduler.cumulative_valid_consumer_exposure = dict(snapshot["cumulative_valid_consumer_exposure"])
        scheduler.admission_order = list(snapshot["admission_order"])
        scheduler.committed_identities = list(snapshot["committed_identities"])
        scheduler.stable_slots = dict(snapshot["stable_slots"])
        scheduler.terminal_slots = dict(snapshot["terminal_slots"])
        scheduler.queue_seed = snapshot["queue_seed"]
        scheduler.queue_epoch = snapshot["queue_epoch"]
        scheduler.queue_permutation = tuple(snapshot["queue_permutation"])
        scheduler.segment_provenance = snapshot["segment_provenance"]
        return scheduler

    def snapshot(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "num_workers": self.num_workers,
            "target_distribution": dict(self.target_distribution),
            "cumulative_valid_consumer_exposure": dict(self.cumulative_valid_consumer_exposure),
            "admission_order": tuple(self.admission_order),
            "committed_identities": tuple(self.committed_identities),
            "stable_slots": dict(self.stable_slots),
            "terminal_slots": dict(self.terminal_slots),
            "queue_seed": self.queue_seed,
            "queue_epoch": self.queue_epoch,
            "queue_permutation": self.queue_permutation,
            "segment_provenance": self.segment_provenance,
        }

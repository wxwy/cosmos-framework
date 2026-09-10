"""CPU/static contracts for canonical ``[B_stream, T]`` Local-Memory batches.

This module deliberately models metadata only.  It is not a producer, packer,
model-forward, runtime-sidecar, or trainer integration point.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from .local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance


class CanonicalSegmentContractError(RuntimeError):
    """Fail-closed error for the canonical batch-level contract double."""


_ATTEMPT_ONE_AUTHORITY = object()


@dataclass(frozen=True)
class ChronologyCountRecord:
    slot_id: int
    episode_id: str
    category: str
    source_digest: str
    consumer_step_start: int
    consumer_step_stop_exclusive: int
    training_stream_end: bool
    manifest_digest: str

    @property
    def planned_n_valid(self) -> int:
        return self.consumer_step_stop_exclusive - self.consumer_step_start

    def validate_row(self, batch: SegmentBatch, row: int) -> int:
        if self.planned_n_valid <= 0:
            raise CanonicalSegmentContractError("chronology valid count must be positive")
        if (
            int(batch.slot_id[row]) != self.slot_id
            or batch.episode_id[row] != self.episode_id
            or batch.category[row] != self.category
            or batch.segment_provenance.source_digest != self.source_digest
            or batch.segment_provenance.manifest_digest != self.manifest_digest
        ):
            raise CanonicalSegmentContractError("chronology row identity/provenance mismatch")
        valid_steps = tuple(int(value) for value in batch.consumer_step[row][batch.consumer_valid[row]].tolist())
        expected_steps = tuple(range(self.consumer_step_start, self.consumer_step_stop_exclusive))
        if valid_steps != expected_steps or int(batch.consumer_valid[row].sum()) != self.planned_n_valid:
            raise CanonicalSegmentContractError("chronology count does not equal valid native consumers")
        return self.planned_n_valid


@dataclass(frozen=True)
class QueueEpochSnapshot:
    queue_seed: int
    epoch: int
    catalog_digest: str
    positions: tuple[tuple[str, int], ...]
    permutations: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.queue_seed < 0 or self.epoch < 0 or not self.catalog_digest:
            raise ValueError("queue snapshot fields are invalid")


def _decimal(value: int) -> bytes:
    if value < 0:
        raise ValueError("queue integers must be non-negative")
    return str(value).encode("ascii")


def queue_digest_preimage(*, queue_seed: int, epoch: int, category: str, canonical_index: int) -> bytes:
    """Return the byte-for-byte v0.3 queue permutation preimage."""
    return b"\0".join(
        (
            b"PSM-WMA/queue/v1",
            _decimal(queue_seed),
            _decimal(epoch),
            category.encode("utf-8"),
            _decimal(canonical_index),
        )
    )


def queue_permutation(*, queue_seed: int, epoch: int, category: str, catalog_size: int) -> tuple[int, ...]:
    if catalog_size < 0:
        raise ValueError("catalog_size must be non-negative")
    return tuple(
        index
        for _, index in sorted(
            (
                (
                    hashlib.sha256(
                        queue_digest_preimage(
                            queue_seed=queue_seed,
                            epoch=epoch,
                            category=category,
                            canonical_index=index,
                        )
                    ).digest(),
                    index,
                )
                for index in range(catalog_size)
            )
        )
    )


@dataclass(frozen=True)
class MicrobatchPlanMember:
    member_index: int
    row_identities: tuple[SegmentIdentity, ...]
    row_provenances: tuple[SegmentProvenance, ...]
    row_chronology: tuple[ChronologyCountRecord, ...]
    row_planned_n_valid: tuple[int, ...]
    planned_n_valid: int
    queue_snapshot: QueueEpochSnapshot
    projected_exposure_before: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        width = len(self.row_identities)
        if (
            width == 0
            or len(self.row_provenances) != width
            or len(self.row_chronology) != width
            or len(self.row_planned_n_valid) != width
            or tuple(identity.slot_id for identity in self.row_identities)
            != tuple(sorted(identity.slot_id for identity in self.row_identities))
            or any(value <= 0 for value in self.row_planned_n_valid)
            or self.planned_n_valid != sum(self.row_planned_n_valid)
        ):
            raise ValueError("MicrobatchPlanMember batch metadata is invalid")
        for identity, record, count in zip(
            self.row_identities, self.row_chronology, self.row_planned_n_valid, strict=True
        ):
            if (
                identity.slot_id != record.slot_id
                or identity.episode_id != record.episode_id
                or identity.category != record.category
                or identity.source_digest != record.source_digest
                or identity.training_stream_end != record.training_stream_end
                or count != record.planned_n_valid
            ):
                raise ValueError("MicrobatchPlanMember identity/chronology mismatch")

    def validate_batch(self, batch: SegmentBatch) -> None:
        if len(self.row_identities) != batch.consumer_valid.shape[0]:
            raise CanonicalSegmentContractError("batch row count differs from frozen member")
        actual_rows = tuple(record.validate_row(batch, row) for row, record in enumerate(self.row_chronology))
        if actual_rows != self.row_planned_n_valid or sum(actual_rows) != self.planned_n_valid:
            raise CanonicalSegmentContractError("batch valid counts differ from frozen member")
        for row, (identity, provenance) in enumerate(
            zip(self.row_identities, self.row_provenances, strict=True)
        ):
            if (
                int(batch.slot_id[row]) != identity.slot_id
                or batch.episode_id[row] != identity.episode_id
                or batch.category[row] != identity.category
                or batch.segment_provenance != provenance
            ):
                raise CanonicalSegmentContractError("batch row identity/provenance differs from frozen member")


@dataclass(frozen=True)
class CanonicalGAWindowPlan:
    members: tuple[MicrobatchPlanMember, ...]
    original_n_valid_window: int
    original_ga_effective: int
    plan_chain_id: str
    attempt: int = 0
    _attempt_authority: object | None = None

    def __post_init__(self) -> None:
        if (
            not self.members
            or self.original_ga_effective != len(self.members)
            or self.original_n_valid_window != sum(member.planned_n_valid for member in self.members)
            or tuple(member.member_index for member in self.members) != tuple(range(len(self.members)))
            or not self.plan_chain_id
            or self.attempt not in (0, 1)
            or (self.attempt == 0 and self._attempt_authority is not None)
            or (self.attempt == 1 and self._attempt_authority is not _ATTEMPT_ONE_AUTHORITY)
        ):
            raise ValueError("CanonicalGAWindowPlan metadata is invalid")

    def objective(
        self, member_index: int, consumer_loss: torch.Tensor, auxiliary_loss: torch.Tensor, actual_n_valid: int
    ) -> torch.Tensor:
        member = self.members[member_index]
        if actual_n_valid != member.planned_n_valid:
            raise CanonicalSegmentContractError("actual gathered count differs from frozen member")
        return (
            member.planned_n_valid / self.original_n_valid_window * consumer_loss
            + auxiliary_loss / self.original_ga_effective
        )

@dataclass(frozen=True)
class BatchWindowSnapshot:
    backward_started: bool
    completed_members: tuple[int, ...]
    slow_grads_cleared: bool
    terminal_failure_code: str | None
    remaining_members_suppressed: bool


class CanonicalBatchWindowTransaction:
    """Batch-level retry/terminal authority; intentionally independent of row APIs."""

    def __init__(self, plan: CanonicalGAWindowPlan) -> None:
        self.plan = plan
        self.backward_started = False
        self.completed_members: list[int] = []
        self.slow_grads_cleared = False
        self.terminal_failure_code: str | None = None
        self.remaining_members_suppressed = False
        self._closed = False

    def retry_first_member_pre_backward(self) -> CanonicalGAWindowPlan:
        if self._closed or self.backward_started or self.completed_members:
            raise CanonicalSegmentContractError("retry requires an unstarted batch window")
        if self.plan.attempt != 0:
            raise CanonicalSegmentContractError("attempt-1 may not retry")
        retry = replace(self.plan, attempt=1, _attempt_authority=_ATTEMPT_ONE_AUTHORITY)
        self._closed = True
        return retry

    def mark_backward_started(self, member_index: int) -> None:
        if (
            self._closed
            or member_index < 0
            or member_index >= len(self.plan.members)
            or member_index != len(self.completed_members)
        ):
            raise CanonicalSegmentContractError("backward must follow the frozen batch window order")
        self.backward_started = True

    def mark_reconciled(self, member_index: int) -> None:
        if (
            self._closed
            or not self.backward_started
            or member_index < 0
            or member_index >= len(self.plan.members)
            or member_index != len(self.completed_members)
        ):
            raise CanonicalSegmentContractError("reconcile requires the current backward member")
        self.completed_members.append(member_index)
        if len(self.completed_members) == len(self.plan.members):
            self._closed = True

    def terminalize(self, member_index: int, code: str) -> None:
        if (
            self._closed
            or member_index < 0
            or member_index >= len(self.plan.members)
            or member_index != len(self.completed_members)
            or not code
        ):
            raise CanonicalSegmentContractError("terminal failure does not match the batch window")
        self.slow_grads_cleared = True
        self.terminal_failure_code = code
        self.remaining_members_suppressed = True
        self._closed = True

    def snapshot(self) -> BatchWindowSnapshot:
        return BatchWindowSnapshot(
            self.backward_started,
            tuple(self.completed_members),
            self.slow_grads_cleared,
            self.terminal_failure_code,
            self.remaining_members_suppressed,
        )


@dataclass(frozen=True)
class NativeConsumerBatch:
    payloads: tuple[Any, ...]
    local_prefixes: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]

    def __post_init__(self) -> None:
        if not self.payloads or len(self.payloads) != len(self.local_prefixes) or len(self.payloads) != len(self.identities):
            raise ValueError("NativeConsumerBatch cardinality is invalid")

    @property
    def item_count(self) -> int:
        return len(self.payloads)

    @classmethod
    def from_segment(
        cls, batch: SegmentBatch, member: MicrobatchPlanMember, local_tokens: torch.Tensor, local_present: torch.Tensor
    ) -> "NativeConsumerBatch":
        member.validate_batch(batch)
        payloads, prefixes, identities = batch.gather_consumers(local_tokens, local_present)
        expected = tuple(
            (identity.slot_id, identity.episode_id, step)
            for identity, record in zip(member.row_identities, member.row_chronology, strict=True)
            for step in range(record.consumer_step_start, record.consumer_step_stop_exclusive)
        )
        result = cls(tuple(payloads), tuple(prefixes), tuple(identities))
        if result.identities != expected or result.item_count != member.planned_n_valid:
            raise CanonicalSegmentContractError("stream-major native gather differs from frozen member")
        return result


@dataclass(frozen=True)
class ProjectedSchedulerState:
    queue_snapshot: QueueEpochSnapshot
    exposure: tuple[tuple[str, int], ...]
    stable_slots: tuple[tuple[int, SegmentIdentity], ...] = ()
    terminal_slots: tuple[tuple[int, SegmentIdentity], ...] = ()
    target_distribution: tuple[tuple[str, float], ...] = ()
    catalog: tuple["CatalogRow", ...] = ()

    def __post_init__(self) -> None:
        positions = dict(self.queue_snapshot.positions)
        if len(positions) != len(self.queue_snapshot.positions) or any(value < 0 for value in positions.values()):
            raise ValueError("queue positions must be unique and non-negative")
        if not self.catalog:
            return
        categories = tuple(sorted({row.identity.category for row in self.catalog}))
        target = dict(self.target_distribution)
        permutations = dict(self.queue_snapshot.permutations)
        if (
            not target
            or tuple(sorted(target)) != categories
            or any(value <= 0.0 for value in target.values())
            or tuple(sorted(permutations)) != categories
        ):
            raise ValueError("catalog requires complete frozen target and epoch permutations")
        for category in categories:
            expected = queue_permutation(
                queue_seed=self.queue_snapshot.queue_seed,
                epoch=self.queue_snapshot.epoch,
                category=category,
                catalog_size=len(self._queue_for(category)),
            )
            if permutations[category] != expected:
                raise ValueError("catalog epoch permutation differs from deterministic authority")

    def exposure_map(self) -> dict[str, int]:
        return dict(self.exposure)

    def _catalog_for(self, category: str) -> tuple["CatalogRow", ...]:
        return tuple(item for item in self.catalog if item.identity.category == category)

    def _queue_for(self, category: str) -> tuple["CatalogRow", ...]:
        """Return canonical fresh episodes only; continuation rows never enter a queue."""
        fresh = tuple(
            row for row in self._catalog_for(category)
            if row.identity.cursor == 0 and row.chronology.consumer_step_start == 0
        )
        ordered = tuple(sorted(fresh, key=lambda row: (row.identity.source_digest, row.identity.episode_id)))
        if len({(row.identity.source_digest, row.identity.episode_id) for row in ordered}) != len(ordered):
            raise CanonicalSegmentContractError("fresh episode queue has duplicate provenance")
        return ordered

    def _permutation(self, category: str) -> tuple[int, ...]:
        frozen = dict(self.queue_snapshot.permutations).get(category)
        catalog = self._queue_for(category)
        expected = queue_permutation(
            queue_seed=self.queue_snapshot.queue_seed,
            epoch=self.queue_snapshot.epoch,
            category=category,
            catalog_size=len(catalog),
        )
        if frozen != expected:
            raise CanonicalSegmentContractError("queue permutation differs from frozen epoch snapshot")
        return expected

    def _admit(self, slot_id: int) -> "CatalogRow":
        stable = dict(self.stable_slots).get(slot_id)
        if stable is not None and slot_id not in dict(self.terminal_slots):
            matches = tuple(
                item for item in self.catalog
                if item.identity.category == stable.category
                and item.identity.episode_id == stable.episode_id
                and item.identity.source_digest == stable.source_digest
                and item.identity.cursor == stable.cursor + 1
            )
            if len(matches) != 1:
                raise CanonicalSegmentContractError("bound stable slot lacks exact continuation")
            return matches[0].bind_slot(slot_id)
        exposure, target, positions = self.exposure_map(), dict(self.target_distribution), dict(self.queue_snapshot.positions)
        choices: list[tuple[float, str, CatalogRow]] = []
        total = sum(exposure.values())
        for category in sorted(target):
            catalog, permutation = self._queue_for(category), self._permutation(category)
            position = positions.get(category, 0)
            if position >= len(permutation):
                continue
            candidate = catalog[permutation[position]]
            choices.append((target[category] - exposure.get(category, 0) / max(total, 1), category, candidate))
        if not choices:
            raise CanonicalSegmentContractError("no legal free-slot queue admission")
        return max(choices, key=lambda item: (item[0], item[1]))[2].bind_slot(slot_id)

    def derive_member(self, *, member_index: int, slot_ids: tuple[int, ...]) -> MicrobatchPlanMember:
        if tuple(sorted(slot_ids)) != slot_ids or len(set(slot_ids)) != len(slot_ids):
            raise CanonicalSegmentContractError("member slot ids must be unique and sorted")
        state = self
        rows: list[CatalogRow] = []
        for slot_id in slot_ids:
            row = state._admit(slot_id)
            rows.append(row)
            state = state._commit_admission(row, row.chronology.planned_n_valid)
        return MicrobatchPlanMember(
            member_index,
            tuple(row.identity for row in rows),
            tuple(row.provenance for row in rows),
            tuple(row.chronology for row in rows),
            tuple(row.chronology.planned_n_valid for row in rows),
            sum(row.chronology.planned_n_valid for row in rows),
            self.queue_snapshot,
            self.exposure,
        )

    def rollover_projected_if_safe(self) -> "ProjectedSchedulerState":
        """Perform the frozen post-member epoch boundary without mutating live state."""
        catalog_sizes = {category: len(self._queue_for(category)) for category, _ in self.target_distribution}
        positions = dict(self.queue_snapshot.positions)
        if not catalog_sizes or any(positions.get(category, 0) < size for category, size in catalog_sizes.items()):
            return self
        if any(slot not in dict(self.terminal_slots) for slot, _ in self.stable_slots):
            return self
        snapshot = self.queue_snapshot
        next_epoch = snapshot.epoch + 1
        permutations = tuple(
            (
                category,
                queue_permutation(
                    queue_seed=snapshot.queue_seed,
                    epoch=next_epoch,
                    category=category,
                    catalog_size=size,
                ),
            )
            for category, size in sorted(catalog_sizes.items())
        )
        return ProjectedSchedulerState(
            QueueEpochSnapshot(snapshot.queue_seed, next_epoch, snapshot.catalog_digest, tuple(), permutations),
            self.exposure,
            target_distribution=self.target_distribution,
            catalog=self.catalog,
        )

    def project_commit(self, member: MicrobatchPlanMember) -> "ProjectedSchedulerState":
        if member.queue_snapshot != self.queue_snapshot or member.projected_exposure_before != self.exposure:
            raise CanonicalSegmentContractError("member does not match projected scheduler frontier")
        state = self
        for identity, count in zip(member.row_identities, member.row_planned_n_valid, strict=True):
            expected = state._admit(identity.slot_id)
            if identity != expected.identity:
                raise CanonicalSegmentContractError("member transition is not the exact projected admission")
            state = state._commit_admission(expected, count)
        return state

    def _commit_admission(self, row: "CatalogRow", count: int) -> "ProjectedSchedulerState":
        """Advance exactly one already-validated projected row transition."""
        identity = row.identity
        exposure = self.exposure_map()
        stable = dict(self.stable_slots)
        terminal = dict(self.terminal_slots)
        positions = dict(self.queue_snapshot.positions)
        if identity.slot_id not in stable or identity.slot_id in terminal:
            positions[identity.category] = positions.get(identity.category, 0) + 1
            terminal.pop(identity.slot_id, None)
        exposure[identity.category] = exposure.get(identity.category, 0) + count
        stable[identity.slot_id] = identity
        if identity.training_stream_end:
            terminal[identity.slot_id] = identity
        return ProjectedSchedulerState(
            QueueEpochSnapshot(
                self.queue_snapshot.queue_seed,
                self.queue_snapshot.epoch,
                self.queue_snapshot.catalog_digest,
                tuple(sorted(positions.items())),
                self.queue_snapshot.permutations,
            ),
            tuple(sorted(exposure.items())),
            tuple(sorted(stable.items())),
            tuple(sorted(terminal.items())),
            self.target_distribution,
            self.catalog,
        )


@dataclass(frozen=True)
class CatalogRow:
    identity: SegmentIdentity
    chronology: ChronologyCountRecord
    provenance: SegmentProvenance

    def bind_slot(self, slot_id: int) -> "CatalogRow":
        """Bind an immutable episode/chronology row to a newly admitted slot."""
        return CatalogRow(
            replace(self.identity, slot_id=slot_id),
            replace(self.chronology, slot_id=slot_id),
            self.provenance,
        )


class CanonicalBatchScheduler:
    """Metadata-only live/projected frontier with all-row atomic commits."""

    def __init__(self, state: ProjectedSchedulerState) -> None:
        self._state = state
        self._frozen_transitions: list[tuple[MicrobatchPlanMember, ProjectedSchedulerState, ProjectedSchedulerState]] = []

    @property
    def snapshot(self) -> ProjectedSchedulerState:
        return self._state

    def freeze_plan(
        self, *, slot_groups: tuple[tuple[int, ...], ...], plan_chain_id: str
    ) -> CanonicalGAWindowPlan:
        """Derive one immutable GA plan from the projected scheduler frontier."""
        state = self._state
        members: list[MicrobatchPlanMember] = []
        transitions: list[tuple[MicrobatchPlanMember, ProjectedSchedulerState, ProjectedSchedulerState]] = []
        for member_index, slot_ids in enumerate(slot_groups):
            before = state
            member = state.derive_member(member_index=member_index, slot_ids=slot_ids)
            state = state.project_commit(member)
            members.append(member)
            state = state.rollover_projected_if_safe()
            transitions.append((member, before, state))
        plan = CanonicalGAWindowPlan(
            tuple(members),
            sum(member.planned_n_valid for member in members),
            len(members),
            plan_chain_id,
        )
        self._frozen_transitions.extend(transitions)
        return plan

    def reconcile_after_backward(self, member: MicrobatchPlanMember, actual_n_valid: int) -> None:
        if actual_n_valid != member.planned_n_valid:
            raise CanonicalSegmentContractError("native actual count differs before atomic commit")
        if not self._frozen_transitions:
            raise CanonicalSegmentContractError("reconcile requires a frozen projected transition")
        frozen, before, after = self._frozen_transitions[0]
        if member is not frozen or self._state != before:
            raise CanonicalSegmentContractError("reconcile member is foreign, stale, or out of frozen order")
        self._state = after
        self._frozen_transitions.pop(0)

    def rollover_if_exhausted(self, catalog_sizes: Mapping[str, int]) -> None:
        if not self._state.catalog and not self._state.target_distribution:
            positions = dict(self._state.queue_snapshot.positions)
            if any(positions.get(category, 0) < size for category, size in catalog_sizes.items()):
                raise CanonicalSegmentContractError("queue epoch is not exhausted")
            if any(slot not in dict(self._state.terminal_slots) for slot, _ in self._state.stable_slots):
                raise CanonicalSegmentContractError("bound continuation takes precedence over epoch rollover")
            snapshot = self._state.queue_snapshot
            permutations = tuple(
                (category, queue_permutation(queue_seed=snapshot.queue_seed, epoch=snapshot.epoch + 1, category=category, catalog_size=size))
                for category, size in sorted(catalog_sizes.items())
            )
            self._state = ProjectedSchedulerState(
                QueueEpochSnapshot(snapshot.queue_seed, snapshot.epoch + 1, snapshot.catalog_digest, tuple(), permutations),
                self._state.exposure,
            )
            return
        authoritative = {
            category: len(self._state._queue_for(category))
            for category, _ in self._state.target_distribution
        }
        if dict(catalog_sizes) != authoritative:
            raise CanonicalSegmentContractError("rollover catalog sizes differ from authoritative queue")
        next_state = self._state.rollover_projected_if_safe()
        if next_state is self._state:
            raise CanonicalSegmentContractError("queue epoch is not exhausted or continuation remains bound")
        self._state = next_state

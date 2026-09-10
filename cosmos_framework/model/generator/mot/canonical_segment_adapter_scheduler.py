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

    def __post_init__(self) -> None:
        if (
            not self.members
            or self.original_ga_effective != len(self.members)
            or self.original_n_valid_window != sum(member.planned_n_valid for member in self.members)
            or tuple(member.member_index for member in self.members) != tuple(range(len(self.members)))
            or not self.plan_chain_id
            or self.attempt not in (0, 1)
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

    def retry_first_member_pre_backward(self, member_index: int) -> "CanonicalGAWindowPlan":
        if self.attempt != 0 or member_index != 0:
            raise CanonicalSegmentContractError("only attempt-0 first member may retry before backward")
        return replace(self, attempt=1)


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

    def exposure_map(self) -> dict[str, int]:
        return dict(self.exposure)

    def project_commit(self, member: MicrobatchPlanMember) -> "ProjectedSchedulerState":
        if member.queue_snapshot != self.queue_snapshot or member.projected_exposure_before != self.exposure:
            raise CanonicalSegmentContractError("member does not match projected scheduler frontier")
        exposure = self.exposure_map()
        stable = dict(self.stable_slots)
        terminal = dict(self.terminal_slots)
        for identity, count in zip(member.row_identities, member.row_planned_n_valid, strict=True):
            exposure[identity.category] = exposure.get(identity.category, 0) + count
            stable[identity.slot_id] = identity
            if identity.training_stream_end:
                terminal[identity.slot_id] = identity
        return ProjectedSchedulerState(
            self.queue_snapshot,
            tuple(sorted(exposure.items())),
            tuple(sorted(stable.items())),
            tuple(sorted(terminal.items())),
        )


class CanonicalBatchScheduler:
    """Metadata-only live/projected frontier with all-row atomic commits."""

    def __init__(self, state: ProjectedSchedulerState) -> None:
        self._state = state

    @property
    def snapshot(self) -> ProjectedSchedulerState:
        return self._state

    def project(self, members: tuple[MicrobatchPlanMember, ...]) -> tuple[ProjectedSchedulerState, ...]:
        state = self._state
        projected: list[ProjectedSchedulerState] = []
        for member in members:
            state = state.project_commit(member)
            projected.append(state)
        return tuple(projected)

    def reconcile_after_backward(self, member: MicrobatchPlanMember, actual_n_valid: int) -> None:
        if actual_n_valid != member.planned_n_valid:
            raise CanonicalSegmentContractError("native actual count differs before atomic commit")
        self._state = self._state.project_commit(member)

    def rollover_if_exhausted(self, catalog_sizes: Mapping[str, int]) -> None:
        positions = dict(self._state.queue_snapshot.positions)
        if any(positions.get(category, 0) < size for category, size in catalog_sizes.items()):
            raise CanonicalSegmentContractError("queue epoch is not exhausted")
        if any(slot not in dict(self._state.terminal_slots) for slot, _ in self._state.stable_slots):
            raise CanonicalSegmentContractError("bound continuation takes precedence over epoch rollover")
        snapshot = self._state.queue_snapshot
        self._state = ProjectedSchedulerState(
            QueueEpochSnapshot(snapshot.queue_seed, snapshot.epoch + 1, snapshot.catalog_digest, tuple()),
            self._state.exposure,
        )

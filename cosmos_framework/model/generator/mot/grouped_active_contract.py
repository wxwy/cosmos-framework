"""A2 grouped-member metadata, without changing the scalar SegmentIdentity ABI.

Rows preserve the frozen scalar selection order. Repeated slots are legal only
as successive segments and are scanned in dependency waves by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from .local_memory_segment import GAWindowPlan, LocalMemoryTransaction, SegmentBatch, SegmentIdentity


@dataclass(frozen=True)
class GroupedPlanMember:
    """Port of MicrobatchPlanMember row authority for asynchronous active queues."""

    row_identities: tuple[SegmentIdentity, ...]
    row_planned_n_valid: tuple[int, ...]
    tbptt_steps: int
    manifest_digest: str
    config_digest: str
    source_digest: str

    def __post_init__(self) -> None:
        if not self.row_identities or len(self.row_identities) != len(self.row_planned_n_valid):
            raise ValueError("group identities/counts are empty or unaligned")
        if self.tbptt_steps <= 0 or any(n != self.tbptt_steps for n in self.row_planned_n_valid):
            raise ValueError("active A2 currently requires whole TBPTT segments")
        if not all((self.manifest_digest, self.config_digest, self.source_digest)):
            raise ValueError("group provenance must be explicit")
        if any(i.source_digest != self.source_digest for i in self.row_identities):
            raise ValueError("group source differs from row source")

    @property
    def planned_n_valid(self) -> int:
        return sum(self.row_planned_n_valid)

    def validate_batch(self, segment: SegmentBatch) -> None:
        segment.validate(self.tbptt_steps)
        if tuple(segment.consumer_valid.shape) != (len(self.row_identities), self.tbptt_steps):
            raise ValueError("group batch shape differs from plan")
        provenance = segment.segment_provenance
        if (provenance.manifest_digest, provenance.config_digest, provenance.source_digest) != (
            self.manifest_digest,
            self.config_digest,
            self.source_digest,
        ):
            raise ValueError("group batch provenance differs from plan")
        for row, identity in enumerate(self.row_identities):
            if (int(segment.slot_id[row]), segment.episode_id[row], segment.category[row]) != (
                identity.slot_id,
                identity.episode_id,
                identity.category,
            ):
                raise ValueError("group row identity differs from plan")
            expected = torch.arange(
                identity.cursor * self.tbptt_steps,
                (identity.cursor + 1) * self.tbptt_steps,
                device=segment.consumer_step.device,
            )
            if not bool(segment.consumer_valid[row].all()) or not torch.equal(segment.consumer_step[row], expected):
                raise ValueError("group row chronology/count differs from plan")


@dataclass(frozen=True)
class GroupedGAWindowPlan(GAWindowPlan):
    members: tuple[GroupedPlanMember, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(m.planned_n_valid != n for m, n in zip(self.members, self.planned_n_valid, strict=True)):
            raise ValueError("group plan count disagrees with row counts")

    def suffix_after_failure(self, failed_index: int) -> GroupedGAWindowPlan:
        if self.attempt != 0 or failed_index != 0:
            raise RuntimeError("A2 only permits one exact first-group retry")
        return replace(self, attempt=1, suffix_snapshot=self.members)


class GroupedLocalMemoryTransaction(LocalMemoryTransaction):
    """Backward receipt only; the owner publishes all row commits atomically."""

    def validate_success(self, index: int, identity: GroupedPlanMember, actual_n_valid: int) -> None:
        self._require_open()
        if index != len(self.completed_members) or self.plan.members[index] is not identity:
            raise ValueError("group backward must follow the exact frozen member")
        if actual_n_valid != self.plan.planned_n_valid[index]:
            raise ValueError("group gathered count differs from plan")

    def successful_backward(self, index: int, identity: GroupedPlanMember, actual_n_valid: int) -> None:
        self.validate_success(index, identity, actual_n_valid)
        self.completed_members.append(identity)


def stack_segments(rows: tuple[SegmentBatch, ...], *, segment_id: int) -> SegmentBatch:
    """Stack source rows, retaining payload objects and all evidence bytes."""
    if not rows or any(row.consumer_valid.shape[0] != 1 for row in rows):
        raise ValueError("group producer requires nonempty single-row segments")
    first = rows[0].segment_provenance
    expected = (first.manifest_digest, first.config_digest, first.source_digest)
    for row in rows:
        p = row.segment_provenance
        if (p.manifest_digest, p.config_digest, p.source_digest) != expected:
            raise ValueError("cannot mix source/config/manifest within a group")
    values: dict[str, Any] = {}
    for field in fields(SegmentBatch):
        name = field.name
        if name == "segment_provenance":
            values[name] = replace(first, segment_id=segment_id)
        elif isinstance(getattr(rows[0], name), torch.Tensor):
            values[name] = torch.cat([getattr(row, name) for row in rows], dim=0)
        else:
            values[name] = tuple(value for row in rows for value in getattr(row, name))
    return SegmentBatch(**values)


def take_rows(segment: SegmentBatch, indices: tuple[int, ...]) -> SegmentBatch:
    values: dict[str, Any] = {}
    for field in fields(SegmentBatch):
        value = getattr(segment, field.name)
        if isinstance(value, torch.Tensor):
            rows = torch.tensor(indices, dtype=torch.long, device=value.device)
            values[field.name] = value.index_select(0, rows)
        elif field.name == "segment_provenance":
            values[field.name] = value
        else:
            values[field.name] = tuple(value[i] for i in indices)
    return SegmentBatch(**values)


def dependency_waves(identities: tuple[SegmentIdentity, ...]) -> tuple[tuple[int, ...], ...]:
    """Stable topological partition: at most one segment per slot in a wave."""
    remaining = list(range(len(identities)))
    waves = []
    while remaining:
        used: set[int] = set()
        wave = []
        later = []
        for index in remaining:
            slot = identities[index].slot_id
            if slot in used:
                later.append(index)
            else:
                used.add(slot)
                wave.append(index)
        waves.append(tuple(wave))
        remaining = later
    return tuple(waves)


def segment_fingerprint(segment: SegmentBatch) -> str:
    """Content-bound retry seal; tensor bytes, not tensor identity or pickle IDs."""
    import hashlib
    from collections.abc import Mapping
    from dataclasses import is_dataclass
    from enum import Enum

    import numpy as np

    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__qualname__.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                digest.update(field.name.encode())
                visit(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                visit(item)
        elif value is None or isinstance(
            value, (str, bytes, bool, int, float, Enum, torch.dtype, torch.device, np.generic)
        ):
            digest.update(repr(value).encode())
        else:
            raise TypeError(f"unsupported A2 retry payload value: {type(value).__qualname__}")
        digest.update(b"\0")

    visit(segment)
    return digest.hexdigest()

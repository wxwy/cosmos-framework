"""C6 synthetic runtime seam delegating all state authority to C5A."""
from __future__ import annotations

import torch

from .c5a_owner_segment import AdmissionAuthority, AdmissionCapability, C5AOwnerSegmentCPU, ReplayRecord
from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


class C6SyntheticRuntimeAdapter:
    """Test-only adapter; it deliberately owns no chronology or replay state."""

    def __init__(self, authority: AdmissionAuthority, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        self._c5a = C5AOwnerSegmentCPU(authority, encoder, core)

    @property
    def c5_write_count(self) -> int:
        return self._c5a.c5_write_count

    def begin_segment(self, owner_key: str) -> None:
        self._c5a.begin(owner_key)

    def admit(self, capability: AdmissionCapability, *, source: dict[str, torch.Tensor], segment_id: str, row_index: int) -> tuple[int, int] | ReplayRecord:
        if not segment_id or row_index < 0:
            raise ValueError("invalid segment coordinates")
        if capability.source_identity != f"{segment_id}:{capability.source_timestep}":
            raise ValueError("segment coordinates do not match source identity")
        return self._c5a.admit(capability, source=source)

    def materialize(self, owner_key: str) -> torch.Tensor:
        return self._c5a.materialize(owner_key)

    def materialize_many(self, owner_keys: list[str], *, valid: torch.Tensor | None = None, done_before: torch.Tensor | None = None) -> torch.Tensor:
        return self._c5a.materialize_many(owner_keys, valid=valid, done_before=done_before)

    def backward_and_mark(self, owner_key: str, segment_outer_loss: torch.Tensor) -> None:
        self._c5a.backward_and_mark(owner_key, segment_outer_loss)

    def backward_and_mark_many(self, owner_keys: list[str], segment_outer_loss: torch.Tensor) -> None:
        self._c5a.backward_and_mark_many(owner_keys, segment_outer_loss)

    def finish(self, owner_key: str, *, terminal: bool) -> None:
        self._c5a.finish(owner_key, terminal=terminal)

    def abort(self, owner_key: str) -> None:
        self._c5a.abort(owner_key)

    def reset(self, owner_key: str) -> None:
        self._c5a.reset(owner_key)

    def done(self, owner_key: str) -> None:
        self._c5a.reset(owner_key)

"""CPU/static adapter for canonical Local-Memory segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .local_memory_segment import SegmentBatch, SegmentIdentity


@dataclass(frozen=True)
class SegmentScanResult:
    local_tokens: torch.Tensor
    local_present: torch.Tensor
    state_out: ContinualTTTFastState
    payloads: tuple[Any, ...]
    locals: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]


class LocalMemorySegmentSidecar:
    """In-memory detached fast-state cache; scheduler identities remain authoritative."""

    def __init__(self) -> None:
        self._records: dict[int, tuple[SegmentIdentity, ContinualTTTFastState]] = {}

    def read(self, identity: SegmentIdentity) -> ContinualTTTFastState | None:
        record = self._records.get(identity.slot_id)
        if record is None:
            return None
        previous, state = record
        if (identity.episode_id != previous.episode_id or identity.source_digest != previous.source_digest
                or identity.cursor != previous.cursor + 1):
            raise ValueError("segment sidecar identity is not a canonical continuation.")
        return state

    def commit(self, identity: SegmentIdentity, state: ContinualTTTFastState) -> None:
        if identity.training_stream_end:
            self._records.pop(identity.slot_id, None)
            return
        self._records[identity.slot_id] = (identity, ContinualTTTFastState(*(value.detach().clone() for value in state)))

    def reset(self, identity: SegmentIdentity) -> None:
        self._records.pop(identity.slot_id, None)


class CanonicalLocalMemorySegmentAdapter:
    def __init__(self, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore, sidecar: LocalMemorySegmentSidecar) -> None:
        self.encoder, self.core, self.sidecar = encoder, core, sidecar

    def scan(self, segment: SegmentBatch, *, identity: SegmentIdentity) -> SegmentScanResult:
        segment.validate(self.core.ttt_tbptt_steps)
        state_in = self.sidecar.read(identity)
        tokens, state_out, present = self.core.scan_segment_masked_encoded_many(
            self.encoder, segment.evidence_visual_summary_prev, segment.evidence_executed_action_prev,
            segment.evidence_valid, state_in, create_graph=True,
        )
        payloads, locals_, identities = segment.gather_consumers(tokens, present)
        return SegmentScanResult(tokens, present, state_out, tuple(payloads), tuple(locals_), tuple(identities))

    def commit(self, identity: SegmentIdentity, result: SegmentScanResult) -> None:
        """Persist detached fast state only after the trainer transaction succeeds."""
        self.sidecar.commit(identity, result.state_out)

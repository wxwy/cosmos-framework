"""CPU/static adapter for canonical Local-Memory segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .local_memory_segment import LocalMemoryTransaction, SegmentBatch, SegmentIdentity


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
        self._pending_scan: tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None = None

    @property
    def pending_scan(self) -> tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None:
        """Read-only pending capability for the CPU/static wiring bridge."""
        return self._pending_scan

    def pending(self) -> tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None:
        """Return the exact graph-bearing pending capability without mutation."""
        return self._pending_scan

    def committed_snapshot(self) -> tuple[tuple[SegmentIdentity, ContinualTTTFastState], ...]:
        """Return detached copies of the committed sidecar frontier."""
        return tuple(
            (identity, ContinualTTTFastState(*(value.detach().to(dtype=torch.float32).clone() for value in state)) )
            for _, (identity, state) in sorted(self.sidecar._records.items())
        )

    def discard_pending(self, identity: SegmentIdentity, transaction: LocalMemoryTransaction, result: SegmentScanResult) -> None:
        pending = self._pending_scan
        if pending is None or pending[0] is not identity or pending[1] is not transaction or pending[2] is not result:
            raise RuntimeError("segment pending discard requires the exact pending capability.")
        self._pending_scan = None

    def scan(
        self, segment: SegmentBatch, *, identity: SegmentIdentity, transaction: LocalMemoryTransaction
    ) -> SegmentScanResult:
        if identity not in transaction.scheduler.admission_order or (
            identity.slot_id, identity.episode_id, identity.cursor
        ) not in transaction.plan.members:
            raise ValueError("segment scan requires an admitted identity in its transaction plan.")
        segment.validate(self.core.ttt_tbptt_steps)
        state_in = self.sidecar.read(identity)
        tokens, state_out, present = self.core.scan_segment_masked_encoded_many(
            self.encoder, segment.evidence_visual_summary_prev, segment.evidence_executed_action_prev,
            segment.evidence_valid, state_in, create_graph=True,
        )
        payloads, locals_, identities = segment.gather_consumers(tokens, present)
        result = SegmentScanResult(tokens, present, state_out, tuple(payloads), tuple(locals_), tuple(identities))
        self._pending_scan = (identity, transaction, result)
        return result

    def commit(self, identity: SegmentIdentity, result: SegmentScanResult, *, transaction: LocalMemoryTransaction) -> None:
        """Persist detached fast state only after the trainer transaction succeeds."""
        pending = self._pending_scan
        if (transaction.terminal_failure_code is not None or transaction.slow_grads_cleared
                or not transaction.completed_members or transaction.completed_members[-1] != identity
                or pending is None or pending[0] != identity or pending[1] is not transaction or pending[2] is not result):
            raise RuntimeError("segment sidecar commit requires successful trainer transaction.")
        self.sidecar.commit(identity, result.state_out)
        self._pending_scan = None

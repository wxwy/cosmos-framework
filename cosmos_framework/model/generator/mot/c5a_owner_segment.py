"""Synthetic CPU contract for R09-B C5A owner/segment transactions.

This module is intentionally isolated from Cosmos production/runtime wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


@dataclass(frozen=True)
class AdmissionCapability:
    owner_key: str
    source_identity: str
    source_timestep: int
    source_bytes: bytes
    provenance_class: str = "R08_COMPLETED_CAUSAL"

    @property
    def digest(self) -> str:
        h = sha256()
        for value in (self.owner_key.encode(), self.source_identity.encode(), str(self.source_timestep).encode(), self.source_bytes):
            h.update(len(value).to_bytes(8, "little"))
            h.update(value)
        return h.hexdigest()

    @property
    def source_key(self) -> tuple[str, str, int, str]:
        return (self.owner_key, self.source_identity, self.source_timestep, self.digest)


@dataclass
class _Pending:
    epoch: int
    next_step: int
    segment_id: int
    rows: list[tuple[AdmissionCapability, torch.Tensor]]
    replay: dict[tuple[str, str, int, str], torch.Tensor]


class C5AOwnerSegmentCPU:
    """Small owner-local transaction wrapper around the approved C5 core."""

    def __init__(self, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        if encoder.evidence_dim != core.evidence_dim:
            raise ValueError("encoder/core evidence dimensions must match")
        self.encoder, self.core = encoder, core
        self.epoch = 0
        self.last_step = -1
        self._state: ContinualTTTFastState | None = None
        self._pending: _Pending | None = None
        self._committed: dict[tuple[str, str, int, str], torch.Tensor] = {}
        self.c5_write_count = 0

    def _lookup(self, cap: AdmissionCapability) -> torch.Tensor | None:
        if cap.provenance_class != "R08_COMPLETED_CAUSAL" or not isinstance(cap.source_bytes, bytes):
            raise ValueError("untrusted admission capability")
        if self._pending is not None and cap.source_key in self._pending.replay:
            return self._pending.replay[cap.source_key]
        return self._committed.get(cap.source_key)

    def begin(self) -> None:
        if self._pending is not None:
            raise RuntimeError("pending transaction already exists")
        self._pending = _Pending(self.epoch, self.last_step + 1, self.last_step // self.core.ttt_tbptt_steps + 1, [], {})

    def admit(self, cap: AdmissionCapability, *, encoded: torch.Tensor) -> tuple[int, int, int] | torch.Tensor:
        if self._pending is None:
            raise RuntimeError("begin() required")
        cached = self._lookup(cap)
        if cached is not None:
            return cached
        if encoded.shape != (self.core.evidence_dim,):
            raise ValueError("C5 evidence must be [256]")
        if self._pending.rows and cap.source_timestep <= self._pending.rows[-1][0].source_timestep:
            raise ValueError("out-of-order source timestep")
        binding = (self._pending.epoch, self._pending.next_step, self._pending.segment_id, len(self._pending.rows))
        self._pending.rows.append((cap, encoded))
        self._pending.next_step += 1
        return binding

    def materialize(self, *, history_visual_summary: torch.Tensor, local_history_action: torch.Tensor,
                    history_age_steps: torch.Tensor, history_dt_s: torch.Tensor, history_mask: torch.Tensor,
                    history_state: torch.Tensor | None = None) -> torch.Tensor:
        if self._pending is None or not self._pending.rows:
            raise RuntimeError("no pending rows")
        evidence_history = self.encoder(history_visual_summary=history_visual_summary, local_history_action=local_history_action,
                                        history_age_steps=history_age_steps, history_dt_s=history_dt_s,
                                        history_mask=history_mask, history_state=history_state)
        if evidence_history.shape[-1] != self.core.evidence_dim:
            raise ValueError("C5 evidence must be [B,256]")
        state = self.core.initial_state(1, device=evidence_history.device) if self._state is None else self._state
        tokens, state, _ = self.core.scan_segment_many(evidence_history[:, -len(self._pending.rows):],
                                                        torch.ones(evidence_history.shape[0], len(self._pending.rows), dtype=torch.bool), state)
        self.c5_write_count += len(self._pending.rows)
        self._state = state
        for index, (cap, _) in enumerate(self._pending.rows):
            self._pending.replay[cap.source_key] = tokens[:, index].detach().clone()
        return tokens

    def commit(self) -> None:
        if self._pending is None:
            raise RuntimeError("no pending transaction")
        self._committed.update({key: value.detach().clone() for key, value in self._pending.replay.items()})
        self.last_step = self._pending.next_step - 1
        self._pending = None

    def abort(self) -> None:
        self._pending = None


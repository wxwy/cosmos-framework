"""Isolated synthetic CPU contract for the R09-B C5A transaction boundary."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


@dataclass(frozen=True)
class AdmissionCapability:
    owner_key: str
    source_identity: str
    source_timestep: int
    source_bytes: bytes
    source_shape: tuple[int, ...]
    source_dtype: str
    _seal: object

    @property
    def digest(self) -> str:
        h = sha256()
        for value in (self.owner_key.encode(), self.source_identity.encode(), self.source_timestep.to_bytes(8, "little", signed=True), self.source_dtype.encode(), repr(self.source_shape).encode(), self.source_bytes):
            h.update(len(value).to_bytes(8, "little")); h.update(value)
        return h.hexdigest()

    @property
    def source_key(self) -> tuple[str, str, int, str]:
        return (self.owner_key, self.source_identity, self.source_timestep, self.digest)


class AdmissionAuthority:
    """Trusted synthetic producer; callers cannot construct a sealed capability."""

    def __init__(self) -> None:
        self._seal = object()

    def issue(self, *, owner_key: str, source_identity: str, source_timestep: int, source: dict[str, torch.Tensor]) -> AdmissionCapability:
        if source_timestep < 0 or not source or any(not isinstance(value, torch.Tensor) for value in source.values()):
            raise ValueError("invalid causal source")
        chunks = []
        for name in sorted(source):
            value = source[name].detach().contiguous().cpu()
            raw = value.numpy().tobytes()
            chunks.append(name.encode() + len(raw).to_bytes(8, "little") + raw)
        first = next(iter(source.values()))
        return AdmissionCapability(owner_key, source_identity, source_timestep, b"".join(chunks), tuple(first.shape), str(first.dtype), self._seal)


@dataclass
class _Pending:
    base_state: ContinualTTTFastState | None
    rows: list[tuple[AdmissionCapability, dict[str, torch.Tensor]]]
    replay: dict[tuple[str, str, int, str], torch.Tensor]


class C5AOwnerSegmentCPU:
    """Owner-local C/P transaction wrapper; never imported by production runtime."""

    def __init__(self, authority: AdmissionAuthority, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        if encoder.evidence_dim != core.evidence_dim:
            raise ValueError("encoder/core evidence dimensions must match")
        self.authority, self.encoder, self.core = authority, encoder, core
        self._state_by_owner: dict[str, ContinualTTTFastState] = {}
        self._pending_by_owner: dict[str, _Pending] = {}
        self._committed: dict[tuple[str, str, int, str], torch.Tensor] = {}
        self._identity_index: dict[tuple[str, str, int], str] = {}
        self.c5_write_count = 0

    def begin(self, owner_key: str) -> None:
        if owner_key in self._pending_by_owner:
            raise RuntimeError("pending transaction already exists")
        self._pending_by_owner[owner_key] = _Pending(self._state_by_owner.get(owner_key), [], {})

    def admit(self, cap: AdmissionCapability, *, source: dict[str, torch.Tensor]) -> tuple[int, int] | torch.Tensor:
        if cap._seal is not self.authority._seal or cap.source_timestep < 0:
            raise ValueError("untrusted admission capability")
        if cap.owner_key not in self._pending_by_owner:
            raise RuntimeError("begin() required")
        key = cap.source_key
        identity = (cap.owner_key, cap.source_identity, cap.source_timestep)
        prior_digest = self._identity_index.get(identity)
        if prior_digest is not None and prior_digest != cap.digest:
            raise ValueError("conflicting source digest")
        pending = self._pending_by_owner[cap.owner_key]
        cached = pending.replay.get(key)
        if cached is None:
            cached = self._committed.get(key)
        if cached is not None:
            return cached
        payload = {name: value.detach().clone() for name, value in source.items()}
        if not payload or any(value.requires_grad for value in payload.values()):
            raise ValueError("source payload must be immutable and graph-free")
        self._identity_index[identity] = cap.digest
        if pending.rows and cap.source_timestep != pending.rows[-1][0].source_timestep + 1:
            raise ValueError("source timestep must be contiguous")
        pending.rows.append((cap, payload))
        return (len(pending.rows) - 1, cap.source_timestep)

    def materialize(self, owner_key: str) -> torch.Tensor:
        pending = self._pending_by_owner.get(owner_key)
        if pending is None or not pending.rows:
            raise RuntimeError("no pending rows")
        encoded = []
        for _, source in pending.rows:
            value = self.encoder(**source)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != self.core.evidence_dim:
                raise ValueError("C5 evidence must be [B,256]")
            encoded.append(value[:, -1])
        evidence = torch.cat(encoded, dim=0)
        state = pending.base_state or self.core.initial_state(len(encoded), device=evidence.device)
        tokens, candidate, _ = self.core.step_many(evidence, state, torch.ones(len(encoded), dtype=torch.bool), create_graph=True)
        pending.base_state = candidate
        for row, (cap, _) in enumerate(pending.rows):
            pending.replay[cap.source_key] = tokens[row].detach().clone()
        self.c5_write_count += len(encoded)
        return tokens

    def commit(self, owner_key: str) -> None:
        pending = self._pending_by_owner.pop(owner_key)
        if pending.base_state is not None:
            self._state_by_owner[owner_key] = ContinualTTTFastState(*(value.detach().clone() for value in pending.base_state))
        self._committed.update({key: value.detach().clone() for key, value in pending.replay.items()})

    def abort(self, owner_key: str) -> None:
        self._pending_by_owner.pop(owner_key)


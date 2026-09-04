"""Isolated synthetic CPU contract for the R09-B C5A transaction boundary."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _canonical_source(source: dict[str, torch.Tensor]) -> bytes:
    """Deterministic field/name/dtype/shape/bytes serialization."""
    chunks: list[bytes] = []
    for name in sorted(source):
        value = source[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("source fields must be tensors")
        value = value.detach().contiguous().cpu()
        name_b, dtype_b, shape_b, raw = name.encode(), str(value.dtype).encode(), repr(tuple(value.shape)).encode(), value.numpy().tobytes()
        for part in (name_b, dtype_b, shape_b, raw):
            chunks.append(len(part).to_bytes(8, "little") + part)
    return b"".join(chunks)


@dataclass(frozen=True)
class AdmissionCapability:
    owner_key: str
    source_identity: str
    source_timestep: int
    source_bytes: bytes
    source_shape: tuple[int, ...]
    source_dtype: str
    source_schema: tuple[tuple[str, str, tuple[int, ...]], ...]
    _seal: object

    @property
    def digest(self) -> str:
        h = sha256()
        for value in (self.owner_key.encode(), self.source_identity.encode(), self.source_timestep.to_bytes(8, "little", signed=True), repr(self.source_schema).encode(), self.source_bytes):
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
        payload = _canonical_source(source)
        schema = tuple((name, str(source[name].dtype), tuple(source[name].shape)) for name in sorted(source))
        first = next(iter(source.values()))
        return AdmissionCapability(owner_key, source_identity, source_timestep, payload, tuple(first.shape), str(first.dtype), schema, self._seal)


@dataclass
class _Pending:
    base_state: ContinualTTTFastState | None
    rows: list[tuple[AdmissionCapability, dict[str, torch.Tensor]]]
    replay: dict[tuple[str, str, int, str], torch.Tensor]
    identity_index: dict[tuple[str, str, int], str]
    phase: str = "COLLECT_RAW"


class C5AOwnerSegmentCPU:
    """Owner-local C/P transaction wrapper; never imported by production runtime."""

    def __init__(self, authority: AdmissionAuthority, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        if encoder.evidence_dim != core.evidence_dim:
            raise ValueError("encoder/core evidence dimensions must match")
        self.authority, self.encoder, self.core = authority, encoder, core
        self.segment_steps = core.ttt_tbptt_steps
        self._state_by_owner: dict[str, ContinualTTTFastState] = {}
        self._last_timestep: dict[str, int] = {}
        self._epoch_by_owner: dict[str, int] = {}
        self._pending_by_owner: dict[str, _Pending] = {}
        self._committed: dict[tuple[str, str, int, str], torch.Tensor] = {}
        self._identity_index: dict[tuple[str, str, int], str] = {}
        self.c5_write_count = 0

    def begin(self, owner_key: str) -> None:
        if owner_key in self._pending_by_owner:
            raise RuntimeError("pending transaction already exists")
        self._pending_by_owner[owner_key] = _Pending(self._state_by_owner.get(owner_key), [], {}, {}, "COLLECT_RAW")

    def admit(self, cap: AdmissionCapability, *, source: dict[str, torch.Tensor]) -> tuple[int, int] | torch.Tensor:
        if cap._seal is not self.authority._seal or cap.source_timestep < 0:
            raise ValueError("untrusted admission capability")
        if cap.owner_key not in self._pending_by_owner:
            raise RuntimeError("begin() required")
        key = cap.source_key
        identity = (cap.owner_key, cap.source_identity, cap.source_timestep)
        pending = self._pending_by_owner[cap.owner_key]
        prior_digest = pending.identity_index.get(identity) or self._identity_index.get(identity)
        if prior_digest is not None and prior_digest != cap.digest:
            raise ValueError("conflicting source digest")
        if _canonical_source(source) != cap.source_bytes:
            raise ValueError("source bytes do not match admission capability")
        schema = tuple((name, str(source[name].dtype), tuple(source[name].shape)) for name in sorted(source))
        if schema != cap.source_schema or cap.digest != AdmissionCapability(cap.owner_key, cap.source_identity, cap.source_timestep, cap.source_bytes, cap.source_shape, cap.source_dtype, cap.source_schema, cap._seal).digest:
            raise ValueError("capability fields are not canonical")
        cached = pending.replay.get(key)
        if cached is None:
            cached = self._committed.get(key)
        if cached is not None:
            return cached
        payload = {name: value.detach().clone() for name, value in source.items()}
        if not payload or any(value.requires_grad for value in payload.values()):
            raise ValueError("source payload must be immutable and graph-free")
        expected = self._last_timestep.get(cap.owner_key, -1) + len(pending.rows) + 1
        if cap.source_timestep != expected:
            raise ValueError("source timestep must be contiguous")
        pending.identity_index[identity] = cap.digest
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
        # 时间轴显式保留为 [B=1,T,D]；每个 timestep 顺序消费同一 owner 的候选 state。
        evidence = torch.stack(encoded, dim=1)
        state = pending.base_state or self.core.initial_state(1, device=evidence.device)
        tokens, candidate, _ = self.core.scan_segment_many(
            evidence, torch.ones(1, evidence.shape[1], dtype=torch.bool, device=evidence.device), state, create_graph=True
        )
        pending.base_state = candidate
        pending.phase = "MATERIALIZED_PENDING"
        for row, (cap, _) in enumerate(pending.rows):
            pending.replay[cap.source_key] = tokens[0, row].detach().clone()
        self.c5_write_count += len(encoded)
        return tokens[0]

    def materialize_many(self, owner_keys: list[str]) -> torch.Tensor:
        """Gather/scatter independent owner rows; temporal axes never become batch state."""
        if not owner_keys or len(set(owner_keys)) != len(owner_keys):
            raise ValueError("owner_keys must be non-empty and unique")
        rows = [self.materialize(owner_key)[-1] for owner_key in owner_keys]
        return torch.stack(rows, dim=0)

    def commit(self, owner_key: str) -> None:
        pending = self._pending_by_owner.get(owner_key)
        if pending is None:
            raise RuntimeError("no pending transaction")
        if pending.phase != "MATERIALIZED_PENDING":
            raise RuntimeError("commit requires one successful materialize/backward")
        pending.phase = "BACKWARD_OK"
        pending = self._pending_by_owner.pop(owner_key)
        if pending.base_state is not None:
            self._state_by_owner[owner_key] = ContinualTTTFastState(*(value.detach().clone() for value in pending.base_state))
        self._last_timestep[owner_key] = pending.rows[-1][0].source_timestep
        self._epoch_by_owner.setdefault(owner_key, 0)
        self._committed.update({key: value.detach().clone() for key, value in pending.replay.items()})
        self._identity_index.update(pending.identity_index)

    def abort(self, owner_key: str) -> None:
        self._pending_by_owner.pop(owner_key)

    def reset(self, owner_key: str) -> None:
        if owner_key in self._pending_by_owner:
            raise RuntimeError("cannot reset owner with pending transaction")
        self._state_by_owner.pop(owner_key, None)
        self._last_timestep.pop(owner_key, None)
        self._committed = {key: value for key, value in self._committed.items() if key[0] != owner_key}
        self._identity_index = {key: value for key, value in self._identity_index.items() if key[0] != owner_key}
        self._epoch_by_owner[owner_key] = self._epoch_by_owner.get(owner_key, 0) + 1

    def finish(self, owner_key: str, *, terminal: bool) -> None:
        """Close one segment; terminal remainder is the sole short-segment exception."""
        pending = self._pending_by_owner.get(owner_key)
        if pending is None:
            raise RuntimeError("no pending transaction")
        count = len(pending.rows)
        if count == 0 and terminal:
            self.abort(owner_key)
            self.reset(owner_key)
            return
        if count == 0 or count > self.segment_steps or (not terminal and count != self.segment_steps):
            raise ValueError("invalid segment length")
        self.materialize(owner_key)
        self.commit(owner_key)
        if terminal:
            self.reset(owner_key)

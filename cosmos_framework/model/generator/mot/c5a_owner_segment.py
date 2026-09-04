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
    epoch: int
    provenance_class: str
    _seal: object

    @property
    def digest(self) -> str:
        h = sha256()
        for value in (self.owner_key.encode(), self.source_identity.encode(), self.source_timestep.to_bytes(8, "little", signed=True), self.epoch.to_bytes(8, "little", signed=True), self.provenance_class.encode(), repr(self.source_schema).encode(), self.source_bytes):
            h.update(len(value).to_bytes(8, "little")); h.update(value)
        return h.hexdigest()

    @property
    def source_key(self) -> tuple[str, str, int, str]:
        return (self.owner_key, self.source_identity, self.source_timestep, self.digest)


@dataclass(frozen=True)
class ReplayRecord:
    value: torch.Tensor
    shape: tuple[int, ...]
    present: bool


class AdmissionAuthority:
    """Trusted synthetic producer; callers cannot construct a sealed capability."""

    def __init__(self) -> None:
        self._issued: dict[int, tuple[AdmissionCapability, bytes]] = {}

    def issue(self, *, owner_key: str, source_identity: str, source_timestep: int, source: dict[str, torch.Tensor], epoch: int = 0, provenance_class: str = "R08_COMPLETED_CAUSAL") -> AdmissionCapability:
        if source_timestep < 0 or epoch < 0 or not source or any(not isinstance(value, torch.Tensor) for value in source.values()):
            raise ValueError("invalid causal source")
        payload = _canonical_source(source)
        schema = tuple((name, str(source[name].dtype), tuple(source[name].shape)) for name in sorted(source))
        first = next(iter(source.values()))
        if provenance_class != "R08_COMPLETED_CAUSAL":
            raise ValueError("unsupported provenance class")
        cap = AdmissionCapability(owner_key, source_identity, source_timestep, payload, tuple(first.shape), str(first.dtype), schema, epoch, provenance_class, object())
        self._issued[id(cap._seal)] = (cap, payload)
        return cap

    def verify(self, cap: AdmissionCapability) -> None:
        record = self._issued.get(id(cap._seal))
        if record is None or record[0] != cap:
            raise ValueError("untrusted admission capability")


@dataclass
class _Pending:
    base_state: ContinualTTTFastState | None
    rows: list[tuple[AdmissionCapability, dict[str, torch.Tensor]]]
    replay: dict[tuple[str, str, int, str], ReplayRecord]
    identity_index: dict[tuple[str, str, int], str]
    phase: str = "COLLECT_RAW"
    witness_grad_fn: object | None = None
    witness: torch.Tensor | None = None
    witness_leaf: torch.Tensor | None = None


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
        self.authority.verify(cap)
        if cap.source_timestep < 0 or cap.epoch != self._epoch_by_owner.get(cap.owner_key, 0) or cap.provenance_class != "R08_COMPLETED_CAUSAL":
            raise ValueError("stale admission capability epoch")
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
        if schema != cap.source_schema:
            raise ValueError("capability fields are not canonical")
        cached = pending.replay.get(key)
        if cached is None:
            cached = self._committed.get(key)
        if cached is not None:
            return cached
        if pending.phase != "COLLECT_RAW":
            raise RuntimeError("unseen admission after materialize")
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
        if pending.phase != "COLLECT_RAW":
            raise RuntimeError("materialize already completed")
        encoded = []
        for _, source in pending.rows:
            value = self.encoder(**source)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != self.core.evidence_dim:
                raise ValueError("C5 evidence must be [B,256]")
            encoded.append(value[:, -1])
        # 时间轴显式保留为 [B=1,T,D]；每个 timestep 顺序消费同一 owner 的候选 state。
        evidence = torch.stack(encoded, dim=1)
        state = pending.base_state or self.core.initial_state(1, device=evidence.device)
        tokens, candidate, present = self.core.scan_segment_many(
            evidence, torch.ones(1, evidence.shape[1], dtype=torch.bool, device=evidence.device), state, create_graph=True
        )
        pending.base_state = candidate
        pending.witness_leaf = torch.ones((), device=tokens.device, requires_grad=True)
        pending.witness = tokens + pending.witness_leaf * 0
        pending.witness_grad_fn = pending.witness.grad_fn
        pending.phase = "MATERIALIZED_PENDING"
        for row, (cap, _) in enumerate(pending.rows):
            value = tokens[0, row].detach().clone()
            pending.replay[cap.source_key] = ReplayRecord(value, tuple(value.shape), bool(present[0, row]))
        self.c5_write_count += len(encoded)
        return pending.witness[0]

    def materialize_many(
        self,
        owner_keys: list[str],
        *,
        valid: torch.Tensor | None = None,
        done_before: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather/scatter independent owner rows as one true B>1 scan."""
        if not owner_keys or len(set(owner_keys)) != len(owner_keys):
            raise ValueError("owner_keys must be non-empty and unique")
        pendings = [self._pending_by_owner.get(owner) for owner in owner_keys]
        if any(p is None or not p.rows or p.phase != "COLLECT_RAW" for p in pendings):
            raise RuntimeError("all owners require raw pending rows")
        steps = {len(p.rows) for p in pendings if p is not None}
        if len(steps) != 1:
            raise ValueError("materialize_many requires equal segment lengths")
        step_count = next(iter(steps))
        batch = len(owner_keys)
        if valid is None:
            valid = torch.ones(batch, step_count, dtype=torch.bool)
        if done_before is None:
            done_before = torch.zeros(batch, dtype=torch.bool)
        if tuple(valid.shape) != (batch, step_count) or valid.dtype != torch.bool:
            raise ValueError("valid must have shape [B,T] and bool dtype")
        if tuple(done_before.shape) != (batch,) or done_before.dtype != torch.bool:
            raise ValueError("done_before must have shape [B] and bool dtype")
        for owner, done in zip(owner_keys, done_before.tolist(), strict=True):
            if done and (owner in self._state_by_owner or owner in self._last_timestep or any(key[0] == owner for key in self._committed)):
                raise RuntimeError("done_before requires explicit owner reset")
        encoded_rows = []
        for pending in pendings:
            assert pending is not None
            encoded_rows.append(torch.stack([self.encoder(**source)[:, -1] for _, source in pending.rows], dim=1)[0])
        evidence = torch.stack(encoded_rows, dim=0)
        states = [p.base_state for p in pendings]
        if all(state is None for state in states):
            state = self.core.initial_state(len(owner_keys), device=evidence.device)
        elif any(state is None for state in states):
            raise RuntimeError("owner state batch is incomplete")
        else:
            state = ContinualTTTFastState(*(torch.cat([s[i].detach() for s in states], dim=0) for i in range(4)))
        state = self.core.reset_mask(state, done_before.to(device=evidence.device))
        tokens, candidate, present = self.core.scan_segment_many(
            evidence, valid.to(device=evidence.device), state, create_graph=True
        )
        for batch_index, pending in enumerate(pendings):
            assert pending is not None
            pending.base_state = ContinualTTTFastState(*(value[batch_index : batch_index + 1] for value in candidate))
            pending.phase = "MATERIALIZED_PENDING"
            pending.witness_leaf = torch.ones((), device=tokens.device, requires_grad=True)
            pending.witness = tokens[batch_index] + pending.witness_leaf * 0
            pending.witness_grad_fn = pending.witness.grad_fn
            for row, (cap, _) in enumerate(pending.rows):
                value = tokens[batch_index, row].detach().clone()
                pending.replay[cap.source_key] = ReplayRecord(value, tuple(value.shape), bool(present[batch_index, row]))
        self.c5_write_count += int(valid.sum().item())
        return tokens[:, -1]

    def commit(self, owner_key: str) -> None:
        pending = self._pending_by_owner.get(owner_key)
        if pending is None:
            raise RuntimeError("no pending transaction")
        if pending.phase != "BACKWARD_OK":
            raise RuntimeError("commit requires one successful materialize/backward")
        pending = self._pending_by_owner.pop(owner_key)
        if pending.base_state is not None:
            self._state_by_owner[owner_key] = ContinualTTTFastState(*(value.detach().clone() for value in pending.base_state))
        self._last_timestep[owner_key] = pending.rows[-1][0].source_timestep
        self._epoch_by_owner.setdefault(owner_key, 0)
        self._committed.update({key: ReplayRecord(record.value.detach().clone(), record.shape, record.present) for key, record in pending.replay.items()})
        self._identity_index.update(pending.identity_index)

    def backward_and_mark(self, owner_key: str, loss: torch.Tensor) -> None:
        pending = self._pending_by_owner.get(owner_key)
        if pending is None or pending.phase != "MATERIALIZED_PENDING":
            raise RuntimeError("backward requires one successful materialize")
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.grad_fn is None:
            raise RuntimeError("backward loss must be a scalar graph")
        if not self._loss_reaches(loss, pending.witness_leaf):
            raise RuntimeError("backward loss is unrelated to materialized segment")
        try:
            loss.backward()
        except Exception:
            raise
        pending.phase = "BACKWARD_OK"

    @staticmethod
    def _loss_reaches(loss: torch.Tensor, witness_leaf: torch.Tensor | None) -> bool:
        stack = [loss.grad_fn]
        while stack:
            function = stack.pop()
            if function is None:
                continue
            variable = getattr(function, "variable", None)
            if variable is not None and witness_leaf is not None and variable.data_ptr() == witness_leaf.data_ptr():
                return True
            stack.extend(next_function for next_function, _ in function.next_functions if next_function is not None)
        return False

    def backward_and_mark_many(self, owner_keys: list[str], loss: torch.Tensor) -> None:
        if not owner_keys or len(set(owner_keys)) != len(owner_keys):
            raise ValueError("owner_keys must be non-empty and unique")
        pendings = [self._pending_by_owner.get(owner) for owner in owner_keys]
        if any(p is None or p.phase != "MATERIALIZED_PENDING" for p in pendings):
            raise RuntimeError("backward requires materialized owners")
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.grad_fn is None:
            raise RuntimeError("backward loss must be a scalar graph")
        if any(not self._loss_reaches(loss, pending.witness_leaf) for pending in pendings if pending is not None):
            raise RuntimeError("backward loss is unrelated to materialized segment")
        loss.backward()
        for pending in pendings:
            assert pending is not None
            pending.phase = "BACKWARD_OK"

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
        if pending.phase == "COLLECT_RAW":
            raise RuntimeError("finish requires materialize and backward")
        if pending.phase == "MATERIALIZED_PENDING":
            raise RuntimeError("finish requires backward")
        if pending.phase != "BACKWARD_OK":
            raise RuntimeError("invalid finish phase")
        self.commit(owner_key)
        if terminal:
            self.reset(owner_key)

"""Stage3A Owner evidence authority, with no Torch/model/simulator imports.

The authenticated Run Owner records callbacks from Registry-managed Actors in
the explicitly approved same-trust domain. Neither this public receipt nor a
Worker HMAC cryptographically proves an env.step. Sim never receives the Owner
control token or the Worker signing key. Public request data cannot create a
completion or enroll a producer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

from cosmos_framework.inference.robocasa_vae_protocol import (
    IDENTITY_FIELDS,
    ProtocolError,
    canonical_json,
    selected_history_step,
    strict_json,
    validate_history_k,
    validate_identity,
    validate_lease,
    validate_raw15,
    validate_visual96,
    verify_feature,
)

FEATURE_EVIDENCE_VERSION = "PSM24-FEATURES-v1"
FEATURE_EVIDENCE_FORMAT = "sim_shared_vae_b1_t1"
POST_STEP_VERSION = "PSM24-POSTSTEP-v1"
SPARSE_EVIDENCE_VERSION = "PSM28-FEATURES-v2"
SPARSE_EVIDENCE_FORMAT = "sim_shared_vae_sparse_b1_t1"
SPARSE_POST_STEP_VERSION = "PSM28-POSTSTEP-v2"
BOUNDARY_VERSION = "PSM24-BOUNDARY-v1"
TRUST_MODEL = "registry_same_trust_domain"
_POST_FIELDS = {
    "version",
    "lease",
    "issued_chunk_id",
    "policy_request_key",
    "source_step",
    "source_rgb_sha256",
    "raw15_sha256",
    "success",
    "clock_domain",
    "episode_deadline_ns",
    "step_completed_ns",
    "actor_id",
    "actor_epoch",
    "trust_model",
}
_BOUNDARY_FIELDS = {
    "version",
    "lease",
    "issued_chunk_id",
    "policy_request_key",
    "source_step",
    "source_rgb_sha256",
    "visual96_sha256",
    "deadline_ns",
    "clock_domain",
    *IDENTITY_FIELDS,
}


def _exact(value, fields, name):
    if type(value) is not dict or value.keys() != set(fields):
        raise ProtocolError(f"{name} requires exact fields")
    return value


def _id(value, name):
    if type(value) is not str or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ProtocolError(f"invalid {name}")
    return value


def _sha(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ProtocolError(f"invalid {name}")
    return value


def _step(value):
    if type(value) is not int or value < 0:
        raise ProtocolError("invalid source step")
    return value


def _ns(value):
    if type(value) is int:
        if 0 <= value < 10**20:
            return value
    elif type(value) is str and 1 <= len(value) <= 20 and all(c in "0123456789" for c in value):
        return int(value)
    raise ProtocolError("invalid absolute monotonic ns")


def _secret(value):
    if type(value) is str:
        value = value.encode("utf-8")
    if type(value) is not bytes or not 16 <= len(value) <= 256:
        raise PermissionError("bounded Owner control capability required")
    return hashlib.sha256(b"PSM24-owner-control\0" + value).digest()


def validate_post_step_receipt(value):
    """Validate public metadata only; authority additionally requires the Owner ledger."""
    sparse = type(value) is dict and value.get("version") == SPARSE_POST_STEP_VERSION
    value = _exact(value, _POST_FIELDS | ({"history_k", "selected"} if sparse else set()), "post-step receipt")
    if (
        value["version"] not in (POST_STEP_VERSION, SPARSE_POST_STEP_VERSION)
        or value["success"] is not True
        or value["trust_model"] != TRUST_MODEL
    ):
        raise ProtocolError("unsupported post-step provenance")
    selected = True
    if sparse:
        k = validate_history_k(value["history_k"])
        step = _step(value["source_step"])
        selected = selected_history_step(step, step // 16 * 16, k)
        if k == 16 or type(value["selected"]) is not bool or value["selected"] != selected:
            raise ProtocolError("post-step selection differs from frozen sparse K")
        if not selected and value["source_rgb_sha256"] is not None:
            raise ProtocolError("skipped history must not carry an RGB payload digest")
    result = {
        "version": value["version"],
        "lease": validate_lease(value["lease"]),
        "issued_chunk_id": _id(value["issued_chunk_id"], "issued_chunk_id"),
        "policy_request_key": _id(value["policy_request_key"], "policy_request_key"),
        "source_step": _step(value["source_step"]),
        "source_rgb_sha256": _sha(value["source_rgb_sha256"], "source_rgb_sha256") if selected else None,
        "raw15_sha256": _sha(value["raw15_sha256"], "raw15_sha256"),
        "success": True,
        "clock_domain": _id(value["clock_domain"], "clock_domain"),
        "actor_id": _id(value["actor_id"], "actor_id"),
        "actor_epoch": _id(value["actor_epoch"], "actor_epoch"),
        "trust_model": TRUST_MODEL,
    }
    if sparse:
        result.update(history_k=k, selected=selected)
    for name in ("episode_deadline_ns", "step_completed_ns"):
        if type(value[name]) is not str:
            raise ProtocolError("receipt deadlines require decimal strings")
        result[name] = str(_ns(value[name]))
        if result[name] != value[name]:
            raise ProtocolError("noncanonical receipt timestamp")
    if int(result["step_completed_ns"]) >= int(result["episode_deadline_ns"]):
        raise ProtocolError("post-step completed after Episode expiry")
    return result


def _boundary_header(value):
    value = _exact(value, _BOUNDARY_FIELDS, "Policy boundary")
    if value["version"] != BOUNDARY_VERSION or _step(value["source_step"]) % 16:
        raise ProtocolError("Policy boundary must use a real H16 boundary step")
    if type(value["deadline_ns"]) is not str or str(_ns(value["deadline_ns"])) != value["deadline_ns"]:
        raise ProtocolError("boundary deadline requires canonical decimal string")
    return {
        "version": BOUNDARY_VERSION,
        "lease": validate_lease(value["lease"]),
        "issued_chunk_id": _id(value["issued_chunk_id"], "issued_chunk_id"),
        "policy_request_key": _id(value["policy_request_key"], "policy_request_key"),
        "source_step": value["source_step"],
        "source_rgb_sha256": _sha(value["source_rgb_sha256"], "source_rgb_sha256"),
        "visual96_sha256": _sha(value["visual96_sha256"], "visual96_sha256"),
        "deadline_ns": value["deadline_ns"],
        "clock_domain": _id(value["clock_domain"], "clock_domain"),
        **validate_identity({name: value[name] for name in IDENTITY_FIELDS}),
    }


def validate_boundary_record(value):
    """Validate Owner-returned public metadata; visual bytes stay private at Policy."""
    value = _exact(value, {"header", "policy_boundary_sha256"}, "boundary record")
    return {
        "header": _boundary_header(value["header"]),
        "policy_boundary_sha256": _sha(value["policy_boundary_sha256"], "policy_boundary_sha256"),
    }


@dataclass(frozen=True, slots=True)
class PolicyBoundary:
    header_json: bytes
    visual96: bytes

    def __post_init__(self):
        header = _boundary_header(strict_json(self.header_json, max_bytes=8192))
        visual = validate_visual96(self.visual96)
        if (
            canonical_json(header) != self.header_json
            or hashlib.sha256(visual).hexdigest() != header["visual96_sha256"]
        ):
            raise ProtocolError("Policy boundary bytes/digest mismatch")
        object.__setattr__(self, "visual96", visual)

    @classmethod
    def create(
        cls,
        *,
        lease,
        request_key,
        chunk_id,
        source_step,
        source_rgb_sha256,
        identity,
        deadline_ns,
        clock_domain,
        visual96,
    ):
        visual = validate_visual96(visual96)
        header = {
            "version": BOUNDARY_VERSION,
            "lease": validate_lease(lease),
            "policy_request_key": request_key,
            "issued_chunk_id": chunk_id,
            "source_step": source_step,
            "source_rgb_sha256": source_rgb_sha256,
            "visual96_sha256": hashlib.sha256(visual).hexdigest(),
            "deadline_ns": str(_ns(deadline_ns)),
            "clock_domain": clock_domain,
            **validate_identity(identity),
        }
        return cls(canonical_json(_boundary_header(header)), visual)

    @property
    def header(self):
        return strict_json(self.header_json, max_bytes=8192)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            b"PSM24-BOUNDARY-v1\0" + struct.pack(">I", len(self.header_json)) + self.header_json + self.visual96
        ).hexdigest()

    def record(self):
        return {"header": self.header, "policy_boundary_sha256": self.fingerprint}


@dataclass(frozen=True, slots=True)
class FeatureEvidenceBatch:
    source_steps: tuple
    visual96: tuple
    raw15: tuple
    wire_digest: str


def feature_evidence_digest(rows):
    if type(rows) is not list or len(rows) > 16:
        raise ProtocolError("bounded feature rows required")
    return hashlib.sha256(b"PSM24-EVIDENCE-v1\0" + canonical_json(rows)).hexdigest()


@dataclass(frozen=True, slots=True)
class _Worker:
    worker_id: str
    epoch: str
    key_id: str
    key: bytes = field(repr=False)
    identity_json: bytes
    clock_domain: str


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk_id: str
    request_key: str
    start: int
    actions: tuple
    deadline: int
    issued_ns: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class _Completion:
    receipt_json: bytes
    raw15: bytes


@dataclass
class _Session:
    lease: dict
    worker: _Worker
    actor_id: str
    actor_epoch: str
    deadline: int
    frontier: int = 0
    last_completed_ns: int = 0
    chunk: _Chunk | None = None
    completed: dict = field(default_factory=dict)
    revoked: bool = False


class FeatureEvidenceOwner:
    """Bounded completion authority; FastState/cache publication remains in the adapter."""

    def __init__(
        self,
        owner_token,
        *,
        clock_ns=time.monotonic_ns,
        max_sessions=1,
        trust_model=TRUST_MODEL,
        policy_identity=None,
        history_k=16,
    ):
        if trust_model != TRUST_MODEL or type(max_sessions) is not int or not 1 <= max_sessions <= 64:
            raise ValueError("explicit bounded Registry same-trust-domain owner required")
        self._control = _secret(owner_token)
        self.clock_ns = clock_ns
        self.max_sessions = max_sessions
        self.history_k = validate_history_k(history_k)
        self._workers = {}
        self._revoked_workers = OrderedDict()
        self._sessions = {}
        self._retired = OrderedDict()
        self._boundary_lookup = None
        self._policy_identity = None if policy_identity is None else canonical_json(validate_identity(policy_identity))
        self._lock = threading.RLock()

    @property
    def identity(self):
        return None if self._policy_identity is None else strict_json(self._policy_identity)

    def attest_policy_identity(self, token, identity):
        """Owner initialization after actual loaded-file/config attestation, never from request metadata."""
        self._auth(token)
        encoded = canonical_json(validate_identity(identity))
        with self._lock:
            if self._policy_identity is not None and self._policy_identity != encoded:
                raise ProtocolError("loaded Policy VAE identity cannot change within an Owner epoch")
            self._policy_identity = encoded

    def _auth(self, token):
        if not hmac.compare_digest(_secret(token), self._control):
            raise PermissionError("Owner control authentication failed")

    def bind_boundary_lookup(self, lookup):
        """Internal wiring only: lookup must read ONE immutable snapshot without waiting/locking."""
        if not callable(lookup) or self._boundary_lookup is not None:
            raise ValueError("one nonblocking committed-boundary lookup required")
        self._boundary_lookup = lookup

    def _session(self, lease):
        public = validate_lease(lease)
        session = self._sessions.get(public["session_id"])
        if session is None or session.lease != public or session.revoked:
            raise ProtocolError("inactive or stale authoritative feature Session")
        if self._workers.get(session.worker.worker_id) is not session.worker:
            raise ProtocolError("Worker enrollment revoked")
        return session

    def _boundary(self, session, source_step):
        if self._boundary_lookup is None:
            raise ProtocolError("committed Policy boundary lookup is not wired")
        boundary = self._boundary_lookup(session.lease["session_id"], source_step)
        if type(boundary) is not PolicyBoundary:
            raise ProtocolError("matching committed Policy boundary is unavailable")
        boundary = PolicyBoundary(boundary.header_json, boundary.visual96)
        header = boundary.header
        identity = strict_json(session.worker.identity_json)
        if (
            header["lease"] != session.lease
            or header["source_step"] != source_step
            or header["clock_domain"] != session.worker.clock_domain
            or any(header[name] != identity[name] for name in IDENTITY_FIELDS)
            or int(header["deadline_ns"]) > session.deadline
        ):
            raise ProtocolError("committed boundary authoritative identity mismatch")
        return boundary

    def enroll_worker(self, token, *, worker_id, worker_epoch, key_id, key, identity, clock_domain, bmax=1, wait_ms=0):
        self._auth(token)
        if type(bmax) is not int or bmax != 1 or type(wait_ms) is not int or wait_ms != 0:
            raise ProtocolError("Stage3A Action evidence requires attested Worker B1/wait0; B4/B8 disabled")
        if type(key) is not bytes or len(key) != 32:
            raise ProtocolError("32-byte private Worker signing key required")
        worker = _Worker(
            _id(worker_id, "worker_id"),
            _id(worker_epoch, "worker_epoch"),
            _id(key_id, "key_id"),
            memoryview(key).tobytes(),
            canonical_json(validate_identity(identity)),
            _id(clock_domain, "clock_domain"),
        )
        with self._lock:
            if self._policy_identity is None or worker.identity_json != self._policy_identity:
                raise ProtocolError("Worker VAE identity does not match attested loaded Policy VAE")
            active = self._workers.get(worker.worker_id)
            previous = active or self._revoked_workers.get(worker.worker_id)
            if previous is not None:
                if previous == worker and active is not None:
                    return {"worker_id": worker.worker_id, "worker_epoch": worker.epoch, "key_id": worker.key_id}
                if previous.epoch == worker.epoch or previous.key_id == worker.key_id or previous.key == worker.key:
                    raise ProtocolError("Worker restart requires fresh epoch, key id and signing key")
                for session in self._sessions.values():
                    if session.worker is previous:
                        session.revoked = True
            if active is None and len(self._workers) >= self.max_sessions:
                raise RuntimeError("feature Worker enrollment capacity exhausted")
            self._workers[worker.worker_id] = worker
            self._revoked_workers.pop(worker.worker_id, None)
        return {"worker_id": worker.worker_id, "worker_epoch": worker.epoch, "key_id": worker.key_id}

    def revoke_worker(self, token, worker_id, worker_epoch):
        self._auth(token)
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or worker.epoch != worker_epoch:
                raise ProtocolError("stale Worker revoke")
            del self._workers[worker_id]
            self._revoked_workers[worker_id] = worker
            while len(self._revoked_workers) > self.max_sessions:
                self._revoked_workers.popitem(last=False)
            for session in self._sessions.values():
                if session.worker is worker:
                    session.revoked = True

    def register_session(self, token, lease, *, worker_id, actor_id, actor_epoch, episode_deadline_ns):
        self._auth(token)
        public = validate_lease(lease)
        deadline = _ns(episode_deadline_ns)
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or deadline <= self.clock_ns():
                raise ProtocolError("active Worker and original future Episode deadline required")
            proposed = _Session(public, worker, _id(actor_id, "actor_id"), _id(actor_epoch, "actor_epoch"), deadline)
            existing = self._sessions.get(public["session_id"])
            if existing is not None:
                if (
                    existing.lease != public
                    or existing.worker is not worker
                    or existing.revoked
                    or (existing.actor_id, existing.actor_epoch, existing.deadline)
                    != (proposed.actor_id, proposed.actor_epoch, deadline)
                ):
                    raise ProtocolError("changed feature Session registration")
                return {"registered": True, "duplicate": True}
            if public["session_id"] in self._retired or len(self._sessions) >= self.max_sessions:
                raise RuntimeError("retired Session or feature Session capacity exhausted")
            self._sessions[public["session_id"]] = proposed
            return {"registered": True, "duplicate": False}

    def boundary_record(self, token, lease, source_step):
        self._auth(token)
        with self._lock:
            return self._boundary(self._session(lease), _step(source_step)).record()

    def validate_boundary_candidate(self, boundary):
        if type(boundary) is not PolicyBoundary:
            raise ProtocolError("PolicyBoundary required")
        boundary = PolicyBoundary(boundary.header_json, boundary.visual96)
        header = boundary.header
        with self._lock:
            session = self._session(header["lease"])
            identity = strict_json(session.worker.identity_json)
            if (
                header["clock_domain"] != session.worker.clock_domain
                or any(header[name] != identity[name] for name in IDENTITY_FIELDS)
                or int(header["deadline_ns"]) > session.deadline
                or self.clock_ns() >= min(int(header["deadline_ns"]), session.deadline)
            ):
                raise ProtocolError("Policy boundary candidate identity/deadline mismatch")
        return boundary

    def validate_request(self, lease, deadline_ns, clock_domain):
        with self._lock:
            session = self._session(lease)
            deadline = _ns(deadline_ns)
            if (
                clock_domain != session.worker.clock_domain
                or deadline > session.deadline
                or self.clock_ns() >= min(deadline, session.deadline)
            ):
                raise ProtocolError("original feature request budget/clock mismatch or expiry")

    @contextmanager
    def publication_guard(self, boundaries):
        """Linearize enrollment revocation against the existing ONE snapshot publication.

        Adapter lock precedes Owner lock. Owner boundary lookup must never take
        the adapter lock, so Owner completion and publication cannot deadlock.
        """
        with self._lock:
            for boundary in boundaries:
                self.validate_boundary_candidate(boundary)
            yield

    def commit_action_chunk(self, token, lease, *, chunk_id, request_key, source_start, actions_f32le, deadline_ns):
        """Called only for an existing committed native Response, after canonicalizing its H16 actions."""
        self._auth(token)
        start, deadline = _step(source_start), _ns(deadline_ns)
        if start % 16 or type(actions_f32le) is not bytes or len(actions_f32le) != 16 * 60:
            raise ProtocolError("committed H16 canonical raw15 chunk required")
        actions = tuple(validate_raw15(actions_f32le[index * 60 : (index + 1) * 60]) for index in range(16))
        with self._lock:
            proposed = _Chunk(
                _id(chunk_id, "chunk_id"), _id(request_key, "request_key"), start, actions, deadline, self.clock_ns()
            )
            session = self._session(lease)
            if proposed == session.chunk:
                return {"committed": True, "duplicate": True}
            if self.clock_ns() >= min(deadline, session.deadline) or deadline > session.deadline:
                raise ProtocolError("original committed Action request deadline expired/extended")
            if start != session.frontier or (session.chunk is not None and start != session.chunk.start + 16):
                raise ProtocolError("Action chunk cannot skip unfinished completed frontier")
            boundary = self._boundary(session, start).header
            if (boundary["issued_chunk_id"], boundary["policy_request_key"], int(boundary["deadline_ns"])) != (
                proposed.chunk_id,
                proposed.request_key,
                proposed.deadline,
            ):
                raise ProtocolError("Action chunk differs from atomically committed Policy boundary")
            session.chunk = proposed
            session.completed.clear()
            return {"committed": True, "duplicate": False}

    def complete_post_step(
        self,
        token,
        lease,
        *,
        chunk_id,
        request_key,
        source_step,
        source_rgb_sha256,
        raw15,
        step_completed_ns,
        actor_id,
        actor_epoch,
    ):
        """Record a successful trusted Actor callback; failed env.step must never call this."""
        self._auth(token)
        step, completed_ns = _step(source_step), _ns(step_completed_ns)
        raw = validate_raw15(raw15)
        with self._lock:
            session = self._session(lease)
            chunk = session.chunk
            if chunk is None or (chunk.chunk_id, chunk.request_key) != (chunk_id, request_key):
                raise ProtocolError("post-step does not name an authoritative committed Action chunk")
            if (actor_id, actor_epoch) != (session.actor_id, session.actor_epoch):
                raise ProtocolError("unregistered Actor provenance")
            if not chunk.start <= step < chunk.start + 16 or raw != chunk.actions[step - chunk.start]:
                raise ProtocolError("post-step action differs from canonical committed Action")
            receipt = validate_post_step_receipt(
                {
                    "version": POST_STEP_VERSION if self.history_k == 16 else SPARSE_POST_STEP_VERSION,
                    "lease": session.lease,
                    "issued_chunk_id": chunk_id,
                    "policy_request_key": request_key,
                    "source_step": step,
                    "source_rgb_sha256": source_rgb_sha256,
                    "raw15_sha256": hashlib.sha256(raw).hexdigest(),
                    "success": True,
                    "clock_domain": session.worker.clock_domain,
                    "episode_deadline_ns": str(session.deadline),
                    "step_completed_ns": str(completed_ns),
                    "actor_id": actor_id,
                    "actor_epoch": actor_epoch,
                    "trust_model": TRUST_MODEL,
                    **(
                        {
                            "history_k": self.history_k,
                            "selected": selected_history_step(step, chunk.start, self.history_k),
                        }
                        if self.history_k != 16
                        else {}
                    ),
                }
            )
            encoded = canonical_json(receipt)
            previous = session.completed.get(step)
            if previous is not None:
                if previous.receipt_json != encoded or previous.raw15 != raw:
                    raise ProtocolError("changed-bytes post-step retry")
                return receipt
            if (
                step != session.frontier
                or completed_ns < max(session.last_completed_ns, chunk.issued_ns)
                or completed_ns > self.clock_ns()
            ):
                raise ProtocolError("post-step completion is missing, reordered or from a future clock")
            if self.clock_ns() >= session.deadline:
                raise ProtocolError("Episode expired before post-step admission")
            if step % 16 == 0:
                header = self._boundary(session, step).header
                if (header["issued_chunk_id"], header["policy_request_key"], header["source_rgb_sha256"]) != (
                    chunk_id,
                    request_key,
                    receipt["source_rgb_sha256"],
                ):
                    raise ProtocolError("executed boundary RGB/Action chunk differs from committed current image")
            session.completed[step] = _Completion(encoded, raw)
            session.frontier = step + 1
            session.last_completed_ns = completed_ns
            return receipt

    def admit(self, payload, *, lease, deadline_ns, clock_domain, previous_step):
        """Read-only admission before native READY; no source data can create completion."""
        if self.history_k != 16:
            if (
                payload.get("evidence_version") != SPARSE_EVIDENCE_VERSION
                or payload.get("evidence_format") != SPARSE_EVIDENCE_FORMAT
                or type(payload.get("local_ttt_history_k")) is not int
                or payload["local_ttt_history_k"] != self.history_k
            ):
                raise ProtocolError("sparse feature selection differs from frozen Owner K")
        elif "local_ttt_history_k" in payload or payload.get("evidence_version") == SPARSE_EVIDENCE_VERSION:
            raise ProtocolError("dense Owner cannot accept sparse selection evidence")
        rows = payload.get("evidence")
        step = payload.get("consumer_step")
        if type(rows) is not list or type(step) is not int or type(previous_step) is not int:
            raise ProtocolError("invalid feature evidence chronology")
        if not 0 <= previous_step <= step or step - previous_step > 16:
            raise ProtocolError("feature history must contain exactly the actual contiguous completed steps")
        steps = tuple(range(previous_step, step))
        if self.history_k != 16 and steps and (previous_step % 16 or len(steps) != 16):
            raise ProtocolError("sparse Feature prepare requires one complete H16 Action chunk")
        if len(rows) != len(steps):
            raise ProtocolError("feature history must contain exactly the actual contiguous completed steps")
        deadline = _ns(deadline_ns)
        with self._lock:
            now = self.clock_ns()
            session = self._session(lease)
            if (
                payload.get("session_id") != session.lease["session_id"]
                or payload.get("episode_id") != session.lease["episode_id"]
            ):
                raise ProtocolError("feature payload Session differs from authenticated native Lease")
            if (
                clock_domain != session.worker.clock_domain
                or deadline > session.deadline
                or now >= min(deadline, session.deadline)
            ):
                raise ProtocolError("feature original deadline/clock domain mismatch or expiry")
            if session.frontier < step:
                raise ProtocolError("feature request includes future or missing env.step completion")
            if not steps and step == 0 and session.frontier != 0:
                raise ProtocolError("completed feature Session cannot reset its causal frontier")
            if steps and (session.chunk is None or session.chunk.start != previous_step):
                raise ProtocolError("feature history does not match the active completed Action chunk")
            visual, actions, owned_rows, selected_sources = [], [], [], []
            identity = strict_json(session.worker.identity_json)
            for source, row in zip(steps, rows, strict=True):
                row = _exact(row, {"source_step", "post_step_receipt", "feature"}, "feature row")
                if type(row["source_step"]) is not int or row["source_step"] != source:
                    raise ProtocolError("feature source sequence was changed/reindexed")
                post = validate_post_step_receipt(row["post_step_receipt"])
                completed = session.completed.get(source)
                if completed is None or canonical_json(post) != completed.receipt_json or post["source_step"] != source:
                    raise ProtocolError("public post-step receipt lacks matching authoritative Owner completion")
                selected = selected_history_step(source, session.chunk.start, self.history_k)
                if self.history_k != 16 and (
                    post.get("history_k") != self.history_k or post.get("selected") != selected
                ):
                    raise ProtocolError("post-step selection differs from authoritative Owner")
                if not selected:
                    if row["feature"] is not None:
                        raise ProtocolError("unselected history must not carry a Worker Feature")
                    owned_rows.append({"source_step": source, "post_step_receipt": post, "feature": None})
                    continue
                selected_sources.append(source)
                if source % 16 == 0:
                    if row["feature"] is not None:
                        raise ProtocolError("Policy boundary must not impersonate a Worker Feature")
                    boundary = self._boundary(session, source)
                    header = boundary.header
                    if any(
                        header[name] != post[name]
                        for name in (
                            "issued_chunk_id",
                            "policy_request_key",
                            "source_step",
                            "source_rgb_sha256",
                            "clock_domain",
                        )
                    ):
                        raise ProtocolError("Policy boundary and actual post-step proof differ")
                    visual.append(boundary.visual96)
                    owned_feature = None
                else:
                    receipt = verify_feature(
                        row["feature"],
                        session.worker.key,
                        expected_lease=session.lease,
                        expected_identity=identity,
                        expected_worker_id=session.worker.worker_id,
                        expected_worker_epoch=session.worker.epoch,
                        expected_key_id=session.worker.key_id,
                        expected_clock_domain=session.worker.clock_domain,
                        now_ns=now,
                    )
                    header = receipt["header"]
                    if any(
                        header[name] != post[name]
                        for name in (
                            "issued_chunk_id",
                            "policy_request_key",
                            "source_step",
                            "source_rgb_sha256",
                            "raw15_sha256",
                            "clock_domain",
                            "episode_deadline_ns",
                        )
                    ):
                        raise ProtocolError("signed Feature differs from actual completed RGB/action/chunk/deadline")
                    signed_action = base64.b64decode(receipt["raw15_f32le_b64"], validate=True)
                    if signed_action != completed.raw15:
                        raise ProtocolError("signed action bytes differ from executed canonical raw15")
                    visual.append(base64.b64decode(receipt["visual96_f32le_b64"], validate=True))
                    owned_feature = receipt
                actions.append(completed.raw15)
                owned_rows.append({"source_step": source, "post_step_receipt": post, "feature": owned_feature})
            digest = feature_evidence_digest(owned_rows)
            self.validate_request(lease, deadline, clock_domain)
            return FeatureEvidenceBatch(tuple(selected_sources), tuple(visual), tuple(actions), digest)

    def end_session(self, token, lease):
        self._auth(token)
        public = validate_lease(lease)
        with self._lock:
            session = self._sessions.get(public["session_id"])
            if session is None:
                if self._retired.get(public["session_id"]) == canonical_json(public):
                    return {"ended": True, "duplicate": True}
                raise ProtocolError("unknown feature Session End")
            if session.lease != public:
                raise ProtocolError("stale feature Session End")
            del self._sessions[public["session_id"]]
            self._retired[public["session_id"]] = canonical_json(public)
            while len(self._retired) > self.max_sessions:
                self._retired.popitem(last=False)
            return {"ended": True, "duplicate": False}

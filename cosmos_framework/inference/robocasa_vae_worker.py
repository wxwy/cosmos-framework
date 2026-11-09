"""Standalone bounded encoder owner; no Policy, FastState, simulator or GPU imports.

The injected encoder is synchronous: one invocation owns all its work until it
returns. Stage1's trusted ``owner_complete`` is mock causal control, not an
attestation that a real environment executed an action. Only an authenticated
local owner wrapper may expose owner methods or ``close`` over IPC.
"""

from __future__ import annotations

import copy
import hashlib
import secrets
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from types import MappingProxyType

from cosmos_framework.inference.robocasa_vae_protocol import (
    IDENTITY_FIELDS,
    MAX_FEATURE_WIRE_BYTES,
    EncoderDrainUnproven,
    Frame,
    canonical_json,
    sign_feature,
    validate_frame,
    validate_identity,
    validate_lease,
    validate_visual96,
)

_OWNER = object()


def _identifier(value, name):
    if type(value) is not str or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _ns(value, name):
    if type(value) is not int or not 0 <= value < 10**20:
        raise ValueError(f"invalid {name}")
    return value


def _sha256(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _capability(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    if type(value) is not bytes or not 16 <= len(value) <= 256:
        raise ValueError("bounded session capability required")
    return hashlib.sha256(b"PSM23-session-capability\0" + value).digest()


@dataclass(frozen=True)
class WorkerConfig:
    bmax: int = 1
    wait_ms: int = 0
    max_items: int = 256
    max_bytes: int = 128 * 1024**2
    max_session_items: int = 16
    max_sessions: int = 16
    max_policy_epochs: int = 1
    max_tombstones: int = 32
    max_chunks: int = 4
    max_trace: int = 256
    max_retired_sessions: int = 16

    def __post_init__(self):
        if type(self.bmax) is not int or self.bmax not in (1, 4, 8):
            raise ValueError("VAE batch cap must be 1, 4 or 8")
        if type(self.wait_ms) is not int or self.wait_ms not in (0, 2, 5) or (self.bmax == 1 and self.wait_ms):
            raise ValueError("VAE wait must be 0/2/5ms; B1 requires zero wait")
        for name in (
            "max_items",
            "max_bytes",
            "max_session_items",
            "max_sessions",
            "max_policy_epochs",
            "max_tombstones",
            "max_chunks",
            "max_trace",
            "max_retired_sessions",
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 2**31:
                raise ValueError("positive bounded Worker capacity required")
        if self.max_session_items > 16:
            raise ValueError("completed evidence per Session is bounded by H16")
        if self.max_policy_epochs > self.max_sessions:
            raise ValueError("Policy epoch capacity cannot exceed physical Session capacity")


@dataclass
class _Chunk:
    request_key: str
    start: int
    count: int
    completed: int

    @property
    def end(self):
        return self.start + self.count


@dataclass
class _Session:
    lease: dict
    capability: bytes
    deadline: int
    serial: int
    state: str = "OPEN"
    error: dict | None = None
    chunks: OrderedDict = field(default_factory=OrderedDict)
    jobs: dict = field(default_factory=dict)
    tombstones: OrderedDict = field(default_factory=OrderedDict)
    boundary_tombstones: OrderedDict = field(default_factory=OrderedDict)
    sparse_boundary: tuple[int, str, str] | None = None
    issued: int = 0
    completed: int = 0
    submitted: int = 0
    consumed: int = 0
    active: int = 0


@dataclass(eq=False)
class _Job:
    session: _Session
    key: str
    fingerprint: str
    step: int
    frame: Frame | None
    size: int
    deadline: int
    ready_at: int
    serial: int
    state: str = "QUEUED"
    active: bool = False
    uncertain: bool = False
    receipt: dict | None = None
    error: dict | None = None


class SharedVAEWorker:
    def __init__(
        self,
        encoder,
        identity,
        signing_key32,
        key_id,
        worker_id,
        worker_epoch=None,
        *,
        clock_domain,
        config=WorkerConfig(),
        clock_ns=time.monotonic_ns,
        autostart=True,
    ):
        if not callable(encoder) or not isinstance(config, WorkerConfig):
            raise ValueError("encoder callable and WorkerConfig required")
        if type(signing_key32) is not bytes or len(signing_key32) != 32:
            raise ValueError("32-byte signing key required")
        self.encoder, self.identity = encoder, MappingProxyType(validate_identity(identity))
        if self.identity["encoder_backend"] == "aot_b1" and config.bmax != 1:
            raise ValueError("aot_b1 identity cannot describe B4/B8")
        self._signing_key = bytes(signing_key32)
        self.key_id = _identifier(key_id, "key id")
        self.worker_id = _identifier(worker_id, "worker id")
        self.worker_epoch = _identifier(worker_epoch or secrets.token_hex(16), "worker epoch")
        self.clock_domain = _identifier(clock_domain, "clock domain")
        self.config, self.clock_ns = config, clock_ns
        self._changed = threading.Condition(threading.RLock())
        self._sessions, self._slots = {}, {}
        self._retired = OrderedDict()
        self._queue = deque()
        self._trace = deque(maxlen=config.max_trace)
        self._trace_overflow = 0
        self._items = self._bytes = self._job_serial = self._session_serial = self._batch_serial = 0
        self._histogram, self._reasons = Counter(), Counter()
        self._last_served = 0
        self._stopping = self._quarantined = False
        self._run = None
        self._policy_epochs = set()
        self._thread = None
        if autostart:
            self.start()

    def start(self):
        with self._changed:
            if self._stopping:
                raise RuntimeError("Worker is closing")
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="sim-shared-vae", daemon=True)
                self._thread.start()
            self._changed.notify_all()
        return self

    def _record(self, event, *, job=None, session=None, **fields):
        if len(self._trace) == self.config.max_trace:
            self._trace_overflow += 1
        row = {"event": event, "at_ns": self.clock_ns(), **fields}
        if job is not None:
            row.update(job=job.serial, session=job.session.serial, source_step=job.step)
        elif session is not None:
            row["session"] = session.serial
        self._trace.append(row)

    def _session(self, session_id, capability=_OWNER):
        _identifier(session_id, "Session id")
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError("unknown or closed Session")
        if capability is not _OWNER and not secrets.compare_digest(session.capability, _capability(capability)):
            raise PermissionError("invalid Session capability")
        return session

    def _open(self, session):
        if self._stopping or self._quarantined or session.state != "OPEN":
            raise RuntimeError("Worker or Session is not accepting new work")
        if self.clock_ns() >= session.deadline:
            self._fail(session, "EPISODE_DEADLINE", "original Episode deadline exhausted")
            raise TimeoutError("original Episode deadline exhausted")

    def owner_register(self, lease, session_capability, episode_deadline_ns):
        lease, capability = validate_lease(lease), _capability(session_capability)
        deadline = _ns(episode_deadline_ns, "Episode deadline")
        with self._changed:
            if self._stopping or self._quarantined:
                raise RuntimeError("Worker is closing/quarantined")
            sid = lease["session_id"]
            session = self._sessions.get(sid)
            if session is not None:
                if (
                    session.lease != lease
                    or session.deadline != deadline
                    or not secrets.compare_digest(session.capability, capability)
                ):
                    raise ValueError("Session registration identity changed")
                return {"registered": True, "duplicate": True, "worker_epoch": self.worker_epoch}
            if self.clock_ns() >= deadline:
                raise TimeoutError("original Episode deadline exhausted")
            if len(self._sessions) >= self.config.max_sessions:
                raise RuntimeError("bounded live Session capacity exhausted")
            run = (lease["run_lease"], lease["manifest_digest"])
            if self._run is not None and self._run != run:
                raise ValueError("foreign Worker run/manifest/Policy epoch")
            if (
                lease["server_epoch"] not in self._policy_epochs
                and len(self._policy_epochs) >= self.config.max_policy_epochs
            ):
                raise ValueError("foreign Worker Policy epoch or bounded Policy capacity exhausted")
            slot = self._slots.get(lease["slot_id"])
            if slot is not None:
                epoch, generation, old_sid = slot
                if epoch != lease["server_epoch"] or lease["generation"] <= generation:
                    raise ValueError("stale Slot generation or changed Policy epoch")
                if old_sid in self._sessions:
                    raise RuntimeError("preceding Slot assignment has not received End ACK")
            elif len(self._slots) >= self.config.max_sessions:
                raise RuntimeError("bounded physical Slot identity capacity exhausted")
            if sid in self._retired:
                raise ValueError("closed Session identity cannot be reused")
            self._session_serial += 1
            session = _Session(lease, capability, deadline, self._session_serial)
            self._sessions[sid] = session
            self._slots[lease["slot_id"]] = (lease["server_epoch"], lease["generation"], sid)
            self._run = run
            self._policy_epochs.add(lease["server_epoch"])
            self._record("REGISTER", session=session)
            self._changed.notify_all()
            return {"registered": True, "duplicate": False, "worker_epoch": self.worker_epoch}

    def owner_issue(self, session_id, chunk_id, request_key, source_start, count):
        chunk_id, request_key = _identifier(chunk_id, "chunk id"), _identifier(request_key, "request key")
        if type(source_start) is not int or source_start < 0 or type(count) is not int or not 1 <= count <= 16:
            raise ValueError("bounded issued source range required")
        with self._changed:
            session = self._session(session_id)
            self._open(session)
            old = session.chunks.get(chunk_id)
            if old is not None:
                if (old.request_key, old.start, old.count) != (request_key, source_start, count):
                    raise ValueError("issued chunk identity changed")
                return {"issued": True, "duplicate": True}
            if source_start != session.issued:
                raise ValueError("issued chunks must be contiguous and never replayed")
            if session.sparse_boundary is not None:
                boundary, boundary_chunk, boundary_request = session.sparse_boundary
                if (
                    (chunk_id, request_key) != (boundary_chunk, boundary_request)
                    or source_start != boundary + 1
                    or count > 15
                    or source_start + count > boundary + 16
                ):
                    raise ValueError("sparse history must follow its exact Policy boundary chunk")
            while len(session.chunks) >= self.config.max_chunks:
                oldest_id, oldest = next(iter(session.chunks.items()))
                if session.consumed < oldest.end or session.completed < oldest.end:
                    raise RuntimeError("bounded issued-chunk metadata capacity exhausted")
                del session.chunks[oldest_id]
            session.chunks[chunk_id] = _Chunk(request_key, source_start, count, source_start)
            session.issued += count
            self._record("ISSUE", session=session, source_start=source_start, count=count)
            self._changed.notify_all()
            return {"issued": True, "duplicate": False}

    def owner_boundary_skip(
        self,
        lease,
        chunk_id,
        request_key,
        source_step,
        source_rgb_sha256,
        raw15_sha256,
        policy_boundary_sha256,
        post_step_receipt,
    ):
        """Register one completed Policy-owned boundary without encoding/signing it.

        The authenticated Run Owner has compared the committed Policy boundary
        and its actual Actor success receipt before this call. This local check
        binds their exact metadata; it neither attests env.step cryptographically
        nor reads another process's Policy cache. Legacy continuous issuance is
        unchanged for Sessions that never use this opt-in owner operation.
        """
        # Keep the standalone Worker import free of Policy/model imports. This
        # shared validator is stdlib-only and returns detached exact metadata.
        from cosmos_framework.inference.robocasa_feature_evidence import validate_post_step_receipt

        if self.config.bmax != 1 or self.config.wait_ms != 0:
            raise ValueError("Stage3A Policy boundary reuse requires VAE B1 with zero wait")
        lease = validate_lease(lease)
        chunk_id, request_key = _identifier(chunk_id, "chunk id"), _identifier(request_key, "request key")
        if type(source_step) is not int or source_step < 0 or source_step % 16:
            raise ValueError("only a true H16 Policy boundary may be skipped")
        rgb_digest = _sha256(source_rgb_sha256, "source RGB digest")
        action_digest = _sha256(raw15_sha256, "executed raw15 digest")
        boundary_digest = _sha256(policy_boundary_sha256, "Policy boundary digest")
        receipt = validate_post_step_receipt(post_step_receipt)
        if (
            receipt["lease"] != lease
            or receipt["issued_chunk_id"] != chunk_id
            or receipt["policy_request_key"] != request_key
            or receipt["source_step"] != source_step
            or receipt["source_rgb_sha256"] != rgb_digest
            or receipt["raw15_sha256"] != action_digest
            or receipt["clock_domain"] != self.clock_domain
        ):
            raise ValueError("Policy boundary and post-step receipt identity mismatch")
        metadata = canonical_json(
            {
                "version": "PSM24-BOUNDARY-SKIP-v1",
                "worker_epoch": self.worker_epoch,
                "lease": lease,
                "issued_chunk_id": chunk_id,
                "policy_request_key": request_key,
                "source_step": source_step,
                "source_rgb_sha256": rgb_digest,
                "raw15_sha256": action_digest,
                "policy_boundary_sha256": boundary_digest,
                "post_step_receipt": receipt,
            }
        )
        with self._changed:
            session = self._session(lease["session_id"])
            self._open(session)
            if session.lease != lease or int(receipt["episode_deadline_ns"]) != session.deadline:
                raise ValueError("Policy boundary belongs to another registered assignment/deadline")
            if int(receipt["step_completed_ns"]) > self.clock_ns():
                raise ValueError("Policy boundary action completion is in the future")
            previous = session.boundary_tombstones.get(source_step)
            duplicate = previous is not None
            if duplicate:
                if previous != metadata:
                    raise ValueError("Policy boundary replay bytes or identity changed")
            else:
                if (session.issued, session.completed, session.submitted, session.consumed) != (source_step,) * 4:
                    raise ValueError("Policy boundary requires four identical exact source frontiers")
                if session.active or session.jobs or any(job.session is session for job in self._queue):
                    raise RuntimeError("Policy boundary requires all prior jobs acknowledged and drained")
                if chunk_id in session.chunks:
                    raise ValueError("Policy boundary cannot reuse an already issued chunk")
                session.boundary_tombstones[source_step] = metadata
                while len(session.boundary_tombstones) > self.config.max_tombstones:
                    session.boundary_tombstones.popitem(last=False)
                session.sparse_boundary = (source_step, chunk_id, request_key)
                session.issued = session.completed = session.submitted = session.consumed = source_step + 1
                self._record("POLICY_BOUNDARY_SKIP", session=session, source_step=source_step, worker_feature=False)
                self._changed.notify_all()
            return {
                "boundary_skipped": True,
                "duplicate": duplicate,
                "source_step": source_step,
                "frontier": source_step + 1,
                "worker_feature": False,
                "encoded": False,
            }

    def owner_complete(self, session_id, chunk_id, completed_through):
        if type(completed_through) is not int or completed_through < 0:
            raise ValueError("exclusive completed frontier required")
        with self._changed:
            session = self._session(session_id)
            self._open(session)
            chunk = session.chunks.get(chunk_id)
            if chunk is None or not chunk.completed <= completed_through <= chunk.end:
                raise ValueError("unissued or regressing completed frontier")
            if chunk.start > session.completed:
                raise ValueError("completed frontier cannot skip an earlier chunk")
            chunk.completed = completed_through
            session.completed = max(session.completed, completed_through)
            self._record("MOCK_COMPLETED", session=session, completed_through=completed_through)
            self._changed.notify_all()
            return {"completed_through": session.completed, "attestation": "owner_mock_control_only"}

    def submit(self, frame, capability):
        frame = validate_frame(frame)  # Expensive CPU validation never holds the admission lock.
        header, fingerprint = frame.header, frame.fingerprint
        size = len(frame.header_json) + len(frame.rgb) + len(frame.raw15) + MAX_FEATURE_WIRE_BYTES
        step, sid = header["source_step"], header["lease"]["session_id"]
        with self._changed:
            session = self._session(sid, capability)
            if header["lease"] != session.lease or any(header[name] != self.identity[name] for name in IDENTITY_FIELDS):
                raise ValueError("Frame assignment or encoder identity mismatch")
            if header["clock_domain"] != self.clock_domain or int(header["episode_deadline_ns"]) != session.deadline:
                raise ValueError("Frame clock domain or original Episode deadline mismatch")
            if self._stopping or self._quarantined or session.state in ("CLOSING", "QUARANTINED"):
                raise RuntimeError("closed/quarantined encoder ownership cannot accept replay")
            old = session.jobs.get(step)
            tombstone = session.tombstones.get(step)
            if old is not None or tombstone is not None:
                previous = old.fingerprint if old is not None else tombstone["fingerprint"]
                if previous != fingerprint:
                    raise ValueError("source identity reused with changed bytes or deadline")
                return old.key if old is not None else tombstone["job_key"]
            self._open(session)
            if step != session.submitted:
                raise ValueError("stale/evicted or noncontiguous source frontier")
            chunk = session.chunks.get(header["issued_chunk_id"])
            if (
                chunk is None
                or chunk.request_key != header["policy_request_key"]
                or not chunk.start <= step < chunk.end
                or step >= chunk.completed
                or step >= session.completed
            ):
                raise ValueError("Frame is not issued and confirmed completed")
            deadline = min(
                int(header["job_deadline_ns"]),
                session.deadline,
                int(header["replan_deadline_ns"]) if header["replan_deadline_ns"] is not None else session.deadline,
            )
            if self.clock_ns() >= deadline:
                raise TimeoutError("original Frame deadline exhausted")
            if len(session.jobs) >= self.config.max_session_items or self._items >= self.config.max_items:
                raise RuntimeError("bounded frame item capacity exhausted")
            if self._bytes + size > self.config.max_bytes:
                raise RuntimeError("bounded frame byte capacity exhausted")
            key = hashlib.sha256(b"PSM23-job\0" + fingerprint.encode("ascii")).hexdigest()
            self._job_serial += 1
            job = _Job(session, key, fingerprint, step, frame, size, deadline, self.clock_ns(), self._job_serial)
            session.jobs[step] = job
            session.submitted += 1
            self._queue.append(job)
            self._items += 1
            self._bytes += size
            self._record("READY", job=job)
            self._changed.notify_all()
            return key

    def _job(self, session, job_key):
        if type(job_key) is not str or len(job_key) != 64 or any(c not in "0123456789abcdef" for c in job_key):
            raise ValueError("SHA256 job identity required")
        for job in session.jobs.values():
            if secrets.compare_digest(job.key, job_key):
                return job
        for tombstone in session.tombstones.values():
            if secrets.compare_digest(tombstone["job_key"], job_key):
                return tombstone
        raise ValueError("unknown or evicted job identity")

    def _fail(self, session, code, message):
        if session.error is None:
            session.error = {"code": code, "message": message}
            self._record("SESSION_ERROR", session=session, code=code)
        if session.state == "OPEN":
            session.state = "FAILED"
        for job in session.jobs.values():
            if job.error is None:
                self._record("JOB_ERROR", job=job, code=session.error["code"])
            job.error = dict(session.error)
            job.receipt = None
            job.state = "ERROR"
        self._queue = deque(job for job in self._queue if job.session is not session)
        self._changed.notify_all()

    def _poll(self, session, job_key):
        job = self._job(session, job_key)
        if isinstance(job, dict):
            return {"status": "already_acked", "job_key": job_key}
        if self.clock_ns() >= job.deadline and session.error is None:
            self._fail(session, "FRAME_DEADLINE", "original Frame deadline exhausted")
        if job.error is not None:
            return {"status": "error", "job_key": job.key, "error": dict(job.error)}
        if job.receipt is None:
            return {"status": "pending", "job_key": job.key}
        return {"status": "ready", "job_key": job.key, "feature": copy.deepcopy(job.receipt)}

    def poll(self, session_id, job_key, capability):
        _identifier(job_key, "job key")
        with self._changed:
            return self._poll(self._session(session_id, capability), job_key)

    def take_ready(self, session_id, capability):
        """Read the first unacknowledged source; arbitrary poll order never advances it."""
        with self._changed:
            session = self._session(session_id, capability)
            job = session.jobs.get(session.consumed)
            if job is None:
                return {"status": "pending", "source_step": session.consumed}
            return {**self._poll(session, job.key), "source_step": session.consumed}

    def _release(self, job):
        if job.uncertain:
            return
        if job.frame is not None:
            self._items -= 1
            self._bytes -= job.size
            job.frame, job.receipt = None, None

    def ack(self, session_id, job_key, capability):
        _identifier(job_key, "job key")
        with self._changed:
            session = self._session(session_id, capability)
            job = self._job(session, job_key)
            if isinstance(job, dict):
                return {"acknowledged": True, "job_key": job_key}
            if job.active or job.uncertain or job.state not in ("READY", "ERROR"):
                raise RuntimeError("cannot acknowledge an unfinished encoder job")
            session.tombstones[job.step] = {"fingerprint": job.fingerprint, "job_key": job.key}
            del session.jobs[job.step]
            self._release(job)
            while session.consumed < session.submitted and session.consumed not in session.jobs:
                session.consumed += 1
            while len(session.tombstones) > self.config.max_tombstones:
                session.tombstones.popitem(last=False)
            self._record("ACK", job=job)
            self._changed.notify_all()
            return {"acknowledged": True, "job_key": job_key}

    def _select(self, now):
        for session in tuple(self._sessions.values()):
            if session.state == "OPEN" and any(now >= job.deadline for job in session.jobs.values()):
                self._fail(session, "FRAME_DEADLINE", "original Frame deadline exhausted")
        if not self._queue:
            return [], None, None
        oldest = min(job.ready_at for job in self._queue)
        wake = oldest + self.config.wait_ms * 1_000_000
        urgent = min(job.deadline for job in self._queue) <= wake
        reason = (
            "B1"
            if self.config.bmax == 1
            else "full"
            if len(self._queue) >= self.config.bmax
            else "deadline"
            if urgent
            else "window"
        )
        if self.config.bmax > 1 and len(self._queue) < self.config.bmax and now < wake and not urgent:
            return [], None, (wake - now) / 1_000_000_000
        # Keep the cursor across dispatches, including B1. A Session with a
        # full H16 backlog cannot monopolize the encoder ahead of ready peers.
        per_session = {}
        for job in self._queue:
            per_session.setdefault(job.session.serial, deque()).append(job)
        serials = sorted(per_session)
        serials = [s for s in serials if s > self._last_served] + [s for s in serials if s <= self._last_served]
        batch = []
        while serials and len(batch) < self.config.bmax:
            remaining = []
            for serial in serials:
                if len(batch) == self.config.bmax:
                    break
                batch.append(per_session[serial].popleft())
                self._last_served = serial
                if per_session[serial]:
                    remaining.append(serial)
            serials = remaining
        selected = set(batch)
        self._queue = deque(job for job in self._queue if job not in selected)
        return batch, reason, None

    def _loop(self):
        while True:
            with self._changed:
                if self._stopping:
                    self._changed.notify_all()
                    return
                batch, reason, wait_s = self._select(self.clock_ns())
                if not batch:
                    self._changed.wait(wait_s)
                    continue
                self._batch_serial += 1
                batch_id = self._batch_serial
                self._histogram[len(batch)] += 1
                self._reasons[reason] += 1
                self._record("DISPATCH", batch=batch_id, size=len(batch), reason=reason)
                for job in batch:
                    job.active = True
                    job.state = "ENCODING"
                    job.session.active += 1
                    self._record("ENCODE_START", job=job, batch=batch_id)
                self._changed.notify_all()
            results, error, unproven, encoder_profile = None, None, False, None
            try:
                if any(job.session.sparse_boundary is not None for job in batch):
                    from cosmos_framework.inference.robocasa_feature_trace import TraceRecorder, use_trace

                    recorder = TraceRecorder(max_events=8, max_bytes=4096)
                    with use_trace(recorder):
                        values = self.encoder([job.frame for job in batch])
                    recorder.resolve_cuda()
                    encoder_profile = recorder.snapshot()
                else:
                    values = self.encoder([job.frame for job in batch])
                if type(values) is not list or len(values) != len(batch):
                    raise ValueError("encoder output cardinality mismatch")
                results = [
                    sign_feature(
                        job.frame,
                        validate_visual96(value),
                        self._signing_key,
                        worker_id=self.worker_id,
                        worker_epoch=self.worker_epoch,
                        key_id=self.key_id,
                    )
                    for job, value in zip(batch, values, strict=True)
                ]
            except BaseException as failure:
                unproven = isinstance(failure, EncoderDrainUnproven)
                error = {
                    "code": "ENCODER_DRAIN_UNPROVEN" if unproven else "ENCODER_ERROR",
                    "message": type(failure).__name__,
                }
            with self._changed:
                now = self.clock_ns()
                if encoder_profile is not None:
                    self._record("ENCODER_PROFILE", batch=batch_id, profile=encoder_profile)
                if unproven:
                    self._quarantined = self._stopping = True
                    for session in self._sessions.values():
                        self._fail(session, error["code"], error["message"])
                    for job in batch:
                        job.uncertain = True
                    self._record("WORKER_QUARANTINE", code=error["code"])
                for index, job in enumerate(batch):
                    job.active = False
                    job.session.active -= 1
                    self._record("ENCODE_END", job=job, batch=batch_id)
                    session = job.session
                    if error is not None:
                        self._fail(session, error["code"], error["message"])
                    elif now >= job.deadline:
                        self._fail(session, "FRAME_DEADLINE", "original Frame deadline exhausted")
                    if session.state == "OPEN" and session.error is None and not self._stopping:
                        job.receipt, job.state = results[index], "READY"
                        self._record("RESULT_READY", job=job, batch=batch_id)
                    else:
                        job.error = dict(session.error or {"code": "CANCELLED", "message": "Session closed"})
                        job.state = "ERROR"
                        job.receipt = None
                        if session.state in ("CLOSING", "QUARANTINED") or self._stopping:
                            self._release(job)
                self._changed.notify_all()

    def _cancel(self, session):
        if session.state != "QUARANTINED":
            session.state = "CLOSING"
        self._queue = deque(job for job in self._queue if job.session is not session)
        for job in session.jobs.values():
            if not job.active:
                self._release(job)
            job.error = dict(session.error or {"code": "CANCELLED", "message": "Session closed"})
            job.receipt, job.state = None, "ERROR"

    def _drain(self, deadline_ns, active):
        # Condition.wait releases the metadata lock; the encoder never needs it
        # while computing. The original absolute deadline is never refreshed.
        while active():
            remaining = deadline_ns - self.clock_ns()
            if remaining <= 0:
                return False
            self._changed.wait(min(remaining / 1_000_000_000, threading.TIMEOUT_MAX))
        return True

    def _end_receipt(self, session, safe):
        return {
            "acknowledged": safe,
            "quiescent": safe,
            "session_id": session.lease["session_id"],
            "worker_id": self.worker_id,
            "worker_epoch": self.worker_epoch,
            "state": "CLOSED" if safe else "QUARANTINED",
            "error": copy.deepcopy(session.error),
        }

    def end(self, session_id, capability, deadline_ns):
        deadline_ns = _ns(deadline_ns, "End deadline")
        supplied = _capability(capability)
        with self._changed:
            retired = self._retired.get(session_id)
            if retired is not None:
                if not secrets.compare_digest(retired[0], supplied):
                    raise PermissionError("invalid Session capability")
                return copy.deepcopy(retired[1])
            session = self._session(session_id, capability)
            self._record("END_REQUEST", session=session)
            self._cancel(session)
            self._changed.notify_all()
            safe = (
                self._drain(deadline_ns, lambda: bool(session.active))
                and session.state != "QUARANTINED"
                and not self._quarantined
            )
            # End retries may wait concurrently. Only the caller still owning
            # this exact live Session may retire it; peers reuse its receipt.
            if self._sessions.get(session_id) is not session:
                retired = self._retired.get(session_id)
                if retired is not None and secrets.compare_digest(retired[0], supplied):
                    return copy.deepcopy(retired[1])
                raise RuntimeError("Session ownership changed while End was draining")
            if not safe:
                if session.error is None:
                    session.error = {"code": "DRAIN_UNPROVEN", "message": "encoder invocation has not drained"}
                session.state = "QUARANTINED"
            receipt = self._end_receipt(session, safe)
            if safe:
                for job in session.jobs.values():
                    self._release(job)
                del self._sessions[session_id]
                self._retired[session_id] = (session.capability, receipt)
                while len(self._retired) > self.config.max_retired_sessions:
                    self._retired.popitem(last=False)
            self._record("END_RESULT", session=session, quiescent=safe)
            self._changed.notify_all()
            return copy.deepcopy(receipt)

    def close(self, deadline_ns):
        """Trusted local owner only; a blocked encoder cannot yield a positive drain."""
        deadline_ns = _ns(deadline_ns, "close deadline")
        with self._changed:
            self._stopping = True
            for session in self._sessions.values():
                self._cancel(session)
            self._changed.notify_all()
            safe = self._drain(deadline_ns, lambda: any(s.active for s in self._sessions.values()))
            self._quarantined |= not safe
            safe = safe and not self._quarantined
            self._record("CLOSE", quiescent=safe)
            return {"closed": safe, "quiescent": safe, "state": "CLOSED" if safe else "QUARANTINED"}

    def info(self):
        with self._changed:
            return {
                "worker_id": self.worker_id,
                "worker_epoch": self.worker_epoch,
                "bmax": self.config.bmax,
                "wait_ms": self.config.wait_ms,
                "sessions": len(self._sessions),
                "known_slots": len(self._slots),
                "retired_sessions": len(self._retired),
                "queued": len(self._queue),
                "active": sum(s.active for s in self._sessions.values()),
                "items": self._items,
                "bytes": self._bytes,
                "tombstones": sum(len(s.tombstones) for s in self._sessions.values()),
                "boundary_tombstones": sum(len(s.boundary_tombstones) for s in self._sessions.values()),
                "issued_chunks": sum(len(s.chunks) for s in self._sessions.values()),
                "batch_histogram": dict(self._histogram),
                "dispatch_reasons": dict(self._reasons),
                "quarantined": self._quarantined,
                "stopping": self._stopping,
                "trace": copy.deepcopy(list(self._trace)),
                "trace_overflow": self._trace_overflow,
                "causal_attestation": "owner_mock_control_only",
                "timing_source": "cpu_monotonic_ns",
            }

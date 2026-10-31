"""Opt-in B=1 inference actor with owned Sessions and replay-aware admission.

One executor serializes model-owned encode/scan/generation. HTTP threads only
validate/queue/cancel; only executor quiescence permits resource release or End
ACK. This module does not create simulators or enable native batching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cosmos_framework.inference.local_memory_state import CachedMemoryResponse

PROTOCOL = "robocasa.session-transactions.b1.v1"
NO_TTT_PROTOCOL = "robocasa.session-transactions.no-ttt.v1"
MAX_BODY = 32 * 1024**2
MAX_RESPONSE = 1024**2


def protocol_for_mode(mode):
    if mode not in ("off", "required"):
        raise ValueError("explicit off|required Local-TTT mode required")
    return PROTOCOL if mode == "required" else NO_TTT_PROTOCOL


def request_frontier(payload, mode):
    """No-TTT has a transaction sequence, never a fabricated Local frontier."""
    if mode == "required":
        local = payload.get("local_memory")
        if not isinstance(local, dict):
            raise ValueError("Local evidence object required")
        value = local.get("consumer_step")
    else:
        inference = payload.get("inference")
        if not isinstance(inference, dict) or set(inference) != {"session_id", "episode_id", "request_index"}:
            raise ValueError("exact No-TTT inference identity required")
        value = inference["request_index"]
    if type(value) is not int or value < 0 or (mode == "off" and value > 100000):
        raise ValueError("bounded integer inference frontier required")
    return value


@dataclass(frozen=True)
class _CommittedResponse:
    frontier: int
    response: CachedMemoryResponse


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("non-finite JSON constant")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def identifier(value: Any, *, limit=128) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ValueError("nonempty bounded identifier required")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def monotonic_clock_domain() -> str:
    """Prove Linux same-boot/same-monotonic-offset comparability, including containers.

    Arbitrary remote monotonic clocks are unsupported. Missing kernel evidence is
    an error, never a wall-clock or receive-time budget fallback.
    """
    info = time.get_clock_info("monotonic")
    if sys.platform != "linux" or not info.monotonic or info.implementation != "clock_gettime(CLOCK_MONOTONIC)":
        raise ValueError("proven Linux CLOCK_MONOTONIC domain required")

    def read(path):
        with Path(path).open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("bounded kernel clock identity required")
        return raw.decode("ascii")

    boot = str(uuid.UUID(read("/proc/sys/kernel/random/boot_id").strip()))
    rows = [line.split() for line in read("/proc/self/timens_offsets").splitlines()]
    offsets = [row for row in rows if row and row[0] == "monotonic"]
    if len(offsets) != 1 or len(offsets[0]) != 3:
        raise ValueError("kernel monotonic namespace offset required")
    seconds, nanos = map(int, offsets[0][1:])
    if not 0 <= nanos < 10**9:
        raise ValueError("invalid kernel monotonic offset")
    return digest({"clock": "linux.CLOCK_MONOTONIC", "boot_id": boot, "offset_ns": seconds * 10**9 + nanos})


def original_deadline(budget: dict, domain: str, now: float, timeout_s: float) -> float:
    if not isinstance(budget, dict) or set(budget) != {"clock_domain", "started_at", "deadline"}:
        raise ValueError("exact original prediction budget required")
    if budget["clock_domain"] != domain:
        raise ValueError("prediction clock domain mismatch")
    start, end = budget["started_at"], budget["deadline"]
    if any(
        type(value) not in (int, float) or abs(value) > 10**15 or not math.isfinite(value) for value in (start, end)
    ):
        raise ValueError("finite original prediction budget required")
    if start < 0 or start > now or end <= start or end > start + timeout_s:
        raise ValueError("invalid original prediction budget or exceeds Server cap")
    if end <= now:
        raise TimeoutError("original prediction deadline exhausted before admission")
    return end


@dataclass(frozen=True)
class SessionLease:
    server_epoch: str
    run_lease: str
    manifest_digest: str
    slot_id: str
    generation: int
    task: str
    episode_index: int
    attempt_id: str
    session_id: str
    episode_id: str
    capability: str

    def identity(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "capability"}


@dataclass(frozen=True)
class Reply:
    status: int
    body: bytes

    def json(self) -> dict:
        return strict_json(self.body)


@dataclass(eq=False)
class _Ticket:
    key: str
    payload_sha256: str
    lease: SessionLease
    payload: dict
    size: int
    deadline: float
    state: str = "QUEUED"
    cancelled: bool = False
    quiescent: bool = False
    update: Any = None


class SessionTransactions:
    def __init__(
        self,
        adapter,
        generate: Callable[[dict, Any], list],
        quiesce: Callable[[], None],
        *,
        owner_token: str,
        run_lease: str,
        manifest_digest: str,
        slot_ids: tuple[str, ...],
        model_lock=None,
        timeout_s=600.0,
        max_bytes=128 * 1024**2,
        clock=time.monotonic,
    ):
        self.mode = adapter.mode
        self.protocol = protocol_for_mode(self.mode)
        if (adapter.memory is not None) != (self.mode == "required"):
            raise ValueError("transaction mode must match actual Local-TTT capability")
        if not 1 <= len(slot_ids) <= 8 or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("one to eight distinct bounded physical Slot identities required")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive finite transaction bounds required")
        if len(manifest_digest) != 64 or any(c not in "0123456789abcdef" for c in manifest_digest):
            raise ValueError("manifest SHA256 required")
        self.adapter, self.generate, self.quiesce = adapter, generate, quiesce
        self.owner_token = identifier(owner_token, limit=256)
        self.run_lease, self.manifest_digest = identifier(run_lease), manifest_digest
        self.slot_ids = tuple(identifier(slot, limit=64) for slot in slot_ids)
        self.timeout_s, self.max_bytes, self.clock = timeout_s, max_bytes, clock
        self.clock_domain = monotonic_clock_domain()
        self.server_epoch = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._model_lock = model_lock if model_lock is not None else threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-ttt-b1")
        self._leases: dict[str, SessionLease] = {}
        self._closed: dict[str, SessionLease] = {}
        self._generations = dict.fromkeys(self.slot_ids, 0)
        self._closing: set[str] = set()
        self._failed: dict[str, dict] = {}
        self._tickets: dict[str, _Ticket] = {}
        self._responses: dict[str, _CommittedResponse] = {}
        self._busy = False
        self._quarantined = False
        self._stopping = False
        self.http_slots = threading.BoundedSemaphore(len(self.slot_ids) + 1)

    def _reply(self, status, value):
        return Reply(status, json_bytes(value))

    def info(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "profile": self.protocol,
                "local_memory_mode": self.mode,
                "server_epoch": self.server_epoch,
                "run_lease": self.run_lease,
                "manifest_digest": self.manifest_digest,
                "slot_ids": list(self.slot_ids),
                "max_native_batch": 1,
                "sessions": len(self._leases),
                "tickets": len(self._tickets),
                "encoded_bytes": sum(ticket.size for ticket in self._tickets.values()),
                "busy": self._busy,
                "quarantined": self._quarantined,
                "stopping": self._stopping,
                "generated_media": "unsupported_in_b1_increment",
                "timeout_s": self.timeout_s,
                "prediction_clock_domain": self.clock_domain,
                "max_encoded_bytes": self.max_bytes,
                "max_item_bytes": MAX_BODY,
                "max_http_requests": len(self.slot_ids) + 1,
            }

    def _healthy(self):
        if self._quarantined or self._stopping:
            raise RuntimeError("transaction actor unavailable; owned Server recovery required")

    def _registration_reply(self, lease):
        return self._reply(
            200,
            {
                "lease": asdict(lease),
                "prediction_clock_domain": self.clock_domain,
                "local_memory_mode": self.mode,
                "profile": self.protocol,
            },
        )

    def register(self, record: dict, owner_token: str) -> Reply:
        if not secrets.compare_digest(owner_token, self.owner_token):
            raise ValueError("registration requires coordinator capability")
        if set(record) != {
            "run_lease",
            "manifest_digest",
            "slot_id",
            "generation",
            "task",
            "episode_index",
            "attempt_id",
        }:
            raise ValueError("exact registration identity required")
        if (record["run_lease"], record["manifest_digest"]) != (self.run_lease, self.manifest_digest):
            raise ValueError("foreign run/manifest registration")
        slot = record["slot_id"]
        if slot not in self.slot_ids or type(record["generation"]) is not int or not 1 <= record["generation"] < 2**63:
            raise ValueError("invalid physical Slot/generation")
        if type(record["episode_index"]) is not int or record["episode_index"] < 0:
            raise ValueError("invalid logical Episode")
        identifier(record["task"])
        identifier(record["attempt_id"])
        with self._lock:
            self._healthy()
            current = self._leases.get(slot)
            if current is not None:
                expected = {key: getattr(current, key) for key in record}
                if record == expected:
                    return self._registration_reply(current)
                raise ValueError("previous physical Slot assignment has not received End ACK")
            if record["generation"] <= self._generations[slot]:
                raise ValueError("stale Slot generation")
            unique = f"{self.server_epoch}:{slot}:{record['generation']}:{uuid.uuid4().hex}"
            lease = SessionLease(
                self.server_epoch,
                **record,
                session_id=unique,
                episode_id=uuid.uuid4().hex,
                capability=secrets.token_urlsafe(32),
            )
            self._generations[slot] = lease.generation
            self._leases[slot] = lease
            self._failed.pop(slot, None)
            return self._registration_reply(lease)

    def _lease(self, proof: dict, *, closed=False) -> SessionLease:
        if not isinstance(proof, dict) or set(proof) != set(SessionLease.__dataclass_fields__):
            raise ValueError("exact assignment proof required")
        lease = self._leases.get(proof["slot_id"])
        if lease is None and closed:
            lease = self._closed.get(proof["slot_id"])
        if (
            lease is None
            or type(proof["generation"]) is not int
            or type(proof["episode_index"]) is not int
            or not isinstance(proof["capability"], str)
            or not secrets.compare_digest(proof["capability"], lease.capability)
            or {k: v for k, v in proof.items() if k != "capability"} != lease.identity()
        ):
            raise ValueError("stale or foreign assignment")
        return lease

    def _usable(self, lease):
        self._healthy()
        if lease.slot_id in self._closing or lease.slot_id in self._failed:
            raise RuntimeError("Episode closing/failed; explicit End and a new Attempt required")

    def _expire(self, ticket):
        if self.clock() >= ticket.deadline:
            ticket.cancelled = True
        if ticket.cancelled:
            raise RuntimeError("inference cancelled/expired; cleanup is not yet proven")

    def _release(self, ticket):
        if not ticket.quiescent or ticket.state not in ("COMMITTED", "ABORTED"):
            raise RuntimeError("cannot release unquiesced inference ownership")
        if self._tickets.get(ticket.key) is ticket:
            del self._tickets[ticket.key]
            ticket.payload, ticket.update = {}, None
            self._changed.notify_all()

    def submit(self, proof: dict, payload: dict, *, budget: dict) -> Reply:
        deadline = original_deadline(budget, self.clock_domain, self.clock(), self.timeout_s)
        raw = json_bytes(payload)
        if len(raw) > MAX_BODY:
            raise ValueError("encoded inference item exceeds bound")
        payload = strict_json(raw)  # Freeze caller-owned structures before admission.
        if not self._payload_keys_valid(payload):
            raise ValueError("B1 policy payload requires observation/state/local evidence; media must be off")
        frontier = request_frontier(payload, self.mode)
        local = payload["local_memory"] if self.mode == "required" else payload["inference"]
        if self.mode == "required" and (not isinstance(local.get("evidence"), list) or len(local["evidence"]) > 16):
            raise ValueError("completed evidence must be bounded by H16")
        payload_sha = hashlib.sha256(raw).hexdigest()
        with self._lock:
            lease = self._lease(proof)
            if (local.get("session_id"), local.get("episode_id")) != (lease.session_id, lease.episode_id):
                raise ValueError("Local evidence belongs to another assignment")
            key = digest({"protocol": self.protocol, **lease.identity(), "frontier": frontier})
            previous = self._responses.get(lease.session_id)
            cached = previous.response if previous else None
            if cached is not None and cached.request_key == key and cached.payload_sha256 != payload_sha:
                raise ValueError("request identity reused with changed payload")
            ticket = self._tickets.get(key)
            if ticket is not None:
                if ticket.lease != lease or ticket.payload_sha256 != payload_sha:
                    raise ValueError("inflight request identity reused with changed payload")
                return self._result(lease, key, payload_sha)
            self._usable(lease)
            if cached is not None and cached.request_key == key:
                return Reply(200, cached.body)  # Post-ACK replay never creates a new Ticket.
            if previous is not None and frontier <= previous.frontier:
                raise ValueError("stale or evicted evidence frontier")
            if previous is None:
                if frontier != 0 or (self.mode == "required" and local["evidence"]):
                    raise ValueError("fresh Episode must start at frontier0 without evidence")
            elif frontier - previous.frontier != (len(local["evidence"]) if self.mode == "required" else 1):
                raise ValueError("bounded contiguous completed evidence required")
            if any(item.lease.slot_id == lease.slot_id for item in self._tickets.values()):
                raise ValueError("one queued/inflight/undelivered request per physical Slot")
            if sum(item.size for item in self._tickets.values()) + len(raw) > self.max_bytes:
                raise RuntimeError("bounded encoded-request capacity exhausted")
            # Include body validation/serialization/lock wait in the SAME original budget.
            original_deadline(budget, self.clock_domain, self.clock(), self.timeout_s)
            ticket = _Ticket(key, payload_sha, lease, payload, len(raw), deadline)
            self._tickets[key] = ticket
            try:
                self._schedule(ticket)
            except BaseException:
                del self._tickets[key]
                raise
            return self._pending(ticket)

    def _payload_keys_valid(self, payload):
        state_key = "local_memory" if self.mode == "required" else "inference"
        return set(payload) == {"image", "prompt", "domain_name", "image_size", "state", state_key}

    def _commit_responses(self, members, caches, *, batch=False):
        # Allocate the complete transaction publication BEFORE any Local commit.
        # Readers/End/ACK use this same actor lock; no response is visible early.
        replacement = dict(self._responses)
        for ticket, cached in zip(members, caches, strict=True):
            replacement[ticket.lease.session_id] = _CommittedResponse(
                request_frontier(ticket.payload, self.mode), cached
            )
        if self.mode == "required":
            if batch:
                self.adapter.commit_many(tuple((t.update, c) for t, c in zip(members, caches, strict=True)))
            else:
                self.adapter.commit(members[0].update, caches[0])
        self._responses = replacement

    def _reset_session(self, session_id):
        if self.mode == "required":
            self.adapter.reset(session_id)
        self._responses.pop(session_id, None)

    def _response_local(self, update):
        return {"local_memory": self.adapter.status(update)} if self.mode == "required" else {}

    def _schedule(self, ticket):
        self._executor.submit(self._execute, ticket)

    def _pending(self, ticket):
        return self._reply(
            202,
            {
                "request_key": ticket.key,
                "payload_sha256": ticket.payload_sha256,
                "state": ticket.state,
                "server_epoch": self.server_epoch,
            },
        )

    def _result(self, lease, key, payload_sha):
        failure = self._failed.get(lease.slot_id)
        if failure is not None and failure["request_key"] == key:
            if failure["payload_sha256"] != payload_sha:
                raise ValueError("failed request payload mismatch")
            return self._reply(409, failure)
        self._usable(lease)
        ticket = self._tickets.get(key)
        if ticket is not None:
            if ticket.lease != lease or ticket.payload_sha256 != payload_sha:
                raise ValueError("wrong inference Ticket binding")
            self._expire(ticket)
            if not ticket.quiescent:
                return self._pending(ticket)
        entry = self._responses.get(lease.session_id)
        cached = entry.response if entry else None
        if cached is None or cached.request_key != key:
            raise ValueError("no retained committed response for this inference")
        if cached.payload_sha256 != payload_sha:
            raise ValueError("response replay payload mismatch")
        return Reply(200, cached.body)

    def result(self, proof, key, payload_sha):
        with self._lock:
            return self._result(self._lease(proof), key, payload_sha)

    def acknowledge(self, proof, key, payload_sha):
        with self._lock:
            lease = self._lease(proof)
            result = self._result(lease, key, payload_sha)
            if result.status != 200:
                raise RuntimeError("cannot acknowledge an unfinished inference")
            ticket = self._tickets.get(key)
            if ticket is not None:
                self._release(ticket)
            return self._reply(200, {"acknowledged": True, "request_key": key})

    def end(self, proof):
        with self._lock:
            lease = self._lease(proof, closed=True)
            if self._leases.get(lease.slot_id) is None:
                return self._reply(200, {"acknowledged": True, **lease.identity()})
            self._closing.add(lease.slot_id)
            owned = [ticket for ticket in self._tickets.values() if ticket.lease == lease]
            for ticket in owned:
                ticket.cancelled = True
            if self._busy or self._quarantined or any(not ticket.quiescent for ticket in owned):
                return self._reply(202, {"acknowledged": False, "state": "draining", **lease.identity()})
            for ticket in owned:
                self._release(ticket)
            self._reset_session(lease.session_id)
            del self._leases[lease.slot_id]
            self._closed[lease.slot_id] = lease
            self._closing.remove(lease.slot_id)
            return self._reply(200, {"acknowledged": True, **lease.identity()})

    def _fence(self, ticket):
        self._usable(ticket.lease)
        self._lease(asdict(ticket.lease))
        self._expire(ticket)

    def _execute(self, ticket):
        committed, drained, abort_failed = False, False, False
        # Quiescence is published only AFTER releasing the model-owner mutex.
        with self._model_lock:
            try:
                with self._lock:
                    self._busy = True
                    ticket.state = "SEALED"
                    self._fence(ticket)
                if self.mode == "required":
                    ticket.update = self.adapter.prepare(ticket.payload)
                with self._lock:
                    ticket.state = "PREPARED"
                    self._fence(ticket)
                    ticket.state = "GENERATING"
                action = self.generate(ticket.payload, ticket.update)
                if (
                    not isinstance(action, list)
                    or len(action) != 16
                    or any(not isinstance(row, list) or len(row) != 15 for row in action)
                    or any(type(v) not in (float, int) or not math.isfinite(v) for row in action for v in row)
                ):
                    raise ValueError("generation must return finite [16,15] raw actions")
                body = json_bytes(
                    {
                        "protocol": self.protocol,
                        "local_memory_mode": self.mode,
                        "request_key": ticket.key,
                        "payload_sha256": ticket.payload_sha256,
                        "assignment": ticket.lease.identity(),
                        "frontier": request_frontier(ticket.payload, self.mode),
                        "action": action,
                        **self._response_local(ticket.update),
                        "native_batch": {"size": 1, "index": 0},
                        "media": {"status": "disabled"},
                    }
                )
                if len(body) > MAX_RESPONSE:
                    raise ValueError("bounded serialized response exceeded")
                cached = CachedMemoryResponse(ticket.key, ticket.payload_sha256, body)
                self.quiesce()  # Device work must complete BEFORE state publication.
                drained = True
                with self._lock:
                    ticket.state = "VALIDATED"
                    self._fence(ticket)
                    self._commit_responses([ticket], [cached])
                    committed = True
                    ticket.state = "COMMITTED"
            except BaseException as error:
                stage = ticket.state
                if not committed and self.mode == "required":
                    try:
                        self.adapter.abort(ticket.update)
                    except BaseException:
                        abort_failed = True
                with self._lock:
                    self._failed[ticket.lease.slot_id] = {
                        "request_key": ticket.key,
                        "payload_sha256": ticket.payload_sha256,
                        "state": "QUARANTINED" if abort_failed else "ABORTED",
                        "failed_stage": stage,
                        "error_type": type(error).__name__,
                        "error": str(error)[:2048],
                        "traceback": traceback.format_exc()[-16384:],
                    }
                    if not committed:
                        ticket.state = "ABORTED"
                logging.getLogger(__name__).exception("Session inference failed at %s key=%s", stage, ticket.key)
            finally:
                if not drained:
                    try:
                        self.quiesce()
                        drained = True
                    except BaseException:
                        logging.getLogger(__name__).exception("Model quiescence unproven; Server quarantined")
        with self._lock:
            if not drained or abort_failed:
                self._quarantined = True
                if ticket.lease.slot_id in self._failed:
                    self._failed[ticket.lease.slot_id].update(state="QUARANTINED", quiescent=False)
                self._changed.notify_all()
                return  # Retain unresolved ownership/candidates until Server decommission.
            ticket.quiescent = True
            if ticket.lease.slot_id in self._failed:
                self._failed[ticket.lease.slot_id]["quiescent"] = True
            self._busy = False
            ticket.update = None
            if not committed:
                self._release(ticket)
            self._changed.notify_all()

    def close(self, timeout_s=10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            self._stopping = True
            for ticket in self._tickets.values():
                ticket.cancelled = True
            while (self._busy or any(not t.quiescent for t in self._tickets.values())) and not self._quarantined:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(remaining)
            safe = not self._busy and all(t.quiescent for t in self._tickets.values()) and not self._quarantined
            if safe:
                try:
                    for ticket in tuple(self._tickets.values()):
                        self._release(ticket)
                    for lease in tuple(self._leases.values()):
                        self._reset_session(lease.session_id)
                        self._closed[lease.slot_id] = lease
                    self._leases.clear()
                    self._closing.clear()
                except BaseException:
                    self._quarantined = True
                    safe = False
        self._executor.shutdown(wait=False, cancel_futures=False)
        return safe

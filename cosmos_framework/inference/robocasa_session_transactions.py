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
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable

from cosmos_framework.inference.local_memory_state import CachedMemoryResponse

PROTOCOL = "robocasa.session-transactions.b1.v1"
MAX_BODY = 32 * 1024**2
MAX_RESPONSE = 1024**2


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
        if adapter.mode != "required" or adapter.memory is None:
            raise ValueError("Session transactions require real required-mode Local-TTT")
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
                "profile": PROTOCOL,
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
                "max_encoded_bytes": self.max_bytes,
                "max_item_bytes": MAX_BODY,
                "max_http_requests": len(self.slot_ids) + 1,
            }

    def _healthy(self):
        if self._quarantined or self._stopping:
            raise RuntimeError("transaction actor unavailable; owned Server recovery required")

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
                    return self._reply(200, {"lease": asdict(current)})
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
            return self._reply(200, {"lease": asdict(lease)})

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

    def submit(self, proof: dict, payload: dict) -> Reply:
        raw = json_bytes(payload)
        if len(raw) > MAX_BODY:
            raise ValueError("encoded inference item exceeds bound")
        payload = strict_json(raw)  # Freeze caller-owned structures before admission.
        if set(payload) != {"image", "prompt", "domain_name", "image_size", "state", "local_memory"}:
            raise ValueError("B1 policy payload requires observation/state/local evidence; media must be off")
        local = payload["local_memory"]
        if not isinstance(local, dict) or type(local.get("consumer_step")) is not int:
            raise ValueError("integer Local consumer frontier required")
        frontier = local["consumer_step"]
        if frontier < 0 or not isinstance(local.get("evidence"), list) or len(local["evidence"]) > 16:
            raise ValueError("completed evidence must be bounded by H16")
        payload_sha = hashlib.sha256(raw).hexdigest()
        with self._lock:
            lease = self._lease(proof)
            if (local.get("session_id"), local.get("episode_id")) != (lease.session_id, lease.episode_id):
                raise ValueError("Local evidence belongs to another assignment")
            key = digest({"protocol": PROTOCOL, **lease.identity(), "frontier": frontier})
            cached = self.adapter.snapshot.responses.get(lease.session_id)
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
            previous = self.adapter.snapshot.fast.get(lease.session_id)
            if previous is not None and frontier <= previous.consumer_step:
                raise ValueError("stale or evicted evidence frontier")
            if previous is None:
                if frontier != 0 or local["evidence"]:
                    raise ValueError("fresh Episode must start at frontier0 without evidence")
            elif frontier - previous.consumer_step != len(local["evidence"]):
                raise ValueError("bounded contiguous completed evidence required")
            if any(item.lease.slot_id == lease.slot_id for item in self._tickets.values()):
                raise ValueError("one queued/inflight/undelivered request per physical Slot")
            if sum(item.size for item in self._tickets.values()) + len(raw) > self.max_bytes:
                raise RuntimeError("bounded encoded-request capacity exhausted")
            ticket = _Ticket(key, payload_sha, lease, payload, len(raw), self.clock() + self.timeout_s)
            self._tickets[key] = ticket
            try:
                self._executor.submit(self._execute, ticket)
            except BaseException:
                del self._tickets[key]
                raise
            return self._pending(ticket)

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
        cached = self.adapter.snapshot.responses.get(lease.session_id)
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
            self.adapter.reset(lease.session_id)
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
                status = self.adapter.status(ticket.update)
                body = json_bytes(
                    {
                        "protocol": PROTOCOL,
                        "request_key": ticket.key,
                        "payload_sha256": ticket.payload_sha256,
                        "assignment": ticket.lease.identity(),
                        "frontier": ticket.payload["local_memory"]["consumer_step"],
                        "action": action,
                        "local_memory": status,
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
                    self.adapter.commit(ticket.update, cached)
                    committed = True
                    ticket.state = "COMMITTED"
            except BaseException as error:
                stage = ticket.state
                if not committed:
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
                        self.adapter.reset(lease.session_id)
                        self._closed[lease.slot_id] = lease
                    self._leases.clear()
                    self._closing.clear()
                except BaseException:
                    self._quarantined = True
                    safe = False
        self._executor.shutdown(wait=False, cancel_futures=False)
        return safe

"""Multi-Session native inference on the existing transaction owner/executor.

Only Local preparation may iterate. Each sealed microbatch invokes generate_many
once and publishes all Fast/Visual/Response entries in one snapshot. Media is
bounded CPU spool on the SAME executor; strict all-media gets decode priority.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
from cosmos_framework.inference.robocasa_session_transactions import (
    MAX_BODY,
    MAX_RESPONSE,
    Reply,
    SessionTransactions,
    digest,
    json_bytes,
    request_frontier,
)

# Reservation frontiers are bounded to0..100000 by the existing wire contract.
# Retain only identity digests, never videos/receipts, until acknowledged End.
MAX_MEDIA_REPLAY_RECORDS = 100001


@dataclass(frozen=True)
class MediaCandidate:
    value: object
    size: int


@dataclass(frozen=True)
class BatchOutput:
    actions: list
    media: tuple = ()


@dataclass(eq=False)
class _Media:
    ticket: object
    candidate: MediaCandidate | None
    deadline: float
    state: str = "QUEUED"
    body: bytes | None = None
    cancelled: bool = False
    required: bool = False
    replan: int = 0
    charge: int = 0
    error: str | None = None


class BatchedSessionTransactions(SessionTransactions):
    def __init__(
        self,
        adapter,
        generate_many,
        quiesce,
        *,
        max_batch_size=2,
        batch_wait_ms=20,
        decode_media=None,
        media_spool_max_jobs=4,
        media_spool_max_bytes=512 * 1024**2,
        media_drain_timeout_s=300.0,
        **kwargs,
    ):
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= 8:
            raise ValueError("native batch cap must be1..8")
        if (
            type(batch_wait_ms) not in (int, float)
            or not math.isfinite(batch_wait_ms)
            or not 1 <= batch_wait_ms <= 1000
        ):
            raise ValueError("bounded batch wait1..1000ms required")
        if (
            type(media_spool_max_jobs) is not int
            or not 1 <= media_spool_max_jobs <= 64
            or type(media_spool_max_bytes) is not int
            or not 1 <= media_spool_max_bytes <= 2**31
            or type(media_drain_timeout_s) not in (int, float)
            or not 0 < media_drain_timeout_s <= 3600
        ):
            raise ValueError("bounded positive media spool/drain settings required")
        super().__init__(adapter, None, quiesce, **kwargs)
        self.generate_many, self.decode_media = generate_many, decode_media
        self.max_batch_size, self.wait_s = max_batch_size, batch_wait_ms / 1000
        self.media_spool_max_jobs, self.media_spool_max_bytes = media_spool_max_jobs, media_spool_max_bytes
        self.media_drain_timeout_s = media_drain_timeout_s
        self._acked_media = {}
        self._queue, self._enqueued, self._media = [], {}, {}
        self._pumping = False
        self._batch_histogram = Counter()
        self._batches = 0
        self._queue_high_water = 0

    def info(self):
        with self._lock:
            return {
                **super().info(),
                "execution_profile": "session_native_batch_v1",
                "max_native_batch": self.max_batch_size,
                "batch_wait_ms": self.wait_s * 1000,
                "batches": self._batches,
                "batch_histogram": dict(self._batch_histogram),
                "queued": len(self._queue),
                "queue_high_water": self._queue_high_water,
                "media_jobs": len(self._media),
                "media_cpu_bytes": sum(
                    j.candidate.size if j.candidate else len(j.body or b"") for j in self._media.values()
                ),
                "generated_media": "bounded_cpu_spool_same_model_executor",
                "media_spool_max_jobs": self.media_spool_max_jobs,
                "media_spool_max_bytes": self.media_spool_max_bytes,
                "media_drain_timeout_s": self.media_drain_timeout_s,
                "strict_media_item_bytes": min(MAX_BODY, self.media_spool_max_bytes),
            }

    def media_room(self):
        with self._lock:
            for key, job in list(self._media.items()):
                if self.clock() >= job.deadline and key not in self._tickets and job.state != "DECODING":
                    del self._media[key]
            used = sum(
                j.charge or (j.candidate.size if j.candidate else len(j.body or b"")) for j in self._media.values()
            )
            return max(0, self.media_spool_max_jobs - len(self._media)), max(0, self.media_spool_max_bytes - used)

    def reserve_media(self, proof, key, payload_sha, replan, frontier):
        """Nonblocking control admission; the client waits, never this executor."""
        if (
            type(replan) is not int
            or not 0 <= replan <= 100000
            or type(frontier) is not int
            or not 0 <= frontier <= 100000
            or not isinstance(payload_sha, str)
            or len(payload_sha) != 64
            or any(c not in "0123456789abcdef" for c in payload_sha)
        ):
            raise ValueError("bounded media reservation identity required")
        with self._lock:
            lease = self._lease(proof)
            self._usable(lease)
            if key != digest({"protocol": self.protocol, **lease.identity(), "frontier": frontier}):
                raise ValueError("media reservation request identity mismatch")
            history = self._acked_media.get(lease.session_id, {})
            acknowledged = history.get(key)
            if acknowledged is not None:
                if acknowledged[0] != payload_sha or acknowledged[2] != replan:
                    raise ValueError("acknowledged media reservation identity reused")
                return self._reply(
                    200, {"reserved": True, "already_saved": True, "request_key": key, "payload_sha256": payload_sha}
                )
            self.media_room()
            job = self._media.get(key)
            if job is not None:
                if job.ticket.lease != lease or job.ticket.payload_sha256 != payload_sha or job.replan != replan:
                    raise ValueError("media reservation identity reused")
                return self._reply(200, {"reserved": True, "request_key": key, "payload_sha256": payload_sha})
            reserved = sum(j.required and j.ticket.lease == lease for j in self._media.values())
            if len(history) + reserved >= MAX_MEDIA_REPLAY_RECORDS:
                return self._reply(
                    200,
                    {
                        "reserved": False,
                        "media": {
                            "status": "MEDIA_INCOMPLETE",
                            "error": "bounded Episode media replay history exhausted",
                        },
                    },
                )
            if self.decode_media is None:
                return self._reply(
                    200, {"reserved": False, "media": {"status": "MEDIA_INCOMPLETE", "error": "decoder unavailable"}}
                )
            room, byte_room = self.media_room()
            charge = min(MAX_BODY, self.media_spool_max_bytes)
            if room < 1 or byte_room < charge:
                return self._reply(429, {"reserved": False, "media": {"status": "backpressure"}})
            ticket = SimpleNamespace(key=key, payload_sha256=payload_sha, lease=lease)
            self._media[key] = _Media(
                ticket,
                None,
                self.clock() + self.media_drain_timeout_s,
                state="RESERVED",
                required=True,
                replan=replan,
                charge=charge,
            )
            return self._reply(200, {"reserved": True, "request_key": key, "payload_sha256": payload_sha})

    def media_capture_budget(self, payload):
        if not payload.get("media", {}).get("required"):
            return 0
        payload_sha = digest(payload)
        with self._lock:
            return next(
                (
                    j.charge
                    for j in self._media.values()
                    if j.required
                    and j.ticket.payload_sha256 == payload_sha
                    and j.state in ("RESERVED", "CAPTURING")
                    and not j.cancelled
                ),
                0,
            )

    def _release(self, ticket):
        super()._release(ticket)
        if ticket.state != "COMMITTED":
            self._media.pop(ticket.key, None)

    def submit(self, proof, payload, *, budget):
        # Independent malformed requests must fail before sealing valid peers.
        if not isinstance(payload, dict):
            raise ValueError("policy object required")
        state = payload.get("state")
        if (
            not isinstance(state, list)
            or len(state) != 15
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in state)
        ):
            raise ValueError("finite state15 required")
        if (
            payload.get("image_size") != 256
            or not isinstance(payload.get("image"), str)
            or not isinstance(payload.get("prompt"), str)
            or payload.get("domain_name") != "robocasa"
        ):
            raise ValueError("native batch requires corrected RoboCasa H16/profile")
        return super().submit(proof, payload, budget=budget)

    def _payload_keys_valid(self, payload):
        if "media" not in payload:
            return super()._payload_keys_valid(payload)
        media = payload["media"]
        return (
            super()._payload_keys_valid({k: v for k, v in payload.items() if k != "media"})
            and isinstance(media, dict)
            and (set(media) == {"replan"} or (set(media) == {"replan", "required"} and media["required"] is True))
            and type(media["replan"]) is int
            and 0 <= media["replan"] <= 100000
        )

    def _schedule(self, ticket):
        self._queue.append(ticket)
        self._enqueued[ticket.key] = time.monotonic()
        ticket.queued_at = self._enqueued[ticket.key]
        self._queue_high_water = max(self._queue_high_water, len(self._queue))
        self._changed.notify_all()
        if not self._pumping:
            self._pumping = True
            try:
                self._executor.submit(self._pump)
            except BaseException:
                self._queue.remove(ticket)
                self._enqueued.pop(ticket.key)
                self._pumping = False
                raise

    def _pump(self):
        while True:
            with self._changed:
                # Strict clients cannot fetch until decode runs. Prioritize these
                # bounded jobs so a busy inference queue cannot starve completion.
                strict_job = next((j for j in self._media.values() if j.required and j.state == "QUEUED"), None)
                if strict_job is not None:
                    members, job = None, strict_job
                    job.state = "DECODING"
                    self._busy = True
                elif self._queue:
                    oldest = self._queue[0]
                    remaining = min(
                        self.wait_s - (time.monotonic() - self._enqueued[oldest.key]), oldest.deadline - self.clock()
                    )
                    if (
                        not oldest.cancelled
                        and len(self._queue) < min(self.max_batch_size, len(self.slot_ids))
                        and remaining > 0
                        and not self._stopping
                    ):
                        self._changed.wait(remaining)
                        continue
                    # Pre-seal expiry/cancel faults only that independent queued member.
                    stale = next(
                        (
                            t
                            for t in self._queue
                            if t.cancelled or self.clock() >= t.deadline or t.lease.slot_id in self._closing
                        ),
                        None,
                    )
                    members = [stale] if stale else self._queue[: self.max_batch_size]
                    for ticket in members:
                        self._queue.remove(ticket)
                        self._enqueued.pop(ticket.key, None)
                    job = None
                else:
                    members = None
                    job = next((j for j in self._media.values() if j.state == "QUEUED"), None)
                    if job is None:
                        self._pumping = False
                        return
                    job.state = "DECODING"
                    self._busy = True  # Reserve media ownership before releasing admission lock.
            if members:
                self._execute_many(members)
            else:
                self._decode(job)
            with self._lock:
                if self._quarantined:
                    self._pumping = False
                    return

    def _execute_many(self, members):
        committed, drained, abort_failed = False, False, False
        batch_id = uuid.uuid4().hex
        with self._model_lock:
            try:
                with self._lock:
                    self._busy = True
                    for t in members:
                        t.state = "SEALED"
                        self._fence(t)
                        if t.key in self._media and self._media[t.key].required:
                            self._media[t.key].state = "CAPTURING"
                prepare_start = time.monotonic()
                for t in members:
                    if self.mode == "required":
                        t.update = self.adapter.prepare(t.payload)
                    t.state = "PREPARED"
                with self._lock:
                    for t in members:
                        self._fence(t)
                        t.state = "GENERATING"
                prepared = time.monotonic()
                output = self.generate_many([t.payload for t in members], [t.update for t in members])
                generated = time.monotonic()
                if not isinstance(output, BatchOutput) or len(output.actions) != len(members):
                    raise ValueError("native batch output cardinality mismatch")
                media_candidates = output.media if len(output.media) == len(members) else ()
                caches, new_jobs = [], {}
                room, byte_room = self.media_room()
                for i, (t, action) in enumerate(zip(members, output.actions, strict=True)):
                    if (
                        not isinstance(action, list)
                        or len(action) != 16
                        or any(
                            not isinstance(r, list)
                            or len(r) != 15
                            or any(type(v) not in (int, float) or not math.isfinite(v) for v in r)
                            for r in action
                        )
                    ):
                        raise ValueError("native batch requires finite [16,15] per member")
                    candidate = media_candidates[i] if media_candidates else None
                    media = {"status": "disabled"}
                    if t.payload.get("media"):
                        required = t.payload["media"].get("required", False)
                        if required:
                            with self._lock:
                                reservation = self._media.get(t.key)
                                valid = (
                                    reservation is not None
                                    and reservation.required
                                    and reservation.ticket.lease == t.lease
                                    and reservation.ticket.payload_sha256 == t.payload_sha256
                                )
                            media = {
                                "status": "MEDIA_INCOMPLETE",
                                "reason": "reserved media candidate unavailable/oversized",
                            }
                            if (
                                valid
                                and candidate is not None
                                and 0 < candidate.size <= reservation.charge
                                and self.decode_media is not None
                            ):
                                new_jobs[t.key] = _Media(
                                    t,
                                    candidate,
                                    self.clock() + self.media_drain_timeout_s,
                                    required=True,
                                    replan=reservation.replan,
                                    charge=reservation.charge,
                                )
                                media = {
                                    "status": "pending",
                                    "ticket": t.key,
                                    "required": True,
                                    "deadline": new_jobs[t.key].deadline,
                                }
                        else:
                            media = {"status": "dropped", "reason": "bounded_spool_unavailable"}
                        if not required and (
                            candidate is not None
                            and room > 0
                            and 0 < candidate.size <= byte_room
                            and self.decode_media is not None
                        ):
                            new_jobs[t.key] = _Media(t, candidate, min(t.deadline, self.clock() + self.timeout_s))
                            room -= 1
                            byte_room -= candidate.size
                            media = {"status": "pending", "ticket": t.key}
                    body = json_bytes(
                        {
                            "protocol": self.protocol,
                            "local_memory_mode": self.mode,
                            "request_key": t.key,
                            "payload_sha256": t.payload_sha256,
                            "assignment": t.lease.identity(),
                            "frontier": request_frontier(t.payload, self.mode),
                            "action": action,
                            **self._response_local(t.update),
                            "native_batch": {"id": batch_id, "size": len(members), "index": i},
                            "timing": {
                                "queue_ms": max(0, prepare_start - t.queued_at) * 1000,
                                "batch_prepare_ms": (prepared - prepare_start) * 1000,
                                "native_generation_ms": (generated - prepared) * 1000,
                            },
                            "media": media,
                        }
                    )
                    if len(body) > MAX_RESPONSE:
                        raise ValueError("bounded batch response exceeded")
                    caches.append(CachedMemoryResponse(t.key, t.payload_sha256, body))
                self.quiesce()
                drained = True
                with self._lock:
                    for t in members:
                        t.state = "VALIDATED"
                        self._fence(t)
                    histogram = self._batch_histogram.copy()
                    histogram[len(members)] += 1
                    batches = self._batches + 1
                    jobs = {**self._media, **new_jobs}  # Respect End/ACK removals while generating.
                    for t in members:
                        if t.payload.get("media", {}).get("required") and t.key not in new_jobs:
                            jobs.pop(t.key, None)
                    self._commit_responses(members, caches, batch=True)
                    committed = True
                    self._media = jobs
                    self._batches = batches
                    self._batch_histogram = histogram
                    for t in members:
                        t.state = "COMMITTED"
            except BaseException as error:
                for t in members:
                    stage = t.state
                    if not committed and self.mode == "required":
                        try:
                            self.adapter.abort(t.update)
                        except BaseException:
                            abort_failed = True
                    with self._lock:
                        self._failed[t.lease.slot_id] = {
                            "request_key": t.key,
                            "payload_sha256": t.payload_sha256,
                            "state": "QUARANTINED" if abort_failed else "ABORTED",
                            "failed_stage": stage,
                            "error_type": type(error).__name__,
                            "error": str(error)[:2048],
                            "traceback": traceback.format_exc()[-16384:],
                        }
                        if not committed:
                            t.state = "ABORTED"
                logging.getLogger(__name__).exception("Native batch %s aborted", batch_id)
            finally:
                if not drained:
                    try:
                        self.quiesce()
                        drained = True
                    except BaseException:
                        logging.getLogger(__name__).exception("Native batch drain unproven")
        with self._lock:
            if not drained or abort_failed:
                self._quarantined = True
                for t in members:
                    if t.lease.slot_id in self._failed:
                        self._failed[t.lease.slot_id].update(state="QUARANTINED", quiescent=False)
                self._changed.notify_all()
                return
            self._busy = False
            for t in members:
                t.quiescent, t.update = True, None
                if t.lease.slot_id in self._failed:
                    self._failed[t.lease.slot_id]["quiescent"] = True
                if not committed:
                    self._release(t)
            self._changed.notify_all()

    def _decode(self, job):
        drained = False
        with self._model_lock:
            with self._lock:
                self._busy = True
                job.state = "DECODING"
            try:
                if job.cancelled or self.clock() >= job.deadline:
                    value = {"status": "expired"}
                else:
                    video = self.decode_media(job.candidate.value)
                    value = {"status": "ready", "video": video}
                    if job.required:
                        value.update(replan=job.replan, assignment=job.ticket.lease.identity())
                body = json_bytes(
                    {"request_key": job.ticket.key, "payload_sha256": job.ticket.payload_sha256, "media": value}
                )
                if len(body) > MAX_BODY:
                    raise ValueError("bounded media response exceeded")
                if job.required and len(body) > job.charge:
                    raise ValueError("decoded media exceeds reserved byte budget")
            except Exception as error:
                if job.required:
                    job.error = str(error)[:1024]
                    body = None  # Failure metadata is not cached video data.
                else:
                    body = json_bytes(
                        {
                            "request_key": job.ticket.key,
                            "payload_sha256": job.ticket.payload_sha256,
                            "media": {"status": "error", "error": str(error)[:1024]},
                        }
                    )
            finally:
                try:
                    self.quiesce()
                    drained = True
                except BaseException:
                    logging.getLogger(__name__).exception("Optional decoder device drain unproven")
        with self._lock:
            if not drained:
                self._quarantined = True
            elif job.cancelled or self.clock() >= job.deadline or self._stopping:
                self._media.pop(job.ticket.key, None)
            else:
                job.body, job.candidate, job.state = body, None, "FAILED" if job.error else "READY"
            self._busy = not drained
            self._changed.notify_all()

    def media_result(self, proof, key, sha):
        with self._lock:
            lease = self._lease(proof)
            job = self._media.get(key)
            if job is None:
                return self._reply(200, {"request_key": key, "payload_sha256": sha, "media": {"status": "unavailable"}})
            if job.ticket.lease != lease or job.ticket.payload_sha256 != sha:
                raise ValueError("foreign media identity")
            if self.clock() >= job.deadline and job.state not in ("DECODING", "CAPTURING"):
                self._media.pop(key, None)
                return self._reply(200, {"request_key": key, "payload_sha256": sha, "media": {"status": "expired"}})
            if job.error:
                return self._reply(
                    200,
                    {
                        "request_key": key,
                        "payload_sha256": sha,
                        "media": {"status": "MEDIA_INCOMPLETE", "error": job.error},
                    },
                )
            return (
                Reply(200, job.body)
                if job.body
                else self._reply(202, {"request_key": key, "payload_sha256": sha, "media": {"status": "pending"}})
            )

    def media_ack(self, proof, key, sha, receipt=None):
        with self._lock:
            lease = self._lease(proof)
            if key in self._media:
                job = self._media[key]
                if job.ticket.lease != self._lease(proof) or job.ticket.payload_sha256 != sha or job.state != "READY":
                    raise ValueError("invalid media ACK")
                if job.required:
                    transfer = {
                        "assignment": lease.identity(),
                        "request_key": key,
                        "payload_sha256": sha,
                        "replan": job.replan,
                    }
                    if (
                        not isinstance(receipt, dict)
                        or len(json_bytes(receipt)) > 16384
                        or receipt.get("status") != "VIDEO_SAVED"
                        or receipt.get("index") != job.replan
                        or receipt.get("binding", {}).get("transfer") != transfer
                        or receipt.get("path") != f"generated/replan_{job.replan:04d}.mp4"
                        or type(receipt.get("frames")) is not int
                        or not 1 <= receipt["frames"] <= 65
                        or type(receipt.get("bytes")) is not int
                        or receipt["bytes"] <= 0
                        or not isinstance(receipt.get("sha256"), str)
                        or len(receipt["sha256"]) != 64
                    ):
                        raise ValueError("strict media ACK requires matching durable VIDEO_SAVED receipt")
                    self._acked_media.setdefault(lease.session_id, {})[key] = (sha, digest(receipt), job.replan)
                del self._media[key]
            elif receipt is not None and self._acked_media.get(lease.session_id, {}).get(key) != (
                sha,
                digest(receipt),
                receipt.get("index"),
            ):
                raise ValueError("unknown or changed strict media ACK replay")
            return self._reply(200, {"acknowledged": True, "request_key": key})

    def _cancel_media(self, lease=None):
        for key, job in list(self._media.items()):
            if lease is None or job.ticket.lease == lease:
                job.cancelled = True
                if job.state != "DECODING":
                    del self._media[key]

    def end(self, proof):
        with self._lock:
            lease = self._lease(proof, closed=True)
            self._cancel_media(lease)
            self._changed.notify_all()
            reply = super().end(proof)
            if reply.json().get("acknowledged") is True:
                self._acked_media.pop(lease.session_id, None)
            return reply

    def close(self, timeout_s=10):
        with self._lock:
            self._cancel_media()
            self._changed.notify_all()
        return super().close(timeout_s)

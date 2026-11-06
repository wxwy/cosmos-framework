"""Bounded, unpublished completed-evidence work for the single model executor.

HTTP admission only freezes and queues bytes. Encode1 and model-owned scans run
through ``run`` on the transaction owner's existing model thread. No background
thread, canonical frontier, response, or independent commit authority lives here.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from cosmos_framework.inference.local_memory_online import ComputedMemoryCandidate, OnlineMemoryRequest
from cosmos_framework.inference.local_memory_pipeline_metrics import DurationWindow
from cosmos_framework.inference.local_memory_pipeline_timing import PipelineTiming
from cosmos_framework.inference.robocasa_composite_visual import (
    composite_frame_digest,
    encode_current_composite_visual96,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.inference.robocasa_session_transactions import SessionLease, json_bytes, strict_json

_PAYLOAD_KEYS = {
    "session_id",
    "episode_id",
    "image_size",
    "preprocess_profile",
    "source_step",
    "composite_image",
    "executed_action",
}


@dataclass(eq=False)
class CompletedWork:
    lease: dict
    payload: dict
    size: int
    wire_digest: str
    queued_at: float
    stage: _Stage
    charged: bool = True
    state: str = "QUEUED"
    profile: dict | None = None


@dataclass(eq=False)
class _Stage:
    lease: dict
    anchor: object
    visual_anchor: object
    works: dict[int, CompletedWork] = field(default_factory=dict)
    candidate: ComputedMemoryCandidate | None = None
    cancelled: bool = False
    failed: str | None = None
    encoded_steps: int = 0


class CompletedEvidencePipeline:
    def __init__(self, memory, decode_row, tokenizer, *, max_items, max_bytes):
        if type(max_items) is not int or max_items <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive bounded completed-pipeline capacity required")
        self.memory, self.decode_row, self.tokenizer = memory, decode_row, tokenizer
        self.max_items, self.max_bytes = max_items, max_bytes
        self._lock = threading.RLock()
        self._stages = {}
        self._leases = {}
        self._queue = deque()
        self._active = None
        self._items = self._bytes = 0
        self._accepted = self._duplicates = self._completed = self._failures = 0
        self._queue_wait_ms = 0.0
        self._decode_prepare_ms = 0.0
        self._timings = {}
        self._latencies = {"queue": DurationWindow(), "decode_prepare_cpu": DurationWindow()}
        self._timing_sources = set()

    @staticmethod
    def _lease(lease):
        if not isinstance(lease, dict) or set(lease) != set(SessionLease.__dataclass_fields__):
            raise ValueError("exact completed assignment proof required")
        frozen = dict(lease)
        for name, value in frozen.items():
            if name in ("generation", "episode_index"):
                if type(value) is not int or value < (1 if name == "generation" else 0):
                    raise ValueError("invalid completed assignment integer identity")
            elif not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
                raise ValueError("invalid completed assignment identity")
        manifest = frozen["manifest_digest"]
        if len(manifest) != 64 or any(c not in "0123456789abcdef" for c in manifest):
            raise ValueError("completed assignment requires manifest SHA256")
        return frozen

    @staticmethod
    def freeze_payload(payload):
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise ValueError("exact completed evidence payload required")
        if (
            type(payload["source_step"]) is not int
            or payload["source_step"] < 0
            or type(payload["image_size"]) is not int
            or payload["image_size"] <= 0
            or payload["preprocess_profile"] != PREPROCESS_PROFILE
            or not isinstance(payload["composite_image"], str)
            or not payload["composite_image"]
        ):
            raise ValueError("invalid completed evidence step/image/profile")
        action = payload["executed_action"]
        if (
            not isinstance(action, list)
            or len(action) != 15
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in action)
        ):
            raise ValueError("completed evidence requires finite canonical raw15")
        raw = json_bytes(payload)
        return strict_json(raw), len(raw), hashlib.sha256(raw).hexdigest()

    def admit(self, lease, payload, anchor, visual_anchor):
        # Serialization/CPU validation precedes the short metadata lock. The
        # actor must authenticate this full lease against its registration first.
        lease = self._lease(lease)
        payload, size, wire_digest = self.freeze_payload(payload)
        session_id, step = lease["session_id"], payload["source_step"]
        if (payload["session_id"], payload["episode_id"]) != (session_id, lease["episode_id"]):
            raise ValueError("completed evidence belongs to another assignment")
        if (
            anchor is None
            or visual_anchor is None
            or (anchor.episode_id, visual_anchor.episode_id) != (lease["episode_id"], lease["episode_id"])
            or (payload["image_size"], payload["preprocess_profile"])
            != (visual_anchor.image_size, visual_anchor.preprocess_profile)
        ):
            raise ValueError("completed evidence has no matching committed episode/profile")
        with self._lock:
            registered = self._leases.get(session_id)
            if registered is not None and registered != lease:
                raise ValueError("completed Session assignment identity changed")
            stage = self._stages.get(session_id)
            if stage is not None and stage.lease != lease:
                raise ValueError("completed Session assignment identity changed")
            if step < anchor.consumer_step:
                if dict(visual_anchor.last_row_digests).get(step) != wire_digest:
                    raise ValueError("stale or changed-bytes completed evidence")
                self._duplicates += 1
                return {"accepted": False, "duplicate": True, "source_step": step}
            if stage is None:
                if visual_anchor.boundary_visual96 is None or visual_anchor.boundary_source_digest is None:
                    raise ValueError("completed boundary frame cache is missing; Encode1 fallback forbidden")
                stage = _Stage(lease, anchor, visual_anchor)
            elif stage.cancelled or stage.failed:
                raise RuntimeError("completed evidence Session is cancelled/failed")
            elif stage.anchor is not anchor or stage.visual_anchor is not visual_anchor:
                raise RuntimeError("completed evidence lost its committed anchor")
            previous = stage.works.get(step)
            if previous is not None:
                if previous.wire_digest != wire_digest:
                    raise ValueError("completed step identity reused with changed bytes")
                self._duplicates += 1
                return {"accepted": False, "duplicate": True, "source_step": step}
            if step != anchor.consumer_step + len(stage.works):
                raise ValueError("completed evidence steps must be contiguous and exactly-once")
            if len(stage.works) >= 16:
                raise RuntimeError("completed evidence per-Session H16 capacity exhausted")
            if self._items >= self.max_items or self._bytes + size > self.max_bytes:
                raise RuntimeError("bounded completed evidence capacity exhausted")
            work = CompletedWork(lease, payload, size, wire_digest, time.monotonic(), stage)
            stage.works[step] = work
            self._stages[session_id] = stage
            self._leases[session_id] = lease
            self._queue.append(work)
            self._items += 1
            self._bytes += size
            self._accepted += 1
            return {"accepted": True, "duplicate": False, "source_step": step}

    def next(self):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("completed model work already active")
            if not self._queue:
                return None
            work = self._queue.popleft()
            self._active = work
            work.state = "ACTIVE"
            waited = max(0.0, time.monotonic() - work.queued_at) * 1000
            self._queue_wait_ms += waited
            self._latencies["queue"].add(waited)
            return work

    def _release(self, work):
        if work.charged:
            self._items -= 1
            self._bytes -= work.size
            work.charged = False

    def _discard_stage(self, stage):
        stage.cancelled = True
        if self._stages.get(stage.lease["session_id"]) is stage:
            del self._stages[stage.lease["session_id"]]
        self._queue = deque(work for work in self._queue if work.stage is not stage)
        for work in stage.works.values():
            if work is not self._active:
                self._release(work)
        stage.works.clear()
        stage.candidate = None

    def cancel(self, lease=None):
        lease = None if lease is None else self._lease(lease)
        with self._lock:
            for stage in tuple(self._stages.values()):
                if lease is None or stage.lease == lease:
                    self._discard_stage(stage)
            return self._active is not None and (lease is None or self._active.lease == lease)

    def cancel_session(self, session_id):
        with self._lock:
            stage = self._stages.get(session_id)
            if stage is not None:
                self._discard_stage(stage)
            self._leases.pop(session_id, None)
            return self._active is not None and self._active.lease["session_id"] == session_id

    def discard(self, work):
        """Model owner calls only after this popped work is proven quiescent."""
        with self._lock:
            self._discard_stage(work.stage)
            if self._active is work:
                self._active = None
            self._release(work)

    def run(self, work):
        with self._lock:
            if self._active is not work:
                raise RuntimeError("completed scan requires the exact active work capability")
            stage = work.stage
            candidate = stage.candidate
            cancelled = stage.cancelled
        timing = PipelineTiming(next(self.memory.encoder.parameters()).device)
        finished = False
        try:
            if cancelled:
                measured = timing.finish()
                finished = True
                return {"cancelled": True, **measured}
            waited_ms = max(0.0, time.monotonic() - work.queued_at) * 1000
            prepare_started = time.perf_counter()
            frame, action = self.decode_row(
                {key: work.payload[key] for key in ("source_step", "composite_image", "executed_action")},
                work.payload["source_step"],
            )
            prepared = prepare_robocasa_composite_frame(frame, work.payload["image_size"])
            decode_prepare_ms = (time.perf_counter() - prepare_started) * 1000
            boundary = work.payload["source_step"] == stage.anchor.consumer_step
            if boundary:
                if composite_frame_digest(prepared.source_uint8) != stage.visual_anchor.boundary_source_digest:
                    raise ValueError("completed boundary frame digest mismatch; Encode1 fallback forbidden")
                visual = stage.visual_anchor.boundary_visual96.detach().cpu().float().clone()
            else:
                _, visual = encode_current_composite_visual96(self.tokenizer, prepared, timing=timing)
                visual = visual.detach().cpu().float().clone()
            request = OnlineMemoryRequest(
                work.lease["session_id"],
                work.lease["episode_id"],
                work.payload["source_step"] + 1,
                (work.payload["source_step"],),
                visual.reshape(1, 96),
                action.reshape(1, 15),
            )
            result = self.memory.compute_completed_step(stage.anchor, candidate, request, timing=timing)
            # Event waits and telemetry synchronization happen outside every
            # admission/Store lock. The caller still owns the model lock.
            measured = timing.finish()
            finished = True
            measured["decode_prepare_cpu_ms"] = decode_prepare_ms
            measured["queue_ms"] = waited_ms
            with self._lock:
                if not stage.cancelled and self._stages.get(work.lease["session_id"]) is stage:
                    stage.candidate = result
                    stage.encoded_steps += int(not boundary)
                    work.state = "DONE"
                    work.profile = measured
                    self._completed += 1
                    self._decode_prepare_ms += decode_prepare_ms
                    self._latencies["decode_prepare_cpu"].add(decode_prepare_ms)
                    self._timing_sources.add(measured["timing_source"])
                    for name, values in measured["stages"].items():
                        total = self._timings.setdefault(name, {"count": 0, "total_ms": 0.0})
                        total["count"] += values["count"]
                        total["total_ms"] += values["total_ms"]
                        self._latencies.setdefault(name, DurationWindow()).add(values["total_ms"])
                else:
                    measured["cancelled"] = True
            return measured
        except BaseException as error:
            # No partially computed state can become a Replan candidate. The
            # transaction actor owns error attribution, device drain and recovery.
            with self._lock:
                self._failures += 1
                stage.failed = type(error).__name__
                self._discard_stage(stage)
            raise
        finally:
            with self._lock:
                if finished and self._active is work:
                    self._active = None
                if finished and stage.cancelled:
                    self._release(work)

    def ready(self, session_id, frontier):
        with self._lock:
            stage = self._stages.get(session_id)
            return bool(
                stage is not None
                and not stage.cancelled
                and not stage.failed
                and (self._active is None or self._active.stage is not stage)
                and stage.candidate is not None
                and stage.candidate.request.consumer_step == frontier
                and all(work.state == "DONE" for work in stage.works.values())
            )

    def candidate(self, session_id, frontier, anchor, visual_anchor, row_digests):
        with self._lock:
            stage = self._stages.get(session_id)
            if (
                not self.ready(session_id, frontier)
                or stage.anchor is not anchor
                or stage.visual_anchor is not visual_anchor
            ):
                raise RuntimeError("completed evidence is not ready at the exact committed anchor/frontier")
            if tuple((step, work.wire_digest) for step, work in stage.works.items()) != row_digests:
                raise ValueError("Replan evidence differs from completed staged bytes")
            return stage.candidate, stage.encoded_steps

    def committed(self, session_id):
        with self._lock:
            stage = self._stages.get(session_id)
            if stage is not None:
                if self._active is not None and self._active.stage is stage:
                    raise RuntimeError("cannot publish an active completed suffix")
                self._discard_stage(stage)

    def profile(self, session_id, frontier):
        """Completed window diagnostics, captured before its atomic commit clears it."""
        with self._lock:
            stage = self._stages.get(session_id)
            if not self.ready(session_id, frontier):
                return {"completed_steps": 0, "independent_encode1_steps": 0, "latencies": {}}
            windows = {}
            for work in stage.works.values():
                measured = work.profile
                if measured is None:
                    raise RuntimeError("completed work has no finalized timing evidence")
                for name, value in (
                    ("queue", measured["queue_ms"]),
                    ("decode_prepare_cpu", measured["decode_prepare_cpu_ms"]),
                ):
                    windows.setdefault(name, DurationWindow(16)).add(value)
                for name, value in measured["stages"].items():
                    windows.setdefault(name, DurationWindow(16)).add(value["total_ms"])
            return {
                "completed_steps": len(stage.works),
                "independent_encode1_steps": stage.encoded_steps,
                "timing_sources": sorted({work.profile["timing_source"] for work in stage.works.values()}),
                "latencies": {name: window.summary() for name, window in windows.items()},
            }

    def info(self):
        with self._lock:
            return {
                "enabled": True,
                "pipeline": "per_frame_v1",
                "queued": len(self._queue),
                "active": int(self._active is not None),
                "items": self._items,
                "bytes": self._bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "accepted": self._accepted,
                "duplicates": self._duplicates,
                "completed": self._completed,
                "failed": self._failures,
                "queue_wait_total_ms": self._queue_wait_ms,
                "decode_prepare_total_cpu_ms": self._decode_prepare_ms,
                "stages": {name: dict(value) for name, value in self._timings.items()},
                "timing_sources": sorted(self._timing_sources),
                "latencies": {name: window.summary() for name, window in self._latencies.items()},
            }

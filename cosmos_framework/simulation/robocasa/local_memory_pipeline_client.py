"""Bounded completed-frame PNG/upload worker; prediction ACK owns its ledger."""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import requests

from cosmos_framework.inference.local_memory_pipeline_metrics import DurationWindow
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.inference.robocasa_session_transactions import json_bytes
from cosmos_framework.simulation.robocasa.eval_utils import b64_png


@dataclass
class _Frame:
    source: int
    image: np.ndarray | None
    action: list[float]
    image_size: int
    profile: str
    fingerprint: bytes
    enqueued_at: float
    row: dict | None = None
    admitted: bool = False


class CompletedEvidencePipeline:
    """One owned worker/session, bounded by the original Episode deadline.

    ``sender(http, frozen_json_bytes, timeout)`` returns an HTTP response. Only
    connection/timeouts retry. Admission ACKs never clear completed evidence.
    """

    def __init__(
        self,
        lease,
        episode_deadline,
        sender,
        *,
        session_factory=requests.Session,
        encode=b64_png,
        max_pending=16,
        request_timeout_s=1.0,
        shutdown_timeout_s=2.0,
    ):
        if type(episode_deadline) not in (int, float) or not math.isfinite(episode_deadline):
            raise ValueError("original bounded Episode deadline required")
        if type(max_pending) is not int or not 1 <= max_pending <= 256:
            raise ValueError("bounded completed-evidence capacity required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (request_timeout_s, shutdown_timeout_s)
        ):
            raise ValueError("positive bounded pipeline timeouts required")
        self.lease = copy.deepcopy(lease)
        self.deadline, self.sender = episode_deadline, sender
        self.session_factory, self.encode = session_factory, encode
        self.max_pending = max_pending
        self.request_timeout_s, self.shutdown_timeout_s = min(1.0, request_timeout_s), shutdown_timeout_s
        self._changed = threading.Condition()
        self._stop = threading.Event()
        self._queue, self._frames = deque(), {}
        self._next, self._ack = 0, 0
        self._thread, self._error = None, None
        self._closed = False
        self._stats = dict(
            enqueued=0, uploaded=0, png_base64_bytes=0, png_seconds=0.0, upload_seconds=0.0, retries=0, queue_full=0
        )
        self._latencies = {name: DurationWindow() for name in ("queue", "png", "upload")}

    def _check(self):
        if self._error is not None:
            raise RuntimeError("completed-evidence worker failed: " + str(self._error)) from self._error
        if self._closed:
            raise RuntimeError("completed-evidence pipeline is closed")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("original Episode deadline exhausted during completed evidence")

    def submit(self, source_step, image, action, *, image_size, preprocess_profile):
        with self._changed:
            self._check()
            if type(source_step) is not int or source_step < self._ack:
                raise ValueError("late completed-evidence source step")
            if source_step == self._next and len(self._frames) >= self.max_pending:
                self._stats["queue_full"] += 1
                raise RuntimeError("completed-evidence queue capacity exhausted")
            image = np.asarray(image)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if image.dtype != np.uint8 or image.shape != (256, 512, 3):
                raise ValueError("completed pipeline requires uint8 [256,512,3]")
            if action.shape != (15,) or not np.isfinite(action).all():
                raise ValueError("completed pipeline requires finite raw15")
            if type(image_size) is not int or image_size <= 0 or preprocess_profile != PREPROCESS_PROFILE:
                raise ValueError("completed pipeline preprocessing identity mismatch")
            image = np.ascontiguousarray(image).copy()
            action = action.tolist()
            fingerprint = hashlib.sha256(
                image.tobytes() + json_bytes([action, image_size, preprocess_profile])
            ).digest()
            previous = self._frames.get(source_step)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise ValueError("completed-evidence source reused with changed bytes")
                return
            if source_step != self._next:
                raise ValueError("completed evidence must be contiguous")
            frame = _Frame(source_step, image, action, image_size, preprocess_profile, fingerprint, time.monotonic())
            self._frames[source_step] = frame
            self._queue.append(frame)
            self._next += 1
            self._stats["enqueued"] += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="robocasa-completed-evidence", daemon=True)
                self._thread.start()
            self._changed.notify_all()

    def _upload(self, http, frame, raw):
        started = time.monotonic()
        while not self._stop.is_set():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("original Episode deadline exhausted during completed upload")
            try:
                reply = self.sender(http, raw, min(self.request_timeout_s, remaining))
            except (requests.Timeout, requests.ConnectionError):
                with self._changed:
                    self._stats["retries"] += 1
                self._stop.wait(min(0.02, max(0, self.deadline - time.monotonic())))
                continue
            if self._stop.is_set():
                return False
            if time.monotonic() >= self.deadline:
                raise TimeoutError("late completed-evidence admission response")
            value = reply.json()
            if time.monotonic() >= self.deadline:
                raise TimeoutError("late completed-evidence response decoding")
            if reply.status_code not in (200, 202):
                detail = (
                    {key: value[key] for key in ("error", "error_type", "failed_stage", "source_step") if key in value}
                    if isinstance(value, dict)
                    else {}
                )
                raise RuntimeError(
                    f"completed-evidence admission rejected: HTTP {reply.status_code}: {str(detail)[:2048]}"
                )
            if (
                not isinstance(value, dict)
                or value.get("accepted") is not True
                or value.get("session_id") != self.lease["session_id"]
                or value.get("episode_id") != self.lease["episode_id"]
                or type(value.get("source_step")) is not int
                or value["source_step"] != frame.source
                or type(value.get("duplicate")) is not bool
            ):
                raise RuntimeError(
                    "completed-evidence admission rejected or identity mismatch: HTTP " + str(reply.status_code)
                )
            with self._changed:
                elapsed = time.monotonic() - started
                self._stats["upload_seconds"] += elapsed
                self._latencies["upload"].add(elapsed * 1000)
            return True
        return False

    def _run(self):
        http = None
        try:
            http = self.session_factory()
            if hasattr(http, "trust_env"):
                http.trust_env = False
            while True:
                with self._changed:
                    self._changed.wait_for(lambda: self._queue or self._stop.is_set())
                    if self._stop.is_set():
                        return
                    frame = self._queue.popleft()
                    self._latencies["queue"].add(max(0.0, time.monotonic() - frame.enqueued_at) * 1000)
                started = time.monotonic()
                encoded = self.encode(frame.image)
                if not isinstance(encoded, str) or not encoded or len(encoded) > 1024**2:
                    raise ValueError("completed-evidence PNG encoder must return nonempty base64 text")
                row = dict(source_step=frame.source, composite_image=encoded, executed_action=frame.action)
                payload = {
                    "session_id": self.lease["session_id"],
                    "episode_id": self.lease["episode_id"],
                    "image_size": frame.image_size,
                    "preprocess_profile": frame.profile,
                    **row,
                }
                raw = json_bytes({"lease": self.lease, "payload": payload})
                with self._changed:
                    frame.image = None
                    self._stats["png_base64_bytes"] += len(encoded)
                    elapsed = time.monotonic() - started
                    self._stats["png_seconds"] += elapsed
                    self._latencies["png"].add(elapsed * 1000)
                if self._stop.is_set() or not self._upload(http, frame, raw):
                    return
                with self._changed:
                    frame.row, frame.admitted = row, True
                    self._stats["uploaded"] += 1
                    self._changed.notify_all()
        except BaseException as error:
            with self._changed:
                self._error = error
                self._stop.set()
                self._queue.clear()
                self._changed.notify_all()
        finally:
            if http is not None and hasattr(http, "close"):
                try:
                    http.close()
                except Exception as error:
                    with self._changed:
                        if self._error is None:
                            self._error = error
            with self._changed:
                self._changed.notify_all()

    def wait(self, frontier):
        with self._changed:
            self._check()
            if type(frontier) is not int or not self._ack <= frontier <= self._next:
                raise ValueError("completed-evidence frontier outside owned ledger")
            while True:
                self._check()
                if all(self._frames[step].admitted for step in range(self._ack, frontier)):
                    break
                self._changed.wait(min(0.05, max(0, self.deadline - time.monotonic())))
            self._check()
            return [copy.deepcopy(self._frames[step].row) for step in range(self._ack, frontier)]

    def acknowledge(self, frontier):
        with self._changed:
            self._check()
            if type(frontier) is not int or not self._ack <= frontier <= self._next:
                raise ValueError("prediction ACK frontier outside completed ledger")
            if any(not self._frames[step].admitted for step in range(self._ack, frontier)):
                raise ValueError("prediction ACK precedes completed-evidence admission")
            for step in range(self._ack, frontier):
                del self._frames[step]
            self._ack = frontier

    def close(self):
        with self._changed:
            self._closed = True
            self._stop.set()
            self._queue.clear()
            self._frames.clear()
            self._changed.notify_all()
            thread = self._thread
        # Cleanup has its own bounded grace, as Session End already does. No
        # inference/upload is revived after the Episode deadline or stop fence.
        deadline = time.monotonic() + self.shutdown_timeout_s
        if thread is not None:
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise TimeoutError("completed-evidence worker did not quiesce before End")
        if self._error is not None:
            raise RuntimeError("completed-evidence worker failed: " + str(self._error)) from self._error

    def stats(self):
        with self._changed:
            return {
                **self._stats,
                "pending": len(self._frames),
                "queued": len(self._queue),
                "closed": self._closed,
                "worker_alive": self._thread is not None and self._thread.is_alive(),
                "latencies": {name: window.summary() for name, window in self._latencies.items()},
            }

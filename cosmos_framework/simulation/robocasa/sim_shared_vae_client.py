"""Opt-in C evidence client: successful Sim steps, scoped Owner relay, B1 VAE.

The controlled Registry Actor shares the Run Owner's trust domain. A callback
does not cryptographically attest arbitrary same-UID simulator code. This module
never receives a global Owner token, Worker control capability, or signing key.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from cosmos_framework.inference.robocasa_feature_evidence import (
    FEATURE_EVIDENCE_FORMAT,
    FEATURE_EVIDENCE_VERSION,
    validate_boundary_record,
    validate_post_step_receipt,
)
from cosmos_framework.inference.robocasa_vae_client import LocalVAEClient, _Client
from cosmos_framework.inference.robocasa_vae_protocol import (
    Frame,
    canonical_json,
    source_digest,
    validate_identity,
    validate_lease,
    validate_raw15,
)
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


class ScopedActorCompletionChannel(_Client):
    """Session-scoped Run relay; no Worker owner methods or credentials."""

    def __init__(self, relay_socket, actor_capability, lease):
        if type(relay_socket) is not str or not relay_socket.endswith(".ctl"):
            raise ValueError("private scoped Actor control socket required")
        if type(actor_capability) is not str or not 16 <= len(actor_capability) <= 256:
            raise ValueError("scoped Actor capability required")
        super().__init__(relay_socket[:-4])
        self.lease = validate_lease(lease)
        self._actor_capability = actor_capability

    def _actor(self, op, fields, deadline_ns):
        return self._call(op, {"actor_capability": self._actor_capability, "lease": self.lease, **fields}, deadline_ns)

    def chunk(self, chunk_id, request_key, payload_sha256, source_start, count, deadline_ns):
        return self._actor(
            "chunk",
            dict(
                chunk_id=chunk_id,
                request_key=request_key,
                payload_sha256=payload_sha256,
                source_start=source_start,
                count=count,
            ),
            deadline_ns,
        )

    def post_step(self, chunk_id, request_key, source_step, source_rgb_sha256, raw15, step_completed_ns, deadline_ns):
        return self._actor(
            "post_step",
            dict(
                chunk_id=chunk_id,
                request_key=request_key,
                source_step=source_step,
                source_rgb_sha256=source_rgb_sha256,
                raw15_f32le_b64=base64.b64encode(raw15).decode("ascii"),
                step_completed_ns=str(step_completed_ns),
            ),
            deadline_ns,
        )

    def end(self, deadline_ns):
        return self._actor("end", {}, deadline_ns)


@dataclass(frozen=True)
class PreparedStep:
    source_step: int
    chunk_id: str
    request_key: str
    rgb: bytes
    raw15: bytes
    source_rgb_sha256: str
    raw15_sha256: str


@dataclass
class _Evidence:
    source_step: int
    post_step_receipt: dict
    frame: Frame | None
    feature: dict | None = None
    job_key: str | None = None
    ready: bool = False


class SimSharedVAEClient(RoboCasaLocalMemoryClient):
    """Bounded private evidence spool; only policy ACK releases receipts."""

    uses_shared_vae = True

    def __init__(
        self,
        lease,
        descriptor,
        episode_deadline_ns,
        predict_timeout_s,
        policy_clock_domain,
        *,
        worker_client=None,
        completion_channel=None,
        end_timeout_s=2.0,
    ):
        super().__init__(enabled=True, max_evidence_steps=16)
        self.lease = validate_lease({k: v for k, v in lease.items() if k != "capability"})
        if descriptor.get("trust_model") != "registry_same_trust_domain":
            raise ValueError("explicit controlled Actor trust model required")
        if type(episode_deadline_ns) is not int or episode_deadline_ns <= 0 or not 0 < predict_timeout_s <= 6000:
            raise ValueError("bounded original Episode and Replan budgets required")
        self.identity = validate_identity(descriptor["identity"])
        self.clock_domain, self.policy_clock_domain = descriptor["clock_domain"], policy_clock_domain
        self.actor_id, self.actor_epoch = descriptor["actor_id"], descriptor["actor_epoch"]
        self.worker_id, self.worker_epoch, self.key_id = (
            descriptor["worker_id"],
            descriptor["worker_epoch"],
            descriptor["key_id"],
        )
        self.worker = worker_client or LocalVAEClient(
            descriptor["worker_socket"], self.lease["session_id"], descriptor["worker_session_capability"]
        )
        self.channel = completion_channel or ScopedActorCompletionChannel(
            descriptor["relay_socket"], descriptor["actor_capability"], self.lease
        )
        if not isinstance(self.worker, LocalVAEClient) or not isinstance(self.channel, ScopedActorCompletionChannel):
            raise ValueError("typed Session Worker client and scoped Actor completion channel required")
        self.episode_deadline_ns, self.predict_timeout_ns = episode_deadline_ns, int(predict_timeout_s * 1e9)
        self.end_timeout_ns = int(end_timeout_s * 1e9)
        if self.end_timeout_ns <= 0:
            raise ValueError("bounded evidence cleanup budget required")
        self._changed = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._queue, self._rows = deque(), {}
        self._thread, self._error, self._prepared = None, None, None
        self._budget, self._frozen, self._boundary, self._pending_boundary = None, None, None, None
        self._begun, self._closed = False, False
        self._ack_frontier = 0
        self._session_end_done, self._end_session_id, self._end_failure = False, None, None
        self._trace = deque(maxlen=256)
        self._profile_trace = None

    def attach_trace(self, recorder):
        self._profile_trace = recorder

    def _event(self, name, **fields):
        now = time.monotonic_ns()
        self._trace.append({"event": name, "at_ns": now, **fields})
        if self._profile_trace is not None:
            self._profile_trace.record(name, now, now, kind="event", **fields)

    def _profile_span(self, name, **fields):
        from contextlib import nullcontext

        return nullcontext() if self._profile_trace is None else self._profile_trace.span(name, **fields)

    def _collect_worker_trace(self, deadline_ns):
        if self._profile_trace is None:
            return
        try:
            with self._profile_span("worker_trace_ipc", producer="sim", blocking="foreground"):
                info = self.worker.info(deadline_ns)
            self._profile_trace.merge("worker", info, clock_domain=self.clock_domain)
        except Exception as error:
            self._profile_trace.mark_incomplete("worker_trace_unavailable")
            self._profile_trace.event("worker_trace_error", error_type=type(error).__name__)

    def _check(self, deadline_ns=None):
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("shared VAE evidence Session is closed")
        if time.monotonic_ns() >= min(self.episode_deadline_ns, deadline_ns or self.episode_deadline_ns):
            raise TimeoutError("original shared VAE Episode/Replan deadline exhausted")

    def begin(self, slot=0, **kwargs):
        if slot != 0 or self._begun:
            raise ValueError("one admitted shared VAE assignment per Sim required")
        super().begin(slot, **kwargs)
        episode = self._get(slot)
        episode.session_id, episode.episode_id = self.lease["session_id"], self.lease["episode_id"]
        self._begun = True

    def start_replan(self, slot=0, started_ns=None):
        with self._changed:
            self._get(slot)
            self._check()
            if self._budget is None:
                started_ns = time.monotonic_ns() if started_ns is None else started_ns
                raw_deadline = min(self.episode_deadline_ns, started_ns + self.predict_timeout_ns)
                deadline_s = raw_deadline / 1e9
                if int(deadline_s * 1e9) > raw_deadline:
                    deadline_s = math.nextafter(deadline_s, -math.inf)
                self._budget = {
                    "started_ns": started_ns,
                    "deadline_ns": int(deadline_s * 1e9),
                    "started_at": started_ns / 1e9,
                    "deadline": deadline_s,
                }
                self._event(
                    "REPLAN_BUDGET",
                    started_ns=started_ns,
                    raw_deadline_ns=raw_deadline,
                    deadline_ns=self._budget["deadline_ns"],
                )
            self._check(self._budget["deadline_ns"])

    @property
    def prediction_budget(self):
        with self._changed:
            if self._budget is None:
                raise RuntimeError("freeze Replan budget before collecting evidence")
            return {
                "clock_domain": self.policy_clock_domain,
                "started_at": self._budget["started_at"],
                "deadline": self._budget["deadline"],
            }

    def _rpc_deadline(self):
        original = self.episode_deadline_ns if self._budget is None else self._budget["deadline_ns"]
        return min(original, time.monotonic_ns() + 250_000_000)

    def accept_policy_result(self, result):
        with self._changed:
            self._check(self._budget["deadline_ns"])
            boundary = validate_boundary_record(result["local_memory"]["committed_boundary"])
            header = boundary["header"]
            if (
                header["lease"] != self.lease
                or header["source_step"] != self._get(0).consumer_step
                or header["policy_request_key"] != result["request_key"]
                or header["issued_chunk_id"] != result["request_key"]
                or header["clock_domain"] != self.clock_domain
                or int(header["deadline_ns"]) != self._budget["deadline_ns"]
                or any(header[name] != value for name, value in self.identity.items())
            ):
                raise ValueError("committed Policy chunk differs from assigned Sim frontier")
            self.channel.chunk(
                header["issued_chunk_id"],
                header["policy_request_key"],
                result["payload_sha256"],
                header["source_step"],
                16,
                self._rpc_deadline(),
            )
            self._pending_boundary = boundary

    def prepare_step(self, slot, source_step, composite_image, executed_raw15):
        with self._changed:
            self._check()
            episode = self._get(slot)
            if self._frozen is not None or self._prepared is not None or self._boundary is None:
                raise RuntimeError("no committed action chunk available for a new Sim step")
            if type(source_step) is not int or source_step != episode.consumer_step or len(self._rows) >= 16:
                raise ValueError("completed evidence source frontier or capacity mismatch")
            header = self._boundary["header"]
            if not header["source_step"] <= source_step < header["source_step"] + 16:
                raise ValueError("Sim step lies outside committed action chunk")
            rgb = self._image(composite_image, "pre-action composite").copy().tobytes()
            raw15 = validate_raw15(np.asarray(executed_raw15, dtype="<f4").reshape(-1).tobytes())
            snapshot = PreparedStep(
                source_step,
                header["issued_chunk_id"],
                header["policy_request_key"],
                rgb,
                raw15,
                source_digest(rgb),
                hashlib.sha256(raw15).hexdigest(),
            )
            if source_step % 16 == 0 and snapshot.source_rgb_sha256 != header["source_rgb_sha256"]:
                raise ValueError("pre-action boundary RGB differs from committed current Policy image")
            self._prepared = snapshot
            return snapshot

    def abort_step(self, slot, snapshot, error):
        with self._changed:
            if self._prepared is snapshot:
                self._prepared = None
            if self._error is None:
                self._error = error
            self._stop.set()
            self._changed.notify_all()

    def complete_step(self, slot, snapshot, step_completed_ns):
        with self._changed:
            self._check()
            episode = self._get(slot)
            if self._prepared is not snapshot or snapshot.source_step != episode.consumer_step:
                raise ValueError("exact prepared successful Sim step required")
            if snapshot.source_step % 16 == 15:
                self.start_replan(slot, step_completed_ns)
            try:
                with self._profile_span("owner_step_completion", source_step=snapshot.source_step):
                    receipt = validate_post_step_receipt(
                        self.channel.post_step(
                            snapshot.chunk_id,
                            snapshot.request_key,
                            snapshot.source_step,
                            snapshot.source_rgb_sha256,
                            snapshot.raw15,
                            step_completed_ns,
                            self._rpc_deadline(),
                        )
                    )
                if (
                    receipt["lease"] != self.lease
                    or receipt["source_step"] != snapshot.source_step
                    or receipt["issued_chunk_id"] != snapshot.chunk_id
                    or receipt["policy_request_key"] != snapshot.request_key
                    or receipt["source_rgb_sha256"] != snapshot.source_rgb_sha256
                    or receipt["raw15_sha256"] != snapshot.raw15_sha256
                    or receipt["actor_id"] != self.actor_id
                    or receipt["actor_epoch"] != self.actor_epoch
                    or receipt["step_completed_ns"] != str(step_completed_ns)
                    or receipt["episode_deadline_ns"] != str(self.episode_deadline_ns)
                    or receipt["clock_domain"] != self.clock_domain
                ):
                    raise ValueError("Owner completion receipt differs from actual successful Sim step")
                frame = (
                    None
                    if snapshot.source_step % 16 == 0
                    else Frame.create(
                        self.lease,
                        snapshot.chunk_id,
                        snapshot.request_key,
                        snapshot.source_step,
                        snapshot.rgb,
                        snapshot.raw15,
                        self.identity,
                        self.episode_deadline_ns,
                        self.episode_deadline_ns,
                        self.clock_domain,
                        replan_deadline_ns=None if self._budget is None else self._budget["deadline_ns"],
                    )
                )
                row = _Evidence(snapshot.source_step, receipt, frame, ready=frame is None)
                self._rows[snapshot.source_step] = row
                if frame is not None:
                    self._queue.append(row)
                    if self._thread is None:
                        self._thread = threading.Thread(target=self._loop, name="sim-shared-vae-features", daemon=True)
                        self._thread.start()
                episode.consumer_step += 1
                self._prepared = None
                self._event("STEP_COMPLETED", source_step=snapshot.source_step, worker_frame=frame is not None)
                self._changed.notify_all()
            except BaseException as error:
                if self._error is None:
                    self._error = error
                self._stop.set()
                self._changed.notify_all()
                raise

    def record_completed(self, *args, **kwargs):
        raise RuntimeError("shared VAE requires the prepared actual env.step success callback")

    def _loop(self):
        try:
            while True:
                with self._changed:
                    self._changed.wait_for(lambda: self._queue or self._stop.is_set())
                    if self._stop.is_set():
                        return
                    row = self._queue.popleft()
                    frame = row.frame
                with self._profile_span("worker_submit_ipc", source_step=row.source_step, blocking="background"):
                    key = self.worker.submit(frame, self._rpc_deadline())
                while not self._stop.is_set():
                    self._check(self._budget["deadline_ns"] if self._budget else None)
                    with self._profile_span("worker_poll_ipc", source_step=row.source_step, blocking="background"):
                        result = self.worker.poll(key, self._rpc_deadline())
                    if result.get("status") == "ready":
                        feature = result["feature"]
                        header = feature["header"]
                        if (
                            result.get("job_key") != key
                            or header["lease"] != self.lease
                            or header["source_step"] != row.source_step
                            or header["worker_id"] != self.worker_id
                            or header["worker_epoch"] != self.worker_epoch
                            or header["key_id"] != self.key_id
                            or header["source_rgb_sha256"] != frame.header["source_rgb_sha256"]
                            or header["raw15_sha256"] != frame.header["raw15_sha256"]
                        ):
                            raise ValueError("Worker feature is not bound to actual completed evidence")
                        with self._changed:
                            row.job_key, row.feature, row.frame, row.ready = key, copy.deepcopy(feature), None, True
                            self._changed.notify_all()
                        break
                    if result.get("status") != "pending":
                        raise RuntimeError(
                            "shared VAE Worker failed completed evidence: " + str(result.get("error", {}))
                        )
                    self._stop.wait(0.002)
        except BaseException as error:
            with self._changed:
                if self._error is None and not self._closed:
                    self._error = error
                self._stop.set()
                self._changed.notify_all()

    def payload(self, slot=0):
        with self._changed:
            self.start_replan(slot)
            episode = self._get(slot)
            self._frozen = episode.consumer_step
            while not all(row.ready for row in self._rows.values()):
                self._check(self._budget["deadline_ns"])
                self._changed.wait(min(0.02, max(0, (self._budget["deadline_ns"] - time.monotonic_ns()) / 1e9)))
            self._check(self._budget["deadline_ns"])
            self._collect_worker_trace(self._rpc_deadline())
            self._check(self._budget["deadline_ns"])
            expected = list(range(self._ack_frontier, self._frozen))
            if len(expected) > 16 or sorted(self._rows) != expected:
                raise ValueError("shared VAE history must contain each actual completed source step")
            payload = super().payload(slot)
            payload.update(
                evidence_version=FEATURE_EVIDENCE_VERSION,
                evidence_format=FEATURE_EVIDENCE_FORMAT,
                evidence=[
                    {
                        "source_step": step,
                        "post_step_receipt": copy.deepcopy(self._rows[step].post_step_receipt),
                        "feature": copy.deepcopy(self._rows[step].feature),
                    }
                    for step in expected
                ],
            )
            return payload

    def acknowledge(self, slot, status):
        with self._changed:
            episode = self._get(slot)
            if (
                not isinstance(status, dict)
                or status.get("session_id") != episode.session_id
                or status.get("episode_id") != episode.episode_id
                or type(status.get("consumer_step")) is not int
                or status["consumer_step"] != self._frozen
                or self._pending_boundary is None
            ):
                raise ValueError("exact shared VAE policy ACK and committed boundary required")
            if canonical_json(validate_boundary_record(status["committed_boundary"])) != canonical_json(
                self._pending_boundary
            ):
                raise ValueError("policy ACK changed committed boundary")
            for row in self._rows.values():
                if (
                    row.job_key is not None
                    and self.worker.ack(row.job_key, self._rpc_deadline()).get("acknowledged") is not True
                ):
                    raise RuntimeError("Worker receipt ACK was not acknowledged")
            super().acknowledge(slot, status)
            self._rows.clear()
            self._ack_frontier = self._frozen
            self._boundary, self._pending_boundary = self._pending_boundary, None
            self._budget, self._frozen = None, None

    def end(self, slot=0):
        with self._changed:
            if self._session_end_done:
                if self._end_failure is not None:
                    raise self._end_failure
                return self._end_session_id
            self._closed = True
            self._stop.set()
            self._prepared = None
            self._queue.clear()
            self._changed.notify_all()
            thread, first_error = self._thread, self._error
        # Cleanup is separately bounded and cannot revive Frame/Replan work.
        deadline = time.monotonic_ns() + self.end_timeout_ns
        cleanup_error = None
        try:
            if thread is not None:
                thread.join(max(0, (deadline - time.monotonic_ns()) / 1e9))
                if thread.is_alive():
                    raise TimeoutError("shared VAE submission thread has not drained")
            receipt = self.channel.end(deadline)
            if receipt.get("acknowledged") is not True or receipt.get("quiescent") is not True:
                raise RuntimeError("shared VAE Session drain is unproven")
            self._session_end_done = True
            self._collect_worker_trace(min(deadline, time.monotonic_ns() + 250_000_000))
            self._end_session_id = super().end(slot)
            if receipt.get("error") is not None:
                raise RuntimeError("shared VAE Session ended after Worker failure: " + str(receipt["error"]))
        except BaseException as error:
            cleanup_error = error
        finally:
            with self._changed:
                self._rows.clear()
        if first_error is not None:
            self._end_failure = first_error
            if cleanup_error is not None:
                first_error.cleanup_error = repr(cleanup_error)
                raise first_error from cleanup_error
            raise first_error
        if cleanup_error is not None:
            self._end_failure = cleanup_error
            raise cleanup_error
        return self._end_session_id

    def diagnostics(self):
        with self._changed:
            return {
                "profile": "sim_shared_vae",
                "pending": len(self._rows),
                "queued": len(self._queue),
                "worker_alive": self._thread is not None and self._thread.is_alive(),
                "trace": list(self._trace),
            }

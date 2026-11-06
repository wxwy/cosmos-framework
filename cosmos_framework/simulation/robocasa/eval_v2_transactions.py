"""Phase B facade for the unchanged native action/evidence loop.

Only the coordinator can register. A Sim binds that admitted assignment once;
HTTP retry is internal to one prediction budget, never an outer Episode retry.
Optional media has a bounded queue and a delivery fence before sink shutdown.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from urllib.parse import urlsplit

import numpy as np
import requests

from cosmos_framework.inference.local_memory_pipeline_metrics import DurationWindow
from cosmos_framework.inference.robocasa_local_memory_contract import EVIDENCE_ACTION_DIM
from cosmos_framework.inference.robocasa_session_transactions import digest, json_bytes
from cosmos_framework.inference.robocasa_transaction_client import TransactionPolicyClient
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient
from cosmos_framework.simulation.robocasa.local_memory_pipeline_client import CompletedEvidencePipeline


class AssignedLocalMemoryClient(RoboCasaLocalMemoryClient):
    def __init__(
        self,
        lease,
        *,
        sender=None,
        session_factory=requests.Session,
        encode=None,
        episode_deadline=None,
        request_timeout_s=1.0,
        shutdown_timeout_s=2.0,
    ):
        super().__init__(enabled=True, max_evidence_steps=16)
        self.lease, self.begun = dict(lease), False
        self._pipeline, self._pipeline_decided, self._frozen = None, False, None
        self._pipeline_closing = False
        self._pipeline_lock = threading.RLock()
        self._step_profile = None
        self._pipeline_options = dict(
            sender=sender,
            session_factory=session_factory,
            episode_deadline=episode_deadline,
            request_timeout_s=request_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
        )
        if encode is not None:
            self._pipeline_options["encode"] = encode

    def begin(self, slot=0, **kwargs):
        if slot != 0 or self.begun:
            raise ValueError("one admitted assignment per Sim Episode required")
        super().begin(slot, **kwargs)
        episode = self._get(slot)
        episode.session_id, episode.episode_id = self.lease["session_id"], self.lease["episode_id"]
        self.begun = True

    def record_completed(self, slot, composite_image, executed_action15):
        started = time.monotonic()
        with self._pipeline_lock:
            if self._pipeline_closing:
                raise RuntimeError("completed-evidence pipeline is closing")
            if self._pipeline is None:
                return super().record_completed(slot, composite_image, executed_action15)
            if self._frozen is not None:
                raise RuntimeError("completed evidence cannot advance during a frozen prediction")
            # Reuse the established RGB/raw15 validation, without synchronous PNG.
            composite = self._image(composite_image, "composite_image")
            action = np.asarray(executed_action15, dtype=np.float32).reshape(-1)
            if action.shape != (EVIDENCE_ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("executed Local-TTT evidence action must be finite raw15")
            episode = self._get(slot)
            self._pipeline.submit(
                episode.consumer_step,
                composite,
                action.tolist(),
                image_size=episode.image_size,
                preprocess_profile=episode.preprocess_profile,
            )
            episode.consumer_step += 1
            self._step_profile["enqueue_cpu"].add((time.monotonic() - started) * 1000)

    def payload(self, slot=0):
        with self._pipeline_lock:
            if self._pipeline_closing:
                raise RuntimeError("completed-evidence pipeline is closing")
            if self._pipeline is None:
                return super().payload(slot)
            episode = self._get(slot)
            self._frozen = episode.consumer_step
            frontier = self._frozen
        rows = self._pipeline.wait(frontier)
        with self._pipeline_lock:
            if self._pipeline_closing:
                raise RuntimeError("completed-evidence pipeline is closing")
            episode.evidence[:] = rows
            return super().payload(slot)

    def acknowledge(self, slot, status):
        with self._pipeline_lock:
            if self._pipeline_closing:
                raise RuntimeError("completed-evidence pipeline is closing")
            episode = self._get(slot)
            if (
                not isinstance(status, dict)
                or status.get("session_id") != episode.session_id
                or status.get("episode_id") != episode.episode_id
                or type(status.get("consumer_step")) is not int
                or status.get("consumer_step") != episode.consumer_step
            ):
                raise ValueError("Local-TTT response does not acknowledge the exact completed-evidence frontier")
            flag = status.get("pipeline")
            if self._pipeline is not None and flag != "per_frame_v1":
                raise ValueError("enabled completed-evidence pipeline capability changed")
            if not self._pipeline_decided:
                if flag not in (None, "per_frame_v1"):
                    raise ValueError("unknown completed-evidence pipeline capability")
                if flag == "per_frame_v1":
                    if (
                        episode.consumer_step != 0
                        or not episode.first_request
                        or self._pipeline_options["sender"] is None
                    ):
                        raise ValueError("pipeline requires initial policy ACK and configured transport")
                    self._pipeline = CompletedEvidencePipeline(self.lease, **self._pipeline_options)
                    self._step_profile = {
                        name: DurationWindow() for name in ("composite_cpu", "sim_step_cpu", "enqueue_cpu")
                    }
                self._pipeline_decided = True
            elif self._pipeline is None and flag is not None:
                raise ValueError("completed-evidence capability changed after initial policy ACK")
            if self._pipeline is not None:
                self._pipeline.acknowledge(episode.consumer_step)
            super().acknowledge(slot, status)
            self._frozen = None

    def close_pipeline(self):
        with self._pipeline_lock:
            pipeline = self._pipeline
            if pipeline is not None:
                self._pipeline_closing = True
        if pipeline is not None:
            pipeline.close()

    def end(self, slot=0):
        self.close_pipeline()
        return super().end(slot)

    def pipeline_stats(self):
        with self._pipeline_lock:
            return (
                {"enabled": False}
                if self._pipeline is None
                else {
                    "enabled": True,
                    **self._pipeline.stats(),
                    "sim_latencies": {name: window.summary() for name, window in self._step_profile.items()},
                }
            )

    @property
    def streaming_enabled(self):
        return self._pipeline is not None

    def record_step_timing(self, composite_seconds, sim_seconds):
        with self._pipeline_lock:
            if self._step_profile is not None:
                self._step_profile["composite_cpu"].add(composite_seconds * 1000)
                self._step_profile["sim_step_cpu"].add(sim_seconds * 1000)


class _Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class PolicyTransport:
    def __init__(
        self,
        settings,
        assignment,
        episode_deadline,
        *,
        transport=None,
        media_transport=None,
        completed_sender=None,
        completed_session_factory=requests.Session,
        completed_encode=None,
    ):
        if not isinstance(episode_deadline, (int, float)) or not math.isfinite(episode_deadline):
            raise ValueError("bounded coordinator Episode deadline required")
        self.settings, self.episode_deadline = settings, episode_deadline
        self.base = settings["server_url"].rstrip("/")
        self.http = transport if transport is not None else requests.Session()
        self.media_http = media_transport if media_transport is not None else requests.Session()
        if hasattr(self.http, "trust_env"):
            self.http.trust_env = False
        if hasattr(self.media_http, "trust_env"):
            self.media_http.trust_env = False
        self.client = TransactionPolicyClient(
            self.base,
            timeout_s=settings["predict_timeout_s"],
            transport=self.http,
            batch_profile=True,
            max_batch_size=settings["max_infer_batch_size"],
            local_memory_mode=settings.get("local_memory_mode", "required"),
            capacity_profile=settings.get("capacity_profile", "legacy8"),
        )
        self.lease = self.client.bind_assignment(assignment)
        self.local = (
            AssignedLocalMemoryClient(
                self.lease,
                sender=completed_sender or self._send_completed,
                session_factory=completed_session_factory,
                encode=completed_encode,
                episode_deadline=episode_deadline,
                request_timeout_s=min(1.0, settings["predict_timeout_s"]),
                shutdown_timeout_s=settings.get("encoder_grace_s", 2.0),
            )
            if self.client.mode == "required"
            else None
        )
        self._failed, self._ended, self.replan = False, False, 0
        self._last_delivered = -1
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._delivery = threading.Lock()
        self._pending = deque()
        self._thread = None
        self.errors = []
        self.media_status = {}
        self.strict_media = settings.get("save_generated_video") == "all"
        self.media_timeout_s = settings.get("media_drain_timeout_s", 300.0)
        if type(self.media_timeout_s) not in (int, float) or not 0 < self.media_timeout_s <= 3600:
            raise ValueError("bounded media drain timeout required")
        self._media_failed = False
        self._active_media = 0

    def _send_completed(self, http, raw, timeout):
        return http.request(
            "POST",
            self.base + "/v2/local-memory/completed",
            data=raw,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    def _error(self, error):
        if len(self.errors) < 16:
            self.errors.append(str(error)[:1024])

    def post(self, url, *, json, timeout, **kwargs):
        if urlsplit(url).path != "/predict" or url.rstrip("/") != self.base + "/predict":
            raise ValueError("native facade cannot redirect assignment to another Server")
        if json.get("_local_memory_command") == "reset":
            if json.get("session_id") != self.lease["session_id"]:
                raise ValueError("foreign Session reset")
            self.end()
            return _Response({"local_memory": {"status": "reset", "session_id": self.lease["session_id"]}})
        if self._failed or self._ended:
            raise RuntimeError("failed/closed Episode requires explicit End and new Attempt")
        if self.local is not None and self.local._pipeline_closing:
            self._failed = True
            raise RuntimeError("completed-evidence pipeline is closing")
        if self.client.mode == "off":
            if "local_memory" in json or "inference" in json:
                raise ValueError("No-TTT native payload cannot supply Local or transaction ownership")
            frontier = self.replan
        else:
            frontier = json.get("local_memory", {}).get("consumer_step")
        if type(frontier) is not int or frontier <= self._last_delivered:
            self._failed = True
            raise ValueError("native Episode cannot execute a delivered frontier twice")
        remaining = self.episode_deadline - time.monotonic()
        if remaining <= 0:
            self._failed = True
            raise TimeoutError("original Episode deadline exhausted")
        self.client.timeout_s = min(timeout, self.settings["predict_timeout_s"], remaining)
        record = dict(json)
        if self.client.mode == "off":
            record["inference"] = {
                "session_id": self.lease["session_id"],
                "episode_id": self.lease["episode_id"],
                "request_index": frontier,
            }
        mode = self.settings.get("save_generated_video", "none")
        if mode == "all" or (mode == "first" and self.replan == 0):
            record["media"] = {"replan": self.replan}
        if self.strict_media:
            record["media"]["required"] = True
            if not self._media_failed:
                try:
                    self._reserve_media(record, frontier)
                except Exception as error:
                    self._media_failed = True
                    self._error("MEDIA_INCOMPLETE: reservation: " + repr(error))
            if self._media_failed:
                # No prediction has been submitted for this frontier yet. Stop
                # diagnostic requests, not native actions or the Local session.
                record.pop("media", None)
                self.media_status[self.replan] = {
                    "status": "MEDIA_INCOMPLETE",
                    "error": "media reservation/prior diagnostic failed",
                }
            remaining = self.episode_deadline - time.monotonic()
            if remaining <= 0:
                self._failed = True
                raise TimeoutError("original Episode deadline exhausted")
            self.client.timeout_s = min(timeout, self.settings["predict_timeout_s"], remaining)
        try:
            result = self.client.predict(record)
            self._last_delivered = frontier
            self.replan += 1
            return _Response(result)
        except BaseException:
            self._failed = True
            raise

    def media(self, result, index, deliver):
        if self.strict_media:
            with self._changed:
                self._active_media += 1
            try:
                return self._strict_media(result, index, deliver)
            finally:
                with self._changed:
                    self._active_media -= 1
                    self._changed.notify_all()
        media = result.get("media", {})
        if media.get("status") != "disabled" and len(self.media_status) < 6250:
            with self._changed:
                self.media_status[index] = {
                    "status": media.get("status", "unavailable"),
                    "ticket": result.get("request_key"),
                }
        if media.get("status") != "pending":
            if media.get("status") not in ("disabled", None):
                self._error("generated media " + str(media))
            return
        if media.get("ticket") != result["request_key"]:
            self._error("foreign optional media ticket")
            self.media_status[index] = {"status": "invalid_media_binding"}
            return
        with self._changed:
            if self._stop.is_set() or len(self._pending) >= 2:
                self._error("bounded optional media fetch dropped")
                self.media_status[index] = {"status": "dropped_fetch_capacity"}
                return
            self._pending.append((result["request_key"], result["payload_sha256"], index, deliver))
            if self._thread is None:
                self._thread = threading.Thread(target=self._media_loop, name="eval-v2-optional-media", daemon=True)
                self._thread.start()
            self._changed.notify_all()

    def _reserve_media(self, payload, frontier):
        identity = {k: v for k, v in self.lease.items() if k != "capability"}
        key = digest({"protocol": self.client.protocol, **identity, "frontier": frontier})
        payload_sha = digest(payload)
        deadline = min(self.episode_deadline, time.monotonic() + self.media_timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop.is_set():
                raise TimeoutError("media spool backpressure deadline exhausted")
            try:
                reply = self.media_http.request(
                    "POST",
                    self.base + "/v2/media/reserve",
                    timeout=min(1.0, remaining),
                    json={
                        "lease": self.lease,
                        "request_key": key,
                        "payload_sha256": payload_sha,
                        "replan": self.replan,
                        "frontier": frontier,
                    },
                )
            except (requests.Timeout, requests.ConnectionError):
                self._stop.wait(min(0.02, max(0, deadline - time.monotonic())))
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("late media reservation response")
            if reply.status_code in (429, 503):
                self._stop.wait(min(0.02, max(0, deadline - time.monotonic())))
                continue
            value = reply.json()
            if (
                reply.status_code != 200
                or value.get("reserved") is not True
                or value.get("request_key") != key
                or value.get("payload_sha256") != payload_sha
            ):
                raise ValueError("media reservation rejected or binding mismatch")
            return

    def _strict_media(self, result, index, deliver):
        key, payload_sha = result.get("request_key"), result.get("payload_sha256")
        previous = self.media_status.get(index)
        if previous and previous.get("status") == "VIDEO_SAVED":
            if previous.get("ticket") != key or previous.get("payload_sha256") != payload_sha:
                self._media_failed = True
                self._error("MEDIA_INCOMPLETE: replan identity reused after VIDEO_SAVED")
                self.media_status[index] = {**previous, "status": "MEDIA_INCOMPLETE", "error": "foreign replay"}
            return
        expected = {k: v for k, v in self.lease.items() if k != "capability"}
        binding = {"assignment": expected, "request_key": key, "payload_sha256": payload_sha, "replan": index}
        self.media_status[index] = {"status": "pending", "ticket": key, "payload_sha256": payload_sha}
        try:
            media = result.get("media", {})
            if self._media_failed or media.get("status") != "pending" or media.get("ticket") != key:
                raise RuntimeError("strict generated media unavailable: " + str(media))
            server_deadline = media.get("deadline", time.monotonic() + self.media_timeout_s)
            if type(server_deadline) not in (int, float) or not math.isfinite(server_deadline):
                raise ValueError("bounded strict media deadline required")
            deadline = min(server_deadline, time.monotonic() + self.media_timeout_s)
            headers = {"X-Cosmos-Assignment": json_bytes(self.lease).decode(), "X-Cosmos-Payload-SHA256": payload_sha}
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    raise TimeoutError("strict generated media deadline exhausted")
                try:
                    reply = self.media_http.request(
                        "GET", self.base + "/v2/media/" + key, headers=headers, timeout=min(1.0, remaining)
                    )
                except (requests.Timeout, requests.ConnectionError):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("late strict media response")
                value = reply.json()
                if value.get("request_key") != key or value.get("payload_sha256") != payload_sha:
                    raise ValueError("strict media response binding mismatch")
                if reply.status_code == 202:
                    self._stop.wait(min(0.02, max(0, deadline - time.monotonic())))
                    continue
                status = value.get("media", {})
                if (
                    reply.status_code != 200
                    or status.get("status") != "ready"
                    or status.get("assignment") != expected
                    or status.get("replan") != index
                ):
                    raise RuntimeError("strict media decode failed: " + str(status.get("status")))
                receipt = deliver(
                    index, {"video": status.get("video"), "_media_binding": binding, "_media_deadline": deadline}
                )
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("status") != "VIDEO_SAVED"
                    or receipt.get("index") != index
                    or receipt.get("binding", {}).get("transfer") != binding
                ):
                    raise ValueError("durable Encoder VIDEO_SAVED receipt required before ACK")
                # The file/receipt remain on disk if this network ACK is lost.
                self.media_status[index] = {
                    "status": "VIDEO_SAVED_ACK_PENDING",
                    "ticket": key,
                    "payload_sha256": payload_sha,
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                }
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("durable video server ACK deadline exhausted")
                    try:
                        ack = self.media_http.request(
                            "POST",
                            self.base + "/v2/media/ack",
                            timeout=min(1.0, remaining),
                            json={
                                "lease": self.lease,
                                "request_key": key,
                                "payload_sha256": payload_sha,
                                "receipt": receipt,
                            },
                        )
                    except (requests.Timeout, requests.ConnectionError):
                        continue
                    value = ack.json()
                    if (
                        time.monotonic() >= deadline
                        or ack.status_code != 200
                        or value.get("acknowledged") is not True
                        or value.get("request_key") != key
                    ):
                        raise ValueError("late/invalid durable media server ACK")
                    self.media_status[index]["status"] = "VIDEO_SAVED"
                    return receipt
        except Exception as error:
            self._media_failed = True
            self._error("MEDIA_INCOMPLETE: " + repr(error))
            self.media_status[index] = {
                **self.media_status[index],
                "status": "MEDIA_INCOMPLETE",
                "error": repr(error)[:1024],
            }
            return None

    def _media_loop(self):
        while not self._stop.is_set():
            with self._changed:
                self._changed.wait_for(lambda: self._pending or self._stop.is_set(), timeout=0.2)
                if self._stop.is_set():
                    return
                if not self._pending:
                    continue
                key, sha, index, deliver = self._pending.popleft()
            headers = {"X-Cosmos-Assignment": json_bytes(self.lease).decode(), "X-Cosmos-Payload-SHA256": sha}
            try:
                while not self._stop.is_set() and time.monotonic() < self.episode_deadline:
                    reply = self.media_http.request("GET", self.base + "/v2/media/" + key, headers=headers, timeout=0.5)
                    value = reply.json()
                    if (
                        reply.status_code not in (200, 202)
                        or value.get("request_key") != key
                        or value.get("payload_sha256") != sha
                    ):
                        raise ValueError("optional media response binding mismatch")
                    if reply.status_code == 202:
                        self._stop.wait(0.05)
                        continue
                    status = value.get("media", {})
                    if status.get("status") == "ready":
                        with self._delivery:
                            if not self._stop.is_set():
                                deliver(index, {"video": status.get("video")})
                                with self._changed:
                                    self.media_status[index] = {"status": "fetched_to_optional_sink", "ticket": key}
                        self.media_http.request(
                            "POST",
                            self.base + "/v2/media/ack",
                            timeout=0.5,
                            json={"lease": self.lease, "request_key": key, "payload_sha256": sha},
                        )
                    else:
                        self._error("optional media " + str(status.get("status")))
                        with self._changed:
                            self.media_status[index] = {"status": str(status.get("status")), "ticket": key}
                    break
            except Exception as error:
                self._error("optional media fetch: " + repr(error))
                with self._changed:
                    self.media_status[index] = {"status": "fetch_error", "ticket": key, "error": repr(error)[:1024]}

    def end(self):
        if self._ended:
            return
        pipeline_error = None
        try:
            if self.local is not None:
                self.local.close_pipeline()
        except Exception as error:
            pipeline_error = error
        finally:
            self.close_media()
        if pipeline_error is not None and self.local.pipeline_stats().get("worker_alive"):
            raise pipeline_error
        # Cleanup has its own bounded budget; never revives inference work.
        self.client.timeout_s = self.settings["predict_timeout_s"]
        try:
            self.client.end()
        except Exception as error:
            if pipeline_error is not None:
                pipeline_error.cleanup_error = {"type": type(error).__name__, "message": str(error)[:1024]}
                raise pipeline_error from error
            raise
        self._ended = True
        if pipeline_error is not None:
            raise pipeline_error

    def close_media(self):
        if self.strict_media:
            # The normal native caller is synchronous. A concurrent End still
            # waits for an active delivery, without holding model/actor locks.
            deadline = time.monotonic() + self.media_timeout_s
            with self._changed:
                while self._active_media and time.monotonic() < deadline:
                    self._changed.wait(min(0.05, max(0, deadline - time.monotonic())))
                if self._active_media:
                    self._media_failed = True
                    self._error("MEDIA_INCOMPLETE: active Encoder delivery did not drain")
                self._stop.set()
            return
        with self._delivery:
            self._stop.set()
        with self._changed:
            for index, status in list(self.media_status.items()):
                if status["status"] == "pending":
                    self.media_status[index] = {**status, "status": "cancelled_at_episode_end"}
            self._changed.notify_all()
            self._pending.clear()
        if self._thread is not None:
            self._thread.join(self.settings.get("encoder_grace_s", 2.0))
            if self._thread.is_alive():
                self._error("optional media fetch thread unquiesced")

    def media_diagnostics(self):
        with self._changed:
            value = {
                "predictions": {str(k): dict(v) for k, v in self.media_status.items()},
                "errors": list(self.errors),
                "unquiesced": self.media_process_fault,
            }
            if self.strict_media:
                saved = sum(s["status"] == "VIDEO_SAVED" for s in self.media_status.values())
                value["completion"] = {
                    "status": "COMPLETE" if saved == self.replan and not self._media_failed else "MEDIA_INCOMPLETE",
                    "expected": self.replan,
                    "acknowledged": saved,
                }
            return value

    @property
    def media_process_fault(self):
        return (self.strict_media and self._active_media > 0) or (self._thread is not None and self._thread.is_alive())

    def close(self):
        try:
            self.end()
        finally:
            self.close_media()
            if hasattr(self.http, "close"):
                self.http.close()
            if not self.media_process_fault and hasattr(self.media_http, "close"):
                self.media_http.close()

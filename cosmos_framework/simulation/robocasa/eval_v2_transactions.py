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

import requests

from cosmos_framework.inference.robocasa_session_transactions import digest, json_bytes
from cosmos_framework.inference.robocasa_transaction_client import TransactionPolicyClient
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


class AssignedLocalMemoryClient(RoboCasaLocalMemoryClient):
    def __init__(self, lease):
        super().__init__(enabled=True, max_evidence_steps=16)
        self.lease, self.begun = dict(lease), False

    def begin(self, slot=0, **kwargs):
        if slot != 0 or self.begun:
            raise ValueError("one admitted assignment per Sim Episode required")
        super().begin(slot, **kwargs)
        episode = self._get(slot)
        episode.session_id, episode.episode_id = self.lease["session_id"], self.lease["episode_id"]
        self.begun = True


class _Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class PolicyTransport:
    def __init__(self, settings, assignment, episode_deadline, *, transport=None, media_transport=None):
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
        profile = settings.get("local_evidence_profile", "server_vae")
        if profile not in ("server_vae", "sim_shared_vae"):
            raise ValueError("explicit server_vae/sim_shared_vae evidence profile required")
        if profile == "sim_shared_vae":
            if self.client.mode != "required" or settings["max_infer_batch_size"] != 1:
                raise ValueError("Stage3A shared VAE requires Local mode required and Policy B1")
            from cosmos_framework.simulation.robocasa.sim_shared_vae_client import SimSharedVAEClient

            self.local = SimSharedVAEClient(
                self.lease,
                assignment["sim_shared_vae"],
                int(episode_deadline * 1e9),
                settings["predict_timeout_s"],
                self.client.clock_domain,
                end_timeout_s=settings.get("shared_vae_end_timeout_s", 5.0),
            )
        else:
            if "sim_shared_vae" in assignment:
                raise ValueError("shared VAE credentials require explicit C profile")
            self.local = AssignedLocalMemoryClient(self.lease) if self.client.mode == "required" else None
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
            if getattr(self.local, "uses_shared_vae", False):
                result = self.client.predict(record, budget=self.local.prediction_budget)
                self.local.accept_policy_result(result)
            else:
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
        local_error = None
        try:
            if getattr(self.local, "uses_shared_vae", False):
                self.local.end(0)
        except BaseException as error:
            local_error = error
        finally:
            self.close_media()
        if local_error is not None and not self.local._session_end_done:
            raise local_error
        # Cleanup has its own bounded budget; never revives inference work.
        self.client.timeout_s = self.settings["predict_timeout_s"]
        try:
            self.client.end()
        except BaseException as error:
            if local_error is not None:
                local_error.cleanup_error = repr(error)
                raise local_error from error
            raise
        self._ended = True
        if local_error is not None:
            raise local_error

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

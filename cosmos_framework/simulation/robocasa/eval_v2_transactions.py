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

from cosmos_framework.inference.robocasa_session_transactions import json_bytes
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
        )
        self.lease = self.client.bind_assignment(assignment)
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
        try:
            result = self.client.predict(record)
            self._last_delivered = frontier
            self.replan += 1
            return _Response(result)
        except BaseException:
            self._failed = True
            raise

    def media(self, result, index, deliver):
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
        self.close_media()
        # Cleanup has its own bounded budget; never revives inference work.
        self.client.timeout_s = self.settings["predict_timeout_s"]
        self.client.end()
        self._ended = True

    def close_media(self):
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
            return {
                "predictions": {str(k): dict(v) for k, v in self.media_status.items()},
                "errors": list(self.errors),
                "unquiesced": self.media_process_fault,
            }

    @property
    def media_process_fault(self):
        return self._thread is not None and self._thread.is_alive()

    def close(self):
        try:
            self.end()
        finally:
            self.close_media()
            if hasattr(self.http, "close"):
                self.http.close()
            if not self.media_process_fault and hasattr(self.media_http, "close"):
                self.media_http.close()

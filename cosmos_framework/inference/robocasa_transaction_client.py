"""Bounded client for the opt-in B=1 Session transaction API, not a simulator."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable

import requests

from cosmos_framework.inference.robocasa_session_transactions import PROTOCOL, digest, json_bytes


class TransactionPolicyClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s=600.0,
        transport=None,
        clock=time.monotonic,
        pause: Callable[[float], None] = time.sleep,
    ):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("positive client deadline required")
        self.base_url, self.timeout_s = base_url.rstrip("/"), timeout_s
        self.transport = transport if transport is not None else requests.Session()
        self.clock, self.pause = clock, pause
        self.lease = None
        self._registration_pending = False

    def _call(self, method, path, deadline, **kwargs):
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("original transaction deadline exhausted")
            try:
                response = self.transport.request(method, self.base_url + path, timeout=remaining, **kwargs)
                break
            except (requests.Timeout, requests.ConnectionError):
                self.pause(min(0.02, max(0, deadline - self.clock())))
        if response.status_code not in (200, 202):
            self.last_error = response.json()
            raise RuntimeError(f"transaction API rejected request: HTTP {response.status_code}: {self.last_error}")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("transaction JSON object required")
        return response.status_code, value

    def register(self, registration: dict, owner_token: str):
        if self.lease is not None or self._registration_pending:
            raise RuntimeError("prior Session has not received End ACK")
        self._registration_pending = True
        _, value = self._call(
            "POST",
            "/v2/session/register",
            self.clock() + self.timeout_s,
            json=registration,
            headers={"X-Cosmos-Owner": owner_token},
        )
        lease = value.get("lease")
        if not isinstance(lease, dict) or any(lease.get(k) != v for k, v in registration.items()):
            raise ValueError("registration ACK identity mismatch")
        self.lease = lease
        self._registration_pending = False
        return dict(lease)

    def predict(self, payload: dict):
        if self.lease is None:
            raise RuntimeError("register a fresh admitted Episode first")
        raw = json_bytes(payload)
        # Frozen bytes prevent retry/ACK from binding a later-mutated caller object.
        payload = json.loads(raw)
        frontier = payload["local_memory"]["consumer_step"]
        expected_key = digest(
            {"protocol": PROTOCOL, **{k: v for k, v in self.lease.items() if k != "capability"}, "frontier": frontier}
        )
        expected_sha = hashlib.sha256(raw).hexdigest()
        deadline = self.clock() + self.timeout_s
        status, result = self._call("POST", "/v2/inference", deadline, json={"lease": self.lease, "payload": payload})
        while status == 202:
            if result.get("request_key") != expected_key or result.get("payload_sha256") != expected_sha:
                raise ValueError("inference Ticket binding mismatch")
            self.pause(min(0.02, max(0, deadline - self.clock())))
            status, result = self._call(
                "GET",
                f"/v2/inference/{expected_key}",
                deadline,
                headers={
                    "X-Cosmos-Assignment": json_bytes(self.lease).decode(),
                    "X-Cosmos-Payload-SHA256": expected_sha,
                },
            )
        identity = {k: v for k, v in self.lease.items() if k != "capability"}
        if (
            result.get("protocol") != PROTOCOL
            or result.get("request_key") != expected_key
            or result.get("payload_sha256") != expected_sha
            or result.get("assignment") != identity
            or result.get("frontier") != frontier
        ):
            raise ValueError("committed inference identity mismatch")
        local = result.get("local_memory", {})
        action = result.get("action")
        if (
            local.get("session_id") != self.lease["session_id"]
            or local.get("episode_id") != self.lease["episode_id"]
            or local.get("consumer_step") != frontier
            or result.get("native_batch") != {"size": 1, "index": 0}
            or not isinstance(action, list)
            or len(action) != 16
            or any(not isinstance(row, list) or len(row) != 15 for row in action)
            or any(type(v) not in (float, int) or not math.isfinite(v) for row in action for v in row)
        ):
            raise ValueError("committed action/Local status contract mismatch")
        _, ack = self._call(
            "POST",
            "/v2/inference/ack",
            deadline,
            json={"lease": self.lease, "request_key": expected_key, "payload_sha256": expected_sha},
        )
        if ack.get("acknowledged") is not True or ack.get("request_key") != expected_key:
            raise ValueError("delivery ACK mismatch")
        return result

    def end(self):
        if self.lease is None:
            if self._registration_pending:
                raise RuntimeError("registration ownership unresolved; owned Server recovery required")
            return
        deadline = self.clock() + self.timeout_s
        while True:
            status, reply = self._call("POST", "/v2/session/end", deadline, json={"lease": self.lease})
            if reply.get("acknowledged") is True:
                if status != 200 or any(reply.get(k) != v for k, v in self.lease.items() if k != "capability"):
                    raise ValueError("Session End ACK identity mismatch")
                self.lease = None
                return
            if status != 202 or reply.get("acknowledged") is not False:
                raise ValueError("invalid deferred Session End reply")
            self.pause(min(0.02, max(0, deadline - self.clock())))

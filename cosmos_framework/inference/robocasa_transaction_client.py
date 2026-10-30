"""Bounded client for the opt-in B=1 Session transaction API, not a simulator."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable

import requests

from cosmos_framework.inference.robocasa_session_transactions import (
    digest,
    json_bytes,
    monotonic_clock_domain,
    protocol_for_mode,
    request_frontier,
)


class TransactionPolicyClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s=600.0,
        transport=None,
        batch_profile=False,
        max_batch_size=1,
        local_memory_mode="required",
        clock=time.monotonic,
        pause: Callable[[float], None] = time.sleep,
    ):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("positive client deadline required")
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= 8:
            raise ValueError("bounded native batch cap required")
        self.batch_profile, self.max_batch_size = batch_profile, max_batch_size
        self.mode = local_memory_mode
        self.protocol = protocol_for_mode(self.mode)
        self.base_url, self.timeout_s = base_url.rstrip("/"), timeout_s
        self.transport = transport if transport is not None else requests.Session()
        self.clock, self.pause = clock, pause
        self.clock_domain = monotonic_clock_domain()
        self.lease = None
        self._registration_pending = False

    def _call(self, method, path, deadline, **kwargs):
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("original transaction deadline exhausted")
            try:
                response = self.transport.request(method, self.base_url + path, timeout=remaining, **kwargs)
                if response.status_code in (429, 503):
                    self.pause(min(0.02, max(0, deadline - self.clock())))
                    continue
                break
            except (requests.Timeout, requests.ConnectionError):
                self.pause(min(0.02, max(0, deadline - self.clock())))
        if self.clock() >= deadline:
            raise TimeoutError("original transaction deadline exhausted during transport")
        if response.status_code not in (200, 202):
            self.last_error = response.json()
            raise RuntimeError(f"transaction API rejected request: HTTP {response.status_code}: {self.last_error}")
        value = response.json()
        if self.clock() >= deadline:
            raise TimeoutError("original transaction deadline exhausted during response decoding")
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
        if value.get("prediction_clock_domain") != self.clock_domain:
            raise ValueError("registration prediction clock domain mismatch")
        self._check_mode(value)
        lease = value.get("lease")
        if not isinstance(lease, dict) or any(lease.get(k) != v for k, v in registration.items()):
            raise ValueError("registration ACK identity mismatch")
        self.lease = lease
        self._registration_pending = False
        return dict(lease)

    def bind_assignment(self, ack):
        """Adopt only the coordinator's exact registration ACK, never auto-register."""
        from cosmos_framework.inference.robocasa_session_transactions import SessionLease

        if self.lease is not None or self._registration_pending:
            raise RuntimeError("previous assignment unresolved")
        if not isinstance(ack, dict) or ack.get("prediction_clock_domain") != self.clock_domain:
            raise ValueError("assignment clock domain mismatch")
        self._check_mode(ack)
        lease = ack.get("lease")
        if not isinstance(lease, dict) or set(lease) != set(SessionLease.__dataclass_fields__):
            raise ValueError("exact coordinator assignment required")
        self.lease = json.loads(json_bytes(lease))
        return dict(self.lease)

    def _check_mode(self, value):
        # Existing required v1 ACKs remain readable. The new No-TTT protocol
        # requires an explicit mode/profile, never an automatic fallback.
        if self.mode == "off" and ("local_memory_mode" not in value or "profile" not in value):
            raise ValueError("explicit No-TTT assignment mode/profile required")
        if (value.get("local_memory_mode", "required"), value.get("profile", self.protocol)) != (
            self.mode,
            self.protocol,
        ):
            raise ValueError("assignment Local-TTT mode/profile mismatch")

    def _batch_valid(self, batch):
        if not self.batch_profile:
            return batch == {"size": 1, "index": 0}
        return (
            isinstance(batch, dict)
            and set(batch) == {"id", "size", "index"}
            and isinstance(batch["id"], str)
            and len(batch["id"]) == 32
            and all(c in "0123456789abcdef" for c in batch["id"])
            and type(batch["size"]) is int
            and 1 <= batch["size"] <= self.max_batch_size
            and type(batch["index"]) is int
            and 0 <= batch["index"] < batch["size"]
        )

    def predict(self, payload: dict):
        # Freeze before encoding/upload; retries reuse these exact absolute values.
        started_at = self.clock()
        deadline = started_at + self.timeout_s
        budget = {"clock_domain": self.clock_domain, "started_at": started_at, "deadline": deadline}
        if self.lease is None:
            raise RuntimeError("register a fresh admitted Episode first")
        raw = json_bytes(payload)
        # Frozen bytes prevent retry/ACK from binding a later-mutated caller object.
        payload = json.loads(raw)
        frontier = request_frontier(payload, self.mode)
        expected_key = digest(
            {
                "protocol": self.protocol,
                **{k: v for k, v in self.lease.items() if k != "capability"},
                "frontier": frontier,
            }
        )
        expected_sha = hashlib.sha256(raw).hexdigest()
        status, result = self._call(
            "POST", "/v2/inference", deadline, json={"lease": self.lease, "payload": payload, "budget": budget}
        )
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
            result.get("protocol") != self.protocol
            or result.get("request_key") != expected_key
            or result.get("payload_sha256") != expected_sha
            or result.get("assignment") != identity
            or result.get("frontier") != frontier
        ):
            raise ValueError("committed inference identity mismatch")
        local = result.get("local_memory", {})
        action = result.get("action")
        local_valid = (
            (
                local.get("session_id") == self.lease["session_id"]
                and local.get("episode_id") == self.lease["episode_id"]
                and local.get("consumer_step") == frontier
            )
            if self.mode == "required"
            else ("local_memory" not in result and result.get("local_memory_mode") == "off")
        )
        if (
            not local_valid
            or not self._batch_valid(result.get("native_batch"))
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

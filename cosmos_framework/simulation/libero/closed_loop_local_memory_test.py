"""HTTP-client Local Memory transaction witnesses without launching LIBERO."""

from __future__ import annotations

import numpy as np
import pytest
import requests

from .closed_loop_eval import ActionEnvironmentClient


class Response:
    def __init__(self, body, status_code=200):
        self._body, self.status_code = body, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._body


def _client():
    return ActionEnvironmentClient("http://unit.test", "libero", "task", 16, 1.0)


def test_http_ack_commits_client_frontier_and_failed_request_keeps_evidence(monkeypatch):
    client = _client()
    reset_sessions = []
    fail_predict = False

    def get(url, **kwargs):
        assert url.endswith("/info")
        return Response({"local_memory": {"enabled": True}})

    def post(url, json=None, **kwargs):
        nonlocal fail_predict
        if url.endswith("/reset"):
            reset_sessions.append(json["session_id"])
            return Response({"status": "reset"})
        if url.endswith("/predict"):
            if fail_predict:
                raise requests.ConnectionError("injected transport failure")
            memory = json.get("local_memory")
            status = None if memory is None else {
                "session_id": memory["session_id"],
                "episode_id": memory["episode_id"],
                "consumer_step": memory["consumer_step"],
            }
            return Response({"action": [[0.0] * 7], "local_memory": [status]})
        raise AssertionError(url)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    client.begin_memory_episode(0)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # Step0 is acknowledged before any completed evidence exists.
    client.predict(image)
    payload = client.memory.payload(0)
    assert payload["consumer_step"] == 0 and payload["evidence"] == [] and not payload["reset"]

    client.record_memory_step(0, image, [0.0] * 6 + [-1.0], gripper_mode="zero_one")
    pending = client.memory.payload(0)
    assert pending["consumer_step"] == 1 and pending["evidence"][0]["gripper_mode"] == "zero_one"

    # A lost response must not discard the only copy of completed evidence.
    fail_predict = True
    with pytest.raises(requests.ConnectionError):
        client.predict(image)
    assert client.memory.payload(0) == pending

    fail_predict = False
    client.predict(image)
    assert client.memory.payload(0)["evidence"] == []
    session = client.memory.payload(0)["session_id"]
    client.end_memory_episode(0)
    assert reset_sessions == [session]


def test_batch_requires_unique_slots_before_http(monkeypatch):
    client = _client()
    client._memory_info_loaded = True
    client.memory.enabled = False
    images = [[np.zeros((16, 16, 3), dtype=np.uint8)]] * 2
    with pytest.raises(ValueError, match="unique slot"):
        client.predict_batch(images, slot_ids=[0, 0])

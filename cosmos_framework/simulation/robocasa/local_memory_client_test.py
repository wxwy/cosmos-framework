from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


def _image(value: int) -> np.ndarray:
    return np.full((256, 512, 3), value, dtype=np.uint8)


def test_completed_evidence_frontier_and_acknowledgement() -> None:
    client = RoboCasaLocalMemoryClient()
    client.begin(0)
    cold = client.payload(0)
    assert cold["consumer_step"] == 0
    assert cold["reset"] is True
    assert cold["evidence"] == []

    client.acknowledge(
        0,
        {
            "session_id": cold["session_id"],
            "episode_id": cold["episode_id"],
            "consumer_step": 0,
        },
    )
    for step in range(16):
        client.record_completed(0, _image(step), [0.1] * 15)
    payload = client.payload(0)
    assert payload["consumer_step"] == 16
    assert payload["reset"] is False
    assert [row["source_step"] for row in payload["evidence"]] == list(range(16))
    assert all(len(row["executed_action"]) == 15 for row in payload["evidence"])
    assert all("composite_image" in row and "left_image" not in row for row in payload["evidence"])
    assert payload["image_size"] == 256 and payload["preprocess_profile"]

    client.acknowledge(
        0,
        {
            "session_id": payload["session_id"],
            "episode_id": payload["episode_id"],
            "consumer_step": 16,
        },
    )
    assert client.payload(0)["evidence"] == []


def test_wrong_frontier_or_action_width_fails() -> None:
    client = RoboCasaLocalMemoryClient()
    client.begin(0)
    with pytest.raises(ValueError, match="raw15"):
        client.record_completed(0, _image(0), [0.0] * 20)
    payload = client.payload(0)
    with pytest.raises(ValueError, match="frontier"):
        client.acknowledge(
            0,
            {
                "session_id": payload["session_id"],
                "episode_id": payload["episode_id"],
                "consumer_step": 1,
            },
        )


def test_unacknowledged_only_and_episode_slot_isolation() -> None:
    client = RoboCasaLocalMemoryClient()
    client.begin(0, image_size=192)
    with pytest.raises(RuntimeError, match="preceding"):
        client.begin(0)
    client.record_completed(0, _image(5), [0.0] * 15)
    pending = client.payload(0)
    assert len(pending["evidence"]) == 1
    client.acknowledge(0, pending)
    assert client.payload(0)["evidence"] == []
    assert client.payload(0)["image_size"] == 192
    old = client.end(0)
    client.begin(0)
    assert client.session_id(0) != old and client.payload(0)["consumer_step"] == 0

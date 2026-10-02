from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


def _image(value: int) -> np.ndarray:
    return np.full((256, 256, 3), value, dtype=np.uint8)


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
        client.record_executed(0, _image(step), _image(100 + step), [0.1] * 15)
    payload = client.payload(0)
    assert payload["consumer_step"] == 16
    assert payload["reset"] is False
    assert [row["source_step"] for row in payload["evidence"]] == list(range(16))
    assert all(len(row["executed_action"]) == 15 for row in payload["evidence"])

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
        client.record_executed(0, _image(0), _image(0), [0.0] * 20)
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

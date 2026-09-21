import pytest

from cosmos_framework.simulation.libero.local_memory_client import ClientLocalMemory


def acknowledge(memory, slot=0):
    payload = memory.payload(slot)
    memory.acknowledge(slot, {key: payload[key] for key in ("session_id", "episode_id", "consumer_step")})


def test_chunk_ledger_contains_every_executed_step_and_survives_request_retry():
    memory = ClientLocalMemory(enabled=True)
    memory.begin()
    first = memory.payload()
    assert first["consumer_step"] == 0 and first["evidence"] == [] and first["reset"]
    acknowledge(memory)
    for step in range(16):
        memory.record_executed(0, f"pre-action-frame-{step}", [0.0] * 6 + [1.0], gripper_mode="zero_one")
    payload = memory.payload()
    assert payload["consumer_step"] == 16
    assert [row["source_step"] for row in payload["evidence"]] == list(range(16))
    assert payload == memory.payload()  # no successful response: keep exact replay bytes
    acknowledge(memory)
    assert memory.payload()["evidence"] == []
    assert not memory.payload()["reset"]


def test_interleaved_env_slots_do_not_share_chronology():
    memory = ClientLocalMemory(enabled=True)
    for slot in (0, 3):
        memory.begin(slot)
        acknowledge(memory, slot)
    memory.record_executed(0, "frame0", [0.0] * 7, gripper_mode="pm_one")
    for _ in range(3):
        memory.record_executed(3, "frame3", [0.0] * 7, gripper_mode="pm_one")
    assert memory.payload(0)["consumer_step"] == 1
    assert memory.payload(3)["consumer_step"] == 3
    old = memory.end(0)
    memory.begin(0)
    assert memory.payload(0)["consumer_step"] == 0
    assert memory.payload(0)["session_id"] == old
    assert memory.payload(3)["consumer_step"] == 3


def test_wrong_server_ack_does_not_drop_pending_evidence():
    memory = ClientLocalMemory(enabled=True)
    memory.begin()
    with pytest.raises(ValueError, match="frontier"):
        memory.acknowledge(0, {"consumer_step": 99})
    assert memory.payload()["reset"]


def test_native_window_ack_retains_only_bounded_causal_tail():
    memory = ClientLocalMemory(enabled=True)
    memory.retain_history_horizon = 2
    memory.begin()
    acknowledge(memory)
    for step in range(3):
        memory.record_executed(0, f"frame-{step}", [0.0] * 7, gripper_mode="pm_one")
    payload = memory.payload()
    assert payload["consumer_step"] == 3
    assert [row["source_step"] for row in payload["evidence"]] == [1, 2]
    acknowledge(memory)
    retained = memory.payload()
    assert [row["source_step"] for row in retained["evidence"]] == [1, 2]
    memory.record_executed(0, "frame-3", [0.0] * 7, gripper_mode="pm_one")
    assert [row["source_step"] for row in memory.payload()["evidence"]] == [2, 3]

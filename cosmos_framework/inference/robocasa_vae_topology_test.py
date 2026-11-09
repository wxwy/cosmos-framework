"""One B1 Worker can serve bounded, separately pinned Policy Server epochs."""

import pytest

from cosmos_framework.inference import robocasa_vae_worker_test as w
from cosmos_framework.inference.robocasa_vae_worker import WorkerConfig

make_worker = w.make_worker


@pytest.mark.parametrize("servers,sims", [(1, 1), (1, 8), (2, 8), (2, 16), (3, 3)])
def test_bounded_shared_worker_keeps_each_policy_session_and_signed_result_separate(make_worker, servers, sims):
    f = make_worker(config=WorkerConfig(bmax=1, wait_ms=0, max_sessions=servers * sims, max_policy_epochs=servers))
    sessions = [
        w._register(f, i, completed=16, lease=w._lease(i, server_epoch=f"policy-{i // sims}"))
        for i in range(servers * sims)
    ]
    frames = [w._frame(f, s, 0, rgb=bytes([i]) * w.RGB_BYTES) for i, s in enumerate(sessions)]
    keys = [f.worker.submit(frame, s.capability) for s, frame in zip(sessions, frames)]
    for s, frame, key in zip(sessions, frames, keys):
        result = w._result(f, s, key)
        w._verify(f, s, frame, result)
        assert f.worker.submit(frame, s.capability) == key
    assert w._histogram(f.worker) == {1: len(sessions)}
    assert all(len(call) == 1 for call in f.encoder.calls)
    first = sessions[0]
    f.worker.end(first.lease["session_id"], first.capability, f.clock() + 3_000_000_000)
    for s, key in zip(sessions[1:], keys[1:]):
        assert w._poll(f, s, key)["status"] == "ready"
    if len(sessions) > 1:
        with pytest.raises((ValueError, PermissionError)):
            f.worker.poll(sessions[1].lease["session_id"], keys[1], first.capability)


def test_policy_epoch_capacity_preserves_foreign_run_slot_epoch_and_stale_generation_guards(make_worker):
    f = make_worker(config=WorkerConfig(max_sessions=4, max_policy_epochs=2))
    w._register(f, 0, lease=w._lease(0, server_epoch="A"))
    w._register(f, 1, lease=w._lease(1, server_epoch="B"))
    for changes in (
        {"server_epoch": "C"},
        {"run_lease": "foreign"},
        {"manifest_digest": "f" * 64},
        {"slot_id": "slot-0", "server_epoch": "B"},
    ):
        with pytest.raises(w.REJECTED):
            w._register(f, 2, lease=w._lease(2, **changes))
    f.worker.end("session-0", "synthetic-session-capability-0-0000000000000000", f.clock() + 3_000_000_000)
    with pytest.raises(w.REJECTED):
        w._register(f, 0, lease=w._lease(0, session_id="replacement", generation=1, server_epoch="A"))
    new = w._register(f, 0, lease=w._lease(0, session_id="replacement", generation=2, server_epoch="A"))
    assert new.lease["generation"] == 2


def test_default_worker_still_pins_a_single_policy_epoch(make_worker):
    f = make_worker()
    w._register(f, 0)
    with pytest.raises(w.REJECTED):
        w._register(f, 1, lease=w._lease(1, server_epoch="foreign"))


@pytest.mark.parametrize("capacity", [0, True, -1, 3])
def test_policy_capacity_is_bounded_by_existing_session_limit(capacity):
    with pytest.raises(ValueError):
        WorkerConfig(max_sessions=2, max_policy_epochs=capacity)

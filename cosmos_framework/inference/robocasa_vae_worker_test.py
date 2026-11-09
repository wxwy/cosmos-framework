"""Standalone synthetic CPU Worker contracts; no Policy/Sim/GPU integration.

The trusted owner_complete calls below are test control inputs. They do not attest
that a real environment action executed. The injected encoder returns deterministic
finite float32 bytes; these tests make no Wan numerical or GPU drain claim.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from cosmos_framework.inference.robocasa_vae_protocol import (
    RGB_BYTES,
    EncoderDrainUnproven,
    Frame,
    ProtocolError,
    source_digest,
    verify_feature,
)
from cosmos_framework.inference.robocasa_vae_worker import SharedVAEWorker, WorkerConfig

CLOCK_DOMAIN = "synthetic-same-boot-clock"
SIGNING_KEY = b"S" * 32
IDENTITY = {
    "vae_weights_sha256": "a" * 64,
    "vae_config_sha256": "b" * 64,
    "preprocess_profile_sha256": "c" * 64,
    "visual96_contract_sha256": "d" * 64,
    "encoder_backend": "eager",
    "encoder_dtype": "bfloat16",
}
REJECTED = (ValueError, RuntimeError, PermissionError, TimeoutError)


class _Clock:
    def __init__(self):
        self.value = time.monotonic_ns()
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance(self, nanoseconds):
        with self.lock:
            self.value += nanoseconds


def _lease(index=0, **changes):
    return {
        "server_epoch": "policy-epoch",
        "run_lease": "run-lease",
        "manifest_digest": "e" * 64,
        "slot_id": f"slot-{index}",
        "generation": 1,
        "task": "CloseFridge",
        "episode_index": index,
        "attempt_id": f"attempt-{index}",
        "session_id": f"session-{index}",
        "episode_id": f"episode-{index}",
        **changes,
    }


def _visual(frame):
    value = int(frame.fingerprint[:6], 16) % 1000 / 1000
    return struct.pack("<96f", *(value + index / 10000 for index in range(96)))


class _Encoder:
    def __init__(self, *, blocked=False, failure=None, invalid=None):
        self.calls = []
        self.changed = threading.Condition()
        self.entered, self.release, self.returned = threading.Event(), threading.Event(), threading.Event()
        self.failure, self.invalid = failure, invalid
        if not blocked:
            self.release.set()

    def __call__(self, frames):
        with self.changed:
            self.calls.append(tuple(frames))
            self.changed.notify_all()
        self.entered.set()
        assert self.release.wait(5), "test did not release synthetic encoder"
        try:
            if self.failure is not None:
                raise self.failure
            if self.invalid is not None:
                return self.invalid(frames)
            return [_visual(frame) for frame in frames]
        finally:
            self.returned.set()


@pytest.fixture
def make_worker():
    created = []

    def make(*, encoder=None, config=None, clock=None, autostart=True, epoch="worker-epoch", identity=None):
        encoder = _Encoder() if encoder is None else encoder
        clock = time.monotonic_ns if clock is None else clock
        worker = SharedVAEWorker(
            encoder,
            dict(IDENTITY) if identity is None else identity,
            SIGNING_KEY,
            "synthetic-key",
            "synthetic-worker",
            worker_epoch=epoch,
            clock_domain=CLOCK_DOMAIN,
            config=WorkerConfig() if config is None else config,
            clock_ns=clock,
            autostart=autostart,
        )
        result = SimpleNamespace(worker=worker, encoder=encoder, clock=clock, epoch=epoch)
        created.append(result)
        return result

    yield make
    for fixture in created:
        fixture.encoder.release.set()
    for fixture in created:
        fixture.worker.close(fixture.clock() + 3_000_000_000)


def _register(fixture, index=0, *, start=0, count=16, completed=None, lease=None, deadline=None):
    lease = _lease(index) if lease is None else lease
    capability = f"synthetic-session-capability-{index}-0000000000000000"
    deadline = fixture.clock() + 30_000_000_000 if deadline is None else deadline
    worker = fixture.worker
    worker.owner_register(lease, capability, deadline)
    worker.owner_issue(lease["session_id"], f"chunk-{start}", f"request-{start}", start, count)
    if completed is not None:
        worker.owner_complete(lease["session_id"], f"chunk-{start}", completed)
    return SimpleNamespace(lease=lease, capability=capability, deadline=deadline, chunk=f"chunk-{start}", start=start)


def _frame(fixture, session, step, *, rgb=None, raw15=None, **changes):
    arguments = {
        "lease": session.lease,
        "chunk_id": session.chunk,
        "request_key": f"request-{session.start}",
        "source_step": step,
        "rgb": bytes([step % 256]) * RGB_BYTES if rgb is None else rgb,
        "raw15": struct.pack("<15f", *([step / 100] * 15)) if raw15 is None else raw15,
        "identity": dict(IDENTITY),
        "episode_deadline_ns": session.deadline,
        "job_deadline_ns": min(session.deadline, fixture.clock() + 20_000_000_000),
        "clock_domain": CLOCK_DOMAIN,
        **changes,
    }
    return Frame.create(**arguments)


def _poll(fixture, session, key):
    return fixture.worker.poll(session.lease["session_id"], key, session.capability)


def _result(fixture, session, key):
    with fixture.worker._changed:
        assert fixture.worker._changed.wait_for(
            lambda: _poll(fixture, session, key)["status"] != "pending", timeout=3
        ), "bounded synthetic worker did not complete"
        return _poll(fixture, session, key)


def _verify(fixture, session, frame, result):
    assert result["status"] == "ready"
    feature = verify_feature(
        result["feature"],
        SIGNING_KEY,
        expected_lease=session.lease,
        expected_identity=IDENTITY,
        expected_worker_id="synthetic-worker",
        expected_worker_epoch=fixture.epoch,
        expected_key_id="synthetic-key",
        expected_clock_domain=CLOCK_DOMAIN,
        now_ns=fixture.clock(),
    )
    assert feature["header"]["source_step"] == frame.header["source_step"]
    assert feature["header"]["source_rgb_sha256"] == frame.header["source_rgb_sha256"]
    assert base64.b64decode(feature["visual96_f32le_b64"]) == _visual(frame)
    assert base64.b64decode(feature["raw15_f32le_b64"]) == frame.raw15
    return feature


def _histogram(worker):
    return {int(size): count for size, count in worker.info()["batch_histogram"].items()}


@pytest.mark.parametrize("bmax,wait_ms", [(1, 0), (4, 0), (4, 2), (4, 5), (8, 0), (8, 2), (8, 5)])
def test_supported_configurations(bmax, wait_ms):
    assert WorkerConfig(bmax=bmax, wait_ms=wait_ms).bmax == bmax


@pytest.mark.parametrize(
    "changes",
    [
        {"bmax": 2},
        {"bmax": True},
        {"wait_ms": 1},
        {"wait_ms": True},
        {"bmax": 1, "wait_ms": 2},
        {"max_items": 0},
        {"max_bytes": 0},
        {"max_session_items": 17},
        {"max_sessions": 0},
        {"max_tombstones": 0},
        {"max_chunks": 0},
        {"max_trace": 0},
        {"max_retired_sessions": 0},
    ],
)
def test_invalid_configuration_fails_before_start(changes):
    with pytest.raises((ValueError, TypeError)):
        WorkerConfig(**changes)


def test_b1_has_no_peer_or_clock_advance_requirement(make_worker):
    fixture = make_worker(clock=_Clock())
    assert fixture.worker.start() is fixture.worker
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    now = fixture.clock()
    key = fixture.worker.submit(frame, session.capability)
    _verify(fixture, session, frame, _result(fixture, session, key))
    assert fixture.clock() == now
    assert [len(batch) for batch in fixture.encoder.calls] == [1]
    assert _histogram(fixture.worker) == {1: 1}


@pytest.mark.parametrize("bmax", [4, 8])
def test_prequeued_mixed_session_timesteps_fill_batch_and_preserve_mapping(make_worker, bmax):
    fixture = make_worker(config=WorkerConfig(bmax=bmax, wait_ms=0))
    sessions = []
    for index in range(bmax):
        session = _register(fixture, index, completed=16)
        sessions.append(session)
        for step in range(index):
            frame = _frame(fixture, session, step)
            key = fixture.worker.submit(frame, session.capability)
            _verify(fixture, session, frame, _result(fixture, session, key))
            fixture.worker.ack(session.lease["session_id"], key, session.capability)
    # A model-owned invocation keeps the next READY cohort queued deterministically.
    sentinel = _register(fixture, bmax, count=1, completed=1)
    fixture.encoder.entered.clear()
    fixture.encoder.release.clear()
    sentinel_key = fixture.worker.submit(_frame(fixture, sentinel, 0), sentinel.capability)
    assert fixture.encoder.entered.wait(3)
    pending = []
    for index, session in enumerate(sessions):
        frame = _frame(fixture, session, index)
        pending.append((session, frame, fixture.worker.submit(frame, session.capability)))
    fixture.encoder.release.set()
    assert _result(fixture, sentinel, sentinel_key)["status"] == "ready"
    for session, frame, key in pending:
        _verify(fixture, session, frame, _result(fixture, session, key))
    assert len(fixture.encoder.calls[-1]) == bmax
    assert {frame.header["source_step"] for frame in fixture.encoder.calls[-1]} == set(range(bmax))
    assert _histogram(fixture.worker)[bmax] == 1
    assert [row for row in fixture.worker.info()["trace"] if row["event"] == "DISPATCH"][-1]["reason"] == "full"


@pytest.mark.parametrize("bmax", [4, 8])
def test_round_robin_does_not_let_hot_session_displace_ready_peer(make_worker, bmax):
    fixture = make_worker(config=WorkerConfig(bmax=bmax, wait_ms=0), autostart=False)
    hot = _register(fixture, completed=16)
    peer = _register(fixture, 1, count=1, completed=1)
    keys = [fixture.worker.submit(_frame(fixture, hot, step), hot.capability) for step in range(bmax)]
    peer_frame = _frame(fixture, peer, 0)
    peer_key = fixture.worker.submit(peer_frame, peer.capability)
    fixture.worker.start()
    _verify(fixture, peer, peer_frame, _result(fixture, peer, peer_key))
    assert len(fixture.encoder.calls[0]) == bmax
    assert any(frame.header["lease"] == peer.lease for frame in fixture.encoder.calls[0])
    for key in keys:
        assert _result(fixture, hot, key)["status"] == "ready"


def test_b1_round_robin_cursor_persists_across_dispatches(make_worker):
    fixture = make_worker(autostart=False)
    sessions = [_register(fixture, index, count=2, completed=2) for index in range(2)]
    pending = []
    for session in sessions:
        for step in range(2):
            pending.append((session, fixture.worker.submit(_frame(fixture, session, step), session.capability)))
    fixture.worker.start()
    for session, key in pending:
        assert _result(fixture, session, key)["status"] == "ready"
    rows = [batch[0].header for batch in fixture.encoder.calls]
    assert [row["lease"]["session_id"] for row in rows] == ["session-0", "session-1", "session-0", "session-1"]
    assert [row["source_step"] for row in rows] == [0, 0, 1, 1]


def test_b4_round_robin_serves_second_cohort_before_hot_first_cohort(make_worker):
    fixture = make_worker(config=WorkerConfig(bmax=4), autostart=False)
    sessions = [_register(fixture, index, count=2, completed=2) for index in range(8)]
    pending = []
    for session in sessions:
        for step in range(2):
            pending.append((session, fixture.worker.submit(_frame(fixture, session, step), session.capability)))
    fixture.worker.start()
    for session, key in pending:
        assert _result(fixture, session, key)["status"] == "ready"
    assert [len(batch) for batch in fixture.encoder.calls] == [4, 4, 4, 4]
    assert [frame.header["lease"]["episode_index"] for frame in fixture.encoder.calls[0]] == list(range(4))
    assert [frame.header["lease"]["episode_index"] for frame in fixture.encoder.calls[1]] == list(range(4, 8))


@pytest.mark.parametrize("wait_ms", [2, 5])
def test_partial_batch_dispatches_from_ready_window_without_barrier(make_worker, wait_ms):
    clock = _Clock()
    fixture = make_worker(config=WorkerConfig(bmax=8, wait_ms=wait_ms), clock=clock, autostart=False)
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    # Frame construction and prior control time do not consume the READY window.
    clock.advance(50_000_000)
    key = fixture.worker.submit(frame, session.capability)
    fixture.worker.start()
    assert not fixture.encoder.entered.is_set()
    clock.advance(wait_ms * 1_000_000)
    with fixture.worker._changed:
        fixture.worker._changed.notify_all()
    _verify(fixture, session, frame, _result(fixture, session, key))
    assert [len(batch) for batch in fixture.encoder.calls] == [1]
    assert _histogram(fixture.worker) == {1: 1}
    trace = fixture.worker.info()["trace"]
    ready = next(row for row in trace if row["event"] == "READY")
    dispatch = next(row for row in trace if row["event"] == "DISPATCH")
    assert dispatch["reason"] == "window"
    assert dispatch["at_ns"] - ready["at_ns"] == wait_ms * 1_000_000


def test_earliest_deadline_dispatches_partial_batch_before_window_expires(make_worker):
    clock = _Clock()
    fixture = make_worker(config=WorkerConfig(bmax=8, wait_ms=5), clock=clock, autostart=False)
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0, job_deadline_ns=clock() + 4_000_000)
    key = fixture.worker.submit(frame, session.capability)
    fixture.worker.start()
    _verify(fixture, session, frame, _result(fixture, session, key))
    trace = fixture.worker.info()["trace"]
    ready = next(row for row in trace if row["event"] == "READY")
    dispatch = next(row for row in trace if row["event"] == "DISPATCH")
    assert dispatch["reason"] == "deadline" and dispatch["size"] == 1
    assert dispatch["at_ns"] == ready["at_ns"]


def test_exact_duplicate_and_post_ack_retry_never_execute_twice(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    assert fixture.encoder.entered.wait(3)
    assert fixture.worker.submit(Frame.from_wire(frame.to_wire()), session.capability) == key
    fixture.encoder.release.set()
    _verify(fixture, session, frame, _result(fixture, session, key))
    assert fixture.worker.submit(frame, session.capability) == key
    fixture.worker.ack(session.lease["session_id"], key, session.capability)
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0
    assert fixture.worker.submit(frame, session.capability) == key
    assert _poll(fixture, session, key)["status"] == "already_acked"
    assert "feature" not in _poll(fixture, session, key)
    assert len(fixture.encoder.calls) == 1


@pytest.mark.parametrize("stage", ["queued", "active", "ready", "acked"])
@pytest.mark.parametrize("changed", ["rgb", "raw15", "deadline"])
def test_same_step_changed_retry_is_rejected_at_every_stage(make_worker, stage, changed):
    encoder = _Encoder(blocked=stage == "active")
    fixture = make_worker(encoder=encoder, autostart=stage != "queued")
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    if stage == "active":
        assert encoder.entered.wait(3)
    elif stage in ("ready", "acked"):
        assert _result(fixture, session, key)["status"] == "ready"
        if stage == "acked":
            fixture.worker.ack(session.lease["session_id"], key, session.capability)
    changes = {
        "rgb": {"rgb": b"x" * RGB_BYTES, "job_deadline_ns": int(frame.header["job_deadline_ns"])},
        "raw15": {
            "raw15": struct.pack("<15f", *([0.5] * 15)),
            "job_deadline_ns": int(frame.header["job_deadline_ns"]),
        },
        "deadline": {"job_deadline_ns": int(frame.header["job_deadline_ns"]) - 1},
    }[changed]
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, session, 0, **changes), session.capability)


def test_ack_tombstone_eviction_cannot_reexecute_old_frontier(make_worker):
    fixture = make_worker(config=WorkerConfig(max_tombstones=1))
    session = _register(fixture, count=2, completed=2)
    frames = [_frame(fixture, session, step) for step in range(2)]
    for frame in frames:
        key = fixture.worker.submit(frame, session.capability)
        assert _result(fixture, session, key)["status"] == "ready"
        fixture.worker.ack(session.lease["session_id"], key, session.capability)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(frames[0], session.capability)
    assert len(fixture.encoder.calls) == 2


def test_issued_chunk_and_trusted_completed_frontier_are_required(make_worker):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=2)
    frame = _frame(fixture, session, 0)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(frame, session.capability)
    fixture.worker.owner_complete(session.lease["session_id"], session.chunk, 1)
    fixture.worker.submit(frame, session.capability)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, session, 1), session.capability)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.owner_complete(session.lease["session_id"], session.chunk, 0)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.owner_complete(session.lease["session_id"], session.chunk, 3)
    fixture.worker.owner_complete(session.lease["session_id"], session.chunk, 2)
    for changes in ({"chunk_id": "unissued"}, {"request_key": "foreign-request"}, {"source_step": 2}):
        with pytest.raises((ValueError, RuntimeError)):
            fixture.worker.submit(_frame(fixture, session, 1, **changes), session.capability)
    assert fixture.worker.info()["items"] == 1 and fixture.encoder.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "server_epoch",
        "run_lease",
        "manifest_digest",
        "slot_id",
        "generation",
        "task",
        "episode_index",
        "attempt_id",
        "session_id",
        "episode_id",
    ],
)
def test_full_registered_lease_is_authoritative(make_worker, field):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=1, completed=1)
    changed = dict(session.lease)
    changed[field] = 2 if field in ("generation", "episode_index") else "f" * 64
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, session, 0, lease=changed), session.capability)
    assert fixture.worker.info()["items"] == 0


@pytest.mark.parametrize("field", ["run_lease", "manifest_digest", "server_epoch"])
@pytest.mark.parametrize("new_slot", [False, True])
def test_registration_cannot_cross_worker_run_manifest_or_policy_epoch(make_worker, field, new_slot):
    fixture = make_worker(autostart=False)
    _register(fixture, count=1, completed=1)
    candidate = _lease(1, slot_id="slot-1" if new_slot else "slot-0")
    candidate[field] = "f" * 64
    with pytest.raises(ValueError, match="run|manifest|epoch"):
        fixture.worker.owner_register(candidate, "new-session-capability-000000000", fixture.clock() + 30_000_000_000)
    assert fixture.worker.info()["sessions"] == fixture.worker.info()["known_slots"] == 1
    _register(fixture, 1, count=1, completed=1)
    assert fixture.worker.info()["sessions"] == 2


@pytest.mark.parametrize("bmax", [4, 8])
def test_aot_b1_identity_cannot_be_advertised_as_dynamic_b4_or_b8(make_worker, bmax):
    with pytest.raises(ValueError, match="aot_b1"):
        make_worker(config=WorkerConfig(bmax=bmax), identity={**IDENTITY, "encoder_backend": "aot_b1"})


def test_register_retry_changed_identity_and_session_slot_capacity(make_worker):
    fixture = make_worker(config=WorkerConfig(max_sessions=1), autostart=False)
    session = _register(fixture, count=1, completed=1)
    retry = fixture.worker.owner_register(session.lease, session.capability, session.deadline)
    assert retry["registered"] is True and retry["duplicate"] is True
    with pytest.raises(ValueError):
        fixture.worker.owner_register(session.lease, "changed-session-capability-000000", session.deadline)
    with pytest.raises(ValueError):
        fixture.worker.owner_register({**session.lease, "generation": 2}, session.capability, session.deadline)
    with pytest.raises(RuntimeError, match="capacity"):
        _register(fixture, 1, count=1, completed=1)
    fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    renewed = _lease(1, slot_id=session.lease["slot_id"], generation=2)
    new_session = _register(fixture, 1, count=1, completed=1, lease=renewed)
    assert new_session.lease["generation"] == 2 and fixture.worker.info()["known_slots"] == 1
    fixture.worker.end(new_session.lease["session_id"], new_session.capability, fixture.clock() + 1_000_000_000)
    with pytest.raises(ValueError, match="generation"):
        fixture.worker.owner_register(session.lease, session.capability, session.deadline)


def test_issue_retry_cannot_change_request_or_source_range(make_worker):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=2)
    reply = fixture.worker.owner_issue(session.lease["session_id"], session.chunk, "request-0", 0, 2)
    assert reply["duplicate"] is True
    for request, start, count in (("changed-request", 0, 2), ("request-0", 1, 2), ("request-0", 0, 3)):
        with pytest.raises(ValueError, match="identity"):
            fixture.worker.owner_issue(session.lease["session_id"], session.chunk, request, start, count)
    with pytest.raises(ValueError, match="contiguous"):
        fixture.worker.owner_issue(session.lease["session_id"], "future-chunk", "future-request", 3, 1)


def test_chunk_metadata_is_bounded_and_eviction_cannot_reissue_old_range(make_worker):
    fixture = make_worker(config=WorkerConfig(max_chunks=1))
    session = _register(fixture, count=1, completed=1)
    key = fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    assert _result(fixture, session, key)["status"] == "ready"
    with pytest.raises(RuntimeError, match="metadata capacity"):
        fixture.worker.owner_issue(session.lease["session_id"], "chunk-1", "request-1", 1, 1)
    fixture.worker.ack(session.lease["session_id"], key, session.capability)
    fixture.worker.owner_issue(session.lease["session_id"], "chunk-1", "request-1", 1, 1)
    fixture.worker.owner_complete(session.lease["session_id"], "chunk-1", 2)
    frame = _frame(fixture, session, 1, chunk_id="chunk-1", request_key="request-1")
    next_key = fixture.worker.submit(frame, session.capability)
    _verify(fixture, session, frame, _result(fixture, session, next_key))
    assert fixture.worker.info()["issued_chunks"] == 1
    with pytest.raises(ValueError, match="contiguous"):
        fixture.worker.owner_issue(session.lease["session_id"], "chunk-0", "request-0", 0, 1)


def test_retired_session_bound_keeps_slot_generation_replay_guard(make_worker):
    fixture = make_worker(config=WorkerConfig(max_sessions=1, max_retired_sessions=1), autostart=False)
    old = _register(fixture, count=1, completed=1)
    fixture.worker.end(old.lease["session_id"], old.capability, fixture.clock() + 1_000_000_000)
    renewed = _register(fixture, 1, count=1, completed=1, lease=_lease(1, slot_id="slot-0", generation=2))
    fixture.worker.end(renewed.lease["session_id"], renewed.capability, fixture.clock() + 1_000_000_000)
    assert fixture.worker.info()["retired_sessions"] == fixture.worker.info()["known_slots"] == 1
    with pytest.raises(ValueError, match="generation"):
        fixture.worker.owner_register(old.lease, old.capability, old.deadline)


@pytest.mark.parametrize("changed", ["capability", "clock", "model", "episode_deadline"])
def test_wrong_capability_clock_model_or_original_budget_never_enqueue(make_worker, changed):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=1, completed=1)
    changes = {
        "capability": {},
        "clock": {"clock_domain": "foreign-clock"},
        "model": {"identity": {**IDENTITY, "vae_weights_sha256": "f" * 64}},
        "episode_deadline": {"episode_deadline_ns": session.deadline + 1},
    }[changed]
    capability = "foreign-session-capability" if changed == "capability" else session.capability
    with pytest.raises(REJECTED):
        fixture.worker.submit(_frame(fixture, session, 0, **changes), capability)
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0


def test_ready_payload_remains_charged_until_ack_and_peer_bound_is_global(make_worker):
    fixture = make_worker(config=WorkerConfig(max_items=1))
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    assert _result(fixture, session, key)["status"] == "ready"
    assert fixture.worker.info()["items"] == 1 and fixture.worker.info()["bytes"] > RGB_BYTES
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, peer, 0), peer.capability)
    fixture.worker.ack(session.lease["session_id"], key, session.capability)
    peer_frame = _frame(fixture, peer, 0)
    peer_key = fixture.worker.submit(peer_frame, peer.capability)
    _verify(fixture, peer, peer_frame, _result(fixture, peer, peer_key))


def test_encoded_byte_bound_rejects_without_partial_enrollment(make_worker):
    fixture = make_worker(config=WorkerConfig(max_bytes=RGB_BYTES), autostart=False)
    session = _register(fixture, count=1, completed=1)
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0


def test_per_session_item_bound_preserves_capacity_for_peer(make_worker):
    fixture = make_worker(config=WorkerConfig(max_session_items=1), autostart=False)
    session, peer = _register(fixture, count=2, completed=2), _register(fixture, 1, count=1, completed=1)
    fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    with pytest.raises(RuntimeError, match="capacity"):
        fixture.worker.submit(_frame(fixture, session, 1), session.capability)
    fixture.worker.submit(_frame(fixture, peer, 0), peer.capability)
    assert fixture.worker.info()["items"] == 2 and fixture.encoder.calls == []


def test_poll_capability_and_pending_ack_cannot_release_or_cross_session(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    assert fixture.encoder.entered.wait(3)
    with pytest.raises(RuntimeError, match="unfinished"):
        fixture.worker.ack(session.lease["session_id"], key, session.capability)
    with pytest.raises(PermissionError):
        fixture.worker.poll(session.lease["session_id"], key, peer.capability)
    with pytest.raises(ValueError, match="unknown"):
        fixture.worker.poll(peer.lease["session_id"], key, peer.capability)
    assert fixture.worker.info()["items"] == 1
    fixture.encoder.release.set()
    result = _result(fixture, session, key)
    result["feature"]["header"]["lease"]["generation"] = 99
    _verify(fixture, session, frame, _poll(fixture, session, key))


@pytest.mark.parametrize("operation", ["submit", "poll", "ack", "take_ready"])
def test_explicit_none_capability_never_selects_trusted_owner_bypass(make_worker, operation):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    with pytest.raises(REJECTED):
        if operation == "submit":
            fixture.worker.submit(frame, None)
        elif operation == "take_ready":
            fixture.worker.take_ready(session.lease["session_id"], None)
        else:
            getattr(fixture.worker, operation)(session.lease["session_id"], key, None)
    assert fixture.worker.info()["items"] == 1


def test_wrong_type_and_omitted_capability_cannot_enter_normal_handlers(make_worker):
    fixture = make_worker(autostart=False)
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    sid = session.lease["session_id"]
    handlers = [
        (fixture.worker.submit, (frame,)),
        (fixture.worker.poll, (sid, key)),
        (fixture.worker.ack, (sid, key)),
        (fixture.worker.take_ready, (sid,)),
    ]
    for handler, arguments in handlers:
        with pytest.raises(TypeError):
            handler(*arguments)
        with pytest.raises(REJECTED):
            handler(*arguments, 42)
    assert fixture.worker.info()["items"] == 1 and fixture.encoder.calls == []


def test_poll_out_of_order_is_safe_but_take_ready_preserves_consumer_order(make_worker):
    fixture = make_worker(config=WorkerConfig(bmax=4, wait_ms=0), autostart=False)
    session = _register(fixture, count=2, completed=2)
    earlier = _frame(fixture, session, 0)
    earlier_key = fixture.worker.submit(earlier, session.capability)
    later = _frame(fixture, session, 1)
    later_key = fixture.worker.submit(later, session.capability)
    fixture.worker.start()
    _verify(fixture, session, later, _result(fixture, session, later_key))
    _verify(fixture, session, earlier, _result(fixture, session, earlier_key))
    assert _poll(fixture, session, later_key)["status"] == "ready"
    first = fixture.worker.take_ready(session.lease["session_id"], session.capability)
    assert first["feature"]["header"]["source_step"] == 0
    assert fixture.worker.take_ready(session.lease["session_id"], session.capability) == first
    assert fixture.worker.info()["items"] == 2
    fixture.worker.ack(session.lease["session_id"], later_key, session.capability)
    assert fixture.worker.take_ready(session.lease["session_id"], session.capability) == first
    assert fixture.worker.info()["items"] == 1
    fixture.worker.ack(session.lease["session_id"], earlier_key, session.capability)
    assert fixture.worker.take_ready(session.lease["session_id"], session.capability) == {
        "status": "pending",
        "source_step": 2,
    }
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0


def test_blocked_encoder_never_holds_admission_lock(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    first = _frame(fixture, session, 0)
    first_key = fixture.worker.submit(first, session.capability)
    assert fixture.encoder.entered.wait(3)
    finished, errors, peer_keys = threading.Event(), [], []

    def admit_peer():
        try:
            peer_keys.append(fixture.worker.submit(_frame(fixture, peer, 0), peer.capability))
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=admit_peer)
    thread.start()
    try:
        assert finished.wait(3), "encoder held the completed-admission lock"
        assert errors == [] and len(peer_keys) == 1
    finally:
        fixture.encoder.release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert _result(fixture, session, first_key)["status"] == "ready"
    assert _result(fixture, peer, peer_keys[0])["status"] == "ready"


@pytest.mark.parametrize(
    "invalid",
    [
        lambda frames: [],
        lambda frames: [b"short"] * len(frames),
        lambda frames: [struct.pack("<96f", *([float("nan")] * 96))] * len(frames),
    ],
)
def test_invalid_encoder_cardinality_shape_and_nonfinite_output_fail_closed(make_worker, invalid):
    fixture = make_worker(encoder=_Encoder(invalid=invalid))
    session = _register(fixture, count=1, completed=1)
    key = fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    result = _result(fixture, session, key)
    assert result["status"] == "error" and "feature" not in result


def test_encoder_error_retains_idempotent_error_without_feature(make_worker):
    fixture = make_worker(encoder=_Encoder(failure=RuntimeError("synthetic encoder failed")))
    session = _register(fixture, count=1, completed=1)
    peer = _register(fixture, 1, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    assert _result(fixture, session, key)["status"] == "error"
    assert fixture.worker.submit(frame, session.capability) == key
    assert len(fixture.encoder.calls) == 1
    assert fixture.worker.info()["quarantined"] is False
    fixture.encoder.failure = None
    peer_frame = _frame(fixture, peer, 0)
    peer_key = fixture.worker.submit(peer_frame, peer.capability)
    _verify(fixture, peer, peer_frame, _result(fixture, peer, peer_key))


def test_unproven_encoder_drain_stickily_quarantines_worker_and_preserves_charge(make_worker):
    # Synthetic typed signal tests ownership policy; no actual device is present.
    fixture = make_worker(
        encoder=_Encoder(failure=EncoderDrainUnproven("synthetic uncertain encoder ownership")), autostart=False
    )
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    first = _frame(fixture, session, 0)
    first_key = fixture.worker.submit(first, session.capability)
    peer_key = fixture.worker.submit(_frame(fixture, peer, 0), peer.capability)
    fixture.worker.start()
    assert _result(fixture, session, first_key)["status"] == "error"
    assert _result(fixture, peer, peer_key)["status"] == "error"
    assert len(fixture.encoder.calls) == 1
    info = fixture.worker.info()
    assert info["active"] == 0 and info["quarantined"] is True and info["stopping"] is True
    assert info["bytes"] > 0
    with pytest.raises(REJECTED):
        fixture.worker.ack(session.lease["session_id"], first_key, session.capability)
    ended = fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    assert ended["acknowledged"] is False and ended["quiescent"] is False
    closed = fixture.worker.close(fixture.clock() + 1_000_000_000)
    assert closed["closed"] is False and closed["quiescent"] is False
    assert fixture.worker.info()["bytes"] > 0 and len(fixture.encoder.calls) == 1


def test_queued_deadline_expiry_and_late_encoder_return_publish_no_feature(make_worker):
    clock = _Clock()
    fixture = make_worker(clock=clock, autostart=False)
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0, job_deadline_ns=clock() + 10)
    key = fixture.worker.submit(frame, session.capability)
    clock.advance(10)
    fixture.worker.start()
    assert _result(fixture, session, key)["status"] == "error"
    assert fixture.encoder.calls == []
    active = make_worker(clock=_Clock(), encoder=_Encoder(blocked=True))
    other = _register(active, count=1, completed=1)
    late = _frame(active, other, 0, job_deadline_ns=active.clock() + 100)
    late_key = active.worker.submit(late, other.capability)
    assert active.encoder.entered.wait(3)
    active.clock.advance(100)
    active.encoder.release.set()
    result = _result(active, other, late_key)
    assert result["status"] == "error" and "feature" not in result


def test_admission_and_replan_use_original_absolute_deadline(make_worker):
    clock = _Clock()
    fixture = make_worker(clock=clock, autostart=False)
    session = _register(fixture, count=1, completed=1)
    with pytest.raises(TimeoutError):
        fixture.worker.submit(_frame(fixture, session, 0, job_deadline_ns=clock()), session.capability)
    frame = _frame(fixture, session, 0, replan_deadline_ns=clock() + 10)
    key = fixture.worker.submit(frame, session.capability)
    clock.advance(10)
    fixture.worker.start()
    result = _result(fixture, session, key)
    assert result["status"] == "error" and result["error"]["code"] == "FRAME_DEADLINE"
    assert fixture.encoder.calls == []
    assert any(
        row["event"] == "JOB_ERROR" and row["code"] == "FRAME_DEADLINE" for row in fixture.worker.info()["trace"]
    )


def test_end_cancels_only_queued_owned_work_and_leaves_peer(make_worker):
    fixture = make_worker(autostart=False)
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    peer_frame = _frame(fixture, peer, 0)
    peer_key = fixture.worker.submit(peer_frame, peer.capability)
    ended = fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    assert ended["acknowledged"] is True and fixture.worker.info()["items"] == 1
    fixture.worker.start()
    _verify(fixture, peer, peer_frame, _result(fixture, peer, peer_key))
    assert all(frame.header["lease"] == peer.lease for batch in fixture.encoder.calls for frame in batch)


def test_clean_end_retires_session_without_affecting_peer(make_worker):
    fixture = make_worker()
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    reply = fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 2_000_000_000)
    assert reply["acknowledged"] is True and reply["quiescent"] is True
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    frame = _frame(fixture, peer, 0)
    key = fixture.worker.submit(frame, peer.capability)
    _verify(fixture, peer, frame, _result(fixture, peer, key))


def test_active_end_deadline_quarantines_and_late_result_cannot_revive_session(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session, peer = _register(fixture, count=1, completed=1), _register(fixture, 1, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    assert fixture.encoder.entered.wait(3)
    reply = fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock())
    assert reply["acknowledged"] is False and reply["quiescent"] is False
    assert reply["state"] == "QUARANTINED"
    assert fixture.worker.info()["bytes"] > 0
    fixture.encoder.release.set()
    assert fixture.encoder.returned.wait(3)
    with fixture.worker._changed:
        assert fixture.worker._changed.wait_for(lambda: fixture.worker.info()["active"] == 0, timeout=3)
    result = _poll(fixture, session, key)
    assert result["status"] == "error" and "feature" not in result
    with pytest.raises((ValueError, RuntimeError)):
        fixture.worker.submit(frame, session.capability)
    peer_frame = _frame(fixture, peer, 0)
    peer_key = fixture.worker.submit(peer_frame, peer.capability)
    _verify(fixture, peer, peer_frame, _result(fixture, peer, peer_key))
    again = fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    assert again["acknowledged"] is False


def test_concurrent_end_retries_wait_for_same_active_owner_and_return_same_receipt(make_worker, monkeypatch):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session = _register(fixture, count=1, completed=1)
    fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    assert fixture.encoder.entered.wait(3)
    both_draining, observer_lock = threading.Event(), threading.Lock()
    original = fixture.worker._drain
    drains, replies, errors = [], [], []

    def observed_drain(deadline, active):
        if threading.current_thread().name.startswith("test-end-retry-"):
            with observer_lock:
                drains.append(threading.current_thread().name)
                if len(drains) == 2:
                    both_draining.set()
        return original(deadline, active)

    monkeypatch.setattr(fixture.worker, "_drain", observed_drain)

    def end():
        try:
            replies.append(
                fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 3_000_000_000)
            )
        except BaseException as error:
            errors.append(error)

    callers = [threading.Thread(target=end, name=f"test-end-retry-{index}") for index in range(2)]
    for caller in callers:
        caller.start()
    try:
        assert both_draining.wait(2), "End callers did not both wait with metadata lock released"
        assert replies == []
    finally:
        fixture.encoder.release.set()
        for caller in callers:
            caller.join(5)
    assert all(not caller.is_alive() for caller in callers)
    assert errors == [] and len(replies) == 2 and replies[0] == replies[1]
    assert replies[0]["acknowledged"] is True and replies[0]["quiescent"] is True
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0


def test_close_cannot_invent_quiescence_for_blocked_encoder(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session = _register(fixture, count=1, completed=1)
    fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    assert fixture.encoder.entered.wait(3)
    reply = fixture.worker.close(fixture.clock())
    assert reply["closed"] is False and reply["quiescent"] is False
    fixture.encoder.release.set()


def test_new_worker_epoch_cannot_accept_old_signed_result_as_current(make_worker):
    fixture = make_worker(epoch="old-worker-epoch")
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    receipt = _verify(fixture, session, frame, _result(fixture, session, key))
    restarted = make_worker(epoch="new-worker-epoch")
    with pytest.raises(ProtocolError, match="worker_epoch"):
        verify_feature(receipt, SIGNING_KEY, expected_worker_epoch=restarted.epoch)
    with pytest.raises((ValueError, RuntimeError)):
        restarted.worker.submit(frame, session.capability)


def test_trace_is_bounded_and_contains_no_authentication_material(make_worker):
    fixture = make_worker(config=WorkerConfig(max_trace=4))
    session = _register(fixture, count=2, completed=2)
    for step in range(2):
        frame = _frame(fixture, session, step)
        key = fixture.worker.submit(frame, session.capability)
        assert _result(fixture, session, key)["status"] == "ready"
        fixture.worker.ack(session.lease["session_id"], key, session.capability)
    trace = fixture.worker.info()["trace"]
    assert 0 < len(trace) <= 4
    serialized = repr(trace)
    assert session.capability not in serialized and SIGNING_KEY.hex() not in serialized
    assert key not in serialized and "request-0" not in serialized
    assert "mac_hex" not in serialized and "visual96_f32le_b64" not in serialized


def test_raw_trace_preserves_monotonic_job_and_end_lifecycle(make_worker):
    fixture = make_worker()
    session = _register(fixture, count=1, completed=1)
    frame = _frame(fixture, session, 0)
    key = fixture.worker.submit(frame, session.capability)
    _verify(fixture, session, frame, _result(fixture, session, key))
    fixture.worker.ack(session.lease["session_id"], key, session.capability)
    fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    trace = fixture.worker.info()["trace"]
    events = [row["event"] for row in trace]
    expected = ["READY", "DISPATCH", "ENCODE_START", "ENCODE_END", "RESULT_READY", "ACK", "END_REQUEST", "END_RESULT"]
    assert [events.index(event) for event in expected] == sorted(events.index(event) for event in expected)
    assert [row["at_ns"] for row in trace] == sorted(row["at_ns"] for row in trace)
    dispatch = next(row for row in trace if row["event"] == "DISPATCH")
    assert dispatch["reason"] == "B1" and dispatch["size"] == 1
    assert len({row["job"] for row in trace if "job" in row}) == 1


def _boundary_session(fixture, index=0):
    """Register without issuing a Worker Frame for the Policy-owned boundary."""
    session = SimpleNamespace(
        lease=_lease(index),
        capability=f"synthetic-boundary-capability-{index}-00000000",
        deadline=fixture.clock() + 30_000_000_000,
        chunk="chunk-0",
        start=0,
    )
    fixture.worker.owner_register(session.lease, session.capability, session.deadline)
    return session


def _boundary_fields(fixture, session, step=0):
    """Synthetic trusted-success metadata; not real Sim execution attestation."""
    rgb_sha = source_digest(bytes([step % 256]) * RGB_BYTES)
    raw_sha = hashlib.sha256(struct.pack("<15f", *([step / 100] * 15))).hexdigest()
    return {
        "lease": dict(session.lease),
        "chunk_id": f"chunk-{step}",
        "request_key": f"request-{step}",
        "source_step": step,
        "source_rgb_sha256": rgb_sha,
        "raw15_sha256": raw_sha,
        "policy_boundary_sha256": hashlib.sha256(struct.pack("<96f", *([step / 100] * 96))).hexdigest(),
        "post_step_receipt": {
            "version": "PSM24-POSTSTEP-v1",
            "lease": dict(session.lease),
            "issued_chunk_id": f"chunk-{step}",
            "policy_request_key": f"request-{step}",
            "source_step": step,
            "source_rgb_sha256": rgb_sha,
            "raw15_sha256": raw_sha,
            "success": True,
            "clock_domain": CLOCK_DOMAIN,
            "episode_deadline_ns": str(session.deadline),
            "step_completed_ns": str(fixture.clock()),
            "actor_id": "synthetic-owned-actor",
            "actor_epoch": "synthetic-actor-epoch",
            "trust_model": "registry_same_trust_domain",
        },
    }


def _frontiers(fixture, session):
    with fixture.worker._changed:
        state = fixture.worker._sessions[session.lease["session_id"]]
        return state.issued, state.completed, state.submitted, state.consumed


def _skip(fixture, fields):
    return fixture.worker.owner_boundary_skip(**fields)


def _encode_sparse_tail(fixture, session, boundary):
    session.chunk, session.start = f"chunk-{boundary}", boundary
    sid = session.lease["session_id"]
    fixture.worker.owner_issue(sid, session.chunk, f"request-{boundary}", boundary + 1, 15)
    fixture.worker.owner_complete(sid, session.chunk, boundary + 16)
    for step in range(boundary + 1, boundary + 16):
        frame = _frame(fixture, session, step)
        key = fixture.worker.submit(frame, session.capability)
        _verify(fixture, session, frame, _result(fixture, session, key))
        fixture.worker.ack(sid, key, session.capability)
    assert _frontiers(fixture, session) == (boundary + 16,) * 4


def test_sparse_boundary_preserves_real_steps_zero_through_thirty_one(make_worker):
    fixture = make_worker()
    session = _boundary_session(fixture)
    zero = _boundary_fields(fixture, session)
    assert _skip(fixture, zero) == {
        "boundary_skipped": True,
        "duplicate": False,
        "source_step": 0,
        "frontier": 1,
        "worker_feature": False,
        "encoded": False,
    }
    assert _frontiers(fixture, session) == (1,) * 4
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0
    assert fixture.encoder.calls == []
    assert _skip(fixture, copy.deepcopy(zero))["duplicate"] is True
    assert _frontiers(fixture, session) == (1,) * 4
    with pytest.raises(ValueError):
        fixture.worker.submit(_frame(fixture, session, 0), session.capability)
    _encode_sparse_tail(fixture, session, 0)
    assert _skip(fixture, zero)["duplicate"] is True
    assert _frontiers(fixture, session) == (16,) * 4
    sixteen = _boundary_fields(fixture, session, 16)
    assert _skip(fixture, sixteen)["frontier"] == 17
    assert _frontiers(fixture, session) == (17,) * 4
    _encode_sparse_tail(fixture, session, 16)
    encoded = [frame.header["source_step"] for batch in fixture.encoder.calls for frame in batch]
    assert encoded == list(range(1, 16)) + list(range(17, 32))
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0
    assert fixture.worker.info()["boundary_tombstones"] == 2
    assert fixture.worker.take_ready(session.lease["session_id"], session.capability) == {
        "status": "pending",
        "source_step": 32,
    }


@pytest.mark.parametrize("bmax", [4, 8])
def test_sparse_boundary_is_disabled_for_experimental_b4_b8(make_worker, bmax):
    fixture = make_worker(config=WorkerConfig(bmax=bmax), autostart=False)
    session = _boundary_session(fixture)
    with pytest.raises(ValueError, match="B1|zero wait"):
        _skip(fixture, _boundary_fields(fixture, session))
    assert _frontiers(fixture, session) == (0,) * 4 and fixture.encoder.calls == []


def test_sparse_boundary_metadata_creates_no_frame_feature_or_hmac(make_worker, monkeypatch):
    from cosmos_framework.inference import robocasa_vae_worker as worker_module

    def forbidden_sign(*args, **kwargs):
        raise AssertionError("Policy-owned boundary must not produce a Worker HMAC")

    monkeypatch.setattr(worker_module, "sign_feature", forbidden_sign)
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    result = _skip(fixture, fields)
    assert result["worker_feature"] is False and result["encoded"] is False
    assert not {"feature", "mac_hex", "visual96_f32le_b64", "job_key"} & result.keys()
    assert fixture.worker.info()["items"] == fixture.worker.info()["bytes"] == 0
    assert fixture.worker.info()["batch_histogram"] == {} and fixture.encoder.calls == []
    with pytest.raises(ValueError, match="unknown|evicted"):
        fixture.worker.poll(session.lease["session_id"], fields["policy_boundary_sha256"], session.capability)


def test_sparse_boundary_concurrent_exact_retry_advances_frontiers_once(make_worker):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    start = threading.Barrier(3)
    replies, errors = [], []

    def skip_once():
        try:
            start.wait(timeout=3)
            replies.append(_skip(fixture, copy.deepcopy(fields)))
        except BaseException as error:
            errors.append(error)

    callers = [threading.Thread(target=skip_once) for _ in range(2)]
    for caller in callers:
        caller.start()
    try:
        start.wait(timeout=3)
    finally:
        for caller in callers:
            caller.join(5)
        start.abort()
    assert all(not caller.is_alive() for caller in callers)
    assert errors == [] and len(replies) == 2
    assert sorted(reply["duplicate"] for reply in replies) == [False, True]
    assert all(reply["frontier"] == 1 and reply["encoded"] is False for reply in replies)
    assert _frontiers(fixture, session) == (1,) * 4
    info = fixture.worker.info()
    assert info["boundary_tombstones"] == 1 and info["queued"] == info["items"] == info["bytes"] == 0
    assert fixture.encoder.calls == []


@pytest.mark.parametrize("step", [True, 1, 15, 32])
def test_sparse_boundary_rejects_arbitrary_or_future_source_step(make_worker, step):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    with pytest.raises(REJECTED):
        _skip(fixture, _boundary_fields(fixture, session, step))
    assert _frontiers(fixture, session) == (0,) * 4
    assert fixture.worker.info()["boundary_tombstones"] == 0 and fixture.encoder.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "chunk_id",
        "request_key",
        "source_rgb_sha256",
        "raw15_sha256",
        "policy_boundary_sha256",
        "post_step_receipt",
        "post_step_time",
    ],
)
def test_sparse_boundary_exact_retry_rejects_changed_metadata(make_worker, field):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    original = _boundary_fields(fixture, session)
    _skip(fixture, original)
    changed = copy.deepcopy(original)
    if field == "post_step_receipt":
        changed[field]["actor_epoch"] = "different-valid-actor-epoch"
    elif field == "post_step_time":
        changed["post_step_receipt"]["step_completed_ns"] = str(
            int(original["post_step_receipt"]["step_completed_ns"]) - 1
        )
    else:
        changed[field] = "f" * 64
        receipt_field = {"chunk_id": "issued_chunk_id", "request_key": "policy_request_key"}.get(field, field)
        if receipt_field in changed["post_step_receipt"]:
            changed["post_step_receipt"][receipt_field] = changed[field]
    with pytest.raises(REJECTED):
        _skip(fixture, changed)
    assert _frontiers(fixture, session) == (1,) * 4
    assert fixture.worker.info()["boundary_tombstones"] == 1 and fixture.encoder.calls == []


def test_sparse_boundary_exact_retry_during_later_active_frame_does_not_rewind(make_worker):
    fixture = make_worker(encoder=_Encoder(blocked=True))
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    _skip(fixture, fields)
    fixture.worker.owner_issue(session.lease["session_id"], "chunk-0", "request-0", 1, 15)
    fixture.worker.owner_complete(session.lease["session_id"], "chunk-0", 2)
    frame = _frame(fixture, session, 1)
    key = fixture.worker.submit(frame, session.capability)
    assert fixture.encoder.entered.wait(3)
    before = _frontiers(fixture, session)
    charged = fixture.worker.info()["bytes"]
    assert _skip(fixture, copy.deepcopy(fields))["duplicate"] is True
    assert _frontiers(fixture, session) == before == (16, 2, 2, 1)
    assert fixture.worker.info()["active"] == 1 and fixture.worker.info()["bytes"] == charged
    assert sum(map(len, fixture.encoder.calls)) == 1
    fixture.encoder.release.set()
    _verify(fixture, session, frame, _result(fixture, session, key))


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "empty",
        "failed",
        "missing_success",
        "wrong_step",
        "foreign_session",
        "wrong_clock",
        "changed_deadline",
        "future_time",
        "extra",
        "wrong_trust",
        "bool_step",
    ],
)
def test_sparse_boundary_requires_exact_successful_post_step_receipt(make_worker, invalid):
    fixture = make_worker(autostart=False, clock=_Clock())
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    receipt = fields["post_step_receipt"]
    if invalid == "missing":
        fields["post_step_receipt"] = None
    elif invalid == "empty":
        fields["post_step_receipt"] = {}
    elif invalid == "failed":
        receipt["success"] = False
    elif invalid == "missing_success":
        del receipt["success"]
    elif invalid == "wrong_step":
        receipt["source_step"] = 1
    elif invalid == "foreign_session":
        receipt["lease"]["session_id"] = "foreign-session"
    elif invalid == "wrong_clock":
        receipt["clock_domain"] = "foreign-clock"
    elif invalid == "changed_deadline":
        receipt["episode_deadline_ns"] = str(session.deadline + 1)
    elif invalid == "future_time":
        receipt["step_completed_ns"] = str(fixture.clock() + 1)
    elif invalid == "extra":
        receipt["unexpected"] = True
    elif invalid == "wrong_trust":
        receipt["trust_model"] = "unattested-public-client"
    else:
        receipt["source_step"] = True
    with pytest.raises(REJECTED):
        _skip(fixture, fields)
    assert _frontiers(fixture, session) == (0,) * 4
    assert fixture.worker.info()["boundary_tombstones"] == fixture.worker.info()["items"] == 0
    assert fixture.encoder.calls == []


@pytest.mark.parametrize("field", ["source_rgb_sha256", "raw15_sha256", "policy_boundary_sha256"])
def test_sparse_boundary_rejects_malformed_digest_before_frontier_update(make_worker, field):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    fields[field] = "G" * 64
    with pytest.raises(REJECTED):
        _skip(fixture, fields)
    assert _frontiers(fixture, session) == (0,) * 4 and fixture.encoder.calls == []


@pytest.mark.parametrize("field", ["server_epoch", "generation", "session_id", "episode_id"])
def test_sparse_boundary_rejects_foreign_lease_and_epoch(make_worker, field):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    fields = _boundary_fields(fixture, session)
    fields["lease"][field] = 2 if field == "generation" else "foreign-identity"
    fields["post_step_receipt"]["lease"] = dict(fields["lease"])
    with pytest.raises(REJECTED):
        _skip(fixture, fields)
    assert _frontiers(fixture, session) == (0,) * 4 and fixture.encoder.calls == []


@pytest.mark.parametrize("state", ["issued", "completed", "queued", "active", "ready_unacked"])
def test_sparse_boundary_cannot_bypass_frontiers_or_owned_jobs(make_worker, state):
    fixture = make_worker(encoder=_Encoder(blocked=state == "active"), autostart=state in ("active", "ready_unacked"))
    session = _register(fixture, count=1, completed=None if state == "issued" else 1)
    if state in ("queued", "active", "ready_unacked"):
        key = fixture.worker.submit(_frame(fixture, session, 0), session.capability)
        if state == "active":
            assert fixture.encoder.entered.wait(3)
        elif state == "ready_unacked":
            assert _result(fixture, session, key)["status"] == "ready"
    before, calls = _frontiers(fixture, session), len(fixture.encoder.calls)
    with pytest.raises(REJECTED):
        _skip(fixture, _boundary_fields(fixture, session))
    assert _frontiers(fixture, session) == before and len(fixture.encoder.calls) == calls
    assert fixture.worker.info()["boundary_tombstones"] == 0


def test_sparse_boundary_does_not_mask_missing_last_completed_action(make_worker):
    fixture = make_worker()
    session = _boundary_session(fixture)
    _skip(fixture, _boundary_fields(fixture, session))
    fixture.worker.owner_issue(session.lease["session_id"], "chunk-0", "request-0", 1, 15)
    fixture.worker.owner_complete(session.lease["session_id"], "chunk-0", 15)
    for step in range(1, 15):
        frame = _frame(fixture, session, step)
        key = fixture.worker.submit(frame, session.capability)
        assert _result(fixture, session, key)["status"] == "ready"
        fixture.worker.ack(session.lease["session_id"], key, session.capability)
    assert _frontiers(fixture, session) == (16, 15, 15, 15)
    with pytest.raises(REJECTED):
        _skip(fixture, _boundary_fields(fixture, session, 16))
    assert _frontiers(fixture, session) == (16, 15, 15, 15)
    assert sum(map(len, fixture.encoder.calls)) == 14


@pytest.mark.parametrize("change", ["chunk", "request", "count16"])
def test_sparse_boundary_next_issue_keeps_same_chunk_request_and_h15(make_worker, change):
    fixture = make_worker(autostart=False)
    session = _boundary_session(fixture)
    _skip(fixture, _boundary_fields(fixture, session))
    with pytest.raises(REJECTED):
        fixture.worker.owner_issue(
            session.lease["session_id"],
            "wrong" if change == "chunk" else "chunk-0",
            "wrong" if change == "request" else "request-0",
            1,
            16 if change == "count16" else 15,
        )
    assert _frontiers(fixture, session) == (1,) * 4
    fixture.worker.owner_issue(session.lease["session_id"], "chunk-0", "request-0", 1, 15)
    assert _frontiers(fixture, session) == (16, 1, 1, 1)


def test_sparse_boundary_tombstone_bound_cannot_revive_evicted_source(make_worker):
    fixture = make_worker(config=WorkerConfig(max_tombstones=1))
    session = _boundary_session(fixture)
    first = _boundary_fields(fixture, session)
    _skip(fixture, first)
    _encode_sparse_tail(fixture, session, 0)
    current = _boundary_fields(fixture, session, 16)
    _skip(fixture, current)
    assert fixture.worker.info()["boundary_tombstones"] == 1
    with pytest.raises(REJECTED):
        _skip(fixture, first)
    assert _skip(fixture, current)["duplicate"] is True
    assert _frontiers(fixture, session) == (17,) * 4
    assert sum(map(len, fixture.encoder.calls)) == 15


def test_sparse_boundary_end_revokes_retry_without_affecting_peer(make_worker):
    fixture = make_worker()
    session = _boundary_session(fixture)
    peer = _register(fixture, 1, count=1, completed=1)
    fields = _boundary_fields(fixture, session)
    _skip(fixture, fields)
    assert _frontiers(fixture, peer) == (1, 1, 0, 0)
    fixture.worker.end(session.lease["session_id"], session.capability, fixture.clock() + 1_000_000_000)
    with pytest.raises(REJECTED):
        _skip(fixture, fields)
    frame = _frame(fixture, peer, 0)
    key = fixture.worker.submit(frame, peer.capability)
    _verify(fixture, peer, frame, _result(fixture, peer, key))
    assert [frame.header["lease"]["session_id"] for batch in fixture.encoder.calls for frame in batch] == [
        peer.lease["session_id"]
    ]

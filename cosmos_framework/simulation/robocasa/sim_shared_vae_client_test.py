"""C route CPU contracts over the actual native loop, immutable Frame and receipts.

The small injected Worker/Owner doubles make no model, GPU, or security-isolation
claim. The real Root relay/Worker IPC is exercised in its own integration tests.
"""

from __future__ import annotations

import copy
import hashlib
import struct
import threading
import time

import numpy as np
import pytest

from cosmos_framework.inference.robocasa_feature_evidence import POST_STEP_VERSION, PolicyBoundary
from cosmos_framework.inference.robocasa_feature_trace import TraceRecorder
from cosmos_framework.inference.robocasa_vae_client import LocalVAEClient
from cosmos_framework.inference.robocasa_vae_protocol import Frame, sign_feature, source_digest
from cosmos_framework.simulation.robocasa import closed_loop_eval as native
from cosmos_framework.simulation.robocasa.eval_utils import canonicalize_raw15_for_env
from cosmos_framework.simulation.robocasa.sim_shared_vae_client import ScopedActorCompletionChannel, SimSharedVAEClient


def _lease():
    return dict(
        server_epoch="policy",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_id="slot",
        generation=1,
        task="CloseFridge",
        episode_index=0,
        attempt_id="attempt",
        session_id="session",
        episode_id="episode",
    )


def _identity():
    return dict(
        vae_weights_sha256="b" * 64,
        vae_config_sha256="c" * 64,
        preprocess_profile_sha256="d" * 64,
        visual96_contract_sha256="e" * 64,
        encoder_backend="eager",
        encoder_dtype="bfloat16",
    )


def _descriptor():
    return dict(
        identity=_identity(),
        clock_domain="clock",
        worker_id="worker",
        worker_epoch="worker-epoch",
        key_id="key",
        actor_id="actor",
        actor_epoch="actor-epoch",
        trust_model="registry_same_trust_domain",
    )


def _image(step):
    return np.full((256, 512, 3), step % 256, dtype=np.uint8)


def _action():
    return [0.2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5]


def _raw():
    return canonicalize_raw15_for_env(np.asarray(_action()), False)[1]


class _Worker(LocalVAEClient):
    def __init__(self, *, pending=False, wrong_epoch=False):
        self.frames, self.features, self.acked = [], {}, []
        self.trace_rows = []
        self.pending, self.wrong_epoch = pending, wrong_epoch

    def submit(self, frame, deadline_ns):
        assert isinstance(frame, Frame) and frame.header["source_step"] % 16
        assert deadline_ns <= int(frame.header["episode_deadline_ns"])
        self.frames.append(frame)
        self.trace_rows.append(
            {
                "event": "SYNTHETIC_FRAME_ADMITTED",
                "at_ns": time.monotonic_ns(),
                "source_step": frame.header["source_step"],
            }
        )
        key = frame.fingerprint
        self.features[key] = sign_feature(
            frame,
            struct.pack("<96f", *([0.25] * 96)),
            b"S" * 32,
            worker_id="worker",
            worker_epoch="stale" if self.wrong_epoch else "worker-epoch",
            key_id="key",
        )
        return key

    def poll(self, key, deadline_ns):
        return (
            {"status": "pending", "job_key": key}
            if self.pending
            else {"status": "ready", "job_key": key, "feature": copy.deepcopy(self.features[key])}
        )

    def ack(self, key, deadline_ns):
        self.acked.append(key)
        return {"acknowledged": True, "job_key": key}

    def info(self, deadline_ns):
        return {"trace": copy.deepcopy(self.trace_rows), "trace_overflow": 0}


class _Channel(ScopedActorCompletionChannel):
    def __init__(self, deadline, *, corrupt=None):
        self.episode_deadline, self.corrupt = deadline, corrupt
        self.chunks, self.completed, self.ends = [], [], 0
        self.env = None

    def chunk(self, chunk_id, request_key, payload_sha256, source_start, count, deadline_ns):
        assert chunk_id == request_key and count == 16
        self.chunks.append((chunk_id, source_start, count))
        return {"authorized": True}

    def post_step(self, chunk_id, request_key, source_step, source_rgb_sha256, raw15, completed_ns, deadline_ns):
        if self.env is not None:
            assert self.env.steps == source_step + 1, "completion callback preceded actual successful env.step"
        assert self.chunks[-1][1] <= source_step < self.chunks[-1][1] + 16
        assert source_step == len(self.completed)
        assert raw15 == _raw().astype("<f4").tobytes()
        receipt = dict(
            version=POST_STEP_VERSION,
            lease=_lease(),
            issued_chunk_id=chunk_id,
            policy_request_key=request_key,
            source_step=source_step,
            source_rgb_sha256=source_rgb_sha256,
            raw15_sha256=hashlib.sha256(raw15).hexdigest(),
            success=True,
            clock_domain="clock",
            episode_deadline_ns=str(self.episode_deadline),
            step_completed_ns=str(completed_ns),
            actor_id="actor",
            actor_epoch="actor-epoch",
            trust_model="registry_same_trust_domain",
        )
        self.completed.append(copy.deepcopy(receipt))
        if self.corrupt is not None:
            receipt[self.corrupt] = "wrong" if self.corrupt in ("actor_epoch", "clock_domain") else "f" * 64
        return receipt

    def end(self, deadline_ns):
        assert deadline_ns > time.monotonic_ns()
        self.ends += 1
        return {"acknowledged": True, "quiescent": True, "error": None}


def _client(*, timeout=2, pending=False, wrong_epoch=False, corrupt=None):
    deadline = time.monotonic_ns() + 10_000_000_000
    worker, channel = _Worker(pending=pending, wrong_epoch=wrong_epoch), _Channel(deadline, corrupt=corrupt)
    client = SimSharedVAEClient(
        _lease(), _descriptor(), deadline, timeout, "clock", worker_client=worker, completion_channel=channel
    )
    return client, worker, channel


@pytest.mark.parametrize("timeout", [5000, 6000])
def test_shared_VAE_accepts_full_existing_Root_prediction_budget_range(timeout):
    client, worker, _ = _client(timeout=timeout)
    assert client.predict_timeout_ns == timeout * 1_000_000_000 and not worker.frames


def test_shared_VAE_rejects_prediction_budget_above_existing_Root_bound():
    with pytest.raises(ValueError, match="budgets"):
        _client(timeout=6001)


def _result(client, step, *, image=None):
    request = f"{step:064x}"
    boundary = PolicyBoundary.create(
        lease=_lease(),
        request_key=request,
        chunk_id=request,
        source_step=step,
        source_rgb_sha256=source_digest((_image(step) if image is None else image).tobytes()),
        identity=_identity(),
        deadline_ns=int(client.prediction_budget["deadline"] * 1e9),
        clock_domain="clock",
        visual96=struct.pack("<96f", *([0.25] * 96)),
    )
    status = dict(
        session_id="session",
        episode_id="episode",
        consumer_step=step,
        prefix_present=bool(step),
        committed_boundary=boundary.record(),
    )
    return dict(
        local_memory=status,
        request_key=request,
        payload_sha256="a" * 64,
        assignment=_lease(),
        action=[_action() for _ in range(16)],
    )


def _issue(client, step):
    client.start_replan()
    payload = client.payload()
    result = _result(client, step)
    client.accept_policy_result(result)
    client.acknowledge(0, result["local_memory"])
    return payload


def _complete(client, step, *, image=None, completed_ns=None):
    prepared = client.prepare_step(0, step, _image(step) if image is None else image, _raw())
    completed_ns = time.monotonic_ns() if completed_ns is None else completed_ns
    client.complete_step(0, prepared, completed_ns)
    return prepared, completed_ns


def test_two_real_source_chunks_have_15_plus_1_no_history_PNG_and_ACK_owned_retention():
    client, worker, channel = _client()
    client.begin()
    try:
        assert _issue(client, 0)["evidence"] == []
        for start in (0, 16):
            for step in range(start, start + 16):
                _complete(client, step)
            payload = client.payload()
            assert payload["consumer_step"] == start + 16
            assert [row["source_step"] for row in payload["evidence"]] == list(range(start, start + 16))
            assert payload["evidence"][0]["feature"] is None
            assert all(row["feature"] is not None for row in payload["evidence"][1:])
            assert all(set(row) == {"source_step", "post_step_receipt", "feature"} for row in payload["evidence"])
            assert all("composite_image" not in row for row in payload["evidence"])
            assert client.payload() == payload and client.diagnostics()["pending"] == 16
            _issue(client, start + 16)
            assert client.diagnostics()["pending"] == 0
        assert [frame.header["source_step"] for frame in worker.frames] == [*range(1, 16), *range(17, 32)]
        assert len(worker.acked) == 30 and len(channel.completed) == 32
    finally:
        client.end()
    assert channel.ends == 1 and not client.diagnostics()["worker_alive"]


def test_pre_action_RGB_and_canonical_raw15_are_owned_before_env_mutation():
    client, _, channel = _client()
    client.begin()
    try:
        _issue(client, 0)
        image, action = _image(0), _raw()
        prepared = client.prepare_step(0, 0, image, action)
        image[:] = 99
        action[:] = -999
        client.complete_step(0, prepared, time.monotonic_ns())
        assert channel.completed[0]["source_rgb_sha256"] == source_digest(_image(0).tobytes())
        assert prepared.raw15 == _raw().astype("<f4").tobytes()
    finally:
        client.end()


def test_attached_trace_records_background_IPC_and_deduplicates_Worker_ring():
    client, _, _ = _client()
    trace = TraceRecorder(clock_domain="clock")
    client.attach_trace(trace)
    client.begin()
    try:
        _issue(client, 0)
        _complete(client, 0)
        _complete(client, 1)
        client.payload()
        client.payload()
        events = trace.snapshot()["events"]
        assert sum(row["producer"] == "worker" for row in events) == 1
        assert any(
            row["phase"] == "worker_submit_ipc" and row["thread_id"] != threading.get_native_id() for row in events
        )
        assert any(row["phase"] == "worker_trace_ipc" for row in events)
    finally:
        client.end()


def test_last_success_starts_original_budget_before_feature_flush_and_never_refreshes():
    client, _, _ = _client()
    client.begin()
    try:
        _issue(client, 0)
        for step in range(15):
            _complete(client, step)
        _, completed_ns = _complete(client, 15)
        original = client.prediction_budget
        assert original["started_at"] == completed_ns / 1e9
        assert original["deadline"] <= (completed_ns + 2_000_000_000) / 1e9
        client.start_replan(0, completed_ns + 1_000_000_000)
        client.payload()
        assert client.prediction_budget == original
    finally:
        client.end()


@pytest.mark.parametrize("field", ["actor_epoch", "clock_domain", "source_rgb_sha256", "raw15_sha256"])
def test_changed_scoped_completion_cannot_advance_consumer_or_submit_Frame(field):
    client, worker, channel = _client(corrupt=field)
    client.begin()
    _issue(client, 0)
    prepared = client.prepare_step(0, 0, _image(0), _raw())
    with pytest.raises(ValueError, match="Owner completion") as result:
        client.complete_step(0, prepared, time.monotonic_ns())
    assert client._get(0).consumer_step == 0 and not worker.frames
    with pytest.raises(ValueError) as cleanup:
        client.end()
    assert cleanup.value is result.value and channel.ends == 1


def test_wrong_boundary_RGB_is_rejected_before_environment_action():
    client, worker, channel = _client()
    client.begin()
    try:
        _issue(client, 0)
        with pytest.raises(ValueError, match="boundary RGB"):
            client.prepare_step(0, 0, _image(1), _raw())
        assert not channel.completed and not worker.frames
    finally:
        client.end()


def test_missing_boundary_row_cannot_be_relabelled_a_valid_partial_history():
    client, _, _ = _client()
    client.begin()
    try:
        _issue(client, 0)
        _complete(client, 0)
        _complete(client, 1)
        with client._changed:
            del client._rows[0]
        with pytest.raises(ValueError, match="each actual completed"):
            client.payload()
    finally:
        client.end()


def test_stale_Worker_epoch_fails_before_Policy_payload_and_preserves_first_error():
    client, _, channel = _client(wrong_epoch=True)
    client.begin()
    _issue(client, 0)
    _complete(client, 0)
    _complete(client, 1)
    with pytest.raises(ValueError, match="Worker feature") as result:
        client.payload()
    with pytest.raises(ValueError) as cleanup:
        client.end()
    assert cleanup.value is result.value and channel.ends == 1


def test_pending_features_expire_original_budget_without_hidden_server_VAE_fallback():
    client, worker, _ = _client(timeout=0.03, pending=True)
    client.begin()
    try:
        _issue(client, 0)
        _complete(client, 0)
        _complete(client, 1)
        with pytest.raises(TimeoutError, match="original shared VAE"):
            client.start_replan(0, time.monotonic_ns() - 100_000_000)
        with pytest.raises(TimeoutError, match="original shared VAE"):
            client.payload()
        assert all(isinstance(frame, Frame) for frame in worker.frames)
    finally:
        client.end()


class _Env:
    def __init__(self, *, fail_at=None):
        self.steps, self.fail_at = 0, fail_at

    def observation(self):
        return {f"{camera}_image": np.full((256, 256, 3), self.steps, dtype=np.uint8) for camera in native.CAMS}

    def reset(self):
        self.steps = 0
        return self.observation()

    def step(self, action):
        if self.steps == self.fail_at:
            raise RuntimeError("actual env.step sentinel")
        self.steps += 1
        return self.observation(), 0.0, False, {}


def _native(monkeypatch, *, target, fail_at=None):
    client, worker, channel = _client()
    env = _Env(fail_at=fail_at)
    channel.env = env
    payloads = []
    monkeypatch.setattr(native, "CAMERA_SET", "left_wrist")
    monkeypatch.setattr(native, "USE_BASE_ACTION", True)
    monkeypatch.setattr(native, "BASE_ENCODING", "raw")
    monkeypatch.setattr(native, "build_state_token", lambda obs: [0.0] * 15)
    monkeypatch.setattr(native, "reset_local_memory", lambda *args, **kwargs: None)

    def predict(*args, local_memory, **kwargs):
        payloads.append(copy.deepcopy(local_memory))
        result = _result(client, local_memory["consumer_step"], image=args[1])
        client.accept_policy_result(result)
        return result

    monkeypatch.setattr(native, "predict", predict)
    return (
        client,
        worker,
        channel,
        env,
        payloads,
        dict(
            server_url="http://cpu",
            image_size=256,
            action_horizon=16,
            max_steps=target,
            latch=1,
            timeout=2,
            use_state=True,
            local_memory_client=client,
            success_check=lambda value: value.steps >= target,
            language_instruction="Close the fridge",
        ),
    )


@pytest.mark.parametrize("target", [1, 4, 16, 32])
def test_actual_native_loop_success_only_completes_executed_steps_and_partial_End(monkeypatch, target):
    client, worker, channel, env, payloads, kwargs = _native(monkeypatch, target=target)
    result = native.run_policy(env, **kwargs)
    assert result[:2] == (True, target)
    assert [row["source_step"] for row in channel.completed] == list(range(target))
    assert [payload["consumer_step"] for payload in payloads] == list(range(0, target, 16))
    assert all(frame.header["source_step"] % 16 for frame in worker.frames)
    assert channel.ends == 1 and not client.diagnostics()["worker_alive"]


@pytest.mark.parametrize("fail_at", [0, 3, 16])
def test_actual_native_env_failure_never_publishes_that_step_or_retries_action(monkeypatch, fail_at):
    client, worker, channel, env, _, kwargs = _native(monkeypatch, target=32, fail_at=fail_at)
    with pytest.raises(RuntimeError, match="actual env.step sentinel"):
        native.run_policy(env, **kwargs)
    assert env.steps == fail_at
    assert [row["source_step"] for row in channel.completed] == list(range(fail_at))
    assert all(frame.header["source_step"] < fail_at for frame in worker.frames)
    assert channel.ends == 1 and not client.diagnostics()["worker_alive"]

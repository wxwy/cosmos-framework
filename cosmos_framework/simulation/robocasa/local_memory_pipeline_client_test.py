"""CPU thread/transport contracts for completed-frame admission, never prediction."""

from __future__ import annotations

import base64
import copy
import io
import json
import threading
import time

import numpy as np
import pytest
import requests
from PIL import Image

from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.inference.robocasa_session_transactions import monotonic_clock_domain, protocol_for_mode
from cosmos_framework.simulation.robocasa.eval_v2_transactions import AssignedLocalMemoryClient, PolicyTransport
from cosmos_framework.simulation.robocasa.local_memory_pipeline_client import CompletedEvidencePipeline


def _lease(session="session"):
    return dict(
        server_epoch="epoch",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_id=session,
        generation=1,
        task="CloseFridge",
        episode_index=0,
        attempt_id="attempt",
        session_id=session,
        episode_id="episode-" + session,
        capability="test-only-capability",
    )


def _image(value=7):
    return np.full((256, 512, 3), value, dtype=np.uint8)


class _Reply:
    def __init__(self, value, status=202):
        self.status_code, self.value = status, value

    def json(self):
        return copy.deepcopy(self.value)


def _accepted(body, *, duplicate=False):
    payload = body["payload"]
    return dict(
        accepted=True,
        session_id=payload["session_id"],
        episode_id=payload["episode_id"],
        source_step=payload["source_step"],
        duplicate=duplicate,
    )


class _HTTP:
    def __init__(self):
        self.calls, self.closed, self.trust_env = [], False, True

    def request(self, method, url, **kwargs):
        assert not self.closed
        self.calls.append((method, url, copy.deepcopy(kwargs)))
        if url.endswith("/v2/session/end"):
            return _Reply(
                {"acknowledged": True, **{k: v for k, v in kwargs["json"]["lease"].items() if k != "capability"}}, 200
            )
        assert url.endswith("/v2/local-memory/completed")
        return _Reply(_accepted(json.loads(kwargs["data"])))

    def close(self):
        self.closed = True


def _send(http, raw, timeout):
    return http.request("POST", "http://cpu/v2/local-memory/completed", data=raw, timeout=timeout)


def _pipeline(*, sender=_send, encode=lambda image: "encoded", max_pending=16, deadline=None, grace=1):
    http = _HTTP()
    pipeline = CompletedEvidencePipeline(
        _lease(),
        time.monotonic() + 5 if deadline is None else deadline,
        sender,
        session_factory=lambda: http,
        encode=encode,
        max_pending=max_pending,
        shutdown_timeout_s=grace,
    )
    return pipeline, http


def _submit(pipeline, source=0, image=None, action=None):
    pipeline.submit(
        source,
        _image() if image is None else image,
        [0.1] * 15 if action is None else action,
        image_size=256,
        preprocess_profile=PREPROCESS_PROFILE,
    )


def _status(client, *, pipeline="per_frame_v1", consumer=None):
    episode = client._get(0)
    value = dict(
        session_id=episode.session_id,
        episode_id=episode.episode_id,
        consumer_step=episode.consumer_step if consumer is None else consumer,
    )
    if pipeline is not None:
        value["pipeline"] = pipeline
    return value


def test_background_server_cause_survives_completed_upload_failure():
    def failed(http, raw, timeout):
        return _Reply({"error": "Encode1 primary failure at source_step=3"}, status=409)

    pipeline, _ = _pipeline(sender=failed)
    _submit(pipeline)
    try:
        with pytest.raises(RuntimeError, match="Encode1 primary failure"):
            pipeline.wait(1)
    finally:
        with pytest.raises(RuntimeError, match="Encode1 primary failure"):
            pipeline.close()


def _assigned(*, encode=lambda image: "encoded", sender=_send, session="session", deadline=None, grace=1):
    http = _HTTP()
    client = AssignedLocalMemoryClient(
        _lease(session),
        sender=sender,
        session_factory=lambda: http,
        encode=encode,
        episode_deadline=time.monotonic() + 5 if deadline is None else deadline,
        shutdown_timeout_s=grace,
    )
    client.begin()
    client.acknowledge(0, _status(client))
    return client, http


def test_record_does_not_wait_for_png_or_upload_and_copies_inputs():
    entered, release = threading.Event(), threading.Event()
    seen = []

    def encode(image):
        entered.set()
        assert release.wait(3)
        seen.append(image.copy())
        return "png-before-mutation"

    client, http = _assigned(encode=encode)
    image, action = _image(9), np.arange(15, dtype=np.float32)
    try:
        client.record_completed(0, image, action)
        assert entered.wait(3)
        # The caller returned while the PNG encoder is still blocked.
        assert client._get(0).consumer_step == 1 and not http.calls
        image[:] = 99
        action[:] = -1
        release.set()
        payload = client.payload()
        assert np.all(seen[0] == 9)
        assert payload["evidence"][0]["executed_action"] == list(range(15))
        assert payload["evidence"][0]["composite_image"] == "png-before-mutation"
        body = json.loads(http.calls[0][2]["data"])
        assert body["lease"] == _lease()
        assert set(body["payload"]) == {
            "session_id",
            "episode_id",
            "source_step",
            "image_size",
            "preprocess_profile",
            "composite_image",
            "executed_action",
        }
        assert body["payload"]["source_step"] == 0
        # Admission is not a policy ACK: original completed evidence is retained.
        assert len(client._get(0).evidence) == client.pipeline_stats()["pending"] == 1
        assert client.payload() == payload
        with pytest.raises(ValueError, match="exact completed-evidence"):
            client.acknowledge(0, _status(client, consumer=0))
        assert client.payload() == payload
        client.acknowledge(0, _status(client))
        assert client.payload()["evidence"] == [] and client.pipeline_stats()["pending"] == 0
    finally:
        release.set()
        client.close_pipeline()
    assert http.closed and not client.pipeline_stats()["worker_alive"]


def test_nonblocking_capacity_includes_admitted_unacknowledged_rows():
    pipeline, http = _pipeline(max_pending=1)
    try:
        _submit(pipeline)
        assert len(pipeline.wait(1)) == 1
        with pytest.raises(RuntimeError, match="capacity"):
            _submit(pipeline, 1)
        assert pipeline.stats()["enqueued"] == 1 and pipeline.stats()["queue_full"] == 1
        pipeline.acknowledge(1)
        _submit(pipeline, 1)
        assert pipeline.wait(2)[0]["source_step"] == 1
    finally:
        pipeline.close()
    assert http.closed


def test_same_source_retry_is_idempotent_changed_bytes_and_late_sources_fail():
    pipeline, http = _pipeline()
    try:
        _submit(pipeline)
        pipeline.wait(1)
        _submit(pipeline)
        assert len(http.calls) == pipeline.stats()["enqueued"] == 1
        with pytest.raises(ValueError, match="changed bytes"):
            _submit(pipeline, image=_image(8))
        with pytest.raises(ValueError, match="contiguous"):
            _submit(pipeline, 2)
        pipeline.acknowledge(1)
        with pytest.raises(ValueError, match="late"):
            _submit(pipeline, 0)
    finally:
        pipeline.close()


@pytest.mark.parametrize("failure", [requests.Timeout, requests.ConnectionError])
def test_network_retry_reuses_same_frozen_bytes_and_original_budget(failure):
    attempts = []

    def sender(http, raw, timeout):
        attempts.append((raw, timeout))
        if len(attempts) == 1:
            raise failure("lost admission ACK")
        return _Reply(_accepted(json.loads(raw), duplicate=True), 200)

    pipeline, _ = _pipeline(sender=sender)
    try:
        _submit(pipeline)
        assert len(pipeline.wait(1)) == 1
        assert len(attempts) == 2 and attempts[0][0] is attempts[1][0]
        assert all(0 < timeout <= 1 for _, timeout in attempts)
        assert pipeline.stats()["retries"] == 1
    finally:
        pipeline.close()


@pytest.mark.parametrize("code", [409, 429, 500, 503])
def test_http_rejection_is_fatal_without_retry_or_fallback(code):
    attempts = []

    def sender(http, raw, timeout):
        attempts.append(raw)
        return _Reply({"accepted": False}, code)

    pipeline, http = _pipeline(sender=sender)
    try:
        _submit(pipeline)
        with pytest.raises(RuntimeError, match="admission rejected"):
            pipeline.wait(1)
        assert len(attempts) == 1 and pipeline.stats()["uploaded"] == 0
        with pytest.raises(RuntimeError, match="worker failed"):
            _submit(pipeline, 1)
    finally:
        with pytest.raises(RuntimeError, match="worker failed"):
            pipeline.close()
    assert http.closed and not pipeline.stats()["worker_alive"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted", False),
        ("session_id", "foreign"),
        ("episode_id", "foreign"),
        ("source_step", True),
        ("source_step", 1),
        ("duplicate", None),
    ],
)
def test_admission_ack_requires_exact_binding(field, value):
    def sender(http, raw, timeout):
        ack = _accepted(json.loads(raw))
        ack[field] = value
        return _Reply(ack)

    pipeline, _ = _pipeline(sender=sender)
    _submit(pipeline)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        pipeline.wait(1)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        pipeline.close()


def test_background_png_failure_propagates_before_replan_and_cleanup():
    def fail(image):
        raise ValueError("PNG sentinel")

    client, http = _assigned(encode=fail)
    client.record_completed(0, _image(), [0.0] * 15)
    with pytest.raises(RuntimeError, match="PNG sentinel"):
        client.payload()
    assert not http.calls
    with pytest.raises(RuntimeError, match="PNG sentinel"):
        client.close_pipeline()
    assert http.closed and not client.pipeline_stats()["worker_alive"]


def test_late_response_cannot_admit_after_original_deadline():
    def sender(http, raw, timeout):
        pipeline.deadline = time.monotonic() - 1
        return _Reply(_accepted(json.loads(raw)))

    pipeline, _ = _pipeline(sender=sender)
    _submit(pipeline)
    with pytest.raises((RuntimeError, TimeoutError), match="late|deadline"):
        pipeline.wait(1)
    pipeline._thread.join(3)
    assert pipeline.stats()["uploaded"] == 0
    with pytest.raises(RuntimeError, match="late"):
        pipeline.close()


def test_expired_deadline_creates_no_worker():
    pipeline, http = _pipeline(deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError, match="original Episode"):
        _submit(pipeline)
    pipeline.close()
    assert pipeline._thread is None and not http.closed


def test_end_cancels_queued_tail_and_never_uploads_after_worker_fence():
    entered, release = threading.Event(), threading.Event()

    def encode(image):
        entered.set()
        assert release.wait(3)
        return "encoded"

    pipeline, http = _pipeline(encode=encode, grace=0.01)
    try:
        _submit(pipeline)
        assert entered.wait(3)
        _submit(pipeline, 1)
        with pytest.raises(TimeoutError, match="did not quiesce"):
            pipeline.close()
        assert pipeline.stats()["queued"] == pipeline.stats()["pending"] == 0
        assert pipeline.stats()["worker_alive"] and not http.closed
        release.set()
        pipeline._thread.join(3)
        pipeline.close()
        assert http.closed and not http.calls and not pipeline.stats()["worker_alive"]
        with pytest.raises(RuntimeError, match="closed"):
            _submit(pipeline, 2)
    finally:
        release.set()
        pipeline._thread.join(3)


@pytest.mark.parametrize("flag", [None, "different"])
def test_enabled_capability_cannot_silently_disappear(flag):
    client, _ = _assigned()
    try:
        client.record_completed(0, _image(), [0.0] * 15)
        original = client.payload()
        with pytest.raises(ValueError, match="capability changed"):
            client.acknowledge(0, _status(client, pipeline=flag))
        assert client.payload() == original and client.pipeline_stats()["pending"] == 1
    finally:
        client.close_pipeline()


def test_legacy_without_initial_capability_stays_synchronous_and_creates_no_worker():
    def forbidden():
        pytest.fail("legacy client created a completed-evidence HTTP session")

    client = AssignedLocalMemoryClient(_lease(), session_factory=forbidden)
    client.begin()
    client.acknowledge(0, _status(client, pipeline=None))
    client.record_completed(0, _image(), [0.0] * 15)
    payload = client.payload()
    assert len(payload["evidence"]) == 1 and client.pipeline_stats() == {"enabled": False}
    with pytest.raises(ValueError, match="capability changed"):
        client.acknowledge(0, _status(client))
    client.acknowledge(0, _status(client, pipeline=None))
    client.end()


@pytest.mark.parametrize("length", [1, 4, 16])
def test_payload_freezes_completed_frontier_and_does_not_invent_unexecuted_tail(length):
    client, _ = _assigned()
    try:
        for step in range(length):
            client.record_completed(0, _image(step), [float(step)] * 15)
        payload = client.payload()
        assert payload["consumer_step"] == length
        assert [row["source_step"] for row in payload["evidence"]] == list(range(length))
        with pytest.raises(RuntimeError, match="frozen prediction"):
            client.record_completed(0, _image(length), [float(length)] * 15)
        assert client._get(0).consumer_step == length
    finally:
        client.close_pipeline()


def test_multiple_sessions_have_independent_workers_http_and_ledgers():
    first, one = _assigned(session="one")
    second, two = _assigned(session="two")
    try:
        for client in (first, second):
            client.record_completed(0, _image(), [0.0] * 15)
        for client, http, session in ((first, one, "one"), (second, two, "two")):
            assert client.payload()["session_id"] == session
            assert json.loads(http.calls[0][2]["data"])["lease"]["session_id"] == session
            assert http.trust_env is False
        first.acknowledge(0, _status(first))
        assert first.pipeline_stats()["pending"] == 0 and second.pipeline_stats()["pending"] == 1
    finally:
        first.close_pipeline()
        second.close_pipeline()


def _transport(*, mode="required", completed_encode=None, completed_session_factory=None, grace=1):
    policy, media, upload = _HTTP(), _HTTP(), _HTTP()
    settings = dict(
        server_url="http://cpu",
        predict_timeout_s=5,
        max_infer_batch_size=1,
        local_memory_mode=mode,
        encoder_grace_s=grace,
    )
    assignment = dict(
        lease=_lease(),
        prediction_clock_domain=monotonic_clock_domain(),
        local_memory_mode=mode,
        profile=protocol_for_mode(mode),
    )
    client = PolicyTransport(
        settings,
        assignment,
        time.monotonic() + 5,
        transport=policy,
        media_transport=media,
        completed_session_factory=completed_session_factory or (lambda: upload),
        completed_encode=completed_encode,
    )
    return client, policy, media, upload


def test_transport_uses_own_http_session_real_png_and_fences_before_End():
    transport, policy, media, upload = _transport()
    local = transport.local
    local.begin()
    local.acknowledge(0, _status(local))
    local.record_completed(0, _image(23), [0.2] * 15)
    payload = local.payload()
    encoded = payload["evidence"][0]["composite_image"]
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(encoded))))
    assert np.all(decoded == 23) and decoded.shape == (256, 512, 3)
    assert not policy.calls and not media.calls and len(upload.calls) == 1
    assert upload.calls[0][2]["headers"] == {"Content-Type": "application/json"}
    stats = local.pipeline_stats()
    assert stats["enqueued"] == stats["uploaded"] == 1 and stats["png_base64_bytes"] == len(encoded)
    transport.close()
    assert upload.closed and policy.closed and media.closed
    assert policy.calls[0][1].endswith("/v2/session/end") and transport._ended
    assert not local.pipeline_stats()["worker_alive"]


def test_transport_unquiesced_pipeline_blocks_End_but_still_cleans_media():
    entered, release = threading.Event(), threading.Event()

    def encode(image):
        entered.set()
        assert release.wait(3)
        return "encoded"

    transport, policy, _, upload = _transport(completed_encode=encode, grace=0.01)
    local = transport.local
    local.begin()
    local.acknowledge(0, _status(local))
    try:
        local.record_completed(0, _image(), [0.0] * 15)
        assert entered.wait(3)
        with pytest.raises(TimeoutError, match="did not quiesce"):
            transport.end()
        assert transport._stop.is_set() and not transport._ended and not policy.calls and not upload.closed
        release.set()
        local._pipeline._thread.join(3)
        transport.close()
        assert transport._ended and upload.closed
    finally:
        release.set()
        local._pipeline._thread.join(3)


def test_no_ttt_has_no_local_client_or_completed_worker():
    def forbidden():
        pytest.fail("No-TTT created completed-evidence worker HTTP")

    transport, _, _, upload = _transport(mode="off", completed_session_factory=forbidden)
    assert transport.local is None
    transport.close()
    assert not upload.calls and not upload.closed


def test_quiescent_pipeline_error_still_Ends_owned_lease_then_propagates():
    def encode(image):
        raise ValueError("first PNG error")

    transport, policy, media, upload = _transport(completed_encode=encode)
    local = transport.local
    local.begin()
    local.acknowledge(0, _status(local))
    local.record_completed(0, _image(), [0.0] * 15)
    with pytest.raises(RuntimeError, match="first PNG error"):
        local.payload()
    with pytest.raises(RuntimeError, match="first PNG error"):
        transport.close()
    assert transport._ended and transport.client.lease is None
    assert policy.calls[0][1].endswith("/v2/session/end")
    assert policy.closed and media.closed and upload.closed
    assert not local.pipeline_stats()["worker_alive"]

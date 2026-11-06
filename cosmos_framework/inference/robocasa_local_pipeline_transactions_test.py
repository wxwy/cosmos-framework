"""Actual CPU actor/adapter integration for the opt-in completed-frame pipeline.

Only Wan and diffusion are doubles. The actor, evidence admission, model-owned
Local encoder/core/scan, immutable snapshots and transaction replay are real.
Actual Wan reuse parity is established separately by the G0 tests.
"""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions, BatchOutput
from cosmos_framework.inference.robocasa_composite_visual import (
    CapturedCurrentVisual,
    composite_frame_digest,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.inference.robocasa_local_memory_policy import RoboCasaLocalMemoryPolicyAdapter
from cosmos_framework.inference.robocasa_session_transactions_test import (
    RealLocalModel,
    ack,
    payload,
    register,
    submit,
    wait_result,
)
from cosmos_framework.inference.robocasa_session_transactions_test import clock_kernel as clock_kernel


def _frame(value):
    return torch.full((3, 256, 512), int(value), dtype=torch.uint8)


def _capture(request):
    """Closed-form result of the explicit constant-image Wan double, no encode."""
    prepared = prepare_robocasa_composite_frame(_frame(request["image"]), request["image_size"])
    normalized = int(request["image"]) / 127.5 - 1.0
    return CapturedCurrentVisual(
        composite_frame_digest(prepared.source_uint8),
        torch.full((96,), normalized, dtype=torch.float32),
        request["image_size"],
        PREPROCESS_PROFILE,
    )


def _completed(lease, step, *, pixel=None, action=0.01):
    return {
        "session_id": lease["session_id"],
        "episode_id": lease["episode_id"],
        "image_size": 256,
        "preprocess_profile": PREPROCESS_PROFILE,
        "source_step": step,
        # Row zero MUST be the actual cold Policy image, whose captured feature
        # is -1 after normalization. Later rows independently use Encode1.
        "composite_image": str((0 if step == 0 else 1 + step % 2) if pixel is None else pixel),
        "executed_action": [action] * 15,
    }


def _request(lease, rows=(), *, state=0.0):
    request = payload(lease, len(rows))
    request["state"][0] = state
    request["local_memory"]["evidence"] = [
        {key: row[key] for key in ("source_step", "composite_image", "executed_action")} for row in rows
    ]
    return request


def _wait(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("bounded CPU pipeline condition did not complete")
        threading.Event().wait(0.005)


def _action_value(request, prefix):
    return request["state"][0] + (0.0 if prefix is None else float(prefix.square().mean()) * 0.01)


@pytest.fixture
def make_pipeline():
    harnesses = []

    def make(*, slots=2, cap=2):
        torch.manual_seed(19)
        model = RealLocalModel().eval()
        harness = SimpleNamespace(
            model=model,
            scan_calls=[],
            generation_calls=[],
            before_scan=None,
            before_encode=None,
            before_generate=None,
            capture_mode="normal",
            drain_fails=False,
            releases=[],
        )
        original_scan = model.net.scan_local_memory
        original_encode = model.tokenizer_vision_gen.encode

        def scan(visual, action, valid, state, *, create_graph=False):
            harness.scan_calls.append(
                {
                    "steps": visual.shape[1],
                    "grad": torch.is_grad_enabled(),
                    "inference": torch.is_inference_mode_enabled(),
                    "create_graph": create_graph,
                }
            )
            if harness.before_scan is not None:
                harness.before_scan(visual, action)
            return original_scan(visual, action, valid, state, create_graph=create_graph)

        def encode(value):
            if harness.before_encode is not None:
                harness.before_encode(value)
            return original_encode(value)

        model.net.scan_local_memory = scan
        model.tokenizer_vision_gen.encode = encode
        # The adapter captures the model-owned scan callable at construction.
        # Install both observational boundaries first so Events and counts
        # observe the exact production path exercised by this fixture.
        adapter = RoboCasaLocalMemoryPolicyAdapter(
            SimpleNamespace(model=model),
            mode="required",
            decode_image=_frame,
            max_sessions=slots,
            pipeline_enabled=True,
        )
        harness.adapter = adapter

        def generate(requests, updates):
            prefixes = [adapter.prefixes(update)[0] for update in updates]
            harness.generation_calls.append(
                {
                    "sessions": [request["local_memory"]["session_id"] for request in requests],
                    "frontiers": [request["local_memory"]["consumer_step"] for request in requests],
                    "prefixes": [None if prefix is None else prefix.detach().clone() for prefix in prefixes],
                }
            )
            if harness.before_generate is not None:
                harness.before_generate(requests, updates)
            visuals = tuple(_capture(request) for request in requests)
            if harness.capture_mode == "missing":
                visuals = ()
            elif harness.capture_mode == "wrong_size":
                visuals = tuple(replace(visual, image_size=128) for visual in visuals)
            actions = [
                [[_action_value(request, prefix)] * 15 for _ in range(16)]
                for request, prefix in zip(requests, prefixes, strict=True)
            ]
            return BatchOutput(actions, current_visuals=visuals)

        def quiesce():
            if harness.drain_fails:
                raise RuntimeError("controlled unproven device drain")

        harness.actor = BatchedSessionTransactions(
            adapter,
            generate,
            quiesce,
            max_batch_size=cap,
            batch_wait_ms=80,
            owner_token="cpu-owner",
            run_lease="run",
            manifest_digest="a" * 64,
            slot_ids=tuple(f"slot{index}" for index in range(slots)),
            model_lock=threading.Lock(),
            timeout_s=10,
        )
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        for event in harness.releases:
            event.set()
        actor = harness.actor
        if actor.info()["quarantined"]:
            assert actor.close(timeout_s=1) is False
            actor._executor.shutdown(wait=True)
        else:
            assert actor.close()
            assert harness.adapter.pipeline_info()["items"] == 0
            assert harness.adapter.pipeline_info()["bytes"] == 0


def _cold(harness, leases):
    actor = harness.actor
    requests = [_request(lease) for lease in leases]
    # Align public admissions under the actual short metadata lock; no random
    # timing window is needed to prove B=2 mapping.
    with actor._lock:
        pending = [submit(actor, lease, request) for lease, request in zip(leases, requests, strict=True)]
    results = [wait_result(actor, lease, ticket) for lease, ticket in zip(leases, pending, strict=True)]
    for lease, result in zip(leases, results, strict=True):
        assert result.json()["local_memory"]["prefix_present"] is False
        ack(actor, lease, result)
    return results


def _admit(harness, lease, rows):
    for row in rows:
        reply = harness.actor.completed(lease, row)
        assert reply.status == 202
        assert reply.json() == {
            "accepted": True,
            "session_id": lease["session_id"],
            "episode_id": lease["episode_id"],
            "source_step": row["source_step"],
            "duplicate": False,
        }


def test_sixteen_completed_rows_feed_real_replan_and_publish_one_snapshot(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter, model = harness.actor, harness.adapter, harness.model
    lease = register(actor)
    _cold(harness, [lease])
    before = adapter.snapshot
    parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rows = [_completed(lease, index) for index in range(16)]
    _admit(harness, lease, rows)
    _wait(lambda: adapter.pipeline_ready(lease["session_id"], 16))
    assert adapter.snapshot is before
    assert before.fast[lease["session_id"]].state is None
    assert model.encoded == 15
    assert harness.scan_calls == [{"steps": 1, "grad": True, "inference": False, "create_graph": False}] * 16

    reference_model = RealLocalModel().eval()
    reference_model.load_state_dict(model.state_dict())
    reference = RoboCasaLocalMemoryPolicyAdapter(
        SimpleNamespace(model=reference_model), mode="required", decode_image=_frame
    )
    reference.commit(reference.prepare(_request(lease)))
    request = _request(lease, rows, state=0.25)
    expected = reference.prepare(request)
    expected_prefix = reference.prefixes(expected)[0].detach().clone()
    entered, release = threading.Event(), threading.Event()
    harness.releases.append(release)

    def generating(requests, updates):
        entered.set()
        assert release.wait(5)

    harness.before_generate = generating
    pending = submit(actor, lease, request)
    assert entered.wait(5)
    assert adapter.snapshot is before  # Includes Fast, Visual and response cache.
    torch.testing.assert_close(harness.generation_calls[-1]["prefixes"][0], expected_prefix, rtol=2e-5, atol=2e-6)
    release.set()
    result = wait_result(actor, lease, pending)
    body = result.json()
    assert body["local_memory"]["encoded_steps"] == 15
    assert body["local_memory"]["adapted_steps"] == 16
    assert body["local_memory"]["pipeline"] == "per_frame_v1"
    assert body["local_memory"]["prefix_present"] is True
    assert body["action"][0][0] == pytest.approx(_action_value(request, expected_prefix), rel=2e-5)
    after = adapter.snapshot
    assert after is not before and before.fast[lease["session_id"]].state is None
    for actual, target in zip(after.fast[lease["session_id"]].state, expected.memory.replacement.state, strict=True):
        torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
        assert not actual.requires_grad and actual.grad_fn is None
    assert after.responses[lease["session_id"]].body == result.body
    assert after.visual[lease["session_id"]].consumer_step == 16
    assert adapter.pipeline_info()["items"] == 0
    assert (model.encoded, model.scan_calls) == (15, 16)
    for name, value in model.state_dict().items():
        assert torch.equal(value, parameters[name])
    reference.abort(expected)
    ack(actor, lease, result)


def test_two_sessions_share_b2_but_not_fast_state_or_response_mapping(make_pipeline):
    harness = make_pipeline()
    actor, adapter = harness.actor, harness.adapter
    leases = [register(actor, f"slot{index}", episode_index=index) for index in range(2)]
    cold = _cold(harness, leases)
    assert {result.json()["native_batch"]["size"] for result in cold} == {2}
    requests = []
    for index, lease in enumerate(leases):
        rows = [_completed(lease, step, action=0.02 + 0.2 * index) for step in range(2)]
        _admit(harness, lease, rows)
        requests.append(_request(lease, rows, state=index + 0.1))
    _wait(lambda: all(adapter.pipeline_ready(lease["session_id"], 2) for lease in leases))
    before = adapter.snapshot
    with actor._lock:
        pending = [submit(actor, lease, req) for lease, req in zip(leases, requests, strict=True)]
    results = [wait_result(actor, lease, p) for lease, p in zip(leases, pending, strict=True)]
    assert len({result.json()["native_batch"]["id"] for result in results}) == 1
    assert [result.json()["native_batch"]["index"] for result in results] == [0, 1]
    assert {result.json()["native_batch"]["size"] for result in results} == {2}
    last = harness.generation_calls[-1]
    assert last["sessions"] == [lease["session_id"] for lease in leases]
    for index, (lease, request, result) in enumerate(zip(leases, requests, results, strict=True)):
        body = result.json()
        assert body["assignment"]["session_id"] == lease["session_id"]
        assert body["action"][0][0] == pytest.approx(_action_value(request, last["prefixes"][index]))
        assert adapter.snapshot.responses[lease["session_id"]].body == result.body
        assert before.fast[lease["session_id"]].state is None
        ack(actor, lease, result)
    states = [adapter.snapshot.fast[lease["session_id"]].state for lease in leases]
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(*states, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(*states, strict=True))
    assert (harness.model.encoded, harness.model.scan_calls) == (2, 4)


def test_exact_completed_and_policy_retries_do_not_repeat_encode_or_scan(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    rows = [_completed(lease, step) for step in range(2)]
    _admit(harness, lease, rows)
    duplicate = actor.completed(lease, copy.deepcopy(rows[0]))
    assert duplicate.status == 200 and duplicate.json()["duplicate"] is True
    request = _request(lease, rows)
    pending = submit(actor, lease, request)
    result = wait_result(actor, lease, pending)
    counts = (harness.model.encoded, harness.model.scan_calls, len(harness.generation_calls))
    snapshot = adapter.snapshot
    replay = submit(actor, lease, copy.deepcopy(request))
    assert replay.status == 200 and replay.body == result.body
    ack(actor, lease, result)
    assert submit(actor, lease, copy.deepcopy(request)).body == result.body
    assert actor.completed(lease, copy.deepcopy(rows[1])).json()["duplicate"] is True
    assert adapter.snapshot is snapshot
    assert counts == (1, 2, 2)
    assert (harness.model.encoded, harness.model.scan_calls, len(harness.generation_calls)) == counts


def test_completed_admission_rejects_out_of_order_stale_lease_and_changed_bytes(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    before = adapter.snapshot
    with pytest.raises(ValueError, match="contiguous|order|step"):
        actor.completed(lease, _completed(lease, 1))
    stale = {**lease, "capability": "stale-capability"}
    with pytest.raises((ValueError, RuntimeError)):
        actor.completed(stale, _completed(lease, 0))
    _admit(harness, lease, [_completed(lease, 0)])
    with pytest.raises(ValueError, match="differ|changed|duplicate|bytes"):
        actor.completed(lease, _completed(lease, 0, action=0.2))
    _wait(lambda: adapter.pipeline_ready(lease["session_id"], 1))
    assert adapter.snapshot is before
    assert harness.model.encoded == 0 and harness.model.scan_calls == 1


@pytest.mark.parametrize("capture_mode", ["missing", "wrong_size"])
def test_cold_generation_requires_matching_captured_boundary_before_commit(make_pipeline, capture_mode):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    before = adapter.snapshot
    harness.capture_mode = capture_mode
    result = wait_result(actor, lease, submit(actor, lease, _request(lease)), expected=409)
    assert result.json()["quiescent"] is True
    assert adapter.snapshot is before
    assert not before.fast and not before.visual and not before.responses
    assert not adapter.memory._pending and not adapter._visual_pending
    assert (harness.model.encoded, harness.model.scan_calls) == (0, 0)


@pytest.mark.parametrize("blocked_stage", ["scan", "encode"])
def test_inflight_device_work_does_not_hold_same_or_other_session_admission(make_pipeline, blocked_stage):
    harness = make_pipeline()
    actor, adapter = harness.actor, harness.adapter
    leases = [register(actor, f"slot{index}") for index in range(2)]
    _cold(harness, leases)
    before = adapter.snapshot
    entered, release = threading.Event(), threading.Event()
    harness.releases.append(release)

    def blocked(*args):
        entered.set()
        assert release.wait(5)

    if blocked_stage == "encode":
        _admit(harness, leases[0], [_completed(leases[0], 0)])
        _wait(lambda: adapter.pipeline_ready(leases[0]["session_id"], 1))
        harness.before_encode = blocked
        blocked_step = 1
    else:
        harness.before_scan = blocked
        blocked_step = 0
    _admit(harness, leases[0], [_completed(leases[0], blocked_step)])
    assert entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            same = pool.submit(actor.completed, leases[0], _completed(leases[0], blocked_step + 1))
            peer = pool.submit(actor.completed, leases[1], _completed(leases[1], 0))
            assert same.result(timeout=1).status == 202
            assert peer.result(timeout=1).status == 202
        assert adapter.snapshot is before
        assert adapter.pipeline_info()["active"] == 1
    finally:
        release.set()
    _wait(lambda: adapter.pipeline_ready(leases[0]["session_id"], blocked_step + 2))
    _wait(lambda: adapter.pipeline_ready(leases[1]["session_id"], 1))


def test_replan_queued_before_completed_suffix_waits_without_owning_model_lock(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    rows = [_completed(lease, step) for step in range(2)]
    # Public Policy admission precedes the asynchronous evidence, so the pump
    # must select completed work rather than wait for itself under model_lock.
    pending = submit(actor, lease, _request(lease, rows))
    _wait(lambda: bool(actor._queue))
    before_calls = len(harness.generation_calls)
    assert before_calls == 1
    _admit(harness, lease, rows)
    result = wait_result(actor, lease, pending)
    assert result.json()["frontier"] == 2
    assert len(harness.generation_calls) == before_calls + 1
    assert (harness.model.encoded, harness.model.scan_calls) == (1, 2)
    assert adapter.pipeline_info()["items"] == 0
    ack(actor, lease, result)


def test_boundary_worker_error_is_attributed_to_its_slot_and_peer_can_replan(make_pipeline):
    harness = make_pipeline()
    actor, adapter = harness.actor, harness.adapter
    bad, peer = [register(actor, f"slot{index}") for index in range(2)]
    _cold(harness, [bad, peer])
    before = adapter.snapshot
    bad_rows = [_completed(bad, 0, pixel=7)]
    pending = submit(actor, bad, _request(bad, bad_rows))
    _admit(harness, bad, bad_rows)
    failure = wait_result(actor, bad, pending, expected=409).json()
    assert failure["quiescent"] is True
    assert failure["request_key"] == pending.json()["request_key"]
    assert failure["payload_sha256"] == pending.json()["payload_sha256"]
    assert failure["failed_stage"] == "LOCAL_PIPELINE" and failure["error_type"] == "ValueError"
    assert "boundary frame digest mismatch" in failure["error"]
    assert actor._failed[bad["slot_id"]]["failed_stage"] == "LOCAL_PIPELINE"
    assert "boundary" in actor._failed[bad["slot_id"]]["error"]
    assert peer["slot_id"] not in actor._failed and actor.info()["quarantined"] is False
    assert adapter.snapshot is before
    assert (harness.model.encoded, harness.model.scan_calls) == (0, 0)
    peer_rows = [_completed(peer, 0)]
    _admit(harness, peer, peer_rows)
    result = wait_result(actor, peer, submit(actor, peer, _request(peer, peer_rows)))
    assert result.json()["assignment"]["session_id"] == peer["session_id"]
    assert adapter.snapshot.fast[bad["session_id"]] is before.fast[bad["session_id"]]
    assert adapter.snapshot.fast[peer["session_id"]].consumer_step == 1
    assert (harness.model.encoded, harness.model.scan_calls) == (0, 1)
    # The queued failed Replan's later fence must preserve the worker's first
    # error instead of replacing it with the secondary Episode-failed guard.
    again = actor.result(bad, pending.json()["request_key"], pending.json()["payload_sha256"])
    assert again.status == 409
    assert again.json()["failed_stage"] == "LOCAL_PIPELINE"
    assert "boundary frame digest mismatch" in again.json()["error"]
    ack(actor, peer, result)


def test_background_failure_before_replan_preserves_cause_on_next_exact_lease_request(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor = harness.actor
    lease = register(actor)
    _cold(harness, [lease])
    row = _completed(lease, 0, pixel=7)
    assert actor.completed(lease, row).status == 202
    _wait(lambda: actor._failed.get(lease["slot_id"], {}).get("quiescent") is True)
    assert not actor._tickets
    failure = actor._failed[lease["slot_id"]]
    assert failure["request_key"] is None and failure["source_step"] == 0
    assert failure["assignment"] == {key: value for key, value in lease.items() if key != "capability"}
    for operation in (
        lambda: submit(actor, lease, _request(lease, [row])),
        lambda: actor.completed(lease, row),
    ):
        with pytest.raises(RuntimeError, match="boundary frame digest mismatch"):
            operation()
    assert not actor._tickets and harness.model.scan_calls == 0


def test_generation_failure_discards_candidate_without_fast_visual_response_half_commit(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    before = adapter.snapshot
    rows = [_completed(lease, step) for step in range(2)]
    _admit(harness, lease, rows)

    def fail_generation(requests, updates):
        assert adapter.status(updates[0])["adapted_steps"] == 2
        assert adapter.snapshot is before
        raise RuntimeError("controlled diffusion failure after completed candidate")

    harness.before_generate = fail_generation
    result = wait_result(actor, lease, submit(actor, lease, _request(lease, rows)), expected=409)
    assert result.json()["quiescent"] is True
    assert adapter.snapshot is before
    assert not adapter.memory._pending and not adapter._visual_pending
    assert adapter.pipeline_info()["items"] == 0 and adapter.pipeline_info()["bytes"] == 0
    assert (harness.model.encoded, harness.model.scan_calls) == (1, 2)


def test_end_after_pop_before_model_lock_retains_charge_until_owner_discards(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    actor._model_lock.acquire()
    try:
        _admit(harness, lease, [_completed(lease, 0)])
        _wait(lambda: adapter.pipeline_info()["active"] == 1)
        charged = adapter.pipeline_info()["bytes"]
        assert charged > 0
        end = actor.end(lease)
        assert end.status == 202 and end.json()["acknowledged"] is False
        assert adapter.pipeline_info()["active"] == 1
        assert adapter.pipeline_info()["bytes"] == charged
    finally:
        actor._model_lock.release()
    _wait(lambda: not actor._busy)
    assert actor.end(lease).json()["acknowledged"] is True
    assert adapter.pipeline_info()["items"] == 0 and adapter.pipeline_info()["bytes"] == 0
    assert not adapter.snapshot.fast and not adapter.snapshot.visual and not adapter.snapshot.responses
    assert (harness.model.encoded, harness.model.scan_calls) == (0, 0)


def test_failed_worker_with_unproven_drain_quarantines_and_retains_owned_raw_bytes(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    before = adapter.snapshot
    harness.drain_fails = True
    _admit(harness, lease, [_completed(lease, 0, pixel=7)])
    _wait(lambda: actor.info()["quarantined"])
    info = adapter.pipeline_info()
    assert info["active"] == 1 and info["items"] == 1 and info["bytes"] > 0
    assert actor._failed[lease["slot_id"]]["quiescent"] is False
    assert actor._failed[lease["slot_id"]]["state"] == "QUARANTINED"
    assert adapter.snapshot is before
    assert actor.end(lease).status == 202
    assert actor.close(timeout_s=0.1) is False
    retained = adapter.pipeline_info()
    assert retained["active"] == 1 and retained["bytes"] == info["bytes"]
    assert (harness.model.encoded, harness.model.scan_calls) == (0, 0)


def test_close_waits_for_active_scan_then_releases_all_completed_ownership(make_pipeline):
    harness = make_pipeline(slots=1, cap=1)
    actor, adapter = harness.actor, harness.adapter
    lease = register(actor)
    _cold(harness, [lease])
    entered, release = threading.Event(), threading.Event()
    harness.releases.append(release)

    def blocked(*args):
        entered.set()
        assert release.wait(5)

    harness.before_scan = blocked
    _admit(harness, lease, [_completed(lease, 0)])
    assert entered.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(actor.close, timeout_s=5)
        _wait(lambda: actor._stopping)
        assert closing.done() is False
        assert adapter.pipeline_info()["active"] == 1
        assert adapter.pipeline_info()["bytes"] > 0
        release.set()
        assert closing.result(timeout=5) is True
    info = adapter.pipeline_info()
    assert info["active"] == 0 and info["queued"] == 0 and info["items"] == 0 and info["bytes"] == 0
    assert not adapter.snapshot.fast and not adapter.snapshot.visual and not adapter.snapshot.responses
    assert not adapter.memory._pending and not adapter._visual_pending

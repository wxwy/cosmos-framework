# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""C/B1 native READY ownership and externally frozen client-budget CPU contracts.

The controlled Feature adapter delegates to the actual CPU Local store. Owner
receipt/boundary authentication has separate integration tests; no fake encoder
result here is claimed to be a real Wan, simulation or hardware measurement.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference import robocasa_session_transactions_test as sessions
from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest
from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions, BatchOutput
from cosmos_framework.scripts import robocasa_transactions_b1_test as legacy

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]
clock_kernel = sessions.clock_kernel
budget_rig = legacy.budget_rig


class _ControlledFeatureAdapter:
    """Scheduling boundary only; state prepare/abort/commit remain real Local code."""

    def __init__(self, base):
        self.base = base
        self.feature_evidence_owner = object()
        self.entered, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()
        self.prepares, self.aborts = [], []
        self.finished_at = None
        self.fail_prepare = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prepare(self, request, *, lease, deadline_ns, clock_domain, request_key):
        self.prepares.append(
            {
                "request": request,
                "lease": lease,
                "deadline_ns": deadline_ns,
                "clock_domain": clock_domain,
                "request_key": request_key,
            }
        )
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("controlled CPU Feature prepare gate was not released")
        if self.fail_prepare:
            raise RuntimeError("controlled Feature validation failure")
        update = self.base.prepare(request)
        self.finished_at = time.monotonic()
        self.finished.set()
        return update

    def abort(self, update):
        self.aborts.append(update)
        return self.base.abort(update)


@pytest.fixture
def feature_actor():
    created = []

    def make(*, fail_prepare=False, fail_generate=False):
        base, model = sessions.make_adapter(2)
        controlled = _ControlledFeatureAdapter(base)
        controlled.fail_prepare = fail_prepare
        generations = []

        def generate(requests, updates):
            assert controlled.finished.is_set()
            assert len(requests) == len(updates) == 1
            generations.append((requests, updates, time.monotonic()))
            if fail_generate:
                raise RuntimeError("controlled native generation failure")
            return BatchOutput([[[0.125] * 15 for _ in range(16)]])

        actor = BatchedSessionTransactions(
            controlled,
            generate,
            lambda: None,
            max_batch_size=1,
            batch_wait_ms=0,
            owner_token="cpu-owner",
            run_lease="run",
            manifest_digest="a" * 64,
            slot_ids=("slot0", "slot1"),
            timeout_s=4,
            model_lock=threading.Lock(),
        )
        created.append((actor, controlled))
        return SimpleNamespace(actor=actor, adapter=controlled, base=base, model=model, generations=generations)

    yield make
    for actor, controlled in created:
        controlled.release.set()
        assert actor.close(timeout_s=5)
        actor._executor.shutdown(wait=True)


def test_c_prepare_is_outside_actor_lock_and_native_ready_starts_after_candidate(feature_actor, monkeypatch):
    rig = feature_actor()
    actor, adapter = rig.actor, rig.adapter
    lease = sessions.register(actor)
    before = rig.base.snapshot
    pending = sessions.submit(actor, lease, sessions.payload(lease))
    assert adapter.entered.wait(2)
    # A different thread currently executes prepare. Checking from this test
    # thread avoids the false proof of reacquiring an RLock in its owning thread.
    assert actor._lock.acquire(timeout=1), "Feature prepare holds the HTTP/End actor lock"
    try:
        ticket = actor._tickets[pending.json()["request_key"]]
        assert ticket.state == "FEATURE_PREPARING"
        assert ticket not in actor._queue and ticket.key not in actor._enqueued
        assert not rig.generations and rig.base.snapshot is before
    finally:
        actor._lock.release()
    condition_waits = []
    original_wait = actor._changed.wait

    def observe_wait(timeout=None):
        condition_waits.append((time.monotonic(), timeout))
        return original_wait(timeout)

    monkeypatch.setattr(actor._changed, "wait", observe_wait)
    adapter.release.set()
    result = sessions.wait_result(actor, lease, pending)
    assert len(adapter.prepares) == len(rig.generations) == 1
    assert ticket.queued_at >= adapter.finished_at
    assert rig.generations[0][2] >= ticket.queued_at
    assert actor.wait_s == 0 and actor.max_batch_size == 1
    # The production READY pump must never wait for peer aggregation at B1/wait0.
    assert not condition_waits
    context = adapter.prepares[0]
    assert context["lease"] == {key: value for key, value in lease.items() if key != "capability"}
    assert context["deadline_ns"] == int(ticket.deadline * 10**9)
    assert context["clock_domain"] == actor.clock_domain and context["request_key"] == ticket.key
    timing = result.json()["timing"]
    assert timing["feature_prepare_ms"] >= 0 and timing["queue_ms"] >= 0
    assert result.json()["native_batch"]["size"] == 1
    assert sessions.ack(actor, lease, result).json()["acknowledged"]
    assert not rig.base.memory._pending and not rig.base._visual_pending
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0


@pytest.mark.parametrize("stage", ["prepare", "generation"])
def test_c_failure_never_publishes_response_or_existing_memory_candidate(feature_actor, stage):
    rig = feature_actor(fail_prepare=stage == "prepare", fail_generate=stage == "generation")
    actor, adapter = rig.actor, rig.adapter
    lease = sessions.register(actor)
    before = rig.base.snapshot
    adapter.release.set()
    pending = sessions.submit(actor, lease, sessions.payload(lease))
    result = sessions.wait_result(actor, lease, pending, expected=409)
    assert result.json()["error_type"] == "RuntimeError"
    assert rig.base.snapshot is before
    assert lease["session_id"] not in actor._responses
    assert not rig.base.memory._pending and not rig.base._visual_pending
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    assert len(rig.generations) == (0 if stage == "prepare" else 1)
    assert adapter.aborts


def test_end_during_feature_prepare_cancels_without_generation_or_pending_state_leak(feature_actor):
    rig = feature_actor()
    actor, adapter = rig.actor, rig.adapter
    lease = sessions.register(actor)
    pending = sessions.submit(actor, lease, sessions.payload(lease))
    assert adapter.entered.wait(2)
    response = actor.end(lease)
    assert response.status == 202 and response.json()["acknowledged"] is False
    adapter.release.set()
    result = sessions.wait_result(actor, lease, pending, expected=409)
    assert result.status == 409 and not rig.generations
    assert actor.end(lease).json()["acknowledged"] is True
    assert not rig.base.memory._pending and not rig.base._visual_pending
    assert not rig.base.snapshot.fast and not rig.base.snapshot.visual and not rig.base.snapshot.responses
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0


@pytest.mark.parametrize("cap,wait", [(8, 0), (4, 0), (1, 1)])
def test_explicit_single_c_rejects_b4_b8_or_intentional_peer_wait_before_allocating_executor(cap, wait):
    base, _model = sessions.make_adapter(1)
    adapter = _ControlledFeatureAdapter(base)
    with pytest.raises(ValueError, match="single|one native Policy member"):
        BatchedSessionTransactions(
            adapter,
            lambda *_args: None,
            lambda: None,
            max_batch_size=cap,
            batch_wait_ms=wait,
            single_latency_arm="C",
            owner_token="cpu-owner",
            run_lease="run",
            manifest_digest="a" * 64,
            slot_ids=("slot0",),
        )


@pytest.mark.parametrize("drop", ["/v2/inference", "/v2/inference/ack", None])
def test_external_budget_includes_prior_payload_wait_and_is_immutable_across_retries_and_ack(budget_rig, drop):
    obj, actor, clock, transport, client, lease = budget_rig
    budget = {"clock_domain": client.clock_domain, "started_at": 0.0, "deadline": 1.0}
    clock.now = 0.6  # Work before calling predict must remain inside this original budget.
    client.timeout_s = 0.1  # A reduced remaining-Episode convenience value is not a new configured cap.
    if drop is not None:
        transport.drop.add(drop)

    def mutate_original(path, _body):
        if path == "/v2/inference":
            budget["deadline"] = 99.0

    transport.before = mutate_original
    result = client.predict(sessions.payload(lease), budget=budget)
    posted = [body["budget"] for path, body in transport.records if path == "/v2/inference"]
    assert posted and all(
        value == {"clock_domain": client.clock_domain, "started_at": 0.0, "deadline": 1.0} for value in posted
    )
    assert result["frontier"] == 0 and len(obj.native_calls) == 1
    inference_and_ack = [call for call in transport.calls if call[1].startswith("/v2/inference")]
    assert any(path == "/v2/inference/ack" for _, path, _ in inference_and_ack)
    assert all(0 < timeout <= 0.4 + 1e-12 for _, _, timeout in inference_and_ack)
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0


def test_expired_external_budget_rejects_before_transport_or_state_allocation(budget_rig):
    obj, actor, clock, transport, client, lease = budget_rig
    budget = {"clock_domain": client.clock_domain, "started_at": 0.0, "deadline": 1.0}
    clock.now = 1.0
    before, calls = obj.local_memory_adapter.snapshot, len(transport.calls)
    with pytest.raises(TimeoutError, match="original prediction deadline"):
        client.predict(sessions.payload(lease), budget=budget)
    assert len(transport.calls) == calls and not obj.native_calls
    assert obj.local_memory_adapter.snapshot is before
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0


@pytest.mark.parametrize("kind", ["domain", "future_start", "exceeds_original_cap"])
def test_external_budget_requires_original_clock_and_configured_total_cap(budget_rig, kind):
    _obj, actor, clock, transport, client, lease = budget_rig
    clock.now = 0.6
    budget = {"clock_domain": client.clock_domain, "started_at": 0.0, "deadline": 1.0}
    if kind == "domain":
        budget["clock_domain"] = "foreign-clock"
    elif kind == "future_start":
        budget["started_at"] = 0.7
    else:
        budget["deadline"] = 1.01
    calls = len(transport.calls)
    with pytest.raises(ValueError):
        client.predict(sessions.payload(lease), budget=budget)
    assert len(transport.calls) == calls and actor.info()["tickets"] == 0


def test_external_budget_ack_after_expiry_is_not_reported_as_delivery_success(budget_rig):
    obj, actor, clock, transport, client, lease = budget_rig
    clock.now = 0.6
    budget = {"clock_domain": client.clock_domain, "started_at": 0.0, "deadline": 1.0}
    transport.after = lambda path, _reply: setattr(clock, "now", 1.01) if path == "/v2/inference/ack" else None
    with pytest.raises(TimeoutError, match="original transaction deadline"):
        client.predict(sessions.payload(lease), budget=budget)
    assert len(obj.native_calls) == 1  # A legal earlier commit is not retrospectively rolled back.
    assert lease["session_id"] in obj.local_memory_adapter.snapshot.fast
    assert actor.info()["tickets"] == 0


def test_client_default_budget_still_starts_at_predict_for_legacy_callers(budget_rig):
    obj, _actor, clock, transport, client, lease = budget_rig
    clock.now = 0.6
    client.predict(sessions.payload(lease))
    posted = [body["budget"] for path, body in transport.records if path == "/v2/inference"]
    assert posted == [{"clock_domain": client.clock_domain, "started_at": 0.6, "deadline": 1.6}]
    assert len(obj.native_calls) == 1


def test_model_owned_cpu_t16_and_sixteen_t1_preserve_state_prefix_loss_and_private_abort():
    _normal_adapter, normal_model = sessions.make_adapter(1)
    _stream_adapter, stream_model = sessions.make_adapter(1)
    memories = []
    for model in (normal_model, stream_model):
        runtime = model.net.local_memory_runtime
        memories.append(OnlineLocalMemory(runtime.encoder, runtime.core, scan_local_memory=model.net.scan_local_memory))
    generator = torch.Generator().manual_seed(245)
    visual, action = torch.randn(16, 96, generator=generator), torch.randn(16, 15, generator=generator)
    before_parameters = [
        {name: value.detach().clone() for name, value in model.named_parameters()}
        for model in (normal_model, stream_model)
    ]
    updates = []
    for index, memory in enumerate(memories):
        cold = memory.prepare(
            OnlineMemoryRequest(
                "session", "episode", 0, (), torch.empty(0, 96), torch.empty(0, 15), reset=True, stream_t1=bool(index)
            )
        )
        memory.commit(cold)
        before = memory.store.snapshot
        update = memory.prepare(
            OnlineMemoryRequest("session", "episode", 16, tuple(range(16)), visual, action, stream_t1=bool(index))
        )
        assert memory.store.snapshot is before  # Preparation never publishes the candidate.
        updates.append(update)
    normal, stream = updates
    for left, right in zip(normal.replacement.state, stream.replacement.state, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(normal.replacement.token, stream.replacement.token, rtol=0, atol=0)
    assert normal.telemetry == stream.telemetry
    assert normal.telemetry["adapted_steps"] == 16
    assert normal_model.scan_calls == 1 and stream_model.scan_calls == 16
    for model, memory, update, parameters in zip(
        (normal_model, stream_model), memories, updates, before_parameters, strict=True
    ):
        before = memory.store.snapshot
        memory.abort(update)
        assert memory.store.snapshot is before and not memory._pending
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
            assert value.grad is None

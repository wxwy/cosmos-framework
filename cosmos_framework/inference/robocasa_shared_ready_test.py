"""Deterministic same-executor READY interleavings; real CPU Local state."""

import threading
import time
from contextlib import contextmanager

import pytest
import torch

from cosmos_framework.inference import robocasa_session_transactions_test as s
from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions, BatchOutput
from cosmos_framework.inference.robocasa_shared_policy_batch_test import FeatureAdapter

clock_kernel = s.clock_kernel


@contextmanager
def actor_for(cap=8, wait=0, timeout=5, generate=None):
    base, model = s.make_adapter(8)
    calls = []

    def default(reqs, updates):
        calls.append(len(reqs))
        return BatchOutput([[[float(i)] * 15 for _ in range(16)] for i in range(len(reqs))])

    actor = BatchedSessionTransactions(
        FeatureAdapter(base),
        generate or default,
        lambda: None,
        max_batch_size=cap,
        batch_wait_ms=wait,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=tuple(f"slot{i}" for i in range(8)),
        timeout_s=timeout,
    )
    try:
        yield actor, base, model, calls
    finally:
        assert actor.close(timeout_s=5)
        actor._executor.shutdown(wait=True)


@pytest.mark.parametrize("cap", range(1, 9))
def test_ready_members_are_bounded_by_config_and_map_to_original_sessions(cap):
    with actor_for(cap) as (actor, base, model, calls):
        leases = [s.register(actor, f"slot{i}") for i in range(8)]
        release = threading.Event()
        blocked = actor._executor.submit(lambda: release.wait(5))
        try:
            pending = [s.submit(actor, l, s.payload(l)) for l in leases]
        finally:
            release.set()
        blocked.result(5)
        replies = [s.wait_result(actor, l, p) for l, p in zip(leases, pending)]
        assert sum(calls) == 8 and max(calls) == cap
        assert calls == [min(cap, 8 - i) for i in range(0, 8, cap)]
        for i, (lease, reply) in enumerate(zip(leases, replies)):
            assert reply.json()["local_memory"]["session_id"] == lease["session_id"]
            assert reply.json()["action"][0][0] == float(i % cap)
            assert s.submit(actor, lease, s.payload(lease)).body == reply.body
            s.ack(actor, lease, reply)
        assert actor.info()["tickets"] == 0 and model.scan_calls == 0


def test_pump_queued_before_peer_preparations_yields_executor_and_makes_actual_b8():
    with actor_for() as (actor, base, model, calls):
        leases = [s.register(actor, f"slot{i}") for i in range(8)]
        pending = []
        original = actor._queue_native

        def enqueue(ticket):
            original(ticket)  # Pump is queued before the other seven prepare tasks.
            if ticket.lease.slot_id == "slot0":
                pending.extend(s.submit(actor, l, s.payload(l)) for l in leases[1:])

        actor._queue_native = enqueue
        first = s.submit(actor, leases[0], s.payload(leases[0]))
        s.wait_result(actor, leases[0], first)
        for lease, reply in zip(leases[1:], pending):
            s.wait_result(actor, lease, reply)
        assert calls == [8] and actor.info()["batch_histogram"] == {8: 1}


def test_no_ready_peer_never_waits_empty_batch_window_and_end_is_session_local():
    with actor_for(wait=1000) as (actor, base, model, calls):
        a, b = s.register(actor), s.register(actor, "slot1")
        started = time.monotonic()
        initial = s.wait_result(actor, a, s.submit(actor, a, s.payload(a)))
        assert time.monotonic() - started < 0.7  # Original empty-window behavior costs1s.
        s.ack(actor, a, initial)
        for lease, value in ((a, 2), (b, 7)):
            if lease is b:
                r = s.wait_result(actor, b, s.submit(actor, b, s.payload(b)))
                s.ack(actor, b, r)
            r = s.wait_result(actor, lease, s.submit(actor, lease, s.payload(lease, 16, 0, value)))
            assert r.json()["local_memory"]["prefix_present"] is True
            s.ack(actor, lease, r)
        before = base.snapshot
        wa, wb = (before.fast[l["session_id"]].state for l in (a, b))
        assert any(not torch.equal(x, y) for x, y in zip(wa, wb))
        assert actor.end(a).json()["acknowledged"]
        assert base.snapshot.fast[b["session_id"]] is before.fast[b["session_id"]]
        assert base.snapshot.visual[b["session_id"]] is before.visual[b["session_id"]]
        assert all(n == 1 for n in calls)


def test_original_deadline_expiring_during_shared_generation_aborts_entire_sealed_batch():
    entered, release = threading.Event(), threading.Event()

    def generate(reqs, updates):
        entered.set()
        assert release.wait(5)
        return BatchOutput([[[0.0] * 15 for _ in range(16)] for _ in reqs])

    with actor_for(timeout=0.2, generate=generate) as (actor, base, model, calls):
        leases = [s.register(actor, f"slot{i}") for i in range(2)]
        initial = base.snapshot
        gate = threading.Event()
        blocked = actor._executor.submit(lambda: gate.wait(5))
        pending = [s.submit(actor, l, s.payload(l)) for l in leases]
        gate.set()
        blocked.result(5)
        assert entered.wait(5)
        try:
            time.sleep(0.25)
        finally:
            release.set()
        for l, p in zip(leases, pending):
            result = s.wait_result(actor, l, p, expected=409)
            assert result.json()["state"] in ("ABORTED", "QUARANTINED")
        assert base.snapshot is initial
        assert actor.info()["requests_finished"] == 0

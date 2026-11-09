"""Issue27 READY scheduling with real Local state, synthetic CPU generation only."""

import threading

from cosmos_framework.inference import robocasa_session_transactions_test as sessions
from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions, BatchOutput
from cosmos_framework.inference.robocasa_session_transactions_test import clock_kernel as clock_kernel


class FeatureAdapter:
    def __init__(self, base):
        self.base = base
        self.feature_evidence_owner = object()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prepare(self, payload, **proof):
        return self.base.prepare(payload)


def test_feature_policy_batch8_uses_real_ready_peers_and_isolated_prefixes():
    base, _ = sessions.make_adapter(8)
    adapter = FeatureAdapter(base)
    batches = []
    actor = BatchedSessionTransactions(
        adapter,
        lambda reqs, updates: (
            batches.append(len(reqs)) or BatchOutput([[[0.125] * 15 for _ in range(16)] for _ in reqs])
        ),
        lambda: None,
        max_batch_size=8,
        batch_wait_ms=0,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=tuple(f"slot{i}" for i in range(8)),
        timeout_s=5,
    )
    release = threading.Event()
    gate = actor._executor.submit(lambda: release.wait(5))
    try:
        leases = [sessions.register(actor, f"slot{i}") for i in range(8)]
        pending = [sessions.submit(actor, l, sessions.payload(l)) for l in leases]
        release.set()
        gate.result(5)
        results = [sessions.wait_result(actor, l, p) for l, p in zip(leases, pending)]
        assert batches == [8] and actor.info()["batch_histogram"] == {8: 1}
        assert len({r.json()["local_memory"]["session_id"] for r in results}) == 8
        for l, r in zip(leases, results):
            sessions.ack(actor, l, r)
    finally:
        release.set()
        actor.close()

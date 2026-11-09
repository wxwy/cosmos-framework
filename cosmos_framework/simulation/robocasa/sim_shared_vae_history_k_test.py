"""Native CPU action loop: skip before RGB capture, not after VAE encoding.

Physics, Owner and the encoder use the existing explicit doubles. The production
loop, Sim client, immutable Frame, causal receipts, ACK and End are real code.
"""

from __future__ import annotations

import time

import pytest

from cosmos_framework.inference.robocasa_feature_evidence import SPARSE_POST_STEP_VERSION
from cosmos_framework.inference.robocasa_session_transactions_test import clock_kernel as clock_kernel
from cosmos_framework.simulation.robocasa import sim_shared_vae_client_test as old
from cosmos_framework.simulation.robocasa.sim_shared_vae_client import SimSharedVAEClient


class SparseChannel(old._Channel):
    def __init__(self, deadline, k):
        super().__init__(deadline)
        self.k = k

    def post_step(self, *args, **kwargs):
        receipt = super().post_step(*args, **kwargs)
        if self.k != 16:
            receipt.update(
                version=SPARSE_POST_STEP_VERSION,
                history_k=self.k,
                selected=receipt["source_step"] % (16 // self.k) == 0,
            )
            self.completed[-1] = receipt.copy()
        return receipt


def client_for(k, **kwargs):
    deadline = time.monotonic_ns() + 10_000_000_000
    worker, channel = old._Worker(**kwargs), SparseChannel(deadline, k)
    descriptor = old._descriptor()
    if k != 16:
        descriptor["local_ttt_history_k"] = k
    client = SimSharedVAEClient(
        old._lease(),
        descriptor,
        deadline,
        2,
        "clock",
        worker_client=worker,
        completion_channel=channel,
        local_ttt_history_k=k,
    )
    return client, worker, channel


def result_for(client, step, **kwargs):
    result = old._result(client, step, **kwargs)
    if client.history_k != 16:
        result["local_memory"]["local_ttt_history_k"] = client.history_k
    return result


def issue(client, step):
    payload = client.payload()
    result = result_for(client, step)
    client.accept_policy_result(result)
    client.acknowledge(0, result["local_memory"])
    return payload


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_two_chunks_keep_all_executed_receipts_but_only_enqueue_k_minus_one_frames(k):
    client, worker, channel = client_for(k)
    client.begin()
    try:
        assert issue(client, 0)["evidence"] == []
        for start in (0, 16):
            for step in range(start, start + 16):
                selected = step % (16 // k) == 0
                snapshot = client.prepare_step(0, step, old._image(step) if selected else None, old._raw())
                assert (snapshot.rgb is not None) is selected
                assert (snapshot.source_rgb_sha256 is not None) is selected
                client.complete_step(0, snapshot, time.monotonic_ns())
            payload = client.payload()
            assert client.payload() == payload  # Retry retains the same rows/budget.
            assert len(payload["evidence"]) == 16
            assert [row["source_step"] for row in payload["evidence"]] == list(range(start, start + 16))
            assert sum(row["feature"] is not None for row in payload["evidence"]) == k - 1
            for row in payload["evidence"]:
                step = row["source_step"]
                if step % (16 // k):
                    assert row["feature"] is None and row["post_step_receipt"]["source_rgb_sha256"] is None
                    assert row["post_step_receipt"]["selected"] is False
            issue(client, start + 16)
        assert [f.header["source_step"] for f in worker.frames] == [
            s for s in range(32) if s % 16 and s % (16 // k) == 0
        ]
        assert len(worker.acked) == 2 * (k - 1) and len(channel.completed) == 32
    finally:
        client.end()
    assert channel.ends == 1 and not client.diagnostics()["worker_alive"]


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("target", [5, 32])
def test_native_loop_skips_composition_before_unselected_steps_and_handles_partial_End(monkeypatch, k, target):
    client, worker, channel = client_for(k)
    result = old._result
    monkeypatch.setattr(old, "_client", lambda: (client, worker, channel))

    def prediction(client, step, **kwargs):
        value = result(client, step, **kwargs)
        if k != 16:
            value["local_memory"]["local_ttt_history_k"] = k
        return value

    monkeypatch.setattr(old, "_result", prediction)
    _, _, _, env, payloads, kwargs = old._native(monkeypatch, target=target)
    compose = old.native.compose
    captures = []

    def observed(obs):
        captures.append(env.steps)
        return compose(obs)

    monkeypatch.setattr(old.native, "compose", observed)
    assert old.native.run_policy(env, **kwargs)[:2] == (True, target)
    assert [r["source_step"] for r in channel.completed] == list(range(target))
    assert all(source % (16 // k) == 0 for source in captures)
    assert len(captures) == 1 + len(range(0, target, 16)) + len(range(0, target, 16 // k))
    assert [p["consumer_step"] for p in payloads] == list(range(0, target, 16))
    assert channel.ends == 1 and not client.diagnostics()["worker_alive"]


def test_unselected_RGB_and_changed_ACK_K_are_rejected_without_losing_retained_rows():
    client, worker, channel = client_for(1)
    client.begin()
    try:
        issue(client, 0)
        zero = client.prepare_step(0, 0, old._image(0), old._raw())
        client.complete_step(0, zero, time.monotonic_ns())
        with pytest.raises(ValueError, match="RGB"):
            client.prepare_step(0, 1, old._image(1), old._raw())
        skipped = client.prepare_step(0, 1, None, old._raw())
        client.complete_step(0, skipped, time.monotonic_ns())
        for step in range(2, 16):
            snapshot = client.prepare_step(0, step, None, old._raw())
            client.complete_step(0, snapshot, time.monotonic_ns())
        client.payload()
        result = result_for(client, 16)
        client.accept_policy_result(result)
        result["local_memory"]["local_ttt_history_k"] = 2
        with pytest.raises(ValueError, match="history K"):
            client.acknowledge(0, result["local_memory"])
        assert client.diagnostics()["pending"] == 16 and not worker.frames
    finally:
        client.end()


def test_actual_eval_connector_accepts_B8_and_propagates_frozen_K_without_a_new_transport():
    from cosmos_framework.inference.robocasa_session_transactions import monotonic_clock_domain
    from cosmos_framework.simulation.robocasa.eval_v2_transactions import PolicyTransport
    from cosmos_framework.simulation.robocasa.eval_v2_transactions_test import settings

    # Production constructor, real TransactionPolicyClient assignment validation,
    # real SimSharedVAEClient construction. Socket methods are not invoked here.
    lease = {**old._lease(), "capability": "native-private-session-capability"}
    descriptor = {
        **old._descriptor(),
        "local_ttt_history_k": 4,
        "worker_socket": "/tmp/unused-history-k.sock",
        "worker_session_capability": "private-worker-session-capability",
        "relay_socket": "/tmp/unused-history-k.ctl",
        "actor_capability": "private-actor-session-capability",
    }
    cfg = {
        **settings(),
        "max_infer_batch_size": 8,
        "local_evidence_profile": "sim_shared_vae",
        "local_ttt_history_k": 4,
    }
    transport = PolicyTransport(
        cfg,
        {"lease": lease, "prediction_clock_domain": monotonic_clock_domain(), "sim_shared_vae": descriptor},
        time.monotonic() + 10,
    )
    assert transport.local.history_k == 4 and transport.client.max_batch_size == 8
    assert isinstance(transport.local, SimSharedVAEClient)

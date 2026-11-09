"""Eight authenticated15+1 histories publish through the existing atomic snapshot.

Receipts are synthetic trusted-owner inputs; this is not env.step/GPU evidence.
The real HMAC verifier, causal T1 math, boundary reuse and commit are exercised.
"""

import copy
import struct

import pytest
import torch

from cosmos_framework.inference import robocasa_local_memory_policy_test as p
from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
from cosmos_framework.inference.robocasa_feature_evidence_test import TOKEN
from cosmos_framework.inference.robocasa_vae_protocol import canonical_json


@pytest.mark.parametrize("count", [2, 8])
def test_signed_multi_session_history_retains_15plus1_t1_atomic_commit_replay_and_end(count):
    first, adapter, model = p._feature_adapter(max_sessions=count)
    histories = [first]
    for i in range(1, count):
        h = copy.copy(first)
        h.lease = dict(
            first.lease, session_id=f"session-{i + 1}", episode_id=f"episode-{i + 1}", slot_id=f"slot-{i + 1}"
        )
        h.visual = lambda step, index=i: struct.pack("<96f", *([step / 16 + index / 32] * 96))
        first.owner.register_session(
            TOKEN, h.lease, worker_id="worker", actor_id="actor", actor_epoch="actor-epoch", episode_deadline_ns=10000
        )
        histories.append(h)

    def commit_round(step, rows):
        before = adapter.snapshot
        updates = [p._prepare_feature(adapter, h, step, r) for h, r in zip(histories, rows)]
        assert adapter.snapshot is before
        for h, u in zip(histories, updates):
            adapter.stage_boundary(u, h.boundary(step))
        responses = [
            CachedMemoryResponse(f"request-{step}", "f" * 64, canonical_json({"local_memory": adapter.status(u)}))
            for u in updates
        ]
        adapter.commit_many(tuple(zip(updates, responses)))
        assert len(adapter.snapshot.responses) == len(adapter.snapshot.visual) == count
        return updates, responses

    commit_round(0, [[] for _ in histories])
    for h in histories:
        h.commit_chunk()
    rows = [h.rows() for h in histories]
    assert all(
        len(r) == 16 and r[0]["feature"] is None and sum(v["feature"] is not None for v in r) == 15 for r in rows
    )
    updates, responses = commit_round(16, rows)
    assert model.scan_calls == ([(1, False)] + [(1, True)] * 15) * count
    assert all(adapter.status(u)["encoded_steps"] == 0 for u in updates)
    states = [adapter.snapshot.fast[h.lease["session_id"]].state for h in histories]
    assert any(not torch.equal(a, b) for a, b in zip(states[0], states[-1]))
    for h, r, cached in zip(histories, rows, responses):
        replay = p._prepare_feature(adapter, h, 16, copy.deepcopy(r))
        assert replay.memory.replay
        adapter.commit(replay, cached)
    assert len(model.scan_calls) == 16 * count
    survivor = histories[-1].lease["session_id"]
    before = adapter.snapshot
    adapter.reset(histories[0].lease["session_id"])
    assert adapter.snapshot.fast[survivor] is before.fast[survivor]
    assert adapter.snapshot.visual[survivor] is before.visual[survivor]

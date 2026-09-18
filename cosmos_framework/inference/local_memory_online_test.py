import copy

import pytest
import torch

from cosmos_framework.inference import local_memory_online as online
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)


def make_memory(**kwargs):
    torch.manual_seed(47)
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore()
    return online.OnlineLocalMemory(encoder, core, **kwargs)


def request(step=0, *, start=0, session="s", episode="e", reset=False):
    n = step - start
    return online.OnlineMemoryRequest(
        session,
        episode,
        step,
        tuple(range(start, step)),
        torch.arange(n * 96, dtype=torch.float32).reshape(n, 96) * 0.0001,
        torch.arange(n * 10, dtype=torch.float32).reshape(n, 10) * 0.001,
        reset,
    )


@pytest.mark.parametrize("steps", [1, 3, 16])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_online_matches_canonical_scan_and_never_mutates_slow_weights(steps, inference_mode):
    memory = make_memory()
    params = list(memory.encoder.parameters()) + list(memory.core.parameters())
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    reference_encoder, reference_core = copy.deepcopy(memory.encoder), copy.deepcopy(memory.core)
    first = memory.prepare(request())
    assert first.token is None
    memory.commit(first)
    req = request(steps)
    with torch.inference_mode(inference_mode):
        update = memory.prepare(req)
    memory.commit(update)
    with torch.enable_grad():
        tokens, state, _ = reference_core.scan_segment_masked_encoded_many(
            reference_encoder,
            req.visual_summary.unsqueeze(0),
            req.executed_action.unsqueeze(0),
            torch.ones(1, steps, dtype=torch.bool),
            create_graph=False,
        )
    torch.testing.assert_close(update.token, tokens[0, -1], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(update.replacement.state, state, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert not actual.requires_grad and actual.grad_fn is None
    for p, value in zip(params, before, strict=True):
        assert torch.equal(p, value) and torch.equal(p.grad, torch.ones_like(p))


def test_prediction_abort_retry_and_changed_bytes_replay():
    memory = make_memory()
    first = memory.prepare(request())
    memory.abort_many((first,))
    assert memory.metadata()["sessions"] == 0
    memory.commit(memory.prepare(request()))
    req = request(3)
    pending = memory.prepare(req)
    assert memory.metadata()["steps"] == {"s": 0}
    memory.abort_many((pending,))
    retry = memory.prepare(req)
    torch.testing.assert_close(retry.token, pending.token, rtol=0, atol=0)
    memory.commit(retry)
    replay = memory.prepare(req)
    assert replay.replay
    memory.commit(replay)
    assert memory.metadata()["steps"] == {"s": 3}
    req.visual_summary[0, 0] += 0.1
    with pytest.raises(ValueError, match="replay"):
        memory.prepare(req)
    assert memory.metadata()["steps"] == {"s": 3}


def test_missing_or_future_evidence_and_episode_switch_fail_closed():
    memory = make_memory()
    memory.commit(memory.prepare(request()))
    with pytest.raises(ValueError, match="exactly once"):
        memory.prepare(request(3, start=1))
    with pytest.raises(ValueError, match="episode changed"):
        memory.prepare(request(1, episode="other"))
    assert memory.metadata()["steps"] == {"s": 0}
    memory.commit(memory.prepare(request(0, episode="other", reset=True)))
    assert memory._records["s"].episode_id == "other"
    assert memory._records["s"].token is None


def test_sessions_are_isolated_and_batch_commit_is_atomic():
    memory = make_memory()
    memory.commit_many((memory.prepare(request(session="a")), memory.prepare(request(session="b"))))
    before_b = tuple(t.clone() for t in memory._records["b"].state)
    a = memory.prepare(request(3, session="a"))
    b = memory.prepare(request(1, session="b"))
    memory.abort_many((b,))
    with pytest.raises(RuntimeError, match="capability"):
        memory.commit_many((a, b))
    assert memory.metadata()["steps"] == {"a": 0, "b": 0}
    memory.commit(a)
    assert memory.metadata()["steps"] == {"a": 3, "b": 0}
    assert all(torch.equal(x, y) for x, y in zip(before_b, memory._records["b"].state))
    with pytest.raises(RuntimeError, match="capability"):
        memory.commit(a)


def test_session_limits_and_reset_pending_guard():
    memory = make_memory(max_sessions=1)
    pending = memory.prepare(request())
    with pytest.raises(RuntimeError, match="limit"):
        memory.prepare(request(session="other"))
    with pytest.raises(RuntimeError, match="pending"):
        memory.reset_session("s")
    memory.commit(pending)
    memory.reset_session("s")
    memory.commit(memory.prepare(request(session="other")))

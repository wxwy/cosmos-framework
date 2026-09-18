"""Online/serving memory equivalence and transaction tests; CPU only."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from .local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
    LocalMemoryRuntime,
)
from .local_memory_online import OnlineLocalMemorySession, OnlineTransition, generate_with_local_memory


def runtime():
    torch.manual_seed(37)
    return LocalMemoryRuntime(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore()
    ).eval()


def transition(step=1, slot=0, episode="e"):
    return OnlineTransition(slot, episode, step, torch.ones(96) * (slot + 1), torch.ones(10) * 0.1)


def initialize(session, slots=(0,)):
    prepared = session.prepare(tuple(OnlineTransition(slot, "e", 0) for slot in slots))
    assert all(token is None for token in prepared.tokens)
    session.commit(prepared)


def test_online_update_matches_training_scan_and_keeps_slow_weights_frozen():
    owner = runtime()
    before = {name: value.detach().clone() for name, value in owner.state_dict().items()}
    session = OnlineLocalMemorySession(owner)
    initialize(session, (0, 1))
    requests = (transition(slot=0), transition(slot=1))
    with torch.inference_mode():
        prepared = session.prepare(requests)
    expected, state, _ = owner.ttt_core.scan_segment_masked_encoded_many(
        owner.evidence_encoder,
        torch.stack([r.previous_visual_summary for r in requests]).unsqueeze(1),
        torch.stack([r.previous_executed_action for r in requests]).unsqueeze(1),
        torch.ones(2, 1, dtype=torch.bool),
        create_graph=False,
    )
    for row, token in enumerate(prepared.tokens):
        torch.testing.assert_close(token, expected[row, 0])
        assert not token.requires_grad
    session.commit(prepared)
    for row in range(2):
        for actual, target in zip(session._records[row][3], state, strict=True):
            torch.testing.assert_close(actual[0], target[row])
            assert not actual.requires_grad
    for name, value in owner.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in owner.parameters())


def test_online_replay_is_idempotent_and_changed_bytes_rejected():
    session = OnlineLocalMemorySession(runtime())
    initialize(session)
    request = transition()
    first = session.prepare((request,))
    value = first.tokens[0].clone()
    session.commit(first)
    prior = session._records[0][3]
    repeated = session.prepare((request,))
    torch.testing.assert_close(repeated.tokens[0], value, rtol=0, atol=0)
    session.commit(repeated)
    for actual, expected in zip(session._records[0][3], prior, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="changed"):
        session.prepare((replace(request, previous_executed_action=torch.ones(10)),))


def test_online_abort_and_reset_do_not_publish_pending_state():
    session = OnlineLocalMemorySession(runtime())
    initialize(session)
    original = session._records[0]
    pending = session.prepare((transition(),))
    assert session._records[0] is original
    with pytest.raises(RuntimeError, match="pending"):
        session.reset(0)
    session.abort(pending)
    assert session._records[0] is original
    with pytest.raises(RuntimeError, match="stale"):
        session.commit(pending)
    session.reset(0)
    assert not session._records
    initialize(session)


@dataclass
class Plan:
    has_local_memory: bool = False


class NativePolicySpy:
    def __init__(self, owner):
        self.net = SimpleNamespace(local_memory_runtime=owner)
        self.config = SimpleNamespace(local_memory_enabled=True)
        self.training = False
        self.fail = False
        self.seen = None

    def generate_samples_from_batch(self, batch, **kwargs):
        self.seen = batch
        if self.fail:
            raise RuntimeError("injected native generation failure")
        return {
            "action": [
                torch.ones(1, 16, 10) * (0.0 if token is None else token.sum()) for token in batch["local_memory"]
            ]
        }


def test_memory_is_in_generation_action_path_and_failure_does_not_commit():
    owner = runtime()
    session = OnlineLocalMemorySession(owner)
    model = NativePolicySpy(owner)
    batch = {"sequence_plan": [Plan()]}
    generate_with_local_memory(model, batch, [OnlineTransition(0, "e", 0)], session)
    assert model.seen["local_memory"] == [None]
    model.fail = True
    with pytest.raises(RuntimeError, match="native"):
        generate_with_local_memory(model, batch, [transition()], session)
    assert session._records[0][1] == 0 and session._pending is None
    model.fail = False
    result = generate_with_local_memory(model, batch, [transition()], session)
    assert model.seen["sequence_plan"][0].has_local_memory
    assert model.seen["local_memory"][0] is not None
    assert result["action"][0].abs().max() > 0
    assert session._records[0][1] == 1
    assert not batch["sequence_plan"][0].has_local_memory
    generate_with_local_memory(model, batch, [transition(2)], session, use_local_tokens=False)
    assert model.seen["local_memory"] == [None]
    assert session._records[0][1] == 2


@pytest.mark.parametrize("bad", ["gap", "foreign_episode", "missing", "nan"])
def test_online_invalid_transition_preserves_committed_state(bad):
    session = OnlineLocalMemorySession(runtime())
    initialize(session)
    original = session._records[0]
    request = {
        "gap": transition(2),
        "foreign_episode": transition(episode="other"),
        "missing": replace(transition(), previous_executed_action=None),
        "nan": replace(transition(), previous_visual_summary=torch.full((96,), float("nan"))),
    }[bad]
    with pytest.raises(ValueError):
        session.prepare((request,))
    assert session._records[0] is original and session._pending is None

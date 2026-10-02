from __future__ import annotations

import copy

import pytest
import torch

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest
from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _runtime() -> OnlineLocalMemory:
    torch.manual_seed(3)
    return OnlineLocalMemory(LocalEvidenceEncoder(action_dim=15), ContinualTTTLocalMemoryCore())


def _request(step: int, start: int, *, reset: bool = False) -> OnlineMemoryRequest:
    n = step - start
    return OnlineMemoryRequest(
        "session",
        "episode",
        step,
        tuple(range(start, step)),
        torch.randn(n, 96),
        torch.randn(n, 15),
        reset,
    )


def test_cold_start_then_completed_evidence_updates_fast_state() -> None:
    memory = _runtime()
    cold = memory.prepare(_request(0, 0, reset=True))
    assert cold.token is None
    assert cold.telemetry["adapted_steps"] == 0
    memory.commit(cold)

    update = memory.prepare(_request(16, 0))
    assert update.token is not None and update.token.shape == (4, 32)
    assert update.telemetry["adapted_steps"] == 16
    assert update.telemetry["fast_update_norm"] > 0
    assert update.telemetry["fast_state_norm"] > 0
    assert update.telemetry["inner_loss_mean"] > 0
    memory.commit(update)
    assert memory.info()["sessions"] == 1


def test_abort_does_not_publish_candidate_state() -> None:
    memory = _runtime()
    cold = memory.prepare(_request(0, 0, reset=True))
    memory.commit(cold)
    before = copy.deepcopy(memory._records["session"])
    update = memory.prepare(_request(16, 0))
    memory.abort(update)
    after = memory._records["session"]
    assert after.consumer_step == before.consumer_step == 0
    assert after.token is before.token is None
    for left, right in zip(after.state, before.state, strict=True):
        torch.testing.assert_close(left, right)


def test_chronology_and_episode_identity_fail_closed() -> None:
    memory = _runtime()
    cold = memory.prepare(_request(0, 0, reset=True))
    memory.commit(cold)
    bad = OnlineMemoryRequest(
        "session",
        "episode",
        16,
        tuple(range(1, 17)),
        torch.randn(16, 96),
        torch.randn(16, 15),
        False,
    )
    with pytest.raises(ValueError, match="contiguous"):
        memory.prepare(bad)

    changed = OnlineMemoryRequest(
        "session",
        "other",
        16,
        tuple(range(16)),
        torch.randn(16, 96),
        torch.randn(16, 15),
        False,
    )
    with pytest.raises(ValueError, match="episode changed"):
        memory.prepare(changed)


def test_slow_parameters_never_receive_grad_buffers() -> None:
    memory = _runtime()
    cold = memory.prepare(_request(0, 0, reset=True))
    memory.commit(cold)
    update = memory.prepare(_request(16, 0))
    assert all(parameter.grad is None for parameter in memory.encoder.parameters())
    assert all(parameter.grad is None for parameter in memory.core.parameters())
    memory.commit(update)

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
    assert after.state is before.state is None


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


def test_online_sequential_update_matches_training_core_scan() -> None:
    torch.manual_seed(11)
    encoder = LocalEvidenceEncoder(action_dim=15)
    core = ContinualTTTLocalMemoryCore()
    reference_encoder = copy.deepcopy(encoder)
    reference_core = copy.deepcopy(core)
    memory = OnlineLocalMemory(encoder, core)

    cold = memory.prepare(OnlineMemoryRequest("s", "e", 0, (), torch.empty(0, 96), torch.empty(0, 15), True))
    memory.commit(cold)
    visual = torch.randn(16, 96)
    action = torch.randn(16, 15)
    update = memory.prepare(OnlineMemoryRequest("s", "e", 16, tuple(range(16)), visual, action, False))

    valid = torch.ones(1, 16, dtype=torch.bool)
    reference_core.reset_telemetry()
    tokens, state, present = reference_core.scan_segment_masked_encoded_many(
        reference_encoder,
        visual.unsqueeze(0),
        action.unsqueeze(0),
        valid,
        state_in=None,
        create_graph=False,
    )
    assert present.all()
    torch.testing.assert_close(update.token, tokens[0, -1].detach(), rtol=2e-5, atol=2e-6)
    for actual, expected in zip(update.replacement.state, state, strict=True):
        torch.testing.assert_close(actual, expected.detach(), rtol=2e-5, atol=2e-6)


def test_model_owned_n20_t8_chunks_match_scalar_reference_and_telemetry() -> None:
    torch.manual_seed(23)
    encoder = LocalEvidenceEncoder(evidence_dim=16, action_dim=15)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=16, local_dim=4, ttt_dim=8, fast_hidden_dim=16, ttt_tbptt_steps=8, k_local=2
    )
    ref_encoder, ref_core = copy.deepcopy(encoder), copy.deepcopy(core)
    calls: list[tuple[int, bool]] = []

    def scan(visual, action, valid, state, *, create_graph=False):
        calls.append((visual.shape[1], state is not None))
        return core.scan_segment_masked_encoded_many(encoder, visual, action, valid, state, create_graph=create_graph)

    memory = OnlineLocalMemory(encoder, core, scan_local_memory=scan)
    cold = memory.prepare(OnlineMemoryRequest("s", "e", 0, (), torch.empty(0, 96), torch.empty(0, 15), True))
    memory.commit(cold)
    visual, action = torch.randn(20, 96), torch.randn(20, 15)
    update = memory.prepare(OnlineMemoryRequest("s", "e", 20, tuple(range(20)), visual, action))
    assert calls == [(8, False), (8, True), (4, True)]
    assert update.telemetry["adapted_steps"] == 20

    ref_core.reset_telemetry()
    state = ref_core.initial_state(1)
    for index in range(20):
        evidence = ref_encoder.encode_segment(
            visual[index : index + 1].unsqueeze(0), action[index : index + 1].unsqueeze(0)
        ).squeeze(1)
        with torch.enable_grad():
            tokens, state = ref_core.step_many(evidence, state, create_graph=False)
        state = OnlineLocalMemory._detach_state(state)
    expected = ref_core.drain_telemetry()
    torch.testing.assert_close(update.token, tokens[0].detach(), rtol=2e-5, atol=2e-6)
    for actual, reference in zip(update.replacement.state, state, strict=True):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    assert update.telemetry["fast_update_norm"] == pytest.approx(
        float(expected["ttt_fast_update_norm_sum"] / expected["ttt_fast_update_norm_count"]), rel=2e-5
    )
    assert update.telemetry["inner_loss_mean"] == pytest.approx(
        float(expected["ttt_inner_loss_sum"] / expected["ttt_inner_loss_count"]), rel=2e-5
    )
    memory.commit(update)
    assert memory._records["s"].consumer_step == 20

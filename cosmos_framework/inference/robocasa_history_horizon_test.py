"""H-derived sampling contracts; non-Native horizons exercise CPU helpers only."""

from copy import deepcopy

import pytest
import torch

from cosmos_framework.evaluation.robocasa_v2.contract import ACTION_CHUNK_SIZE
from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest
from cosmos_framework.inference.robocasa_feature_evidence import FeatureEvidenceOwner
from cosmos_framework.inference.robocasa_local_memory_policy_test import _Model
from cosmos_framework.inference.robocasa_vae_protocol import selected_history_step, validate_history_k
from cosmos_framework.inference.robocasa_vae_worker import WorkerConfig
from cosmos_framework.simulation.robocasa.sim_shared_vae_client import SimSharedVAEClient
from cosmos_framework.simulation.robocasa.sim_shared_vae_client_test import _descriptor, _lease


@pytest.mark.parametrize("h", range(1, 49))
def test_all_divisors_default_dense_first_frame_and_relative_source_identity(h):
    assert validate_history_k(None, h) == h
    divisors = [k for k in range(1, h + 1) if h % k == 0]
    if h == 12:
        assert divisors == [1, 2, 3, 4, 6, 12]
    if h == 24:
        assert divisors == [1, 2, 3, 4, 6, 8, 12, 24]
    for k in [None, *divisors]:
        expected_count = h if k is None else k
        assert validate_history_k(k, h) == expected_count
        for start in (0, 5, 2 * h):
            selected = [s for s in range(start, start + h) if selected_history_step(s, start, k, h)]
            assert selected == list(range(start, start + h, h // expected_count))
            assert selected[0] == start and len(selected) == expected_count
            assert len(selected[1:]) == expected_count - 1
            for outside in (start - 1, start + h):
                with pytest.raises(ValueError):
                    selected_history_step(outside, start, k, h)


@pytest.mark.parametrize("h", [1, 12, 16, 24])
@pytest.mark.parametrize("k", [0, -1, True, False, 1.0, "1", [], {}])
def test_invalid_K_is_not_coerced_even_when_numerically_equal_to_a_divisor(h, k):
    with pytest.raises(ValueError):
        validate_history_k(k, h)


@pytest.mark.parametrize("h,k", [(12, 5), (16, 3), (24, 5), (1, 2), (12, 24)])
def test_non_divisors_are_rejected_in_validation_and_sampling(h, k):
    with pytest.raises(ValueError):
        validate_history_k(k, h)
    with pytest.raises(ValueError):
        selected_history_step(0, 0, k, h)


@pytest.mark.parametrize("h", [None, 0, -1, True, False, 12.0, "12"])
def test_H_must_be_a_positive_integer_Action_Chunk_length(h):
    with pytest.raises(ValueError):
        validate_history_k(None, h)


@pytest.mark.parametrize("h,k", [(12, None), (12, 3), (24, 6), (16, None)])
def test_existing_T1_consumer_uses_bound_H_for_default_sampling_and_session_fence(h, k):
    torch.manual_seed(28)
    model = _Model()
    runtime = model.net.local_memory_runtime
    memory = OnlineLocalMemory(
        runtime.encoder, runtime.core, scan_local_memory=model.net.scan_local_memory, action_horizon=h
    )
    empty = OnlineMemoryRequest("s", "e", 0, (), torch.empty(0, 96), torch.empty(0, 15), True, True, k)
    cold = memory.prepare(empty)
    memory.commit(cold)
    count = validate_history_k(k, h)
    sources = tuple(range(0, h, h // count))
    request = OnlineMemoryRequest("s", "e", h, sources, torch.zeros(count, 96), torch.zeros(count, 15), False, True, k)
    update = memory.prepare(request)
    assert update.telemetry["adapted_steps"] == count and len(model.scan_calls) == count
    memory.commit(update)
    snapshot = memory.store.snapshot
    memory.action_horizon = h * 2  # Simulate a forbidden in-Session model identity change.
    with pytest.raises(ValueError, match="K/H"):
        memory.prepare(empty)
    assert memory.store.snapshot is snapshot and len(model.scan_calls) == count


def test_native_participants_default_K_to_native_contract_and_reject_H_mismatch():
    owner = FeatureEvidenceOwner("test-private-owner-capability")
    assert owner.history_k == owner.action_horizon == ACTION_CHUNK_SIZE
    cfg = WorkerConfig()
    assert cfg.history_k == cfg.action_horizon == ACTION_CHUNK_SIZE
    assert WorkerConfig(history_k=None) == WorkerConfig(history_k=ACTION_CHUNK_SIZE)
    with pytest.raises(ValueError, match="Native"):
        WorkerConfig(history_k=3, action_horizon=12)
    with pytest.raises(ValueError, match="Native"):
        FeatureEvidenceOwner("test-private-owner-capability", history_k=3, action_horizon=12)
    desc = deepcopy(_descriptor())
    desc["action_chunk_size"] = 12
    with pytest.raises(ValueError, match="Horizon"):
        SimSharedVAEClient(_lease(), desc, 10_000, 1, "clock", local_ttt_history_k=4)

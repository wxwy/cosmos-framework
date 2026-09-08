from __future__ import annotations

import copy
from types import MappingProxyType

import pytest
import torch

from cosmos_framework.callbacks.local_memory_telemetry import (
    LocalMemoryTelemetryProducer,
    LocalTelemetrySnapshot,
)


def _snapshot(**changes: object) -> LocalTelemetrySnapshot:
    values: dict[str, object] = {
        "local_tokens": torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]),
        "fast_state": torch.tensor([[3.0, 4.0], [5.0, 12.0]]),
        "fast_update": torch.tensor([[8.0, 15.0]]),
        "consumer_valid_count": 4,
        "local_present_count": 1,
        "admitted_segments": 3,
        "committed_segments": 2,
        "pad_rows": 5,
        "terminal_remainders": 1,
        "initialized_fraction": 0.25,
        "segment_progress_mean": 0.75,
        "txn_backward_success": 1,
        "txn_commit": 2,
        "txn_transient_failure": 3,
        "txn_suffix_retry_begin": 4,
        "txn_scaler_skip": 5,
        "txn_slow_optimizer_step": 6,
        "txn_retry_exhausted": 7,
        "txn_identity_failure": 8,
        "txn_numerical_failure": 9,
        "txn_outer_failure": 10,
    }
    values.update(changes)
    return LocalTelemetrySnapshot(**values)  # type: ignore[arg-type]


def test_full_snapshot_has_exact_frozen_schema_and_values() -> None:
    result = LocalMemoryTelemetryProducer().record(_snapshot())
    expected = {
        "local/token/l2_mean", "local/token/l2_max", "local/token/abs_max", "local/token/present_fraction",
        "local/fast/state_l2_mean", "local/fast/state_l2_max", "local/fast/update_l2_mean", "local/fast/update_l2_max",
        "local/fast/finite_fraction", "local/fast/initialized_fraction", "local/fast/segment_progress_mean",
        "local/exposure/admitted_segments", "local/exposure/valid_consumers", "local/exposure/pad_rows",
        "local/exposure/segments_committed", "local/exposure/terminal_remainders",
        "local/txn/backward_success", "local/txn/commit", "local/txn/transient_failure", "local/txn/suffix_retry_begin",
        "local/txn/scaler_skip", "local/txn/slow_optimizer_step", "local/txn/retry_exhausted", "local/txn/identity_failure",
        "local/txn/numerical_failure", "local/txn/outer_failure",
    }
    assert isinstance(result, MappingProxyType)
    assert set(result) == expected
    assert result["local/token/l2_mean"] == pytest.approx(9.0)
    assert result["local/token/l2_max"] == pytest.approx(13.0)
    assert result["local/token/abs_max"] == 12.0
    assert result["local/token/present_fraction"] == 0.25
    assert result["local/fast/state_l2_mean"] == pytest.approx(9.0)
    assert result["local/fast/state_l2_max"] == 13.0
    assert result["local/fast/update_l2_mean"] == 17.0
    assert result["local/fast/finite_fraction"] == 1.0
    assert "local/transaction/commit" not in result
    assert "local/token_vs_consumer_hidden/l2_ratio" not in result
    assert "local/exposure/by_slot/0" not in result


def test_absent_tokens_and_fast_observations_have_presence_matrix() -> None:
    result = LocalMemoryTelemetryProducer().record(
        _snapshot(local_tokens=None, local_present_count=0, consumer_valid_count=0, fast_state=None, fast_update=None,
                  initialized_fraction=None, segment_progress_mean=None)
    )
    assert result["local/token/present_fraction"] == 0.0
    assert not any(key.startswith("local/token/l2") or key.startswith("local/fast/") for key in result)


@pytest.mark.parametrize("field", ["fast_state", "fast_update"])
def test_fast_observation_accepts_each_optional_field_independently(field: str) -> None:
    changes = {"fast_state": None, "fast_update": None, field: torch.tensor([[6.0, 8.0]])}
    result = LocalMemoryTelemetryProducer().record(_snapshot(**changes))
    name = "state" if field == "fast_state" else "update"
    assert result[f"local/fast/{name}_l2_mean"] == 10.0
    assert result["local/fast/finite_fraction"] == 1.0


@pytest.mark.parametrize(
    "value",
    [torch.tensor([1.0]), torch.ones(1, 1, 1), torch.empty(0, 1), torch.empty(1, 0), torch.tensor([[float("nan")]])],
)
def test_fast_observation_rejects_invalid_rank_empty_or_nonfinite(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="fast_state"):
        LocalMemoryTelemetryProducer().record(_snapshot(fast_state=value))


def test_fast_observation_rejects_dtype_and_noncontiguous_without_mutation() -> None:
    value = torch.ones(2, 2, dtype=torch.float64)
    version = value._version
    with pytest.raises(ValueError, match="fast_state"):
        LocalMemoryTelemetryProducer().record(_snapshot(fast_state=value))
    assert value._version == version and value.grad is None
    noncontiguous = torch.ones(2, 2).transpose(0, 1)
    with pytest.raises(ValueError, match="fast_state"):
        LocalMemoryTelemetryProducer().record(_snapshot(fast_state=noncontiguous))


@pytest.mark.parametrize("field,value", [("initialized_fraction", -0.1), ("segment_progress_mean", float("inf")), ("initialized_fraction", 1)])
def test_optional_fraction_is_strict_python_float_in_unit_interval(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        LocalMemoryTelemetryProducer().record(_snapshot(**{field: value}))


def test_snapshot_count_and_token_contract_fail_closed() -> None:
    producer = LocalMemoryTelemetryProducer()
    with pytest.raises(ValueError, match="committed_segments"):
        producer.record(_snapshot(committed_segments=4))
    with pytest.raises(ValueError, match="local_tokens"):
        producer.record(_snapshot(local_tokens=None))
    with pytest.raises(ValueError, match="local_present_count"):
        producer.record(_snapshot(local_tokens=torch.ones(2, 2)))
    with pytest.raises(ValueError, match="local_present_count"):
        producer.record(_snapshot(local_present_count=5))


def test_successful_record_does_not_mutate_or_retain_snapshot_tensors() -> None:
    local_tokens = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]], requires_grad=True)
    fast_state = torch.tensor([[3.0, 4.0]], requires_grad=True)
    fast_update = torch.tensor([[8.0, 15.0]], requires_grad=True)
    tensors = (local_tokens, fast_state, fast_update)
    for tensor in tensors:
        tensor.grad = torch.full_like(tensor, 2.0)
    values = tuple(tensor.detach().clone() for tensor in tensors)
    grads = tuple(tensor.grad.detach().clone() for tensor in tensors)
    versions = tuple(tensor._version for tensor in tensors)
    requires_grad = tuple(tensor.requires_grad for tensor in tensors)
    external_metadata = {"owner": "fixture", "nested": ["unchanged", 3]}
    metadata_before = copy.deepcopy(external_metadata)
    rng_before = torch.get_rng_state().clone()
    snapshot = _snapshot(local_tokens=local_tokens, fast_state=fast_state, fast_update=fast_update)
    producer = LocalMemoryTelemetryProducer()

    first = producer.record(snapshot)
    second = producer.record(snapshot)

    assert dict(first) == dict(second)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert external_metadata == metadata_before
    assert producer.__dict__ == {}
    for tensor, value, grad, version, requires in zip(tensors, values, grads, versions, requires_grad, strict=True):
        assert torch.equal(tensor.detach(), value)
        assert tensor.requires_grad is requires
        assert tensor._version == version
        assert tensor.grad is not None and torch.equal(tensor.grad, grad)

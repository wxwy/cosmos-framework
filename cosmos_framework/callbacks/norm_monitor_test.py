import pytest
import torch
from torch import nn

from cosmos_framework.callbacks.norm_monitor import (
    LOCAL_SLOW_SELECTOR_GROUPS,
    NormMonitor,
    _build_group_payloads,
    _group_metric_values,
    _reduce_group_payloads,
    _selector_group_for_name,
    _validate_selector_groups,
)


def test_local_slow_selector_groups_match_only_canonical_names():
    assert _selector_group_for_name(LOCAL_SLOW_SELECTOR_GROUPS, "local_history_runtime.encoder.weight") == "encoder"
    assert _selector_group_for_name(LOCAL_SLOW_SELECTOR_GROUPS, "local_history_runtime.recurrent_backend.weight") == "core"
    assert _selector_group_for_name(LOCAL_SLOW_SELECTOR_GROUPS, "local_memory2llm.weight") == "projector"
    assert _selector_group_for_name(LOCAL_SLOW_SELECTOR_GROUPS, "local_memory_modality_embed") == "modality_embed"
    assert _selector_group_for_name(LOCAL_SLOW_SELECTOR_GROUPS, "local_memory_modality_embed_extra") is None


def test_selector_groups_fail_closed_on_overlap_and_mutation():
    with pytest.raises(ValueError):
        _validate_selector_groups({"a": ("local.",), "b": ("local.value",)})
    with pytest.raises(ValueError):
        _validate_selector_groups({"bad-name": ("value.",)})
    with pytest.raises(ValueError):
        _validate_selector_groups({"group": ("",)})
    with pytest.raises(ValueError):
        _validate_selector_groups({"encoder": ("wrong.",)})
    source = dict(LOCAL_SLOW_SELECTOR_GROUPS)
    frozen = _validate_selector_groups(source)
    source["encoder"] = ("other.",)
    assert frozen["encoder"] == ("local_history_runtime.encoder.",)


def test_none_selector_groups_preserve_legacy_predicate():
    monitor = NormMonitor()
    assert monitor._should_track_param("layers.0.moe_gen.weight")
    assert monitor._should_track_param("layers.0.k_norm_und_for_gen.weight")
    assert not monitor._should_track_param("layers.0.moe_gen.net_ema.weight")
    assert not monitor._should_track_param("local_memory2llm.weight")


def test_group_payloads_count_each_selected_parameter_once():
    encoder_weight = nn.Parameter(torch.tensor([3.0, 4.0]))
    encoder_weight.grad = torch.zeros_like(encoder_weight)
    named_parameters = {
        "local_history_runtime.encoder.weight": encoder_weight,
        "local_history_runtime.encoder.bias": nn.Parameter(torch.tensor([12.0])),
        "local_history_runtime.recurrent_backend.weight": nn.Parameter(torch.tensor([5.0])),
        "local_memory2llm.weight": nn.Parameter(torch.tensor([6.0])),
        "local_memory_modality_embed": nn.Parameter(torch.tensor([7.0])),
        "local_history_runtime.encoder.net_ema.weight": nn.Parameter(torch.tensor([99.0])),
        "fast_state.weight": nn.Parameter(torch.tensor([99.0])),
    }
    payloads = _build_group_payloads(LOCAL_SLOW_SELECTOR_GROUPS, named_parameters)

    assert payloads["encoder"].tolist() == [169.0, 0.0, 1.0]
    assert payloads["core"].tolist() == [25.0, 0.0, 0.0]
    assert payloads["projector"].tolist() == [36.0, 0.0, 0.0]
    assert payloads["modality_embed"].tolist() == [49.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        _build_group_payloads(
            {"first": ("local_history_runtime.encoder.",), "second": ("local_history_runtime.encoder.",)},
            {"local_history_runtime.encoder.weight": encoder_weight},
        )


def test_group_payload_reducer_uses_one_sum_per_group_and_supports_synthetic_ranks():
    payloads = {
        "encoder": torch.tensor([4.0, 0.0, 1.0]),
        "core": torch.tensor([9.0, 0.0, 0.0]),
        "projector": torch.tensor([16.0, 0.0, 0.0]),
        "modality_embed": torch.tensor([25.0, 36.0, 1.0]),
    }
    peer_payloads = {
        "encoder": torch.tensor([5.0, 0.0, 1.0]),
        "core": torch.tensor([7.0, 0.0, 0.0]),
        "projector": torch.tensor([20.0, 4.0, 1.0]),
        "modality_embed": torch.tensor([11.0, 0.0, 0.0]),
    }
    calls: list[str] = []

    def synthetic_sum(payload: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        group = next(group for group, value in payloads.items() if value is payload)
        calls.append(group)
        payload.add_(peer_payloads[group])

    _reduce_group_payloads(payloads, synthetic_sum)

    assert calls == list(payloads)
    assert _group_metric_values(payloads["encoder"]) == (3.0, 0.0)
    assert _group_metric_values(payloads["core"]) == (16.0**0.5, None)
    assert _group_metric_values(payloads["projector"]) == (36.0**0.5, 2.0)
    assert _group_metric_values(payloads["modality_embed"]) == (36.0**0.5, 6.0)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (torch.tensor([4.0, 0.0, 0.0]), (2.0, None)),
        (torch.tensor([4.0, 0.0, 2.0]), (2.0, 0.0)),
        (torch.tensor([5.0, 4.0, 1.0]), (5.0**0.5, 2.0)),
        (torch.tensor([4.0, 9.0, 1.0]), (2.0, 3.0)),
    ],
)
def test_group_metric_values_preserve_grad_presence(payload, expected):
    assert _group_metric_values(payload) == expected

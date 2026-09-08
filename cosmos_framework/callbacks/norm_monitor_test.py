import pytest
import torch

from cosmos_framework.callbacks.norm_monitor import (
    LOCAL_SLOW_SELECTOR_GROUPS,
    NormMonitor,
    _group_metric_values,
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

import pytest
import torch
import importlib
from types import SimpleNamespace

import cosmos_framework.model.generator.omni_mot_model as omni_mot_model
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.mot.local_evidence import (
    LocalEvidenceEncoder,
    LocalHistoryRuntime,
    StatelessLocalReplayReadout,
)


def _runtime() -> LocalHistoryRuntime:
    return LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=5, hidden_dim=8),
    )


def _inputs(batch: int = 3, horizon: int = 4):
    mask = torch.zeros(batch, horizon, dtype=torch.bool)
    if horizon:
        mask[0, : min(2, horizon)] = True
        if batch > 2:
            mask[2, :] = True
    return dict(
        history_visual_summary=torch.randn(batch, horizon, 4),
        local_history_action=torch.randn(batch, horizon, 3),
        history_age_steps=torch.arange(horizon).repeat(batch, 1),
        history_dt_s=torch.ones(batch, horizon, 1),
        history_mask=mask,
    )


def test_runtime_gates_mixed_batch_and_trace_shapes() -> None:
    tokens, present, evidence = _runtime()(**_inputs())
    assert tuple(tokens.shape) == (3, 1, 5)
    assert present.tolist() == [True, False, True]
    assert tuple(evidence.shape) == (3, 4, 8)
    assert torch.equal(tokens[1], torch.zeros_like(tokens[1]))
    assert torch.isfinite(tokens).all()


def test_runtime_h0_control_is_local_absent() -> None:
    inputs = _inputs(horizon=1)
    inputs["history_mask"] = torch.zeros(3, 1, dtype=torch.bool)
    tokens, present, _ = _runtime()(**inputs)
    assert not present.any()
    assert torch.equal(tokens, torch.zeros_like(tokens))


def test_runtime_rejects_nonfinite_dt() -> None:
    inputs = _inputs()
    inputs["history_dt_s"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="history_dt_s must be finite"):
        _runtime()(**inputs)


def test_model_injection_h0_without_history_fields_is_absent() -> None:
    model = object.__new__(OmniMoTModel)
    model.config = SimpleNamespace(local_history_horizon=0)
    model.local_history_runtime = object()
    plans = [SequencePlan(has_text=True, has_local_memory=True)]
    data_batch = {}
    model._inject_local_history(data_batch, plans)
    assert data_batch["local_memory"] == [None]
    assert plans[0].has_local_memory is False


def test_model_injection_preserves_one_local_token_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omni_mot_model, "DEVICE", torch.device("cpu"))
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(local_history_horizon=2, local_history_state_enabled=False)
    model.local_history_runtime = _runtime()
    plans = [SequencePlan(has_text=True), SequencePlan(has_text=True)]
    data_batch = _inputs(batch=2, horizon=2)
    data_batch["history_mask"] = torch.tensor([[True, False], [False, False]])
    model._inject_local_history(data_batch, plans)
    assert tuple(data_batch["local_memory"][0].shape) == (1, 5)
    assert data_batch["local_memory"][1] is None
    assert [plan.has_local_memory for plan in plans] == [True, False]


def test_edge_config_history_and_dummy_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_all import (
        _action_policy_libero_edge_model_config,
    )

    monkeypatch.setenv("PSM_R08_LOCAL_HISTORY_ENABLED", "1")
    monkeypatch.setenv("PSM_LOCAL_DUMMY_ENABLED", "0")
    cfg = _action_policy_libero_edge_model_config()
    assert cfg["local_history_enabled"] is True
    assert cfg["local_memory_enabled"] is True
    monkeypatch.setenv("PSM_LOCAL_DUMMY_ENABLED", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _action_policy_libero_edge_model_config()


def test_edge_config_selects_only_r08_runtime_and_r07_local_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSM_R08_LOCAL_HISTORY_ENABLED", "1")
    monkeypatch.setenv("PSM_LOCAL_DUMMY_ENABLED", "0")
    from cosmos_framework.configs.base.experiment.action.posttrain_config import action_policy_libero_edge_all as recipe

    recipe = importlib.reload(recipe)
    cfg = recipe._action_policy_libero_edge_model_config()
    selected = recipe.action_policy_libero_edge_all["optimizer"]["keys_to_select"]
    assert "local_history_runtime" in selected
    assert {"local_memory2llm", "local_memory_modality_embed"}.issubset(selected)
    runtime_names = [f"local_history_runtime.{name}" for name, _ in _runtime().named_parameters()]
    local_names = ["local_memory2llm.weight", "local_memory2llm.bias", "local_memory_modality_embed"]
    selected_names = [name for name in runtime_names + local_names if any(key in name for key in selected)]
    assert set(selected_names) == set(runtime_names + local_names)
    assert cfg["local_history_enabled"] is True

"""Native-loss preservation and A2 configuration boundaries (CPU-only)."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from .production_active_wiring import ActiveNativeBatchInputs


def test_active_loss_retains_native_modality_weights_and_separates_auxiliary():
    vision = torch.tensor(2.0, requires_grad=True)
    action = torch.tensor(3.0, requires_grad=True)
    auxiliary = torch.tensor(0.5, requires_grad=True)
    calls = []

    def native(data, iteration, *, _psm_local_override):
        calls.append((len(data["x"]), iteration, _psm_local_override))
        output = {"flow_matching_loss_vision": vision, "flow_matching_loss_action": action, "aux_loss_gen": auxiliary}
        return output, 10 * vision + 7 * action + auxiliary

    model = SimpleNamespace(
        config=object(),
        net=torch.nn.Linear(1, 1),
        training_step=native,
        _psm_reduced_loss=OmniMoTModel._psm_reduced_loss,
        _PSM_AUXILIARY_LOSS_KEYS=OmniMoTModel._PSM_AUXILIARY_LOSS_KEYS,
    )
    inputs = ActiveNativeBatchInputs(({"x": torch.ones(1)},) * 3, (None,) * 3, ((0, "a", 0), (1, "b", 0), (2, "c", 0)))
    result = OmniMoTModel._run_active_local_memory_native_forward(model, inputs, 11)
    torch.testing.assert_close(result.primary_consumer_mean, torch.tensor(41.0))
    torch.testing.assert_close(result.auxiliary_loss, auxiliary)
    assert calls == [(3, 11, inputs.locals)]
    result.primary_consumer_mean.backward()
    torch.testing.assert_close(vision.grad, torch.tensor(10.0))
    torch.testing.assert_close(action.grad, torch.tensor(7.0))
    torch.testing.assert_close(auxiliary.grad, torch.tensor(0.0))


@pytest.mark.parametrize("b_stream", [4, 8, 12])
@pytest.mark.parametrize("layout", ["single", "a2"])
def test_recipe_derives_group_geometry_and_exact_optimizer_scope(monkeypatch, b_stream, layout):
    from .config_checkpoint_contract import SELECTORS
    from .ttt_lifecycle_test import _BASE_ENVS, _reload_recipe

    env = {
        **_BASE_ENVS,
        "PSM_R09_B_TTT_ENABLED": "1",
        "PSM_R09_B_TTT_ACTIVE": "1",
        "PSM_R09_B_TTT_ACTIVE_GA": "16",
        "PSM_R09_B_TTT_B_STREAM": str(b_stream),
        "PSM_R09_B_TTT_MEMBER_LAYOUT": layout,
    }
    recipe = _reload_recipe(monkeypatch, env)
    config = recipe.action_policy_libero_edge_all
    callback = config["trainer"]["callbacks"]["r09_b_active_wiring"]
    assert config["trainer"]["grad_accum_iter"] == (16 if layout == "a2" else b_stream * 16)
    assert callback["b_stream"] == b_stream
    assert callback["group_size"] == (b_stream if layout == "a2" else 1)
    assert callback["member_layout"] == layout
    assert config["optimizer"]["keys_to_select"] == (
        list(recipe.action_policy_libero_all_nano["optimizer"]["keys_to_select"]) + list(SELECTORS)
    )

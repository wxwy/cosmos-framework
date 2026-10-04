from __future__ import annotations

import pytest

from cosmos_framework.inference.robocasa_local_memory_contract import validate_local_memory_eval_contract


@pytest.mark.parametrize("action_horizon", [1, 4, 8, 16])
def test_required_local_accepts_configurable_replan_horizon_up_to_t16(action_horizon: int) -> None:
    validate_local_memory_eval_contract(
        local_memory_mode="required",
        camera_set="left_wrist",
        use_base_action=True,
        base_encoding="raw",
        use_state=True,
        action_horizon=action_horizon,
    )


@pytest.mark.parametrize("action_horizon", [0, -1, 17, 32])
def test_required_local_rejects_horizon_outside_t16(action_horizon: int) -> None:
    with pytest.raises(ValueError, match="action-horizon"):
        validate_local_memory_eval_contract(
            local_memory_mode="required",
            camera_set="left_wrist",
            use_base_action=True,
            base_encoding="raw",
            use_state=True,
            action_horizon=action_horizon,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_set": "wrist_lr"},
        {"use_base_action": False},
        {"base_encoding": "ego"},
        {"use_state": False},
    ],
)
def test_required_local_rejects_training_contract_drift(kwargs: dict) -> None:
    base = {
        "local_memory_mode": "required",
        "camera_set": "left_wrist",
        "use_base_action": True,
        "base_encoding": "raw",
        "use_state": True,
        "action_horizon": 16,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        validate_local_memory_eval_contract(**base)


def test_off_mode_does_not_impose_local_checkpoint_contract() -> None:
    validate_local_memory_eval_contract(
        local_memory_mode="off",
        camera_set="wrist_lr",
        use_base_action=False,
        base_encoding="ego",
        use_state=False,
        action_horizon=0,
    )

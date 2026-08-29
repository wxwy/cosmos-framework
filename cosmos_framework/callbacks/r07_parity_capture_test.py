# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib
import json

import pytest
import torch

from cosmos_framework.callbacks.r07_parity_capture import R07ParityCaptureCallback


def test_r07_parity_summary_hashes_bfloat16_bit_patterns() -> None:
    reference = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    same = reference.clone()
    changed = reference.clone()
    changed[0] = 3.0

    summary = R07ParityCaptureCallback._tensor_summary(reference)

    assert summary["sha256"] == R07ParityCaptureCallback._tensor_summary(same)["sha256"]
    assert summary["sha256"] != R07ParityCaptureCallback._tensor_summary(changed)["sha256"]
    assert summary["mean"] == -0.5
    assert summary["l2_norm"] > 0.0


def test_history_mask_summary_accepts_production_batch_shapes() -> None:
    mask = torch.tensor([[True, False]])
    assert isinstance(R07ParityCaptureCallback._tensor_or_list_summary(mask), dict)
    assert isinstance(R07ParityCaptureCallback._tensor_or_list_summary([mask]), list)
    assert isinstance(R07ParityCaptureCallback._tensor_or_list_summary([[mask]]), list)
    with pytest.raises(TypeError, match="history_mask"):
        R07ParityCaptureCallback._tensor_or_list_summary([None])


def test_r07_parity_callback_is_not_registered_without_output_env(monkeypatch) -> None:
    monkeypatch.delenv("PSM_R07_PARITY_OUTPUT", raising=False)
    module = importlib.import_module(
        "cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_all"
    )
    module = importlib.reload(module)

    assert "r07_parity_capture" not in module.action_policy_libero_edge_all["trainer"]["callbacks"]


def test_r07_parity_callback_captures_effective_action_sigma(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("cosmos_framework.callbacks.r07_parity_capture.distributed.is_rank0", lambda: True)
    output_path = tmp_path / "parity.json"
    callback = R07ParityCaptureCallback(str(output_path))
    tensor = torch.tensor([[1.0]], dtype=torch.bfloat16)
    indexes = torch.tensor([0], dtype=torch.long)

    callback.on_training_step_end(
        model=None,
        data_batch={"history_mask": torch.tensor([[True]])},
        output_batch={
            "flow_matching_loss_vision": torch.tensor(1.0),
            "flow_matching_loss_action": torch.tensor(2.0),
            "x0": [tensor],
            "xt": [tensor],
            "sigma": tensor,
            "r07_parity_sigma_vision_effective": [tensor],
            "r07_parity_x0_action": [tensor],
            "r07_parity_xt_action": [tensor],
            "r07_parity_sigma_action_effective": [tensor],
            "r07_parity_text_ids": indexes,
            "r07_parity_text_indexes": indexes,
            "r07_parity_vision_indexes": indexes,
            "r07_parity_action_indexes": indexes,
            "split_lens": [1],
            "attn_modes": ["full"],
            "model_pred": [tensor],
            "r07_parity_preds_action": [tensor],
            "r07_parity_position_ids": indexes.reshape(1, 1),
        },
        loss=torch.tensor(3.0),
        iteration=1,
    )

    payload = json.loads(output_path.read_text())
    assert "sigma_action_effective" in payload
    assert payload["sigma_action_effective"][0]["sha256"] == R07ParityCaptureCallback._tensor_summary(tensor)["sha256"]


def test_r07_parity_callback_optionally_writes_sensitivity_tensors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("cosmos_framework.callbacks.r07_parity_capture.distributed.is_rank0", lambda: True)
    output_path = tmp_path / "parity.json"
    tensor_output_path = tmp_path / "sensitivity.pt"
    callback = R07ParityCaptureCallback(str(output_path), str(tensor_output_path))
    tensor = torch.tensor([[1.0]], dtype=torch.bfloat16)
    indexes = torch.tensor([0], dtype=torch.long)

    callback.on_training_step_end(
        model=None,
        data_batch={"local_memory": [tensor], "history_mask": torch.tensor([[True]])},
        output_batch={
            "flow_matching_loss_vision": torch.tensor(1.0),
            "flow_matching_loss_action": torch.tensor(2.0),
            "x0": [tensor],
            "xt": [tensor],
            "sigma": tensor,
            "r07_parity_sigma_vision_effective": [tensor],
            "r07_parity_x0_action": [tensor],
            "r07_parity_xt_action": [tensor],
            "r07_parity_sigma_action_effective": [tensor],
            "r07_parity_text_ids": indexes,
            "r07_parity_text_indexes": indexes,
            "r07_parity_vision_indexes": indexes,
            "r07_parity_action_indexes": indexes,
            "split_lens": [1],
            "attn_modes": ["full"],
            "model_pred": [tensor],
            "r07_parity_preds_action": [tensor],
            "r07_parity_position_ids": indexes.reshape(1, 1),
        },
        loss=torch.tensor(3.0),
        iteration=1,
    )

    tensor_payload = torch.load(tensor_output_path, weights_only=True)
    assert tensor_payload["schema_version"] == "r07_sensitivity_tensors_v1"
    assert torch.equal(tensor_payload["preds_vision"][0], tensor)
    assert torch.equal(tensor_payload["preds_action"][0], tensor)
    assert torch.equal(tensor_payload["local_memory"][0], tensor)

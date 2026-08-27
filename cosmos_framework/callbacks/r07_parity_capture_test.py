# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib

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


def test_r07_parity_callback_is_not_registered_without_output_env(monkeypatch) -> None:
    monkeypatch.delenv("PSM_R07_PARITY_OUTPUT", raising=False)
    module = importlib.import_module(
        "cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_all"
    )
    module = importlib.reload(module)

    assert "r07_parity_capture" not in module.action_policy_libero_edge_all["trainer"]["callbacks"]

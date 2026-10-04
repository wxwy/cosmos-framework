# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared wire contract for RoboCasa online Local-TTT evidence."""

from __future__ import annotations

EVIDENCE_VERSION = "b1_causal_endpoint_visual96_executed_action15_v5"
EVIDENCE_FORMAT = "robocasa_dual_camera_rgb_raw15_v2"
EVIDENCE_ACTION_DIM = 15
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256


def validate_local_memory_eval_contract(
    *,
    local_memory_mode: str,
    camera_set: str,
    use_base_action: bool,
    base_encoding: str,
    use_state: bool,
    action_horizon: int,
) -> None:
    """Fail closed on the V3 RoboCasa Local-TTT evaluator/checkpoint ABI."""
    if local_memory_mode != "required":
        return
    if camera_set != "left_wrist":
        raise ValueError("V3 Local-TTT requires --camera-set left_wrist")
    if not use_base_action or base_encoding != "raw":
        raise ValueError("V3 Local-TTT requires --use-base-action --base-encoding raw")
    if not use_state:
        raise ValueError("V3 Local-TTT formal checkpoint requires --use-state")
    if type(action_horizon) is not int or not 1 <= action_horizon <= 16:
        raise ValueError("V3 Local-TTT requires 1 <= --action-horizon <= 16 (Local TBPTT T=16)")

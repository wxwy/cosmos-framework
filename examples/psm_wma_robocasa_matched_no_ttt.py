# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Passive telemetry for the matched profile; Local does not exist in this model."""

from examples.psm_wma_robocasa_corrected_telemetry import GroupedPlanObserver


class NoTTTPlanObserver(GroupedPlanObserver):
    def _materialize_core(self, trainer):
        return {
            "training_profile": "matched_no_ttt_v1",
            "local_memory_enabled": False,
            "fast_state_status": "DISABLED_NOT_INSTANTIATED",
        }

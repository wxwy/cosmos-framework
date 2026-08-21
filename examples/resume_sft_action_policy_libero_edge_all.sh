#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Deprecated compatibility wrapper for the unified 4in1 SFT launcher.
#
# Usage:
#   bash examples/resume_sft_action_policy_libero_edge_all.sh [iter_dir]
# Explicit iter_dir is retained for compatibility. Prefer the unified launcher,
# which auto-selects the largest iter_* directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "WARNING: examples/resume_sft_action_policy_libero_edge_all.sh is deprecated; forwarding to the unified launcher." >&2
[[ $# -le 1 ]] || { echo "ERROR: expected at most one [iter_dir] argument" >&2; exit 2; }
if [[ $# -eq 1 ]]; then
    export AUTO_RESUME_CHECKPOINT="$1"
fi
exec bash examples/launch_sft_action_policy_libero_edge_all.sh

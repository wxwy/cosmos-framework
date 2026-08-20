#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Resume the 4in1 SFT training from a saved iteration (no cosmos source modified).
#
# Resume mechanics:
#   - checkpoint.load_path=<iter dir>          load weights+optimizer from the save
#   - checkpoint.load_training_state=True      restore optimizer / scheduler / scaler
# Log goes to outputs/train/logs/resume_libero_4in1_from_<iter>.log.
#
# Usage:
#   bash examples/resume_sft_action_policy_libero_edge_all.sh [iter_dir]
#   (default iter_dir=iter_000000275)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT_DIR="outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints"
FROM_ITER="${1:-iter_000000275}"
CKPT="$PWD/$CKPT_DIR/$FROM_ITER"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

export LOG_FILENAME="resume_libero_4in1_from_${FROM_ITER}.log"
export EXTRA_TAIL_OVERRIDES="checkpoint.load_path=$CKPT checkpoint.load_training_state=True"

echo ">>> $(date '+%H:%M:%S') Resuming from $CKPT"
exec bash examples/launch_sft_action_policy_libero_edge_all.sh
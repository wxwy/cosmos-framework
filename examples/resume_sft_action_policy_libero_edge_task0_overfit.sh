#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Resume the task0-overfit SFT training from a saved iteration, attaching the
# per-part loss monitor (examples/_loss_monitor.py) so the vision / action
# flow-matching losses are printed separately to stdout for the rest of the run.
#
# Resume mechanics (no cosmos source modified):
#   - checkpoint.load_path=<iter dir>          load weights+optimizer from the save
#   - checkpoint.load_training_state=True      restore optimizer / scheduler / scaler
#   - +trainer.callbacks.loss_monitor._target_=examples._loss_monitor.LossMonitor
#                                              attach the loss monitor (Hydra + prefix
#                                              appends a NEW callback key)
# Log goes to outputs/train/logs/resume_task0_overfit_from_<iter>.log (the original
# run's log file is preserved via LOG_FILENAME override).
#
# Usage:
#   bash examples/resume_sft_action_policy_libero_edge_task0_overfit.sh [iter_dir]
#   (default iter_dir=iter_000000100)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT_DIR="outputs/train/cosmos3_action_libero/action_sft/edge_libero_task0_overfit/checkpoints"
FROM_ITER="${1:-iter_000000100}"
CKPT="$PWD/$CKPT_DIR/$FROM_ITER"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

export LOG_FILENAME="resume_task0_overfit_from_${FROM_ITER}.log"
export EXTRA_TAIL_OVERRIDES="+trainer.callbacks.loss_monitor._target_=examples._loss_monitor.LossMonitor checkpoint.load_path=$CKPT checkpoint.load_training_state=True"

echo ">>> $(date '+%H:%M:%S') Resuming from $CKPT with loss-monitor callback"
exec bash examples/launch_sft_action_policy_libero_edge_task0_overfit.sh

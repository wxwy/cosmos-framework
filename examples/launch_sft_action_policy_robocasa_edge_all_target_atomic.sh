#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail

# RoboCasa365 target-atomic Edge baseline.  This launcher deliberately keeps
# Local/TTT disabled; it establishes the matched RoboCasa baseline and reusable
# dataloader-resume path before the Local-TTT recipe is layered on top.

: "${TOML_FILE:=examples/toml/sft_config/action_policy_robocasa_edge_all_target_atomic.toml}"
: "${RUN_NAME:=edge_robocasa365_target_atomic}"
: "${ROBOCASA_SUITE:=robocasa365_target_atomic}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp}"
: "${EDGE_POLICY_CHECKPOINT:=/disk/rl/models/Cosmos3-Edge-Policy-DROID}"
: "${NPROC_PER_NODE:=8}"
: "${ROBOCASA_NUM_WORKERS:=2}"
: "${ROBOCASA_PREFETCH_FACTOR:=4}"

export NPROC_PER_NODE
export ROBOCASA_SUITE EDGE_POLICY_CHECKPOINT ROBOCASA_NUM_WORKERS ROBOCASA_PREFETCH_FACTOR
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-}"
export ROBOCASA_LATENT_CACHE_ROOT="${ROBOCASA_LATENT_CACHE_ROOT:-}"

if [[ -z "$ROBOCASA_ROOT" ]]; then
    echo "ERROR: ROBOCASA_ROOT must point to one flat RoboCasa365 v3 mirror root" >&2
    exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    [[ -f "$ROBOCASA_ROOT/meta/info.json" ]] || {
        echo "ERROR: missing RoboCasa v3 metadata: $ROBOCASA_ROOT/meta/info.json" >&2
        exit 2
    }
    if [[ -n "$ROBOCASA_LATENT_CACHE_ROOT" ]]; then
        [[ -f "$ROBOCASA_LATENT_CACHE_ROOT/dataset_manifest.json" ]] || {
            echo "ERROR: missing latent-cache manifest: $ROBOCASA_LATENT_CACHE_ROOT/dataset_manifest.json" >&2
            exit 2
        }
        echo ">>> RoboCasa vision path: exact-window latent cache ($ROBOCASA_LATENT_CACHE_ROOT)"
    else
        echo ">>> RoboCasa vision path: online video decode + online VAE"
    fi
fi

# Keep every RoboCasa baseline/history flag explicitly off.  The later Local-TTT
# launcher will own these settings rather than inheriting ambient shell state.
export PSM_HISTORY_MODE=none
export PSM_LOCAL_DUMMY_ENABLED=0
export PSM_R08_LOCAL_HISTORY_ENABLED=0
export PSM_E003_RECENT_HISTORY_CONTROL=0
export PSM_R09_A1_ENABLED=0
export PSM_R09_B1_TTT_ENABLED=0
export PSM_R09_B_TTT_ENABLED=0
export PSM_R09_B_TTT_ACTIVE=0

TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES:-})

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_ROOT_FOR_RESUME="${OUTPUT_ROOT:-$REPO_ROOT/outputs/train}"
[[ "$OUTPUT_ROOT_FOR_RESUME" = /* ]] || OUTPUT_ROOT_FOR_RESUME="$REPO_ROOT/$OUTPUT_ROOT_FOR_RESUME"
CHECKPOINT_ROOT="$OUTPUT_ROOT_FOR_RESUME/cosmos3_action_robocasa/action_sft/$RUN_NAME/checkpoints"

if [[ "${DISABLE_AUTO_RESUME:-0}" == "1" ]]; then
    echo ">>> FRESH start (DISABLE_AUTO_RESUME=1; ignoring $CHECKPOINT_ROOT)"
    if [[ -e "$CHECKPOINT_ROOT/latest_checkpoint.txt" ]]; then
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo ">>> FRESH start (dry run): would set aside latest_checkpoint.txt"
        else
            mv "$CHECKPOINT_ROOT/latest_checkpoint.txt"                "$CHECKPOINT_ROOT/latest_checkpoint.txt.disabled-$(date '+%Y%m%d%H%M%S')"
        fi
    fi
else
    CHECKPOINT_CANDIDATES=()
    if [[ -d "$CHECKPOINT_ROOT" ]]; then
        while IFS= read -r checkpoint; do
            CHECKPOINT_CANDIDATES+=("$checkpoint")
        done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print | sort -V)
    fi
    if (( ${#CHECKPOINT_CANDIDATES[@]} > 0 )); then
        SELECTED_CHECKPOINT="${CHECKPOINT_CANDIDATES[${#CHECKPOINT_CANDIDATES[@]} - 1]}"
        echo ">>> RESUME from $(basename "$SELECTED_CHECKPOINT")"
        TAIL_OVERRIDES+=(
            "checkpoint.load_path=$SELECTED_CHECKPOINT"
            "checkpoint.load_training_state=True"
        )
    else
        echo ">>> FRESH start (no iter_* checkpoint under $CHECKPOINT_ROOT)"
    fi
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

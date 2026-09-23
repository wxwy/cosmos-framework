#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail

# RoboCasa365 target-atomic N=100 active Local-TTT, formal 8-GPU route.

: "${TOML_FILE:=examples/toml/sft_config/action_policy_robocasa_edge_all_target_atomic_localmem_active.toml}"
: "${ROBOCASA_SUITE:=robocasa365_target_atomic}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp}"
: "${EDGE_POLICY_CHECKPOINT:=/disk/rl/models/Cosmos3-Edge-Policy-DROID}"
: "${NPROC_PER_NODE:=8}"

# User-facing runtime knobs.
: "${ROBOCASA_NUM_WORKERS:=4}"   # Active producer prefetch threads per rank.
: "${SAVE_ITER:=}"
: "${TTT_TBPTT_STEPS:=16}"
: "${TTT_DIM:=64}"
: "${TTT_FAST_HIDDEN_DIM:=256}"
: "${TTT_K_LOCAL:=4}"
: "${TTT_INNER_LR:=0.1}"
: "${TTT_B_STREAM:=8}"
: "${TTT_ACTIVE_GA:=2}"
: "${RUN_NAME:=local_ttt_robocasa365_target_atomic_n100_fsdp8_k${TTT_K_LOCAL}}"

export NPROC_PER_NODE
export ROBOCASA_SUITE EDGE_POLICY_CHECKPOINT
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-}"
export ROBOCASA_LATENT_CACHE_ROOT="${ROBOCASA_LATENT_CACHE_ROOT:-}"

if ! [[ "$ROBOCASA_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROBOCASA_NUM_WORKERS must be a non-negative integer, got: $ROBOCASA_NUM_WORKERS" >&2
    exit 2
fi
if [[ -n "$SAVE_ITER" ]] && { ! [[ "$SAVE_ITER" =~ ^[0-9]+$ ]] || (( SAVE_ITER <= 0 )); }; then
    echo "ERROR: SAVE_ITER must be a positive integer when set, got: $SAVE_ITER" >&2
    exit 2
fi
if [[ ! "$TTT_K_LOCAL" =~ ^(1|4|8|16)$ ]]; then
    echo "ERROR: TTT_K_LOCAL must be one of 1,4,8,16, got: $TTT_K_LOCAL" >&2
    exit 2
fi

if [[ -z "$ROBOCASA_ROOT" ]]; then
    echo "ERROR: ROBOCASA_ROOT must point to the flat RoboCasa365 target-atomic v3 mirror" >&2
    exit 2
fi
if [[ -z "$ROBOCASA_LATENT_CACHE_ROOT" ]]; then
    echo "ERROR: active Local-TTT requires ROBOCASA_LATENT_CACHE_ROOT" >&2
    exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    [[ -f "$ROBOCASA_ROOT/meta/info.json" ]] || {
        echo "ERROR: missing RoboCasa v3 metadata: $ROBOCASA_ROOT/meta/info.json" >&2
        exit 2
    }
    [[ -f "$ROBOCASA_LATENT_CACHE_ROOT/dataset_manifest.json" ]] || {
        echo "ERROR: missing latent-cache manifest: $ROBOCASA_LATENT_CACHE_ROOT/dataset_manifest.json" >&2
        exit 2
    }
fi

# Active Local-TTT contract.
export PSM_HISTORY_MODE=ttt
export PSM_LOCAL_DUMMY_ENABLED=0
export PSM_R08_LOCAL_HISTORY_ENABLED=1
export PSM_E003_RECENT_HISTORY_CONTROL=0
export PSM_R09_A1_ENABLED=0
export PSM_R09_B1_TTT_ENABLED=0
export PSM_R09_B_TTT_ENABLED=1
export PSM_R09_B_TTT_ACTIVE=1
export PSM_R09_B_TTT_MEMBER_LAYOUT=a2
export PSM_R09_B_TTT_TBPTT_STEPS="$TTT_TBPTT_STEPS"
export PSM_R09_B_TTT_DIM="$TTT_DIM"
export PSM_R09_B_TTT_FAST_HIDDEN_DIM="$TTT_FAST_HIDDEN_DIM"
export PSM_R09_B_TTT_K_LOCAL="$TTT_K_LOCAL"
export PSM_R09_B_TTT_INNER_LR="$TTT_INNER_LR"
export PSM_R09_B_TTT_B_STREAM="$TTT_B_STREAM"
export PSM_R09_B_TTT_ACTIVE_GA="$TTT_ACTIVE_GA"

# On the active route the real samples come from the canonical producer, not
# PyTorch DataLoader workers. Reuse the familiar knob to control producer
# prefetch concurrency.
export PSM_ACTIVE_PREFETCH_DEPTH="$ROBOCASA_NUM_WORKERS"

TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES:-})
if [[ -n "$SAVE_ITER" ]]; then
    TAIL_OVERRIDES+=("checkpoint.save_iter=$SAVE_ITER")
fi

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

echo ">>> Local-TTT: T=$TTT_TBPTT_STEPS dim=$TTT_DIM fast_hidden=$TTT_FAST_HIDDEN_DIM K=$TTT_K_LOCAL inner_lr=$TTT_INNER_LR B_stream=$TTT_B_STREAM active_GA=$TTT_ACTIVE_GA prefetch=$ROBOCASA_NUM_WORKERS"
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

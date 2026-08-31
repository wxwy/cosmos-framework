#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_edge_all — Edge-Policy-DROID
# warm-start LIBERO-all (4-suite) action-policy SFT (单卡 A100-80GB 版). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_libero_edge_all.toml.
#
# 官方 all_nano 超参全保留（bs=128/rank、74000 tokens、lr 5e-5、cycle 16000/500、selective AC）；
# 单卡差异：dp_shard=-1 + grad_accum_iter=16（全局批 128x1x16=2048 与官方一致）。
# 默认使用 exact-window latent cache；base = Edge-Policy-DROID DCP；tokenizer = 本地 Edge 包
# （EDGE_POLICY_CHECKPOINT）。若存在训练 checkpoint 则自动从最大 iter_* 恢复。
#
# Required env vars:
#   LIBERO_ROOT           LIBERO_LeRobot_v3 PARENT dir（含 4 个 suite 子目录，no default）
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp
#   EDGE_POLICY_CHECKPOINT default: /disk/rl/models/Cosmos3-Edge-Policy-DROID (本地 Edge 包, 零下载)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth (本地转化原生 VAE)
#   NPROC_PER_NODE        default: 1 (单卡)
#   EXTRA_TAIL_OVERRIDES  额外 Hydra override 字符串（smoke 用），如 "trainer.max_iter=5 trainer.logging_iter=1"
#   OUTPUT_ROOT           default: outputs/train
#   LIBERO_LATENT_CACHE_ROOT default: /disk/rl/data/LIBERO_LeRobot_v3_cosmos_exact_window_shared_vae_v1
#   LIBERO_LATENT_CACHE_VERIFY_RATIO default: 0.001 (0 disables online verification)
#   LIBERO_NUM_WORKERS    default: 12
#   LIBERO_PREFETCH_FACTOR default: 4 (num_workers=0 时自动为 None)
#   DISABLE_AUTO_RESUME   set 1 to ignore saved iter_* checkpoints and start fresh
#   DRY_RUN               set 1 to print the resolved command without running it
#
# Usage:
#   LIBERO_ROOT=/disk/rl/data/LIBERO_LeRobot_v3 \
#     bash examples/launch_sft_action_policy_libero_edge_all.sh
#   # smoke:
#   LIBERO_ROOT=/disk/rl/data/LIBERO_LeRobot_v3 \
#     EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 trainer.logging_iter=1" \
#     bash examples/launch_sft_action_policy_libero_edge_all.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_edge_all.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp}"
: "${EDGE_POLICY_CHECKPOINT:=/disk/rl/models/Cosmos3-Edge-Policy-DROID}"
: "${NPROC_PER_NODE:=1}"
: "${LIBERO_LATENT_CACHE_ROOT:=/disk/rl/data/LIBERO_LeRobot_v3_cosmos_exact_window_shared_vae_v1}"
: "${LIBERO_LATENT_CACHE_VERIFY_RATIO:=0.001}"
: "${LIBERO_NUM_WORKERS:=12}"
: "${LIBERO_PREFETCH_FACTOR:=4}"

# EDGE_POLICY_CHECKPOINT 被 experiment 配置经 ${oc.env:...} 解析（tokenizer_type 本地包路径），
# 必须 export 才能被 torchrun 子进程继承；共享 launcher 不处理它（只处理 BASE_CHECKPOINT_PATH/WAN_VAE_PATH）。
export EDGE_POLICY_CHECKPOINT
export LIBERO_LATENT_CACHE_ROOT LIBERO_LATENT_CACHE_VERIFY_RATIO LIBERO_NUM_WORKERS LIBERO_PREFETCH_FACTOR
if [[ "${DRY_RUN:-0}" != "1" && ! -d "$LIBERO_LATENT_CACHE_ROOT" ]]; then
    echo "ERROR: LIBERO_LATENT_CACHE_ROOT not found: $LIBERO_LATENT_CACHE_ROOT" >&2
    exit 1
fi

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT}/<suite> (a LOCAL LeRobot PARENT dir);
# export it so torchrun (launched in this shell) inherits it.
export LIBERO_ROOT="${LIBERO_ROOT:-}"

# 视频解码依赖 cu13 运行库（libnppicc.so.13）；确保在 LD_LIBRARY_PATH 前置。
if [[ -z "${LD_LIBRARY_PATH:-}" || "$LD_LIBRARY_PATH" != *cu13/lib* ]]; then
    CU13_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"
    export LD_LIBRARY_PATH="$CU13_LIB:${LD_LIBRARY_PATH:-}"
fi

EXTRA_DATASET_CHECK='for _s in libero_spatial libero_object libero_goal libero_10; do [[ -f "$LIBERO_ROOT/$_s/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be the LIBERO_LeRobot_v3 parent dir containing all 4 suites (missing $_s; got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --local-dir <dir> (then LIBERO_ROOT=<dir>). See docs/action_policy_libero_posttrain.md" >&2; exit 1; }; done'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell.
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT_FOR_RESUME="${OUTPUT_ROOT:-$REPO_ROOT/outputs/train}"
[[ "$OUTPUT_ROOT_FOR_RESUME" = /* ]] || OUTPUT_ROOT_FOR_RESUME="$REPO_ROOT/$OUTPUT_ROOT_FOR_RESUME"
CHECKPOINT_ROOT="$OUTPUT_ROOT_FOR_RESUME/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints"

_print_resume_candidates() {
    local -a candidates=("$@")
    local start=$(( ${#candidates[@]} > 3 ? ${#candidates[@]} - 3 : 0 ))
    local candidate
    echo ">>> AUTO-RESUME candidates (latest up to 3):"
    for ((i = start; i < ${#candidates[@]}; ++i)); do
        candidate="${candidates[i]}"
        echo ">>>   $(basename "$candidate")"
    done
}

if [[ "${DISABLE_AUTO_RESUME:-0}" == "1" ]]; then
    echo ">>> FRESH start (DISABLE_AUTO_RESUME=1; ignoring $CHECKPOINT_ROOT)"
elif [[ -n "${AUTO_RESUME_CHECKPOINT:-}" ]]; then
    SELECTED_CHECKPOINT="$AUTO_RESUME_CHECKPOINT"
    [[ "$SELECTED_CHECKPOINT" = /* ]] || SELECTED_CHECKPOINT="$CHECKPOINT_ROOT/$SELECTED_CHECKPOINT"
    [[ -d "$SELECTED_CHECKPOINT" ]] || { echo "ERROR: explicit resume checkpoint not found: $SELECTED_CHECKPOINT" >&2; exit 1; }
    echo ">>> DEPRECATED explicit resume selected: $(basename "$SELECTED_CHECKPOINT")"
    echo ">>> RESUME from $(basename "$SELECTED_CHECKPOINT") ($SELECTED_CHECKPOINT)"
    TAIL_OVERRIDES+=("checkpoint.load_path=$SELECTED_CHECKPOINT" "checkpoint.load_training_state=True")
else
    CHECKPOINT_CANDIDATES=()
    if [[ -d "$CHECKPOINT_ROOT" ]]; then
        while IFS= read -r checkpoint; do
            CHECKPOINT_CANDIDATES+=("$checkpoint")
        done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print | sort -V)
    fi
    if (( ${#CHECKPOINT_CANDIDATES[@]} == 0 )); then
        echo ">>> FRESH start (no iter_* checkpoint under $CHECKPOINT_ROOT)"
    else
        _print_resume_candidates "${CHECKPOINT_CANDIDATES[@]}"
        SELECTED_CHECKPOINT="${CHECKPOINT_CANDIDATES[${#CHECKPOINT_CANDIDATES[@]} - 1]}"
        echo ">>> RESUME from $(basename "$SELECTED_CHECKPOINT") ($SELECTED_CHECKPOINT)"
        TAIL_OVERRIDES+=("checkpoint.load_path=$SELECTED_CHECKPOINT" "checkpoint.load_training_state=True")
    fi
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_edge_task0_overfit — Edge-Policy-DROID
# warm-start 单任务 overfit（LIBERO task 0，单卡 A100-80GB 版）。Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_libero_edge_task0_overfit.toml.
#
# 与 warmstart 的差异只在 dataset（SingleTaskLIBEROLeRobotDataset 过滤 task_index=0）与
# overfit schedule（max_iter=200, save_iter=50, warmup 20 / cycle 200）；模型/权重/参数全同。
# 单卡差异：dp_shard=-1 + grad_accum_iter=16（全局批 128x1x16=2048 与官方一致）。
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO-10 LeRobot dir, e.g. <dir>/libero_10 (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp
#   EDGE_POLICY_CHECKPOINT default: /disk/rl/models/Cosmos3-Edge-Policy-DROID (本地 Edge 包, 零下载)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth (本地转化原生 VAE)
#   NPROC_PER_NODE        default: 1 (单卡)
#   EXTRA_TAIL_OVERRIDES  额外 Hydra override 字符串，如 "trainer.max_iter=5 trainer.logging_iter=1"
#   OUTPUT_ROOT           default: outputs/train
#
# Usage:
#   LIBERO_ROOT=/disk/data/LIBERO_LeRobot_v3/libero_10 \
#     bash examples/launch_sft_action_policy_libero_edge_task0_overfit.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_edge_task0_overfit.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp}"
: "${EDGE_POLICY_CHECKPOINT:=/disk/rl/models/Cosmos3-Edge-Policy-DROID}"
: "${NPROC_PER_NODE:=1}"

# EDGE_POLICY_CHECKPOINT 被 experiment 配置经 ${oc.env:...} 解析（tokenizer_type 本地包路径），
# 必须 export 才能被 torchrun 子进程继承；共享 launcher 不处理它（只处理 BASE_CHECKPOINT_PATH/WAN_VAE_PATH）。
export EDGE_POLICY_CHECKPOINT

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} directly (a LOCAL LeRobot dir);
# export it so torchrun (launched in this shell) inherits it.
export LIBERO_ROOT="${LIBERO_ROOT:-}"

# 视频解码依赖 cu13 运行库（libnppicc.so.13）；确保在 LD_LIBRARY_PATH 前置。
if [[ -z "${LD_LIBRARY_PATH:-}" || "$LD_LIBRARY_PATH" != *cu13/lib* ]]; then
    CU13_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"
    export LD_LIBRARY_PATH="$CU13_LIB:${LD_LIBRARY_PATH:-}"
fi

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). See /disk/rl/psm_wma/datasets/README.md" >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell.
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

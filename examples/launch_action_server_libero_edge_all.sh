#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Start the Action HTTP inference server for a 4in1 (action_policy_libero_edge_all)
# SFT checkpoint (DCP), for LIBERO closed-loop eval over the 4-suite fixed tasks.
# Mirrors launch_action_server_libero_edge_task0.sh; only the experiment name and
# LIBERO_ROOT / CHECKPOINT_PATH defaults differ.
#
# Model:  a saved SFT checkpoint under edge_libero_4in1/checkpoints/iter_* (CHECKPOINT_PATH)
# Config: experiment module action_policy_libero_edge_all (DCP+MODULE branch)
# Port:   8000
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export EDGE_POLICY_CHECKPOINT="${EDGE_POLICY_CHECKPOINT:-/disk/rl/models/Cosmos3-Edge-Policy-DROID}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
# 服务哪个 checkpoint 由 CHECKPOINT_PATH 决定。watcher 会对每个 iter%250==0 的 iter 覆写。
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PWD/outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints/iter_000000050}"
# 视觉(视频)生成的去噪步数：默认 50（同 task0 server）。环境变量 NUM_STEPS 可临时覆盖。
export NUM_STEPS="${NUM_STEPS:-50}"
# edge_all 继承 action_policy_libero_all_nano，数据 root 用 ${oc.env:LIBERO_ROOT}；
# 服务端不加载训练数据，但 OmegaConf 急切解析插值（含 4 个 suite 子路径），缺了会崩。
# 与 4in1 训练同一路径：LIBERO_LeRobot_v3 PARENT 目录。
export LIBERO_ROOT="${LIBERO_ROOT:-/disk/rl/data/LIBERO_LeRobot_v3}"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# sitecustomize shim: 把 guardrails 默认打成 False，跳过未安装的重型 guardrail 依赖链
# (nltk/better_profanity/retinaface/qwen3guard)。不修改任何 cosmos 源码。
export PYTHONPATH="$PWD/examples/_server_shim${PYTHONPATH:+:$PYTHONPATH}"

# DCP 加载分支 (inference.py) 要求 config_file_type==MODULE：默认 config-file=configs/base/config.py
# 自动推断为 MODULE，配合 --experiment 解析 edge_all 实验。无需 --config-file-type。
# tokenizer_type / vae_path 由 edge 实验模块的 ${oc.env:...} 解析；这里再显式 override 一遍兜底。
exec .venv/bin/python -m cosmos_framework.scripts.action_policy_server_libero \
  --experiment action_policy_libero_edge_all \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --experiment-overrides "model.config.vlm_config.tokenizer.tokenizer_type=$EDGE_POLICY_CHECKPOINT" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --no-use-ema-weights \
  --action-normalization quantile_rot \
  --action-stats-path cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
  --raw-action-dim 10 --fps 20 --port 8000 --num-steps "$NUM_STEPS"

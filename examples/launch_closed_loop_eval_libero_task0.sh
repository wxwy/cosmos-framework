#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Run LIBERO closed-loop eval against the Action HTTP server (task 0 = sim task_id 2).
# Uses the RLinf python (has LIBERO sim + robosuite/mujoco) with osmesa CPU rendering.
#
# Headless env facts that make this work (documented, not hacks):
#   - MUJOCO_GL=osmesa + PYOPENGL_PLATFORM=osmesa  -> CPU rendering (no EGL, no GPU contention)
#   - LD_PRELOAD system libstdc++                  -> conda libstdc++ is too old for Mesa's libLLVM-15
#   - examples/_eval_client_launcher.py            -> RLinf python imports cosmos_framework from repo,
#                                                     falling back to cosmos venv site-packages for
#                                                     pure-python deps (loguru, ...) RLinf lacks
#
# Args pass through to closed_loop_eval.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# 绕过 clash 代理（否则 localhost 的 predict 走 127.0.0.1:7897，冷启动首步会 30s 读超时）
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

exec /disk/rl/RLinf/.venv/bin/python examples/_eval_client_launcher.py \
  --server_url "${SERVER_URL:-http://localhost:8000}" \
  --task_suite "${TASK_SUITE:-libero_10}" \
  --task_ids "${TASK_IDS:-2}" \
  --num_trials_per_task "${NUM_TRIALS:-3}" \
  --camera agentview,wrist \
  --image_size 256 \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --action_dim 10 \
  --timeout "${EVAL_TIMEOUT:-120}" \
  --action_horizon "${ACTION_HORIZON:-8}" \
  --save_gifs --gif_fps 20 \
  --save_mp4 --mp4_fps 20 \
  --save_pred_mp4 \
  --output_dir "${OUTPUT_DIR:-results/libero_closed_loop_task0}" \
  "$@"

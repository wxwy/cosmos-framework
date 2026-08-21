#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 在 tmux 会话中继续 libero4in1 正式 SFT 训练（从指定迭代 resume）。
#
# 背景：容器 zshrc 把 `tmux` 别名成坏插件函数 _zsh_tmux_plugin_run，直接敲 `tmux`
# 会报 command not found；这里固定用 /usr/bin/tmux 全路径绕过。
#
# 用法：
#   bash examples/tmux_resume_sft_libero_edge_all.sh [iter_dir] [tmux_session_name]
#   默认 iter=iter_000000275, session=sft_4in1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FROM_ITER="${1:-iter_000000275}"
SESSION="${2:-sft_4in1}"
CKPT="$PWD/outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints/$FROM_ITER"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

# 必需/默认资产（可在调用前 export 覆盖）。
: "${LIBERO_ROOT:=/disk/rl/data/LIBERO_LeRobot_v3}"
: "${BASE_CHECKPOINT_PATH:=$PWD/examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp}"
: "${EDGE_POLICY_CHECKPOINT:=/disk/rl/models/Cosmos3-Edge-Policy-DROID}"

[[ -d "$LIBERO_ROOT" ]]            || { echo "ERROR: LIBERO_ROOT not found: $LIBERO_ROOT" >&2; exit 1; }
[[ -d "$BASE_CHECKPOINT_PATH" ]]   || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -d "$EDGE_POLICY_CHECKPOINT" ]] || { echo "ERROR: EDGE_POLICY_CHECKPOINT not found: $EDGE_POLICY_CHECKPOINT" >&2; exit 1; }

if /usr/bin/tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists (attach or kill first)" >&2
  exit 1
fi

# tmux session 环境继承自 tmux *server*（由更早的 ds/fq 会话启动），本脚本的 export
# 不会传到 session 里。因此把全部 env 直接内联进命令串，保证在 session 内生效。
/usr/bin/tmux new-session -d -s "$SESSION" -x 200 -y 50 \
  "cd $PWD && export PATH=$PWD/.venv/bin:\$PATH && export LIBERO_ROOT='$LIBERO_ROOT' && export BASE_CHECKPOINT_PATH='$BASE_CHECKPOINT_PATH' && export EDGE_POLICY_CHECKPOINT='$EDGE_POLICY_CHECKPOINT' && bash examples/resume_sft_action_policy_libero_edge_all.sh $FROM_ITER"

echo ">>> tmux session '$SESSION' started; attach with: /usr/bin/tmux attach -t $SESSION"
echo ">>> log: outputs/train/logs/resume_libero_4in1_from_${FROM_ITER}.log"
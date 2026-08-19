#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 在 tmux 会话中继续 task0-overfit SFT 训练（从指定迭代 resume）。
# 使用前需先停掉旧的 nohup 训练（resume_sft_... 或 launch_sft_... 起的进程）。
#
# 背景：容器 zshrc 把 `tmux` 别名成坏插件函数 _zsh_tmux_plugin_run，直接敲 `tmux`
# 会报 command not found；这里固定用 /usr/bin/tmux 全路径绕过。
#
# 用法：
#   bash examples/tmux_resume_sft_task0_overfit.sh [iter_dir] [tmux_session_name]
#   默认 iter=iter_000000150, session=sft_overfit
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FROM_ITER="${1:-iter_000000150}"
SESSION="${2:-sft_overfit}"
CKPT="$PWD/outputs/train/cosmos3_action_libero/action_sft/edge_libero_task0_overfit/checkpoints/$FROM_ITER"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

if /usr/bin/tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists (attach or kill first)" >&2
  exit 1
fi

export LOG_FILENAME="resume_task0_overfit_from_${FROM_ITER}.log"
export EXTRA_TAIL_OVERRIDES="+trainer.callbacks.loss_monitor._target_=examples._loss_monitor.LossMonitor checkpoint.load_path=$CKPT checkpoint.load_training_state=True"

/usr/bin/tmux new-session -d -s "$SESSION" -x 200 -y 50 \
  "cd $PWD && export PATH=$PWD/.venv/bin:\$PATH && bash examples/resume_sft_action_policy_libero_edge_task0_overfit.sh $FROM_ITER"

echo ">>> tmux session '$SESSION' started; attach with: /usr/bin/tmux attach -t $SESSION"
echo ">>> log: outputs/train/logs/resume_task0_overfit_from_${FROM_ITER}.log"

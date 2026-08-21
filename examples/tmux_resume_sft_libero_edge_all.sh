#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Deprecated compatibility wrapper for the unified tmux 4in1 SFT launcher.
#
# 背景：容器 zshrc 把 `tmux` 别名成坏插件函数 _zsh_tmux_plugin_run，直接敲 `tmux`
# 会报 command not found；这里固定用 /usr/bin/tmux 全路径绕过。
#
# 用法：
#   bash examples/tmux_resume_sft_libero_edge_all.sh [iter_dir] [tmux_session_name]
# 显式 iter_dir 仅为兼容旧调用；新调用请使用 tmux_launch 自动选择最大 iter_*。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "WARNING: examples/tmux_resume_sft_libero_edge_all.sh is deprecated; forwarding to tmux_launch." >&2
[[ $# -le 2 ]] || { echo "ERROR: expected [iter_dir] [tmux_session_name]" >&2; exit 2; }
if [[ $# -ge 1 && -n "$1" ]]; then
    export AUTO_RESUME_CHECKPOINT="$1"
fi
SESSION="${2:-sft_4in1}"
exec bash examples/tmux_launch_sft_libero_edge_all.sh "$SESSION"

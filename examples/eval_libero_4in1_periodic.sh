#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 周期性闭环 eval watcher：监控 4in1 SFT 训练，每保存到一个 iter%200==0 的
# checkpoint 时，自动起 action server 并对 4 个 suite 各一个固定任务跑一次
# closed-loop eval（复用 launch_closed_loop_eval_libero_task0.sh 的 env 覆写），
# 完成后写 .done 标记，避免重复跑。
#
# 训练不动（save_iter=50 继续跑），本脚本只读 checkpoint 目录、到点起 eval。
# 与训练并发共享 GPU（用户已确认并发模式）。
#
# 挂到独立 tmux 会话：
#   /usr/bin/tmux new-session -d -s eval_4in1 "cd $PWD && bash examples/eval_libero_4in1_periodic.sh"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT_DIR="outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints"
RESULTS_ROOT="results/libero_closed_loop_4in1"   # watcher 在下面按 iter_XXXX 加子目录
EVAL_STRIDE=200            # 每 200 步测一次
NUM_TRIALS=1               # 每 suite 的 task0 测 1 次 trial
POLL_INTERVAL_INITIAL=300    # 启动后立即一轮，之后每 5min 查一次（避免错过首批 iter%200==0）
POLL_INTERVAL_LATER=18000    # 看到第一个 ≥200 checkpoint 后改 5h 轮询（200 步 ≈ 5h）
SERVER_READY_TIMEOUT=600   # server 就绪最长等待（秒）
SERVER_PORT=8000

# 固定任务：所有 suite 统一评测 benchmark task0。
SUITES=(
  "libero_spatial:0"
  "libero_object:0"
  "libero_goal:0"
  "libero_10:0"
)

log() { echo "[$(date '+%F %T')] $*"; }

# 列出已保存 checkpoint 的 iter 数值（9 位补零），升序。
list_iters() {
  ls -1 "$CHECKPOINT_DIR" 2>/dev/null \
    | sed -n 's/^iter_\([0-9]\{9\}\)$/\1/p' | sort -n
}

# 清理可能残留的 action server（上次 eval 中断/未杀干净）。只匹配 server 进程，
# 不碰训练进程（cosmos_framework.scripts.train）。
clear_stale_server() {
  if pgrep -f "cosmos_framework.scripts.action_policy_server_libero" >/dev/null 2>&1; then
    log "  清理残留 action server"
    pkill -f "cosmos_framework.scripts.action_policy_server_libero" 2>/dev/null || true
    sleep 3
  fi
}

# 对单个 iter 跑完整 4-suite eval。
run_eval() {
  local it="$1"
  local iter_dir; iter_dir="iter_$(printf '%09d' "$it")"
  local ckpt="$PWD/$CHECKPOINT_DIR/$iter_dir"
  local out_root="$RESULTS_ROOT/$iter_dir"
  [[ -d "$ckpt" ]] || return 1

  log "=== eval $iter_dir 开始 ==="
  mkdir -p "$out_root"
  clear_stale_server

  # 后台起 server（launch_..._all.sh 内部 exec python，$! 即 python PID）。
  CHECKPOINT_PATH="$ckpt" bash examples/launch_action_server_libero_edge_all.sh \
    > "outputs/train/logs/action_server_4in1_$iter_dir.log" 2>&1 &
  local server_pid=$!

  # 等健康：GET / -> {"status":"ok"}。
  local ready=0 t=0
  while (( t < SERVER_READY_TIMEOUT )); do
    if curl -sf "http://localhost:$SERVER_PORT/" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "  !! server 提前退出，见 outputs/train/logs/action_server_4in1_$iter_dir.log"
      break
    fi
    sleep 5; t=$((t+5))
  done

  if [[ "$ready" != 1 ]]; then
    log "  !! server $iter_dir 未就绪（超时/退出），跳过，下轮重试"
    kill "$server_pid" 2>/dev/null || true
    return 1
  fi
  log "  server $iter_dir 就绪"

  # 依次跑 4 个 suite 的固定任务。
  local entry suite task_id
  for entry in "${SUITES[@]}"; do
    suite="${entry%%:*}"; task_id="${entry##*:}"
    log "  -> $suite task $task_id"
    TASK_SUITE="$suite" TASK_IDS="$task_id" NUM_TRIALS="$NUM_TRIALS" \
      OUTPUT_DIR="$out_root/$suite" \
      bash examples/launch_closed_loop_eval_libero_task0.sh \
      || log "  !! $suite eval 失败(exit $?)，继续下一个 suite"
  done

  # 关 server，写完成标记。
  kill "$server_pid" 2>/dev/null || true
  sleep 3
  clear_stale_server
  touch "$out_root/.done"
  log "=== eval $iter_dir 完成（结果: $out_root/{libero_spatial,libero_object,libero_goal,libero_10}/summary.json）==="
}

log "watcher 启动：监控 $CHECKPOINT_DIR，每 $EVAL_STRIDE 步对 4-suite 固定任务测一次"
POLL_INTERVAL="$POLL_INTERVAL_INITIAL"
while true; do
  for it in $(list_iters); do
    it_num=$((10#$it))   # 去前导零
    (( it_num % EVAL_STRIDE == 0 )) || continue
    done_marker="$RESULTS_ROOT/iter_$(printf '%09d' "$it_num")/.done"
    [[ -f "$done_marker" ]] && continue
    run_eval "$it_num"
  done
  # 看到第一个 ≥200 倍数 checkpoint 后切到低频轮询
  if [[ "$POLL_INTERVAL" == "$POLL_INTERVAL_INITIAL" ]]; then
    for it in $(list_iters); do
      it_num=$((10#$it))
      (( it_num >= EVAL_STRIDE && it_num % EVAL_STRIDE == 0 )) && { POLL_INTERVAL="$POLL_INTERVAL_LATER"; break; }
    done
  fi
  sleep "$POLL_INTERVAL"
done

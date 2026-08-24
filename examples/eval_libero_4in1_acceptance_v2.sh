#!/usr/bin/env bash
# 正式验收仿真测试（EVAL-LIBERO-4IN1-ACCEPTANCE）：
# 对 4in1 SFT 的 200 倍数 checkpoint 做全量闭环评测：
#   4 suite × 全部 10 个任务 × 10 trials，denoise num_steps=30，单 episode 上限 700 步。
# 输出：results/libero_closed_loop_4in1_acceptance/iter_XXXXXXXXX/<suite>/summary.json
#
# 与训练并发共享 GPU/内存（训练不动）。内存水位准入：cgroup memory.current
# ≥ MEM_GATE_BYTES 时等待，避免与 ckpt 保存叠加 OOM。
#
# 挂到独立 tmux 会话：
#   /usr/bin/tmux new-session -d -s eval_acceptance "cd $PWD && bash examples/eval_libero_4in1_acceptance.sh"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT_DIR="outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints"
RESULTS_ROOT="results/libero_closed_loop_4in1_acceptance"
NUM_STEPS=30               # denoise 步数（server 端）
MAX_STEPS=700              # 单 episode 仿真步数上限
NUM_TRIALS=10              # 每任务 10 次
SERVER_READY_TIMEOUT=600
SERVER_PORT=8000
MEM_GATE_GB=115            # 内存水位准入阈值（cgroup 上限 128.8G）
MEM_POLL_S=60
SUITES=(libero_spatial libero_object libero_goal libero_10)

# 评测顺序：200 倍数 ckpt 从新到旧（最新权重结论最有价值，旧的补训练动态曲线）。
CKPT_ITERS=${CKPT_ITERS:-"$(ls -1 "$CHECKPOINT_DIR" | sed -n 's/^iter_\([0-9]\{9\}\)$/\1/p' | sort -rn | awk '$1 % 200 == 0')"}

log() { echo "[$(date '+%F %T')] $*"; }

mem_ok() {
  local cur; cur=$(cat /sys/fs/cgroup/memory.current)
  (( cur < MEM_GATE_GB * 1024 * 1024 * 1024 ))
}

wait_mem() {
  local what="$1"
  while ! mem_ok; do
    log "  内存水位超 ${MEM_GATE_GB}G，等待 ${MEM_POLL_S}s 后再 $what"
    sleep "$MEM_POLL_S"
  done
}

clear_stale_server() {
  if pgrep -f "cosmos_framework.scripts.action_policy_server_libero" >/dev/null 2>&1; then
    log "  清理残留 action server"
    pkill -f "cosmos_framework.scripts.action_policy_server_libero" 2>/dev/null || true
    sleep 3
  fi
}

run_ckpt() {
  local it="$1"
  local iter_dir; iter_dir="iter_$(printf '%09d' "$it")"
  local ckpt="$PWD/$CHECKPOINT_DIR/$iter_dir"
  local out_root="$RESULTS_ROOT/$iter_dir"
  [[ -d "$ckpt" ]] || { log "!! $iter_dir 不存在，跳过"; return 1; }

  log "=== acceptance eval $iter_dir 开始（num_steps=$NUM_STEPS, max_steps=$MAX_STEPS, trials=$NUM_TRIALS, 全 10 任务）==="
  mkdir -p "$out_root"
  wait_mem "启动 $iter_dir server"
  clear_stale_server

  CHECKPOINT_PATH="$ckpt" NUM_STEPS="$NUM_STEPS" bash examples/launch_action_server_libero_edge_all.sh \
    > "outputs/train/logs/action_server_acceptance_$iter_dir.log" 2>&1 &
  local server_pid=$!

  local ready=0 t=0
  while (( t < SERVER_READY_TIMEOUT )); do
    if curl -sf "http://localhost:$SERVER_PORT/" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "  !! server 提前退出，见 outputs/train/logs/action_server_acceptance_$iter_dir.log"
      break
    fi
    sleep 5; t=$((t+5))
  done

  if [[ "$ready" != 1 ]]; then
    log "  !! server $iter_dir 未就绪，跳过该 ckpt"
    kill "$server_pid" 2>/dev/null || true
    return 1
  fi
  log "  server $iter_dir 就绪"

  local suite
  for suite in "${SUITES[@]}"; do
    [[ -f "$out_root/$suite/summary.json" ]] && { log "  -> $suite 已有 summary，跳过"; continue; }
    wait_mem "$suite"
    log "  -> $suite（全部任务 × $NUM_TRIALS trials）"
    # TASK_IDS 显式列全 10 任务（launch 脚本 ${TASK_IDS:-2} 会把空串当未设而退回 2）
    TASK_SUITE="$suite" TASK_IDS="0,1,2,3,4,5,6,7,8,9" NUM_TRIALS="$NUM_TRIALS" \
      OUTPUT_DIR="$out_root/$suite" \
      bash examples/launch_closed_loop_eval_libero_task0.sh --max_steps "$MAX_STEPS" \
      || log "  !! $suite eval 失败(exit $?)，继续下一个 suite"
  done

  kill "$server_pid" 2>/dev/null || true
  sleep 3
  clear_stale_server

  # 只有 4 个 suite 的 summary.json 齐全才写 .done；任一失败留待下次补跑。
  local missing=()
  for suite in "${SUITES[@]}"; do
    [[ -f "$out_root/$suite/summary.json" ]] || missing+=("$suite")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$out_root/.done"
    log "=== acceptance eval $iter_dir 完成（结果: $out_root/<suite>/summary.json）==="
  else
    log "=== acceptance eval $iter_dir 未完成，缺: ${missing[*]}；不写 .done，留待补跑 ==="
  fi
}

log "acceptance 评测启动：ckpt 顺序 = $(echo $CKPT_ITERS | tr '\n' ' ')"
for it in $CKPT_ITERS; do
  it_num=$((10#$it))
  [[ -f "$RESULTS_ROOT/iter_$(printf '%09d' "$it_num")/.done" ]] && continue
  run_ckpt "$it_num"
done
log "全部验收评测完成"

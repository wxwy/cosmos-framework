#!/usr/bin/env bash
# LIBERO 4in1 正式验收（4090 / EVAL-LIBERO-4IN1-ACCEPTANCE）。
#
# 每个 checkpoint 启动一个 action server；每个 suite 最多并行 6 个 closed-loop
# client。一个 client 串行完成一个 task 的全部 trials，并写入独占 task 目录，随后
# 由本脚本校验并原子汇总为 suite/summary.json。仅四个 suite 均完整时写 .done。
#
# 默认输出与冒烟输出必须分开。task #16 示例：
#   CKPT_ITERS=2800 SUITES=libero_spatial NUM_TRIALS=1 \
#   RESULTS_ROOT=results/libero_closed_loop_4in1_acceptance_4090_smoke \
#   bash examples/eval_libero_4in1_acceptance_4090.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-results/libero_closed_loop_4in1_acceptance_4090}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
NUM_STEPS="${NUM_STEPS:-30}"
MAX_STEPS="${MAX_STEPS:-700}"
NUM_TRIALS="${NUM_TRIALS:-10}"
N_WORKERS="${N_WORKERS:-6}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_PORT="${SERVER_PORT:-8000}"
MEM_GATE_GB="${MEM_GATE_GB:-50}"
MEM_POLL_S="${MEM_POLL_S:-60}"
CKPT_ITERS="${CKPT_ITERS:-2800 2600 2400 2200 2000 1800 1600 1400 1200 1000 800 600 400 200}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
REQUIRED_SUITES=(libero_spatial libero_object libero_goal libero_10)

log() { echo "[$(date '+%F %T')] $*"; }

mem_ok() {
  local cur
  cur=$(cat /sys/fs/cgroup/memory.current)
  (( cur < MEM_GATE_GB * 1024 * 1024 * 1024 ))
}

wait_mem() {
  local what="$1"
  while ! mem_ok; do
    log "  内存水位超 ${MEM_GATE_GB}G，等待 ${MEM_POLL_S}s 后再 $what"
    sleep "$MEM_POLL_S"
  done
}

task_summary_complete() {
  local summary_path="$1"
  local task_id="$2"
  "$PYTHON_BIN" - "$summary_path" "$task_id" "$NUM_TRIALS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_id = int(sys.argv[2])
num_trials = int(sys.argv[3])
try:
    payload = json.loads(path.read_text())
    selected = [int(value) for value in payload["selected_task_ids"]]
    result = payload["task_results"]
    valid = (
        selected == [task_id]
        and len(result) == 1
        and int(result[0]["task_id"]) == task_id
        and int(result[0]["episodes"]) == num_trials
    )
except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
sys.exit(0 if valid else 1)
PY
}

merge_suite_summary() {
  local suite_dir="$1"
  local suite="$2"
  "$PYTHON_BIN" - "$suite_dir" "$suite" "$NUM_TRIALS" <<'PY'
import json
import os
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
suite = sys.argv[2]
num_trials = int(sys.argv[3])
results = []
run_contract = None

for task_id in range(10):
    path = suite_dir / "tasks" / f"task_{task_id:03d}" / "summary.json"
    try:
        payload = json.loads(path.read_text())
        selected = [int(value) for value in payload["selected_task_ids"]]
        task_results = payload["task_results"]
        if (
            selected != [task_id]
            or len(task_results) != 1
            or int(task_results[0]["task_id"]) != task_id
            or int(task_results[0]["episodes"]) != num_trials
        ):
            raise ValueError("task summary schema or trial count mismatch")
        contract = {
            "action_space": payload["action_space"],
            "rotation_space": payload["rotation_space"],
            "action_dim": payload["action_dim"],
        }
        if run_contract is None:
            run_contract = contract
        elif contract != run_contract:
            raise ValueError(f"task {task_id} action contract mismatch: {contract}")
        results.append(task_results[0])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"incomplete {suite} task {task_id}: {path}: {exc}") from exc

total_episodes = sum(int(result["episodes"]) for result in results)
total_successes = sum(int(result["successes"]) for result in results)
summary = {
    "task_suite": suite,
    "selected_task_ids": list(range(10)),
    "num_trials_per_task": num_trials,
    **run_contract,
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    "task_results": results,
}
tmp_path = suite_dir / ".summary.json.tmp"
tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(tmp_path, suite_dir / "summary.json")
PY
}

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    log "  停止 action server (pid=$server_pid)"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

start_server() {
  local ckpt="$1"
  local iter_dir="$2"
  local server_log="outputs/train/logs/action_server_acceptance_4090_${iter_dir}.log"
  local ready=0 elapsed=0

  if curl -sf "http://localhost:$SERVER_PORT/" >/dev/null 2>&1; then
    log "  !! 端口 $SERVER_PORT 已有服务响应；为避免误接 checkpoint，拒绝启动"
    return 1
  fi

  wait_mem "启动 $iter_dir server"
  CHECKPOINT_PATH="$ckpt" NUM_STEPS="$NUM_STEPS" \
    bash examples/launch_action_server_libero_edge_all.sh >"$server_log" 2>&1 &
  server_pid=$!

  while (( elapsed < SERVER_READY_TIMEOUT )); do
    if curl -sf "http://localhost:$SERVER_PORT/" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "  !! server 提前退出，见 $server_log"
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if (( ! ready )); then
    log "  !! server $iter_dir 未在 ${SERVER_READY_TIMEOUT}s 内就绪"
    stop_server
    return 1
  fi
  log "  server $iter_dir 就绪 (pid=$server_pid)"
}

run_suite() {
  local suite="$1"
  local out_root="$2"
  local suite_dir="$out_root/$suite"
  local task_id task_dir worker_log pid status=0
  local -a active_pids=()

  mkdir -p "$suite_dir/tasks"
  log "  -> $suite：10 task × $NUM_TRIALS trials，最多 $N_WORKERS 个 worker"
  for task_id in {0..9}; do
    task_dir="$suite_dir/tasks/task_$(printf '%03d' "$task_id")"
    if task_summary_complete "$task_dir/summary.json" "$task_id"; then
      log "     task $task_id 已完成，跳过"
      continue
    fi

    while (( ${#active_pids[@]} >= N_WORKERS )); do
      pid="${active_pids[0]}"
      if ! wait "$pid"; then
        status=1
        log "     !! worker pid=$pid 失败；继续收集其余 task"
      fi
      active_pids=("${active_pids[@]:1}")
    done

    wait_mem "启动 $suite task $task_id worker"
    mkdir -p "$task_dir"
    worker_log="$suite_dir/worker_task_$(printf '%03d' "$task_id").log"
    log "     启动 task $task_id"
    TASK_SUITE="$suite" TASK_IDS="$task_id" NUM_TRIALS="$NUM_TRIALS" \
      OUTPUT_DIR="$task_dir" SERVER_URL="http://localhost:$SERVER_PORT" \
      bash examples/launch_closed_loop_eval_libero_task0.sh --max_steps "$MAX_STEPS" \
      >"$worker_log" 2>&1 &
    active_pids+=("$!")
  done

  for pid in "${active_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
      log "     !! worker pid=$pid 失败"
    fi
  done
  if (( status )); then
    log "  !! $suite 存在失败 worker，不汇总 summary"
    return 1
  fi

  if ! merge_suite_summary "$suite_dir" "$suite"; then
    log "  !! $suite task summary 不完整或不兼容，不汇总"
    return 1
  fi
  log "  -> $suite 完成：$suite_dir/summary.json"
}

all_required_summaries_present() {
  local out_root="$1"
  local suite
  for suite in "${REQUIRED_SUITES[@]}"; do
    [[ -f "$out_root/$suite/summary.json" ]] || return 1
  done
}

run_ckpt() {
  local it="$1"
  local iter_dir="iter_$(printf '%09d' "$it")"
  local ckpt="$PWD/$CHECKPOINT_DIR/$iter_dir"
  local out_root="$RESULTS_ROOT/$iter_dir"
  local suite

  [[ -d "$ckpt" ]] || { log "!! $iter_dir 不存在，跳过"; return 1; }
  if [[ -f "$out_root/.done" ]] && all_required_summaries_present "$out_root"; then
    log "=== $iter_dir 已完整验收，跳过 ==="
    return 0
  fi

  log "=== $iter_dir 开始：steps=$NUM_STEPS, max_steps=$MAX_STEPS, trials=$NUM_TRIALS, workers=$N_WORKERS ==="
  mkdir -p "$out_root" "outputs/train/logs"
  if ! start_server "$ckpt" "$iter_dir"; then
    return 1
  fi
  for suite in $SUITES; do
    run_suite "$suite" "$out_root" || log "  !! $iter_dir/$suite 未完成，保留 task 输出供下次续跑"
  done
  stop_server

  if all_required_summaries_present "$out_root"; then
    touch "$out_root/.done"
    log "=== $iter_dir 全部 4 suite 验收完成 ==="
  else
    log "=== $iter_dir 未完成全部 4 suite；不写 .done ==="
  fi
}

if (( N_WORKERS < 1 )); then
  log "!! N_WORKERS 必须 >= 1，当前为 $N_WORKERS"
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  log "!! Python 不存在或不可执行：$PYTHON_BIN"
  exit 2
fi

log "4090 acceptance 启动：ckpt 倒序=[$CKPT_ITERS]；suite 顺序=[$SUITES]；MEM_GATE=${MEM_GATE_GB}G"
for it in $CKPT_ITERS; do
  run_ckpt "$((10#$it))"
done
log "4090 acceptance driver 已结束"

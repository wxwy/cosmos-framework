#!/usr/bin/env bash
# 一次性推理测试：iter_000001800,denoise num_steps=12,仿真单 episode 上限 700 步。
# 输出到 results/libero_closed_loop_4in1_steps12/iter_000001800/<suite>/。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT="$PWD/outputs/train/cosmos3_action_libero/action_sft/edge_libero_4in1/checkpoints/iter_000001800"
OUT_ROOT="results/libero_closed_loop_4in1_steps12/iter_000001800"
SERVER_LOG="outputs/train/logs/action_server_steps12_iter_000001800.log"
PORT=8000

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUT_ROOT"

log "启动 server: ckpt=iter_000001800 num_steps=12"
CHECKPOINT_PATH="$CKPT" NUM_STEPS=12 bash examples/launch_action_server_libero_edge_all.sh \
  > "$SERVER_LOG" 2>&1 &
server_pid=$!

ready=0; t=0
while (( t < 600 )); do
  if curl -sf "http://localhost:$PORT/" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    log "!! server 提前退出,见 $SERVER_LOG"; exit 1
  fi
  sleep 5; t=$((t+5))
done
[[ "$ready" == 1 ]] || { log "!! server 就绪超时"; kill "$server_pid" 2>/dev/null; exit 1; }
log "server 就绪,开始 4-suite 评测(max_steps=700, num_trials=1, task0)"

for suite in libero_spatial libero_object libero_goal libero_10; do
  log "-> $suite"
  TASK_SUITE="$suite" TASK_IDS=0 NUM_TRIALS=1 OUTPUT_DIR="$OUT_ROOT/$suite" \
    bash examples/launch_closed_loop_eval_libero_task0.sh --max_steps 700 \
    || log "!! $suite eval 失败(exit $?)"
done

kill "$server_pid" 2>/dev/null || true
sleep 3
pkill -f "cosmos_framework.scripts.action_policy_server_libero" 2>/dev/null || true
log "完成: $OUT_ROOT/{libero_spatial,libero_object,libero_goal,libero_10}/summary.json"

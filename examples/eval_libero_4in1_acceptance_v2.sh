#!/usr/bin/env bash
# 正式验收仿真测试（EVAL-LIBERO-4IN1-ACCEPTANCE）：
# 对 4in1 SFT 的 200 倍数 checkpoint 做全量闭环评测：
#   4 suite × 独立 server/client 并行；每 suite 内 num_envs 向量 batch。
#   默认 denoise num_steps=30，保持正式 baseline 推理语义不变。
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
SERVER_PORT_BASE=${SERVER_PORT_BASE:-8000}
SERVER_GPUS=${SERVER_GPUS:-"4,5,6,7"}
SERVER_STAGGER_S=${SERVER_STAGGER_S:-6}
CLIENT_STAGGER_S=${CLIENT_STAGGER_S:-6}
NUM_ENVS=${NUM_ENVS:-4}
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

  log "=== acceptance eval $iter_dir 开始（4 server 并行, num_envs=$NUM_ENVS, num_steps=$NUM_STEPS）==="
  mkdir -p "$out_root"
  wait_mem "启动 $iter_dir servers"
  clear_stale_server

  IFS=',' read -r -a gpu_list <<< "$SERVER_GPUS"
  if (( ${#gpu_list[@]} < ${#SUITES[@]} )); then
    log "!! SERVER_GPUS=$SERVER_GPUS 少于 4 个 suite 所需 GPU"
    return 1
  fi

  local server_pids=() server_ports=() server_ready=()
  local idx suite gpu port server_log
  for idx in "${!SUITES[@]}"; do
    suite="${SUITES[$idx]}"
    gpu="${gpu_list[$idx]}"
    port=$((SERVER_PORT_BASE + idx))
    server_ports[$idx]="$port"
    server_log="outputs/train/logs/action_server_acceptance_${iter_dir}_${suite}.log"
    log "  启动 server[$idx] suite=$suite physical_gpu=$gpu port=$port"
    CUDA_VISIBLE_DEVICES="$gpu" SERVER_PORT="$port" \
      SERVER_OUTPUT_DIR="/tmp/cosmos3_action_server_${iter_dir}_${suite}_${port}" \
      CHECKPOINT_PATH="$ckpt" NUM_STEPS="$NUM_STEPS" \
      bash examples/launch_action_server_libero_edge_all.sh > "$server_log" 2>&1 &
    server_pids[$idx]=$!
    server_ready[$idx]=0
    if (( idx + 1 < ${#SUITES[@]} )); then sleep "$SERVER_STAGGER_S"; fi
  done

  local failed=0 t pid
  for idx in "${!SUITES[@]}"; do
    port="${server_ports[$idx]}"; pid="${server_pids[$idx]}"; t=0
    while (( t < SERVER_READY_TIMEOUT )); do
      if curl -sf "http://localhost:$port/" >/dev/null 2>&1; then
        server_ready[$idx]=1
        break
      fi
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      sleep 5; t=$((t+5))
    done
    if [[ "${server_ready[$idx]}" != 1 ]]; then
      log "  !! ${SUITES[$idx]} server port=$port 未就绪"
      failed=1
    else
      log "  server[$idx] ${SUITES[$idx]} port=$port 就绪"
    fi
  done

  if (( failed != 0 )); then
    for pid in "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    clear_stale_server
    return 1
  fi

  # Four suite clients run concurrently. Each client still batches its own
  # vectorized envs into one /predict_batch request. CLIENT_STAGGER_S intentionally
  # offsets their inference phases to smooth contention with synchronous training.
  local client_pids=() client_rc=()
  for idx in "${!SUITES[@]}"; do
    suite="${SUITES[$idx]}"; port="${server_ports[$idx]}"
    if [[ -f "$out_root/$suite/summary.json" ]]; then
      log "  -> $suite 已有 summary，跳过 client"
      client_pids[$idx]=""
      continue
    fi
    mkdir -p "$out_root/$suite"
    log "  -> 并行 client[$idx] $suite port=$port num_envs=$NUM_ENVS"
    SERVER_URL="http://localhost:$port" TASK_SUITE="$suite" \
      TASK_IDS="0,1,2,3,4,5,6,7,8,9" NUM_TRIALS="$NUM_TRIALS" \
      OUTPUT_DIR="$out_root/$suite" \
      bash examples/launch_closed_loop_eval_libero_task0.sh \
        --max_steps "$MAX_STEPS" --num_envs "$NUM_ENVS" \
        > "$out_root/$suite/eval.log" 2>&1 &
    client_pids[$idx]=$!
    if (( idx + 1 < ${#SUITES[@]} )); then sleep "$CLIENT_STAGGER_S"; fi
  done

  for idx in "${!SUITES[@]}"; do
    pid="${client_pids[$idx]:-}"
    [[ -z "$pid" ]] && continue
    if wait "$pid"; then
      client_rc[$idx]=0
      log "  <- ${SUITES[$idx]} client 完成"
    else
      client_rc[$idx]=$?
      log "  !! ${SUITES[$idx]} client 失败(exit ${client_rc[$idx]})"
    fi
  done

  for pid in "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 3
  clear_stale_server

  local missing=()
  for suite in "${SUITES[@]}"; do
    [[ -f "$out_root/$suite/summary.json" ]] || missing+=("$suite")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$out_root/.done"
    log "=== acceptance eval $iter_dir 完成（4 suite 并行结果齐全）==="
  else
    log "=== acceptance eval $iter_dir 未完成，缺: ${missing[*]}；partial/actions/predictions 已保留 ==="
  fi
}

log "acceptance 评测启动：ckpt 顺序 = $(echo $CKPT_ITERS | tr '\n' ' ')"
for it in $CKPT_ITERS; do
  it_num=$((10#$it))
  [[ -f "$RESULTS_ROOT/iter_$(printf '%09d' "$it_num")/.done" ]] && continue
  run_ckpt "$it_num"
done
log "全部验收评测完成"

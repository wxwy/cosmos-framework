#!/usr/bin/env bash
set -euo pipefail

CHILD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$CHILD/.." && pwd)"
cd "$CHILD"

: "${ROBOCASA_ROOT:?set ROBOCASA_ROOT}"
: "${ROBOCASA_LATENT_CACHE_ROOT:?set ROBOCASA_LATENT_CACHE_ROOT}"
: "${EDGE_POLICY_CHECKPOINT:?set EDGE_POLICY_CHECKPOINT}"
: "${WAN_VAE_PATH:?set WAN_VAE_PATH}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

SAVE_ITER="${SAVE_ITER:-500}"
TTT_TBPTT_STEPS="${TTT_TBPTT_STEPS:-16}"
TTT_DIM="${TTT_DIM:-64}"
TTT_FAST_HIDDEN_DIM="${TTT_FAST_HIDDEN_DIM:-256}"
TTT_K_LOCAL="${TTT_K_LOCAL:-4}"
TTT_INNER_LR="${TTT_INNER_LR:-0.1}"
TTT_B_STREAM="${TTT_B_STREAM:-8}"
TTT_ACTIVE_GA="${TTT_ACTIVE_GA:-2}"
ROBOCASA_NUM_WORKERS="${ROBOCASA_NUM_WORKERS:-6}"

[[ "$SAVE_ITER" == "500" ]] || { echo "SAVE_ITER must be 500" >&2; exit 2; }
[[ "$TTT_TBPTT_STEPS" == "16" ]] || { echo "TTT_TBPTT_STEPS must be 16" >&2; exit 2; }
[[ "$TTT_DIM" == "64" ]] || { echo "TTT_DIM must be 64" >&2; exit 2; }
[[ "$TTT_FAST_HIDDEN_DIM" == "256" ]] || { echo "TTT_FAST_HIDDEN_DIM must be 256" >&2; exit 2; }
[[ "$TTT_K_LOCAL" == "4" ]] || { echo "TTT_K_LOCAL must be 4" >&2; exit 2; }
[[ "$TTT_INNER_LR" == "0.1" ]] || { echo "TTT_INNER_LR must be 0.1" >&2; exit 2; }
[[ "$TTT_B_STREAM" == "8" ]] || { echo "TTT_B_STREAM must be 8" >&2; exit 2; }
[[ "$TTT_ACTIVE_GA" == "2" ]] || { echo "TTT_ACTIVE_GA must be 2" >&2; exit 2; }
[[ "$ROBOCASA_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ROBOCASA_NUM_WORKERS must be positive" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "CUDA_VISIBLE_DEVICES must be 0,1,2,3,4,5,6,7" >&2
  exit 2
}

export SAVE_ITER TTT_TBPTT_STEPS TTT_DIM TTT_FAST_HIDDEN_DIM TTT_K_LOCAL
export TTT_INNER_LR TTT_B_STREAM TTT_ACTIVE_GA ROBOCASA_NUM_WORKERS

EXPECTED_ROOT="${H3F_EXPECTED_ROOT:-$(git -C "$ROOT" rev-parse HEAD)}"
EXPECTED_CHILD="${H3F_EXPECTED_CHILD:-$(git -C "$CHILD" rev-parse HEAD)}"
JOB_NAME="${H3F_JOB_NAME:-edge_local_target_atomic_30k}"
READINESS_STEPS="${H3F_READINESS_STEPS:-}"

if [[ -n "$READINESS_STEPS" ]]; then
  PHASE="fresh"
  ATTEMPT=1
  GROUP="h3f_edge_local_h100_readiness"
  EXTRA_ARGS=(--readiness-steps "$READINESS_STEPS")
else
  GROUP="h3f_edge_local_h100"
  EXTRA_ARGS=()
  JOB_DIR="$OUTPUT_ROOT/psm_wma_v3/$GROUP/$JOB_NAME"
  LATEST="$JOB_DIR/checkpoints/latest_checkpoint.txt"
  if [[ -f "$LATEST" ]]; then
    PHASE="resume"
    max_attempt=1
    if [[ -d "$JOB_DIR/h3f_evidence" ]]; then
      while IFS= read -r name; do
        value="${name#attempt_}"
        value="${value%%_*}"
        if [[ "$value" =~ ^[0-9]+$ ]] && ((10#$value > max_attempt)); then
          max_attempt=$((10#$value))
        fi
      done < <(find "$JOB_DIR/h3f_evidence" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')
    fi
    ATTEMPT=$((max_attempt + 1))
  elif [[ -e "$JOB_DIR" ]]; then
    echo "H3-F job exists without latest_checkpoint.txt; refuse implicit fresh overwrite: $JOB_DIR" >&2
    exit 3
  else
    PHASE="fresh"
    ATTEMPT=1
  fi
fi

echo "H3-F launch: phase=$PHASE attempt=$ATTEMPT job=$JOB_NAME output=$OUTPUT_ROOT"
echo "H3-F pair: root=$EXPECTED_ROOT child=$EXPECTED_CHILD"
if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
  echo "NOTE: BASE_CHECKPOINT_PATH is compatibility input; H3-F effective warmstart is the frozen Stage-A H100 authority."
fi
echo "NOTE: ROBOCASA_NUM_WORKERS=$ROBOCASA_NUM_WORKERS is contract metadata; grouped planner/binder materialization is synchronous."

export PYTHONPATH="$CHILD${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  examples/psm_wma_robocasa_h3f.py \
  --phase "$PHASE" \
  --attempt "$ATTEMPT" \
  --output-root "$OUTPUT_ROOT" \
  --job-name "$JOB_NAME" \
  --expected-root "$EXPECTED_ROOT" \
  --expected-child "$EXPECTED_CHILD" \
  "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# E003 bounded recent-history matched control.
# This formal launcher is pinned to H=64 so smoke runs cannot contaminate the
# auto-resume namespace with a different history horizon.
export PSM_R08_LOCAL_HISTORY_ENABLED=1
export PSM_E003_RECENT_HISTORY_CONTROL=1
export PSM_R08_LOCAL_HISTORY_HORIZON=64
export PSM_R08_HISTORY_MODE=normal
export PSM_LOCAL_DUMMY_ENABLED=0
export PSM_R09_A1_ENABLED=0
export PSM_R09_B1_TTT_ENABLED=0
export PSM_R09_B_TTT_ENABLED=0
export PSM_R09_B_TTT_ACTIVE=0
unset PSM_R09_A1_PROBE_OUTPUT PSM_R09_B1_PROBE_OUTPUT PSM_R09_B2_STREAM_MANIFEST_ROOT

export TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_edge_all_recent_history.toml}"
export RUN_NAME="${RUN_NAME:-edge_libero_4in1_recent_history_h64}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_edge_all.sh"

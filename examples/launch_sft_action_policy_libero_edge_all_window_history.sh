#!/usr/bin/env bash
set -euo pipefail

# E003-WINDOW bounded native history control.
# H=16 completed observations/actions are provided through the native
# multi-vision/action conditioning path. No Local token or learned compressor.
export PSM_HISTORY_MODE=window
export PSM_RECENT_HISTORY_HORIZON=16
export PSM_R08_LOCAL_HISTORY_ENABLED=0
export PSM_E003_RECENT_HISTORY_CONTROL=0
export PSM_LOCAL_DUMMY_ENABLED=0
export PSM_R09_A1_ENABLED=0
export PSM_R09_B1_TTT_ENABLED=0
export PSM_R09_B_TTT_ENABLED=0
export PSM_R09_B_TTT_ACTIVE=0
unset PSM_R09_A1_PROBE_OUTPUT PSM_R09_B1_PROBE_OUTPUT PSM_R09_B2_STREAM_MANIFEST_ROOT

# Cached history latents are the formal training source. Online cache
# verification would compare history placeholders rather than source pixels,
# so keep it disabled for this route.
export LIBERO_LATENT_CACHE_VERIFY_RATIO=0

# Formal WINDOW-H16 training is an 8-rank FSDP run. The TOML carries GA=16,
# giving 16 native samples/rank x 8 ranks x GA16 = 2048 consumers/update.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export TOML_FILE="examples/toml/sft_config/action_policy_libero_edge_all_window_history.toml"
export RUN_NAME="edge_libero_4in1_window_history_h16"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_edge_all.sh"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge-Policy-DROID warm-start recipe for LIBERO action-policy SFT."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG


def _action_policy_libero_edge_model_config() -> dict:
    """Return the Edge config with the admission-smoke settings from R04."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["ema"]["enabled"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    cfg["vlm_config"]["tokenizer"]["repository"] = "${oc.env:EDGE_POLICY_CHECKPOINT}"
    return cfg


action_policy_libero_edge_warmstart = copy.deepcopy(action_policy_libero_nano)
action_policy_libero_edge_warmstart["job"].update(
    name="action_policy_libero_edge_warmstart",
)
action_policy_libero_edge_warmstart["model"]["config"] = _action_policy_libero_edge_model_config()

# Preserve the Policy-DROID action heads. LIBERO owns domain 5; DROID domain 8
# and every other untouched embedding row must survive the warm-start.
action_policy_libero_edge_warmstart["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]

# DomainAwareLinear stores all domains in one Embedding parameter. A LIBERO-only
# batch gives non-domain-5 rows zero gradients, while AdamW would still decay
# them. Disable decay only for the two domain-aware projections.
action_policy_libero_edge_warmstart["optimizer"]["weight_decay_skip_patterns"] = [
    r"action2llm\.(fc|bias)\.weight$",
    r"llm2action\.(fc|bias)\.weight$",
]

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_warmstart",
    node=action_policy_libero_edge_warmstart,
)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge-Policy-DROID warm-start recipe for LIBERO action-policy SFT."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.stdout_loss_logger import StdoutLossLogger
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.utils.lazy_config import LazyCall as L


def _action_policy_libero_edge_model_config() -> dict:
    """Return the Edge config with the admission-smoke settings from R04."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["ema"]["enabled"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    cfg["vlm_config"]["tokenizer"].update(
        repository=None,
        revision=None,
        tokenizer_type="${oc.env:EDGE_POLICY_CHECKPOINT}",
    )
    return cfg


action_policy_libero_edge_warmstart = copy.deepcopy(action_policy_libero_nano)
action_policy_libero_edge_warmstart["job"].update(
    name="action_policy_libero_edge_warmstart",
)
action_policy_libero_edge_warmstart["model"]["config"] = _action_policy_libero_edge_model_config()

# 单卡（dp=1）：官方 8 rank × 4 worker/rank = 32 worker，单卡只剩 4 个。
# 提高 per-rank worker 数补吞吐。机器 cgroup 限 13 CPU，8 worker + 主进程足够。
action_policy_libero_edge_warmstart["dataloader_train"]["dataloader"]["num_workers"] = 13

# Admission smoke is fully offline. Keep the safety/monitor callbacks, but do
# not instantiate the basic group because it unconditionally initializes W&B.
for _default in action_policy_libero_edge_warmstart["defaults"]:
    if "override /callbacks" in _default:
        _default["override /callbacks"] = ["optimization", "job_monitor"]
        break

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

# The ``basic`` callback group is disabled to avoid W&B initialization, but that
# also removes loss logging. Add a lightweight stdout-only logger so the training
# log file captures the loss curve.
action_policy_libero_edge_warmstart["trainer"]["callbacks"]["stdout_loss_logger"] = L(StdoutLossLogger)(
    every_n=1,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_warmstart",
    node=action_policy_libero_edge_warmstart,
)

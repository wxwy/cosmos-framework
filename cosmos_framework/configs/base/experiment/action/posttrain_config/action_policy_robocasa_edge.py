# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge-Policy-DROID DCP 初始化的 RoboCasa raw15 原生基线。"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_nano import (
    action_policy_robocasa_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG


def _robocasa_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["ema"]["enabled"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [33]
    cfg["tokenizer"]["vae_path"] = "${oc.env:WAN_VAE_PATH}"
    cfg["vlm_config"]["model_name"] = "${oc.env:EDGE_POLICY_CHECKPOINT}"
    cfg["vlm_config"]["tokenizer"].update(
        repository=None,
        revision=None,
        tokenizer_type="${oc.env:EDGE_POLICY_CHECKPOINT}",
    )
    return cfg


# 只复用 upstream 数据/优化器/训练骨架；整个模型配置替换为 Edge，避免混入 Nano backbone。
action_policy_robocasa_edge = copy.deepcopy(action_policy_robocasa_nano)
action_policy_robocasa_edge["job"]["name"] = "action_policy_robocasa_edge"
action_policy_robocasa_edge["model"]["config"] = _robocasa_edge_model_config()
action_policy_robocasa_edge["checkpoint"].update(
    load_path="${oc.env:BASE_CHECKPOINT_PATH}",
    keys_to_skip_loading=["net_ema."],
    strict_resume=True,
)
# 2026-09-25 DCP planner 在校验缺失键前移除 skip 项；其余头/生成器必须完整加载。
# DROID 的 15fps → RoboCasa 的 20fps 仅改变数据合同，仍用 64维、32 domains 的既有头。

# RoboCasa 只更新 domain30；避免 AdamW 衰减其余 domain 的预训练投影行。
action_policy_robocasa_edge["optimizer"]["weight_decay_skip_patterns"] = [
    r"action2llm\.(fc|bias)\.weight$",
    r"llm2action\.(fc|bias)\.weight$",
]

ConfigStore.instance().store(
    group="experiment", package="_global_", name="action_policy_robocasa_edge", node=action_policy_robocasa_edge
)

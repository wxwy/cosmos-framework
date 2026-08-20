# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge-Policy-DROID warm-start recipe for LIBERO-all (4-suite) action-policy SFT.

Feeds ``LIBEROLeRobotDataset`` (frame-wise-relative rot6d, ``quantile_rot``,
concat_view third-person + wrist) on all 4 LIBERO suites (equal mix:
``libero_spatial`` / ``libero_object`` / ``libero_goal`` / ``libero_10``) and
trains the generation + action heads from the local Edge-Policy-DROID base.

This is ``action_policy_libero_all_nano`` re-pointed at the Edge model, applying
the same single-GPU / head-preservation changes as
``action_policy_libero_edge_warmstart`` (which does the same on ``libero_10``
alone). ``LIBERO_ROOT`` is the LIBERO_LeRobot_v3 PARENT dir (containing all 4
suites). See docs/action_policy_libero_posttrain.md.
"""

import copy

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano import (
    action_policy_libero_all_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L

_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _action_policy_libero_edge_model_config() -> dict:
    """Edge model config (identical to edge_warmstart): capped packed tokens,
    selective AC, compile off, fresh diffusion-expert init, local Edge tokenizer."""
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


def _action_policy_libero_edge_dataloader():
    """Single-GPU equal mix over the 4 LIBERO suites.

    ``RankPartitionedDataLoader`` assigns one suite per rank and asserts
    ``world_size >= len(datasets)`` — impossible on world_size=1. On a single
    GPU, reproduce the same "each optimizer step sees every suite in equal
    measure" balance with ``IterativeJointDataLoader(seed=None)`` round-robin:
    ``grad_accum_iter=16`` batches = 4 full 1:1:1:1 cycles, so every optimizer
    step sees each suite exactly 4 times. Each suite keeps its own DataLoader;
    the datasets default to ``shard_world_size=1``/``shard_rank=0`` so its
    workers shard episodes disjointly within the suite.
    """

    def _suite_dataset(_suite):
        return L(get_action_libero_sft_dataset)(
            root="${oc.env:LIBERO_ROOT}/" + _suite,
            fps=20,
            chunk_length=16,
            image_size=256,  # concat_view -> 256x512
            mode="wam",
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            pose_coordinate_frame="native",
            action_normalization="quantile_rot",
            val_ratio=0.01,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            resolution=None,
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        )

    return L(IterativeJointDataLoader)(
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,  # None disables token packing (use max_samples_per_batch)
        max_samples_per_batch=128,  # peak-mem bound; global = 128 x 1 x grad_accum 16 = 2048
        sound_latent_fps=0,
        audio_sample_rate=48000,
        seed=None,  # deterministic round-robin 1:1:1:1 (balanced per grad-accum window)
        dataloaders={
            _suite: dict(
                ratio=1,
                dataloader=L(torch.utils.data.DataLoader)(
                    dataset=_suite_dataset(_suite),
                    batch_size=1,
                    in_order=False,
                    num_workers=30,  # 4 suites x 3 = 12 workers + main = 13 (cgroup CPU budget)
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=3,
                    sampler=None,
                ),
            )
            for _suite in _SUITES
        },
    )


action_policy_libero_edge_all = copy.deepcopy(action_policy_libero_all_nano)
action_policy_libero_edge_all["job"].update(
    name="action_policy_libero_edge_all",
)
action_policy_libero_edge_all["model"]["config"] = _action_policy_libero_edge_model_config()

# 单卡多 suite：替换 RankPartitionedDataLoader（world_size>=4 断言不满足）为
# IterativeJointDataLoader 轮询等权混合（每 grad-accum 窗口 16 批 = 4 套 × 4 次）。
action_policy_libero_edge_all["dataloader_train"] = _action_policy_libero_edge_dataloader()

# Admission smoke is fully offline. Keep the safety/monitor callbacks, but do
# not instantiate the basic group because it unconditionally initializes W&B.
for _default in action_policy_libero_edge_all["defaults"]:
    if "override /callbacks" in _default:
        _default["override /callbacks"] = ["optimization", "job_monitor"]
        break

# Preserve the Policy-DROID action heads. LIBERO owns domain 5; DROID domain 8
# and every other untouched embedding row must survive the warm-start.
action_policy_libero_edge_all["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]

# DomainAwareLinear stores all domains in one Embedding parameter. A LIBERO-only
# batch gives non-domain-5 rows zero gradients, while AdamW would still decay
# them. Disable decay only for the two domain-aware projections.
action_policy_libero_edge_all["optimizer"]["weight_decay_skip_patterns"] = [
    r"action2llm\.(fc|bias)\.weight$",
    r"llm2action\.(fc|bias)\.weight$",
]

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_all",
    node=action_policy_libero_edge_all,
)

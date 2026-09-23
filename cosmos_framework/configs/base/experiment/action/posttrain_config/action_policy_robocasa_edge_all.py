# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge-Policy-DROID warm-start recipe for RoboCasa365 v3 action-policy SFT.

One flat RoboCasa365 v3 mirror is trained at a time, selected by
``ROBOCASA_SUITE``.  All six supported suite keys share the same data path,
resume contract, and model recipe.  The first formal PSM-WMA run uses
``robocasa365_target_atomic``.
"""

from __future__ import annotations

import copy
import os

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.action_dataloader_state import ActionIterableShuffleStateCallback
from cosmos_framework.callbacks.stdout_loss_logger import StdoutLossLogger
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano import (
    action_policy_libero_all_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.model.generator.vision_vae import (
    LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES,
    LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS,
)
from cosmos_framework.utils.lazy_config import LazyCall as L


_ROBOCASA_SUITES = (
    "robocasa365_pretrain_atomic",
    "robocasa365_pretrain_mg",
    "robocasa365_pretrain_composite",
    "robocasa365_target_atomic",
    "robocasa365_target_composite_seen",
    "robocasa365_target_composite_unseen",
)


def _robocasa_suite() -> str:
    suite = os.environ.get("ROBOCASA_SUITE", "robocasa365_target_atomic").strip()
    if suite not in _ROBOCASA_SUITES:
        raise ValueError(f"ROBOCASA_SUITE must be one of {_ROBOCASA_SUITES}, got {suite!r}")
    return suite


def _robocasa_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["ema"]["enabled"] = False
    cfg["tokenizer"]["encode_exact_durations"] = LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS
    cfg["tokenizer"]["encode_chunk_frames"] = LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES

    # Baseline recipe: history/local-memory routes are explicitly disabled.
    # Local-TTT is added later as a separate matched recipe.
    cfg["history_mode"] = "none"
    cfg["local_memory_enabled"] = False
    cfg["local_history_enabled"] = False
    cfg["local_ttt_enabled"] = False
    cfg["local_history_horizon"] = 0
    cfg["local_memory_dim"] = None

    cfg["vlm_config"]["tokenizer"].update(
        repository=None,
        revision=None,
        tokenizer_type="${oc.env:EDGE_POLICY_CHECKPOINT}",
    )
    return cfg


def _robocasa_dataloader() -> object:
    suite = _robocasa_suite()
    num_workers = int(os.environ.get("ROBOCASA_NUM_WORKERS", "12"))
    prefetch_factor = int(os.environ.get("ROBOCASA_PREFETCH_FACTOR", "4"))
    if num_workers < 0:
        raise ValueError(f"ROBOCASA_NUM_WORKERS must be non-negative, got {num_workers}")
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError(
            f"ROBOCASA_PREFETCH_FACTOR must be positive when workers are enabled, got {prefetch_factor}"
        )

    dataset = L(get_action_robocasa_sft_dataset)(
        root="${oc.env:ROBOCASA_ROOT}",
        suite=suite,
        fps=20.0,
        chunk_length=16,
        mode="wam",
        camera_set="left_wrist",
        use_state=True,
        use_base_action=True,
        base_encoding="raw",
        action_normalization=None,
        split="train",
        split_val_ratio=0.01,
        split_seed=42,
        resolution=None,
        max_action_dim="${model.config.max_action_dim}",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        cfg_dropout_rate=0.1,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
        shuffle_state_name=suite,
        sample_stride=1,
        # Optional: unset/empty means decode source video and run the VAE online.
        # A non-empty path switches RoboCasaLeRobotDataset to exact-window cached
        # latents and skips source-video decoding.
        latent_cache_root=os.environ.get("ROBOCASA_LATENT_CACHE_ROOT") or None,
    )

    return L(IterativeJointDataLoader)(
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=128,
        sound_latent_fps=0,
        audio_sample_rate=48000,
        seed=None,
        lazy_initialize_child_iterators=True,
        dataloaders={
            suite: dict(
                ratio=1,
                dataloader=L(torch.utils.data.DataLoader)(
                    dataset=dataset,
                    batch_size=1,
                    in_order=False,
                    num_workers=num_workers,
                    persistent_workers=num_workers > 0,
                    pin_memory=True,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    sampler=None,
                ),
            )
        },
    )


action_policy_robocasa_edge_all = copy.deepcopy(action_policy_libero_all_nano)
action_policy_robocasa_edge_all["job"].update(
    project="cosmos3_action_robocasa",
    group="action_sft",
    name="edge_robocasa365",
    wandb_mode="disabled",
)
action_policy_robocasa_edge_all["model"]["config"] = _robocasa_edge_model_config()
action_policy_robocasa_edge_all["dataloader_train"] = _robocasa_dataloader()
action_policy_robocasa_edge_all["dataloader_val"] = None

# One-rank recipe; the TOML restores the official 2048-consumer/update scale by GA=16.
action_policy_robocasa_edge_all["model"]["config"]["parallelism"]["data_parallel_shard_degree"] = -1
action_policy_robocasa_edge_all["model"]["config"]["parallelism"]["data_parallel_replicate_degree"] = 1

# Preserve the DROID-trained action heads. RoboCasa uses a different embodiment
# domain but the shared DomainAwareLinear embedding parameters must not decay
# untouched domain rows.
action_policy_robocasa_edge_all["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_robocasa_edge_all["optimizer"]["weight_decay_skip_patterns"] = [
    r"action2llm\.(fc|bias)\.weight$",
    r"llm2action\.(fc|bias)\.weight$",
]

# Fully offline training callbacks: no W&B, but keep optimizer safety/monitoring.
for _default in action_policy_robocasa_edge_all["defaults"]:
    if "override /callbacks" in _default:
        _default["override /callbacks"] = ["optimization", "job_monitor"]
        break

action_policy_robocasa_edge_all["trainer"]["callbacks"]["stdout_loss_logger"] = L(StdoutLossLogger)(
    every_n=1,
)
action_policy_robocasa_edge_all["trainer"]["callbacks"]["action_dataloader_state"] = L(
    ActionIterableShuffleStateCallback
)()

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa_edge_all",
    node=action_policy_robocasa_edge_all,
)

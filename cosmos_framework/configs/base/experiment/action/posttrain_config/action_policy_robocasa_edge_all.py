# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa365 Edge-Policy-DROID + active Local-TTT training recipe.

RoboCasa's formal PSM-WMA training route starts directly with Local-TTT.  The
target-atomic N=100 latent cache is consumed by the canonical segment producer;
the outer dataloader is only a trainer clock on the active route.
"""

from __future__ import annotations

import copy
import os

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.stdout_loss_logger import StdoutLossLogger
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano import (
    action_policy_libero_all_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.model.generator.mot.active_local_memory_launch import ActiveLocalMemoryLaunchCallback
from cosmos_framework.model.generator.mot.config_checkpoint_contract import SELECTORS as TTT_SLOW_GROUP_SELECTORS
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


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _robocasa_suite() -> str:
    suite = os.environ.get("ROBOCASA_SUITE", "robocasa365_target_atomic").strip()
    if suite not in _ROBOCASA_SUITES:
        raise ValueError(f"ROBOCASA_SUITE must be one of {_ROBOCASA_SUITES}, got {suite!r}")
    return suite


def _robocasa_edge_local_ttt_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["ema"]["enabled"] = False
    cfg["tokenizer"]["encode_exact_durations"] = LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS
    cfg["tokenizer"]["encode_chunk_frames"] = LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES

    # Formal RoboCasa route: active continual Local-TTT from the first run.
    cfg["history_mode"] = "ttt"
    cfg["local_memory_enabled"] = True
    cfg["local_memory_dim"] = 32
    cfg["local_history_enabled"] = True
    cfg["local_history_backend"] = "ttt_fast_weight"
    cfg["local_history_horizon"] = 0
    cfg["local_history_evidence_dim"] = 256
    cfg["local_history_action_dim"] = 20
    cfg["local_history_state_enabled"] = False
    cfg["local_history_canonical_evidence"] = False
    cfg["local_ttt_enabled"] = True
    cfg["ttt_tbptt_steps"] = _env_int("PSM_R09_B_TTT_TBPTT_STEPS", 16)
    cfg["ttt_inner_lr"] = _env_float("PSM_R09_B_TTT_INNER_LR", 0.1)
    cfg["ttt_dim"] = _env_int("PSM_R09_B_TTT_DIM", 64)
    cfg["ttt_fast_hidden_dim"] = _env_int("PSM_R09_B_TTT_FAST_HIDDEN_DIM", 256)
    cfg["k_local"] = _env_int("PSM_R09_B_TTT_K_LOCAL", 4)
    if cfg["k_local"] not in {1, 4, 8, 16}:
        raise ValueError("PSM_R09_B_TTT_K_LOCAL must be one of 1,4,8,16")
    cfg["local_evidence_feature_version"] = "causal_visual96_executed_action20_v1"
    cfg["local_fast_state_dtype"] = "fp32"
    cfg["local_runtime_resume_mode"] = "slow_only_no_mid_episode_resume"

    cfg["vlm_config"]["tokenizer"].update(
        repository=None,
        revision=None,
        tokenizer_type="${oc.env:EDGE_POLICY_CHECKPOINT}",
    )
    return cfg


def _robocasa_dataset(*, iterable_shuffle: bool):
    suite = _robocasa_suite()
    return L(get_action_robocasa_sft_dataset)(
        root="${oc.env:ROBOCASA_ROOT}",
        suite=suite,
        fps=20.0,
        chunk_length=16,
        mode="wam",
        camera_set="left_wrist",
        use_state=True,
        use_base_action=True,
        base_encoding="ego",
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
        iterable_shuffle=iterable_shuffle,
        episode_shuffle_seed=42,
        shuffle_state_name=suite,
        sample_stride=1,
        latent_cache_root=os.environ.get("ROBOCASA_LATENT_CACHE_ROOT") or None,
    )


def _robocasa_active_dataloader():
    suite = _robocasa_suite()
    active_datasets = {suite: _robocasa_dataset(iterable_shuffle=False)}

    # Active Local-TTT ignores this batch content; one map-style sample is enough
    # to drive the trainer fetch/device path. The canonical producer owns the real
    # B_stream x TBPTT training samples.
    outer_dataset = _robocasa_dataset(iterable_shuffle=False)
    loader = L(IterativeJointDataLoader)(
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        sound_latent_fps=0,
        audio_sample_rate=48000,
        seed=None,
        lazy_initialize_child_iterators=True,
        dataloaders={
            suite: dict(
                ratio=1,
                dataloader=L(torch.utils.data.DataLoader)(
                    dataset=outer_dataset,
                    batch_size=1,
                    in_order=True,
                    num_workers=0,
                    persistent_workers=False,
                    pin_memory=True,
                    prefetch_factor=None,
                    sampler=None,
                ),
            )
        },
    )
    return loader, active_datasets


action_policy_robocasa_edge_all = copy.deepcopy(action_policy_libero_all_nano)
action_policy_robocasa_edge_all["job"].update(
    project="cosmos3_action_robocasa",
    group="action_sft",
    name="local_ttt_robocasa365_target_atomic_n100_fsdp8_k4",
    wandb_mode="disabled",
)
action_policy_robocasa_edge_all["model"]["config"] = _robocasa_edge_local_ttt_model_config()
action_policy_robocasa_edge_all["dataloader_train"], _robocasa_active_datasets = _robocasa_active_dataloader()
action_policy_robocasa_edge_all["dataloader_val"] = None

# 8-GPU FSDP: B_stream=8, T=16, GA=2 gives
# 8 slots x 16 consumers x 2 members x 8 ranks = 2048 consumers/update.
action_policy_robocasa_edge_all["model"]["config"]["parallelism"]["data_parallel_shard_degree"] = -1
action_policy_robocasa_edge_all["model"]["config"]["parallelism"]["data_parallel_replicate_degree"] = 1
active_ga = _env_int("PSM_R09_B_TTT_ACTIVE_GA", 2)
b_stream = _env_int("PSM_R09_B_TTT_B_STREAM", 8)
action_policy_robocasa_edge_all["trainer"]["grad_accum_iter"] = active_ga

# Train the inherited generation/action heads together with the Local-TTT slow owner.
baseline_selectors = list(action_policy_libero_all_nano["optimizer"]["keys_to_select"])
action_policy_robocasa_edge_all["optimizer"]["keys_to_select"] = (
    baseline_selectors + list(TTT_SLOW_GROUP_SELECTORS)
)

# Preserve DROID action heads on warm-start; do not decay untouched domain rows.
action_policy_robocasa_edge_all["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_robocasa_edge_all["optimizer"]["weight_decay_skip_patterns"] = [
    r"action2llm\.(fc|bias)\.weight$",
    r"llm2action\.(fc|bias)\.weight$",
]

for default in action_policy_robocasa_edge_all["defaults"]:
    if "override /callbacks" in default:
        default["override /callbacks"] = ["optimization", "job_monitor"]
        break

action_policy_robocasa_edge_all["trainer"]["callbacks"]["stdout_loss_logger"] = L(StdoutLossLogger)(every_n=1)
action_policy_robocasa_edge_all["trainer"]["callbacks"]["r09_b_active_wiring"] = L(
    ActiveLocalMemoryLaunchCallback
)(
    suite_datasets=_robocasa_active_datasets,
    b_stream=b_stream,
    member_layout="a2",
    group_size=b_stream,
    ttt_tbptt_steps=action_policy_robocasa_edge_all["model"]["config"]["ttt_tbptt_steps"],
    manifest_digest=f"{_robocasa_suite()}-n100",
    config_digest=(
        "robocasa-local-ttt-"
        f"t{action_policy_robocasa_edge_all['model']['config']['ttt_tbptt_steps']}-"
        f"d{action_policy_robocasa_edge_all['model']['config']['ttt_dim']}-"
        f"h{action_policy_robocasa_edge_all['model']['config']['ttt_fast_hidden_dim']}-"
        f"k{action_policy_robocasa_edge_all['model']['config']['k_local']}-v1"
    ),
    source_digest=os.environ.get("ROBOCASA_LATENT_CACHE_ROOT") or "robocasa-exact-window-cache",
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa_edge_all",
    node=action_policy_robocasa_edge_all,
)

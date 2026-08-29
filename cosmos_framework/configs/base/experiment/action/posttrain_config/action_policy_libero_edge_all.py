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
import os

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.online_vae_probe import OnlineVAEProbeCallback
from cosmos_framework.callbacks.r07_parity_capture import R07ParityCaptureCallback
from cosmos_framework.callbacks.r07_runtime_probe import R07RuntimeProbeCallback
from cosmos_framework.callbacks.r08_gate_a_probe import R08GateAProbeCallback
from cosmos_framework.callbacks.r08_gate_b_provenance import R08GateBProvenanceCallback
from cosmos_framework.callbacks.stdout_loss_logger import StdoutLossLogger
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano import (
    action_policy_libero_all_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.model.generator.vision_vae import (
    LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES,
    LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS,
)
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
    cfg["tokenizer"]["encode_exact_durations"] = LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS
    cfg["tokenizer"]["encode_chunk_frames"] = LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES
    local_dummy_enabled = os.environ.get("PSM_LOCAL_DUMMY_ENABLED", "0") == "1"
    local_history_enabled = os.environ.get("PSM_R08_LOCAL_HISTORY_ENABLED", "0") == "1"
    if local_dummy_enabled and local_history_enabled:
        raise ValueError("PSM_LOCAL_DUMMY_ENABLED and PSM_R08_LOCAL_HISTORY_ENABLED are mutually exclusive")
    local_dummy_mode = os.environ.get("PSM_LOCAL_DUMMY_MODE", "normal")
    if local_dummy_mode not in {"normal", "zero", "shuffle"}:
        raise ValueError(f"unsupported PSM_LOCAL_DUMMY_MODE: {local_dummy_mode}")
    cfg["local_memory_enabled"] = local_dummy_enabled or local_history_enabled
    cfg["local_history_enabled"] = local_history_enabled
    cfg["local_history_horizon"] = int(os.environ.get("PSM_R08_LOCAL_HISTORY_HORIZON", "16"))
    if cfg["local_history_horizon"] < 0:
        raise ValueError("PSM_R08_LOCAL_HISTORY_HORIZON must be non-negative")
    cfg["local_memory_dim"] = int(os.environ.get("PSM_LOCAL_DUMMY_DIM", "32")) if cfg["local_memory_enabled"] else None
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

    local_history_enabled = os.environ.get("PSM_R08_LOCAL_HISTORY_ENABLED", "0") == "1"

    def _suite_dataset(_suite):
        latent_cache_root = os.environ.get("LIBERO_LATENT_CACHE_ROOT")
        cache_kwargs = (
            {
                "latent_cache_root": f"{latent_cache_root}/{_suite}",
                # cache 模式默认完全跳过 VAE；仅在显式设置大于 0 的比例时做在线抽检。
                "latent_cache_verify_ratio": float(os.environ.get("LIBERO_LATENT_CACHE_VERIFY_RATIO", "0.0")),
            }
            if latent_cache_root
            else {}
        )
        if os.environ.get("LIBERO_MAX_EPISODES"):
            max_episodes = int(os.environ["LIBERO_MAX_EPISODES"])
            if max_episodes <= 0:
                raise ValueError(f"LIBERO_MAX_EPISODES must be a positive integer, got {max_episodes}")
            cache_kwargs["max_episodes"] = max_episodes
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
            local_dummy_enabled=os.environ.get("PSM_LOCAL_DUMMY_ENABLED", "0") == "1",
            local_dummy_tokens=int(os.environ.get("PSM_LOCAL_DUMMY_TOKENS", "1")),
            local_dummy_dim="${model.config.local_memory_dim}",
            local_dummy_mode=os.environ.get("PSM_LOCAL_DUMMY_MODE", "normal"),
            local_history_horizon=int(os.environ.get("PSM_R08_LOCAL_HISTORY_HORIZON", "16"))
            if local_history_enabled
            else 0,
            **cache_kwargs,
        )

    _num_workers = int(os.environ.get("LIBERO_NUM_WORKERS", "32"))
    _prefetch_factor = int(os.environ.get("LIBERO_PREFETCH_FACTOR", "4"))
    if _num_workers > 0 and _prefetch_factor <= 0:
        raise ValueError(
            f"LIBERO_PREFETCH_FACTOR must be a positive integer when LIBERO_NUM_WORKERS > 0, got {_prefetch_factor}"
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
        local_memory_shuffle=os.environ.get("PSM_LOCAL_DUMMY_MODE", "normal") == "shuffle",
        dataloaders={
            _suite: dict(
                ratio=1,
                dataloader=L(torch.utils.data.DataLoader)(
                    dataset=_suite_dataset(_suite),
                    batch_size=1,
                    in_order=False,
                    num_workers=_num_workers,
                    persistent_workers=_num_workers > 0,
                    pin_memory=True,
                    prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
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

# 本配方继承 Nano 的选择式 optimizer allowlist；保留全部原生选择，仅在启用时加入 R07 Local 参数。
if action_policy_libero_edge_all["model"]["config"]["local_memory_enabled"]:
    action_policy_libero_edge_all["optimizer"]["keys_to_select"] += [
        "local_memory2llm",
        "local_memory_modality_embed",
    ]
if action_policy_libero_edge_all["model"]["config"]["local_history_enabled"]:
    action_policy_libero_edge_all["optimizer"]["keys_to_select"].append("local_history_runtime")

if os.environ.get("PSM_R09_A1_ENABLED", "0") == "1":
    action_policy_libero_edge_all["optimizer"]["keys_to_select"] = [
        "local_history_runtime.encoder",
        "local_history_runtime.recurrent_backend",
        "local_history_runtime.readout",
        "local_memory2llm",
        "local_memory_modality_embed",
    ]

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

# The ``basic`` callback group is disabled to avoid W&B initialization, but that
# also removes loss logging. Add a lightweight stdout-only logger so the training
# log file captures the loss curve.
action_policy_libero_edge_all["trainer"]["callbacks"]["stdout_loss_logger"] = L(StdoutLossLogger)(
    every_n=1,
)

_r07_probe_output = os.environ.get("PSM_R07_RUNTIME_PROBE_OUTPUT")
if _r07_probe_output:
    action_policy_libero_edge_all["trainer"]["callbacks"]["r07_runtime_probe"] = L(R07RuntimeProbeCallback)(
        output_path=_r07_probe_output,
    )

_r08_probe_output = os.environ.get("PSM_R08_GATE_A_PROBE_OUTPUT")
if _r08_probe_output:
    action_policy_libero_edge_all["trainer"]["callbacks"]["r08_gate_a_probe"] = L(R08GateAProbeCallback)(
        output_path=_r08_probe_output,
    )

_r08_device_monitor_every_n = os.environ.get("PSM_R08_GATE_A_DEVICE_MONITOR_EVERY_N")
if _r08_device_monitor_every_n:
    action_policy_libero_edge_all["trainer"]["callbacks"]["device_monitor"]["every_n"] = int(
        _r08_device_monitor_every_n
    )

_r07_parity_output = os.environ.get("PSM_R07_PARITY_OUTPUT")
if _r07_parity_output:
    action_policy_libero_edge_all["trainer"]["callbacks"]["r07_parity_capture"] = L(R07ParityCaptureCallback)(
        output_path=_r07_parity_output,
        tensor_output_path=os.environ.get("PSM_R07_PARITY_TENSOR_OUTPUT"),
    )

if os.environ.get("PSM_R08_GATE_B_CAPTURE_ONLY", "0") == "1":
    action_policy_libero_edge_all["trainer"]["callbacks"].pop("termination_signal_checkpoint", None)

_r08_gate_b_provenance = os.environ.get("PSM_R08_GATE_B_PROVENANCE_OUTPUT")
if _r08_gate_b_provenance:
    action_policy_libero_edge_all["trainer"]["callbacks"]["r08_gate_b_provenance"] = L(R08GateBProvenanceCallback)(
        output_path=_r08_gate_b_provenance
    )

# Disabled unless explicitly requested. The probe captures pre-normalization uint8
# frames and the exact online VAE outputs for offline-cache parity verification.
_probe_output = os.environ.get("ONLINE_VAE_PROBE_OUTPUT")
if _probe_output:
    action_policy_libero_edge_all["trainer"]["callbacks"]["online_vae_probe"] = L(OnlineVAEProbeCallback)(
        output_dir=_probe_output,
        max_samples=int(os.environ.get("ONLINE_VAE_PROBE_MAX_SAMPLES", "200")),
    )

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_all",
    node=action_policy_libero_edge_all,
)

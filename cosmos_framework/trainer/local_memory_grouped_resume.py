# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-D per-rank DCP dataloader 组件中的严格 grouped Local runtime 状态。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTFastState
from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLiveState, GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.local_memory_segment import (
    RankLocalSegmentScheduler,
    SegmentIdentity,
    SegmentProvenance,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import CatalogFrontier
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io

FORMAT = "psm_v3_h3d_grouped_local_v1"


def _profile(window: GroupedLocalMemoryWindow) -> tuple[Any, ...]:
    runtime = window.model.net.local_memory_runtime
    encoder, core = runtime.encoder, runtime.core
    return (
        window.planner.rank,
        window.planner.world_size,
        window.planner.seed,
        encoder.visual_proj.in_features,
        encoder.action_proj.in_features,
        encoder.evidence_dim,
        core.ttt_dim,
        core.fast_hidden_dim,
        core.local_dim,
        core.k_local,
        core.ttt_tbptt_steps,
        core.inner_lr,
        2,  # active_GA
        64,  # Edge action model-space
        32,  # policy chunk
        33,  # policy consumer frames
    )


def snapshot_grouped_local_state(
    window: GroupedLocalMemoryWindow, *, iteration: int, config_digest: str
) -> dict[str, Any]:
    if window.plan is not None or type(iteration) is not int or iteration < 0 or not config_digest:
        raise RuntimeError("只能在成功 optimizer step 后保存 committed grouped 状态")
    sidecar = {
        identity.slot_id: (
            identity,
            provenance,
            tuple(value.detach().to("cpu", dtype=torch.float32).clone() for value in fast),
        )
        for identity, provenance, fast in window.live.sidecar.snapshot()
    }
    return {
        "format": FORMAT,
        "iteration": iteration,
        "manifest_digest": window.planner.catalog.manifest_digest,
        "config_digest": config_digest,
        "profile": _profile(window),
        "frontier": window.live.frontier,
        "scheduler": window.live.scheduler._committed.copy(),
        "sidecar": sidecar,
    }


def restore_grouped_local_state(
    window: GroupedLocalMemoryWindow, state: dict[str, Any], *, iteration: int, config_digest: str
) -> None:
    """所有字段先在候选对象验证；最后一个引用赋值才改变 live。"""
    if window.plan is not None or not isinstance(state, dict):
        raise RuntimeError("pending window 或非法 checkpoint 不可恢复")
    if (
        set(state)
        != {
            "format",
            "iteration",
            "manifest_digest",
            "config_digest",
            "profile",
            "frontier",
            "scheduler",
            "sidecar",
        }
        or state["format"] != FORMAT
        or type(state["iteration"]) is not int
        or state["iteration"] != iteration
        or state["manifest_digest"] != window.planner.catalog.manifest_digest
        or state["config_digest"] != config_digest
        or state["profile"] != _profile(window)
    ):
        raise ValueError("Local checkpoint schema/iteration/manifest/config/profile 不匹配")
    frontier, committed, saved = state["frontier"], state["scheduler"], state["sidecar"]
    if type(frontier) is not CatalogFrontier or type(committed) is not dict or type(saved) is not dict:
        raise ValueError("Local checkpoint frontier/scheduler/sidecar 类型不合法")
    window.planner.plan_window(frontier)
    expected_slots = {window.planner.rank * 8 + index for index in range(8)}
    if set(committed) != expected_slots or not set(saved).issubset(expected_slots):
        raise ValueError("Local checkpoint 的8个 stable slot 不完整或越界")
    core = window.model.net.local_memory_runtime.core
    by_source_digest = {episode.source_digest: episode for episode in window.planner.episodes}
    records = {}
    for local_slot, slot in enumerate(frontier.slots):
        slot_id = window.planner.rank * 8 + local_slot
        identity = committed[slot_id]
        if type(identity) is not SegmentIdentity or identity.slot_id != slot_id:
            raise ValueError("Local checkpoint scheduler identity 不合法")
        if slot.next_segment_id != identity.segment_id + 1:
            raise ValueError("Local checkpoint segment_id 不连续")
        if slot.uid is None:
            episode = by_source_digest.get(identity.source_digest)
            if (
                not identity.training_stream_end
                or slot_id in saved
                or episode is None
                or identity.episode_id != episode.key.episode_id
                or identity.category != episode.key.task
                or identity.cursor != episode.segment_count - 1
            ):
                raise ValueError("terminal slot 不得保留 fast state")
            continue
        episode = window.planner.by_uid[slot.uid]
        if (
            identity.training_stream_end
            or slot.cursor != identity.cursor + 1
            or identity.episode_id != episode.key.episode_id
            or identity.category != episode.key.task
            or identity.source_digest != episode.source_digest
            or slot_id not in saved
        ):
            raise ValueError("Local checkpoint slot→episode/cursor/source 不一致")
        record = saved[slot_id]
        if type(record) is not tuple or len(record) != 3:
            raise ValueError("Local checkpoint sidecar record 不合法")
        recorded_identity, provenance, values = record
        if (
            recorded_identity != identity
            or type(provenance) is not SegmentProvenance
            or provenance.manifest_digest != state["manifest_digest"]
            or provenance.config_digest != config_digest
            or provenance.source_digest != identity.source_digest
            or provenance.segment_id != identity.segment_id
            or type(values) is not tuple
            or len(values) != 4
        ):
            raise ValueError("Local checkpoint provenance/fast state 不匹配")
        if any(
            not isinstance(value, torch.Tensor)
            or isinstance(value, nn.Parameter)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise ValueError("Local checkpoint fast state 必须为 CPU finite fp32 普通 tensor")
        fast = ContinualTTTFastState(*(value.to(core.slot_queries.device).clone() for value in values))
        core.validate_state(fast, 1)
        records[slot_id] = (identity, provenance, fast)
    sidecar = LocalMemorySegmentSidecar()
    sidecar._records = records
    scheduler = RankLocalSegmentScheduler()
    scheduler._committed = committed.copy()
    window._live = GroupedLiveState(frontier, sidecar, scheduler)


class GroupedLocalMemoryStateCallback(Callback):
    """占用本 profile 的 DCP dataloader rank-pkl；现有 RNG 仍由 trainer 组件保存。"""

    checkpoint_component = "dataloader"

    def __init__(self, trainer: Any) -> None:
        super().__init__()
        self.trainer = trainer

    def has_checkpoint_state(self) -> bool:
        return hasattr(self.trainer, "_grouped_planner")

    def state_dict(self) -> dict[str, Any]:
        window = getattr(self.trainer, "_grouped_window", None)
        if window is None:
            raise RuntimeError("Local window 尚未建立；禁止零步 checkpoint")
        return snapshot_grouped_local_state(
            window,
            iteration=self.trainer._grouped_completed_iteration,
            config_digest=self.trainer._grouped_config_digest,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.has_checkpoint_state() or getattr(self.trainer, "_grouped_window", None) is not None:
            raise RuntimeError("Local checkpoint 必须在创建 grouped window 前加载")
        self.trainer._pending_grouped_resume = deepcopy(state_dict)


def require_dcp_grouped_resume_component(checkpointer: Any) -> bool:
    """同 job 恢复必须有本 rank 的 dataloader state；Stage-A warm-start 不需要。"""
    keys, source = checkpointer.keys_to_resume_during_load()
    if source is None:
        return False
    if source.warm_start:
        if checkpointer.load_training_state:
            raise ValueError("H3-D Stage-A warm-start 禁止继承外部训练状态")
        return False
    if "dataloader" not in keys:
        raise FileNotFoundError("同 job Local 恢复缺少 DCP dataloader 组件")
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    name = f"rank_{rank}.pkl"
    path = (
        f"{source.path.rstrip('/')}/dataloader/{name}"
        if source.uses_object_store
        else str(Path(source.path) / "dataloader" / name)
    )
    if not easy_io.exists(path, backend_key=source.backend_key):
        raise FileNotFoundError(f"同 job Local 恢复缺少本 rank 状态：{path}")
    return True

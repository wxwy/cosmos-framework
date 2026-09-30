# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-C 两个 grouped member 的候选 Local 状态与 optimizer 后发布边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentBatch,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
    SegmentScanResult,
)
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import (
    CatalogFrontier,
    GroupedWindowPlan,
    RankLocalGroupedPlanner,
    gather_same_index,
)

NativeBatchLoss = Callable[[tuple[Any, ...], tuple[torch.Tensor | None, ...], int], torch.Tensor]
Backward = Callable[[torch.Tensor, bool], None]
OptimizerStep = Callable[[], bool]


@dataclass(frozen=True)
class GroupedLiveState:
    frontier: CatalogFrontier
    sidecar: LocalMemorySegmentSidecar
    scheduler: RankLocalSegmentScheduler


class GroupedLocalMemoryWindow:
    """两段 scan/backward 在 overlay 内推进；成功 optimizer step 后只换一个 live 引用。"""

    def __init__(self, model: nn.Module, planner: RankLocalGroupedPlanner) -> None:
        self.model, self.planner = model, planner
        self._live = GroupedLiveState(
            planner.initial_frontier(), LocalMemorySegmentSidecar(), RankLocalSegmentScheduler()
        )
        self._plan: GroupedWindowPlan | None = None
        self._candidate: GroupedLiveState | None = None
        self._member_index = 0
        runtime = model.net.local_memory_runtime
        if runtime.encoder.action_proj.in_features != 15 or runtime.core.ttt_tbptt_steps != 16:
            raise ValueError("H3-C 要求冻结 raw15/T16 Local runtime")
        scan = getattr(model.net, "scan_local_memory", None)
        if any(isinstance(parameter, DTensor) for parameter in runtime.parameters()) and (
            scan is None or not getattr(model.net, "_local_memory_scan_fsdp_registered", False)
        ):
            raise RuntimeError("FSDP Local 参数要求注册 model-owned scan_local_memory")
        self._scan = scan

    @property
    def live(self) -> GroupedLiveState:
        return self._live

    @property
    def plan(self) -> GroupedWindowPlan | None:
        return self._plan

    def begin(self) -> GroupedWindowPlan:
        if self._candidate is not None:
            raise RuntimeError("已有 pending grouped window")
        reset_telemetry = getattr(self.model.net.local_memory_runtime.core, "reset_telemetry", None)
        if reset_telemetry is not None:
            reset_telemetry()
        plan = self.planner.plan_window(self._live.frontier)
        sidecar = LocalMemorySegmentSidecar()
        sidecar._records = {
            identity.slot_id: (identity, provenance, state)
            for identity, provenance, state in self._live.sidecar.snapshot()
        }
        scheduler = RankLocalSegmentScheduler()
        scheduler._committed = self._live.scheduler._committed.copy()
        self._candidate = GroupedLiveState(plan.candidate_frontier, sidecar, scheduler)
        self._plan, self._member_index = plan, 0
        return plan

    def _require_pending(self) -> tuple[GroupedWindowPlan, GroupedLiveState]:
        if self._plan is None or self._candidate is None:
            raise RuntimeError("grouped window 尚未 begin 或已终止")
        return self._plan, self._candidate

    def run_member(
        self,
        segments: tuple[SegmentBatch, ...],
        native_loss: NativeBatchLoss,
        backward: Backward,
    ) -> torch.Tensor:
        plan, candidate = self._require_pending()
        member_index = self._member_index
        if member_index >= 2 or len(segments) != 8:
            self.abort()
            raise ValueError("grouped window 只接受两个各8 slot member")
        requests = plan.members[member_index]
        expected_slots = tuple(request.identity.slot_id for request in requests)
        if tuple(int(segment.slot_id[0]) for segment in segments) != expected_slots:
            self.abort()
            raise ValueError("member segment slot 顺序不匹配冻结 plan")
        runtime = self.model.net.local_memory_runtime
        pending: list[tuple[CanonicalLocalMemorySegmentAdapter, LocalMemoryTransaction, SegmentScanResult, Any]] = []
        try:
            for request, segment in zip(requests, segments, strict=True):
                identity = request.identity
                transaction = LocalMemoryTransaction(
                    GAWindowPlan((identity.member,), (request.valid_count,)), candidate.scheduler
                )
                adapter = CanonicalLocalMemorySegmentAdapter(
                    runtime.encoder,
                    runtime.core,
                    candidate.sidecar,
                    scan_local_memory=self._scan,
                )
                result = adapter.scan(segment, identity=identity, transaction=transaction)
                pending.append((adapter, transaction, result, identity))
            if sum(len(result.payloads) for _, _, result, _ in pending) != plan.member_counts[member_index]:
                raise RuntimeError("member scan 有效 consumer 数与 plan 不一致")
            active_indices = [index for index in range(16) if any(bool(s.consumer_valid[0, index]) for s in segments)]
            total = None
            for index in active_indices:
                batch = gather_same_index(segments, index)
                prefixes = tuple(
                    pending[row][2].locals[index]
                    for row, segment in enumerate(segments)
                    if bool(segment.consumer_valid[0, index])
                )
                loss = native_loss(batch.payloads, prefixes, index)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
                    raise ValueError("native callback 必须返回可微标量 outer loss")
                if not bool(torch.isfinite(loss)):
                    raise ValueError("native outer loss 非有限")
                weighted = loss * (len(batch.payloads) / plan.n_window)
                backward(weighted, index != active_indices[-1])
                detached = weighted.detach()
                total = detached if total is None else total + detached
            for adapter, transaction, result, identity in pending:
                transaction.successful_backward(identity, result)
                adapter.commit(identity, result, transaction=transaction)
            self._member_index += 1
            assert total is not None
            return total
        except Exception:
            for adapter, transaction, result, identity in pending:
                if adapter._pending is not None:
                    adapter.discard_pending(identity, result, transaction=transaction)
            self.abort()
            raise

    def finish(self, optimizer_step: OptimizerStep) -> None:
        plan, candidate = self._require_pending()
        if self._member_index != 2:
            self.abort()
            raise RuntimeError("两个 grouped member 均完成 backward 才能 optimizer step")
        try:
            if optimizer_step() is not True:
                raise RuntimeError("optimizer 未执行成功；禁止发布 Local 状态")
        except Exception:
            self.abort()
            raise
        # candidate 的 scan/transaction 已全部完成；单个引用切换发布所有 slot 与 frontier。
        self._live = candidate
        self._plan, self._candidate, self._member_index = None, None, 0

    def abort(self) -> None:
        self._plan, self._candidate, self._member_index = None, None, 0

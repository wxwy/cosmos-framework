# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected grouped Local scan 的候选状态与 optimizer 后发布边界。"""

from __future__ import annotations

from contextlib import nullcontext
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
    BatchedLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowCatalogFrontier,
    ExactWindowGroupedPlan,
    ExactWindowRankPlanner,
)

# Third argument is one exact (slot_id, local_t_index) coordinate per consumer.
# Packed model input uses slot-major order, as V2 A2; audit can reconstruct the
# previous index-major chronology without changing consumer data or RNG here.
NativeBatchLoss = Callable[
    [tuple[Any, ...], tuple[torch.Tensor | None, ...], tuple[tuple[int, int], ...]], torch.Tensor
]
Backward = Callable[[torch.Tensor, bool], None]
OptimizerStep = Callable[[], bool]


@dataclass(frozen=True)
class PackedNativeMemberInputs:
    """V2-A2 slot-major consumer assembly. Does NOT detach Local prefix tensors."""

    payloads: tuple[Any, ...]
    prefixes: tuple[torch.Tensor | None, ...]
    positions: tuple[tuple[int, int], ...]


def pack_native_member(
    segments: tuple[SegmentBatch, ...],
    results: tuple[Any, ...],
    identities: tuple[Any, ...],
    *,
    ttt_tbptt_steps: int,
    expected_count: int,
) -> PackedNativeMemberInputs:
    """Merge B independently-scanned rows into one packed native forward.

    Each row contributes its valid temporal consumers in chronological order.
    No artificial padded consumers, copies of Local prefixes or graph detaches.
    """
    if not segments or not (len(segments) == len(results) == len(identities)):
        raise ValueError("packed member rows/results/identities do not match")
    payloads: list[Any] = []
    prefixes: list[torch.Tensor | None] = []
    positions: list[tuple[int, int]] = []
    for segment, result, identity in zip(segments, results, identities, strict=True):
        steps = segment.consumer_valid[0].nonzero(as_tuple=False).flatten().tolist()
        if (
            len(steps) != len(result.payloads)
            or len(steps) != len(result.locals)
            or len(steps) != len(result.identities)
        ):
            raise ValueError("packed row valid-count / prefix / consumer identities mismatch")
        for local_index, payload, prefix, consumer in zip(
            steps, result.payloads, result.locals, result.identities, strict=True
        ):
            slot_id, episode_uid, absolute_step = consumer
            if (
                slot_id != identity.slot_id
                or episode_uid != identity.episode_id
                or absolute_step != identity.cursor * ttt_tbptt_steps + local_index
                or int(segment.consumer_step[0, local_index]) != absolute_step
                or (prefix is None) != (absolute_step == 0)
                or payload is None
            ):
                raise ValueError("packed consumer identity/S0/prefix/chronology mismatch")
            payloads.append(payload)
            prefixes.append(prefix)
            positions.append((slot_id, local_index))
    if len(payloads) != expected_count or expected_count <= 0 or len(set(positions)) != expected_count:
        raise ValueError("packed member actual consumer count/coordinates mismatch")
    return PackedNativeMemberInputs(tuple(payloads), tuple(prefixes), tuple(positions))


@dataclass(frozen=True)
class GroupedLiveState:
    frontier: ExactWindowCatalogFrontier
    sidecar: LocalMemorySegmentSidecar
    scheduler: RankLocalSegmentScheduler


class GroupedLocalMemoryWindow:
    """Grouped scan/backward 在 overlay 内推进；成功 optimizer step 后发布。"""

    def __init__(self, model: nn.Module, planner: ExactWindowRankPlanner) -> None:
        self.model, self.planner = model, planner
        self._live = GroupedLiveState(
            planner.initial_frontier(), LocalMemorySegmentSidecar(), RankLocalSegmentScheduler()
        )
        self._plan: ExactWindowGroupedPlan | None = None
        self._candidate: GroupedLiveState | None = None
        self._member_index = 0
        runtime = model.net.local_memory_runtime
        if (
            runtime.encoder.action_proj.in_features != 15
            or runtime.core.ttt_tbptt_steps != planner.catalog.ttt_tbptt_steps
        ):
            raise ValueError("Local runtime raw15/T 必须匹配 corrected planner")
        scan = getattr(model.net, "scan_local_memory", None)
        if any(isinstance(parameter, DTensor) for parameter in runtime.parameters()) and (
            scan is None or not getattr(model.net, "_local_memory_scan_fsdp_registered", False)
        ):
            raise RuntimeError("FSDP Local 参数要求注册 model-owned scan_local_memory")
        if scan is None:
            raise RuntimeError("grouped Local 要求 model-owned scan_local_memory")
        self._scan = scan

    @property
    def live(self) -> GroupedLiveState:
        return self._live

    @property
    def plan(self) -> ExactWindowGroupedPlan | None:
        return self._plan

    def begin(self) -> ExactWindowGroupedPlan:
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

    def _require_pending(self) -> tuple[ExactWindowGroupedPlan, GroupedLiveState]:
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
        if member_index >= self.planner.active_ga or len(segments) != self.planner.b_stream:
            self.abort()
            raise ValueError("grouped member B/GA 与 planner 不匹配")
        requests = plan.members[member_index]
        expected_slots = tuple(request.identity.slot_id for request in requests)
        if tuple(int(segment.slot_id[0]) for segment in segments) != expected_slots:
            self.abort()
            raise ValueError("member segment slot 顺序不匹配冻结 plan")
        runtime = self.model.net.local_memory_runtime
        adapter = BatchedLocalMemorySegmentAdapter(runtime.encoder, runtime.core, candidate.sidecar, self._scan)
        try:
            identities = tuple(request.identity for request in requests)
            transactions = tuple(
                LocalMemoryTransaction(GAWindowPlan((identity.member,), (request.valid_count,)), candidate.scheduler)
                for identity, request in zip(identities, requests, strict=True)
            )
            timing = getattr(self, "_telemetry_stage", None)
            with timing("local_scan") if timing is not None else nullcontext():
                results = adapter.scan(segments, identities=identities, transactions=transactions)
            if sum(len(result.payloads) for result in results) != plan.member_counts[member_index]:
                raise RuntimeError("member scan 有效 consumer 数与 plan 不一致")
            # V2 A2 alignment: one slot-major Packed Native Forward/Backward
            # per GA member, irrespective of T. Full B8×T16 -> Batch128.
            packed = pack_native_member(
                segments,
                results,
                identities,
                ttt_tbptt_steps=self.planner.catalog.ttt_tbptt_steps,
                expected_count=plan.member_counts[member_index],
            )
            loss = native_loss(packed.payloads, packed.prefixes, packed.positions)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
                raise ValueError("packed native callback must return differentiable scalar outer loss")
            if not bool(torch.isfinite(loss)):
                raise ValueError("packed native outer loss is non-finite")
            # Preserve exact V3 consumer-weighted GA objective; terminal
            # partial members use their real count, never an assumed 128.
            weighted = loss * (len(packed.payloads) / plan.n_window)
            backward(weighted, False)
            if any(
                parameter.grad is not None
                and not bool(
                    torch.isfinite(
                        parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                    ).all()
                )
                for parameter in self.model.parameters()
            ):
                raise ValueError("packed native backward gradient is non-finite")
            for row, (transaction, result, identity) in enumerate(zip(transactions, results, identities, strict=True)):
                transaction.successful_backward(identity, result)
                adapter.commit(row, identity, result, transaction=transaction)
            self._member_index += 1
            return weighted.detach()
        except Exception:
            for row, (identity, transaction, result, _) in tuple(adapter._pending.items()):
                adapter.discard_pending(row, identity, result, transaction=transaction)
            self.abort()
            raise

    def finish(self, optimizer_step: OptimizerStep) -> None:
        plan, candidate = self._require_pending()
        if self._member_index != self.planner.active_ga:
            self.abort()
            raise RuntimeError("所有 grouped member 完成 backward 才能 optimizer step")
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

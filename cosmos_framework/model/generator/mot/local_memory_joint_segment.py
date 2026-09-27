# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-A 单段 joint autograd；optimizer 与 fast-state 发布留给 trainer Gate。"""

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
    SegmentIdentity,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
    SegmentScanResult,
)

NativeLoss = Callable[[Any, torch.Tensor | None, int], torch.Tensor]


@dataclass(frozen=True)
class PreparedJointSegment:
    identity: SegmentIdentity
    transaction: LocalMemoryTransaction
    scan_result: SegmentScanResult
    valid_count: int


class SingleSegmentNativeJointAutograd:
    """保留 scan→native loss 的完整计算图，不执行 optimizer 或 commit。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        sidecar: LocalMemorySegmentSidecar | None = None,
        scheduler: RankLocalSegmentScheduler | None = None,
    ) -> None:
        self.model = model
        self.sidecar = sidecar if sidecar is not None else LocalMemorySegmentSidecar()
        self.scheduler = scheduler if scheduler is not None else RankLocalSegmentScheduler()
        runtime = model.net.local_memory_runtime
        scan = getattr(model.net, "scan_local_memory", None)
        if any(isinstance(parameter, DTensor) for parameter in runtime.parameters()) and (
            scan is None or not getattr(model.net, "_local_memory_scan_fsdp_registered", False)
        ):
            raise RuntimeError("FSDP-owned Local parameters require registered model scan_local_memory")
        self.adapter = CanonicalLocalMemorySegmentAdapter(
            runtime.encoder, runtime.core, self.sidecar, scan_local_memory=scan
        )
        if self.adapter.encoder is not runtime.encoder or self.adapter.core is not runtime.core:
            raise RuntimeError("Joint adapter must reuse model-owned encoder/core instances")
        self._prepared: PreparedJointSegment | None = None

    def prepare(self, segment: SegmentBatch, identity: SegmentIdentity, plan: GAWindowPlan) -> PreparedJointSegment:
        if self._prepared is not None or self.adapter._pending is not None:
            raise RuntimeError("已有 pending joint segment")
        if plan.ga_effective != 1 or plan.members[0] != identity.member:
            raise ValueError("H3-A 只接受单 member plan；grouped GA 属于 H3-C")
        if segment.consumer_valid.shape != (1, self.adapter.core.ttt_tbptt_steps):
            raise ValueError("H3-A 要求单 slot、完整 T16 geometry")
        segment.validate(self.adapter.core.ttt_tbptt_steps)
        count = int(segment.consumer_valid.sum())
        if count != plan.planned_n_valid[0]:
            raise ValueError("plan 与 segment 的有效 consumer 数不匹配")
        transaction = LocalMemoryTransaction(plan, self.scheduler)
        scan_result = self.adapter.scan(segment, identity=identity, transaction=transaction)
        if len(scan_result.payloads) != count or len(scan_result.locals) != count:
            self.adapter.discard_pending(identity, scan_result, transaction=transaction)
            raise RuntimeError("scan consumer 数与 plan 不一致")
        prepared = PreparedJointSegment(identity, transaction, scan_result, count)
        self._prepared = prepared
        return prepared

    def forward_prepared(self, prepared: PreparedJointSegment, native_loss: NativeLoss) -> torch.Tensor:
        if self._prepared is not prepared:
            raise RuntimeError("joint forward 要求精确 pending capability")
        try:
            losses = []
            for index, (payload, prefix) in enumerate(
                zip(prepared.scan_result.payloads, prepared.scan_result.locals, strict=True)
            ):
                loss = native_loss(payload, prefix, index)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
                    raise ValueError("native callback 必须返回可微的标量 outer loss")
                if not torch.isfinite(loss).all():
                    raise ValueError("native outer loss 必须有限")
                losses.append(loss)
            return torch.stack(losses).mean()
        except Exception:
            self.discard(prepared)
            raise

    def discard(self, prepared: PreparedJointSegment) -> None:
        if self._prepared is not prepared:
            raise RuntimeError("joint discard 要求精确 pending capability")
        self.adapter.discard_pending(prepared.identity, prepared.scan_result, transaction=prepared.transaction)
        self._prepared = None

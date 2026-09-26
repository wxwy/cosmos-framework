# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B0 CPU segment、窗口权重与事务合同；没有 trainer、producer 或 checkpoint 依赖。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SegmentProvenance:
    manifest_digest: str
    config_digest: str
    source_digest: str
    segment_id: int

    def __post_init__(self) -> None:
        if not all((self.manifest_digest, self.config_digest, self.source_digest)) or self.segment_id < 0:
            raise ValueError("segment provenance 必须包含有效 digest 和非负 segment_id")


@dataclass(frozen=True)
class SegmentBatch:
    consumer_visual_summary: torch.Tensor
    consumer_payload: tuple[tuple[Any | None, ...], ...]
    consumer_valid: torch.Tensor
    consumer_step: torch.Tensor
    evidence_visual_summary_prev: torch.Tensor
    evidence_executed_action_prev: torch.Tensor
    evidence_valid: torch.Tensor
    evidence_source_step: torch.Tensor
    slot_id: torch.Tensor
    episode_id: tuple[str, ...]
    category: tuple[str, ...]
    segment_provenance: SegmentProvenance

    def validate(self, ttt_tbptt_steps: int) -> None:
        if self.consumer_visual_summary.ndim != 3 or self.consumer_visual_summary.shape[-1] != 96:
            raise ValueError("consumer_visual_summary 必须为 [B,T,96]")
        batch, steps, _ = self.consumer_visual_summary.shape
        if batch <= 0 or not 1 <= steps <= ttt_tbptt_steps:
            raise ValueError("segment 的 B/T 超出范围")
        device = self.consumer_valid.device
        for value, dtype in (
            (self.consumer_valid, torch.bool),
            (self.evidence_valid, torch.bool),
            (self.consumer_step, torch.long),
            (self.evidence_source_step, torch.long),
        ):
            if value.shape != (batch, steps) or value.dtype != dtype or value.device != device:
                raise ValueError("segment mask/step 的形状、dtype 或设备不匹配")
        if self.evidence_visual_summary_prev.shape != (batch, steps, 96):
            raise ValueError("evidence visual 必须为 [B,T,96]")
        action = self.evidence_executed_action_prev
        if action.ndim != 3 or action.shape[:2] != (batch, steps) or action.shape[-1] <= 0:
            raise ValueError("evidence action 必须为 [B,T,D] 且 D>0")
        if self.slot_id.shape != (batch,) or self.slot_id.dtype != torch.long:
            raise ValueError("slot_id 必须为 [B] int64")
        if (self.slot_id < 0).any() or self.slot_id.unique().numel() != batch:
            raise ValueError("segment slot_id 必须非负且互异")
        if len(self.episode_id) != batch or len(self.category) != batch or not all((*self.episode_id, *self.category)):
            raise ValueError("episode/category 必须匹配 B 且非空")
        if len(self.consumer_payload) != batch or any(len(row) != steps for row in self.consumer_payload):
            raise ValueError("consumer_payload 必须为 [B,T]")
        valid, evidence = self.consumer_valid, self.evidence_valid
        if not torch.equal(self.evidence_source_step.eq(-1), ~evidence):
            raise ValueError("无 evidence 时 source step 必须且只能为 -1")
        if torch.any(evidence & ~valid) or torch.any(valid & self.consumer_step.lt(0)):
            raise ValueError("PAD 不得有 evidence；有效 step 必须非负")
        if not torch.equal(evidence, valid & self.consumer_step.gt(0)):
            raise ValueError("S0 无 evidence；非 S0 必须有 previous evidence")
        if torch.any(evidence & self.evidence_source_step.ne(self.consumer_step - 1)):
            raise ValueError("evidence_source_step 必须为 consumer_step-1")
        for row in range(batch):
            count = int(valid[row].sum())
            if not torch.equal(valid[row], torch.arange(steps, device=device) < count):
                raise ValueError("有效 consumer 必须连续，PAD 只能位于尾部")
            row_steps = self.consumer_step[row, :count]
            if count > 1 and not torch.all(row_steps[1:] - row_steps[:-1] == 1):
                raise ValueError("同 slot 的 consumer step 必须连续")
            for index in range(steps):
                if (self.consumer_payload[row][index] is not None) != bool(valid[row, index]):
                    raise ValueError("payload 存在性必须匹配 consumer_valid")

    def gather_consumers(
        self,
        local_tokens: torch.Tensor,
        local_present: torch.Tensor,
    ) -> tuple[tuple[Any, ...], tuple[torch.Tensor | None, ...], tuple[tuple[int, str, int], ...]]:
        if local_tokens.ndim != 4 or local_tokens.shape[:2] != self.consumer_valid.shape:
            raise ValueError("Local tokens 必须为 [B,T,K,D]")
        if local_present.dtype != torch.bool or not torch.equal(
            local_present.to(self.evidence_valid.device), self.evidence_valid
        ):
            raise ValueError("Local present 必须精确匹配 evidence_valid")
        payloads, locals_, identities = [], [], []
        for row, index in self.consumer_valid.nonzero(as_tuple=False).tolist():
            payloads.append(self.consumer_payload[row][index])
            locals_.append(local_tokens[row, index] if bool(local_present[row, index]) else None)
            identities.append((int(self.slot_id[row]), self.episode_id[row], int(self.consumer_step[row, index])))
        return tuple(payloads), tuple(locals_), tuple(identities)


@dataclass(frozen=True)
class GAWindowPlan:
    members: tuple[tuple[int, str, int], ...]
    planned_n_valid: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.members or len(self.members) != len(self.planned_n_valid):
            raise ValueError("GA plan 的 members/counts 必须非空且对齐")
        if len(set(self.members)) != len(self.members):
            raise ValueError("GA plan 不得重复 member")
        if any(type(count) is not int or count <= 0 for count in self.planned_n_valid):
            raise ValueError("planned_n_valid 必须为正整数")

    @property
    def n_window(self) -> int:
        return sum(self.planned_n_valid)

    @property
    def ga_effective(self) -> int:
        return len(self.members)

    def objective(
        self,
        index: int,
        consumer_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        actual_n_valid: int,
    ) -> torch.Tensor:
        if not 0 <= index < self.ga_effective or actual_n_valid != self.planned_n_valid[index]:
            raise ValueError("actual count/index 与冻结 GA plan 不匹配")
        # 已完成窗口归一化，调用方不得再次除 GA；本层不构造任何 Cosmos loss。
        return consumer_loss * (actual_n_valid / self.n_window) + auxiliary_loss / self.ga_effective


@dataclass(frozen=True)
class SegmentIdentity:
    slot_id: int
    episode_id: str
    category: str
    cursor: int
    segment_id: int
    source_digest: str
    training_stream_end: bool = False

    def __post_init__(self) -> None:
        if min(self.slot_id, self.cursor, self.segment_id) < 0 or not all(
            (self.episode_id, self.category, self.source_digest)
        ):
            raise ValueError("SegmentIdentity 的编号必须非负，身份字段必须非空")

    @property
    def member(self) -> tuple[int, str, int]:
        return self.slot_id, self.episode_id, self.cursor


class RankLocalSegmentScheduler:
    """只验证 committed frontier 的单 slot 连续性，不实现数据采样或 grouped driver。"""

    def __init__(self) -> None:
        self._committed: dict[int, SegmentIdentity] = {}

    def validate(self, identity: SegmentIdentity) -> None:
        previous = self._committed.get(identity.slot_id)
        if previous is None or previous.training_stream_end:
            if identity.cursor != 0:
                raise ValueError("新 episode 必须从 cursor0 开始")
        elif (
            identity.episode_id != previous.episode_id
            or identity.category != previous.category
            or identity.source_digest != previous.source_digest
            or identity.cursor != previous.cursor + 1
        ):
            raise ValueError("continuation 必须保持 episode/category/source 且 cursor+1")


class LocalMemoryTransaction:
    """B0 CPU 成功通知；成功标记本身不发布 sidecar/frontier。"""

    def __init__(self, plan: GAWindowPlan, scheduler: RankLocalSegmentScheduler) -> None:
        self.plan, self.scheduler = plan, scheduler
        self.completed_members: list[SegmentIdentity] = []
        self._pending: tuple[SegmentIdentity, object, int] | None = None
        self._backward_succeeded = False
        self._closed = False
        self.failure_code: str | None = None

    def prepare(self, identity: SegmentIdentity, capability: object, count: int) -> None:
        index = len(self.completed_members)
        if self._closed or self._pending is not None or index >= self.plan.ga_effective:
            raise RuntimeError("transaction 关闭、已有 pending 或已完成")
        if identity.member != self.plan.members[index] or count != self.plan.planned_n_valid[index]:
            raise ValueError("identity/count 不匹配当前 plan member")
        self.scheduler.validate(identity)
        self._pending = (identity, capability, count)
        self._backward_succeeded = False

    def require_pending(self, identity: SegmentIdentity, capability: object) -> None:
        if self._pending is None or self._pending[0] is not identity or self._pending[1] is not capability:
            raise RuntimeError("必须提供精确的 identity/result capability")

    def successful_backward(self, identity: SegmentIdentity, capability: object) -> None:
        self.require_pending(identity, capability)
        if self._closed or self._backward_succeeded:
            raise RuntimeError("transaction 已关闭或重复 backward 成功通知")
        self._backward_succeeded = True

    def validate_commit(self, identity: SegmentIdentity, capability: object) -> None:
        self.require_pending(identity, capability)
        if self._closed or not self._backward_succeeded:
            raise RuntimeError("commit 需要未失败的 outer backward 成功通知")
        self.scheduler.validate(identity)

    def _publish(self, identity: SegmentIdentity) -> None:
        # adapter 先完成全部校验与 detach/clone；此处只执行无计算的元数据发布。
        self.scheduler._committed[identity.slot_id] = identity
        self.completed_members.append(identity)
        self._pending = None
        self._backward_succeeded = False
        self._closed = len(self.completed_members) == self.plan.ga_effective

    def terminal_failure(self, code: str = "outer_failure") -> None:
        self.failure_code = code
        self._closed = True
        self._backward_succeeded = False

    def grad_scaler_skip(self) -> None:
        self.terminal_failure("grad_scaler_skip")

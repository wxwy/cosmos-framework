# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B0 单 slot 的 scan/commit 边界；core 批量算法独立于未来 grouped driver。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    LocalMemoryTransaction,
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)


@dataclass(frozen=True)
class SegmentScanResult:
    local_tokens: torch.Tensor
    local_present: torch.Tensor
    state_out: ContinualTTTFastState
    payloads: tuple[Any, ...]
    locals: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]


class LocalMemorySegmentSidecar:
    def __init__(self) -> None:
        self._records: dict[int, tuple[SegmentIdentity, SegmentProvenance, ContinualTTTFastState]] = {}

    def read(self, identity: SegmentIdentity, provenance: SegmentProvenance) -> ContinualTTTFastState | None:
        record = self._records.get(identity.slot_id)
        if record is None:
            if identity.cursor != 0:
                raise ValueError("continuation 缺少 committed fast state")
            return None
        previous, old_provenance, state = record
        if (
            identity.episode_id != previous.episode_id
            or identity.category != previous.category
            or identity.source_digest != previous.source_digest
            or identity.cursor != previous.cursor + 1
            or provenance.manifest_digest != old_provenance.manifest_digest
            or provenance.config_digest != old_provenance.config_digest
        ):
            raise ValueError("sidecar continuation 身份或 provenance 不匹配")
        # 不向调用者暴露 live tensor，防止 scan 外的原位操作污染 committed state。
        return ContinualTTTLocalMemoryCore.detach_state(state)

    def snapshot(self) -> tuple[tuple[SegmentIdentity, SegmentProvenance, ContinualTTTFastState], ...]:
        return tuple(
            (identity, provenance, ContinualTTTLocalMemoryCore.detach_state(state))
            for _, (identity, provenance, state) in sorted(self._records.items())
        )


class CanonicalLocalMemorySegmentAdapter:
    def __init__(
        self,
        encoder: LocalEvidenceEncoder,
        core: ContinualTTTLocalMemoryCore,
        sidecar: LocalMemorySegmentSidecar,
        scan_local_memory: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, ContinualTTTFastState | None],
            tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor],
        ]
        | None = None,
    ) -> None:
        if encoder.evidence_dim != core.evidence_dim:
            raise ValueError("encoder/core evidence_dim 不匹配")
        self.encoder, self.core, self.sidecar = encoder, core, sidecar
        self.scan_local_memory = scan_local_memory
        self._pending: tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult, SegmentProvenance] | None = (
            None
        )

    def scan(
        self,
        segment: SegmentBatch,
        *,
        identity: SegmentIdentity,
        transaction: LocalMemoryTransaction,
    ) -> SegmentScanResult:
        if self._pending is not None:
            raise RuntimeError("已有 pending scan，必须先 commit 或 discard")
        segment.validate(self.core.ttt_tbptt_steps)
        if segment.consumer_valid.shape[0] != 1:
            raise ValueError("B0 adapter 只接单 slot；批量 scan 在 core 验证")
        if segment.evidence_executed_action_prev.shape[-1] != self.encoder.action_proj.in_features:
            raise ValueError("segment action 维度必须精确匹配 encoder.action_proj.in_features")
        provenance = segment.segment_provenance
        if (
            int(segment.slot_id[0]) != identity.slot_id
            or segment.episode_id[0] != identity.episode_id
            or segment.category[0] != identity.category
            or provenance.segment_id != identity.segment_id
            or provenance.source_digest != identity.source_digest
        ):
            raise ValueError("segment 与 identity 不匹配")
        count = int(segment.consumer_valid.sum())
        if count <= 0 or (count != self.core.ttt_tbptt_steps and not identity.training_stream_end):
            raise ValueError("非 terminal segment 必须包含完整 TBPTT consumers")
        if int(segment.consumer_step[0, 0]) != identity.cursor * self.core.ttt_tbptt_steps:
            raise ValueError("consumer 起点必须匹配 cursor*TBPTT")
        transaction.scheduler.validate(identity)
        state = self.sidecar.read(identity, provenance)
        device = self.core.slot_queries.device
        visual = segment.evidence_visual_summary_prev.to(device)
        action = segment.evidence_executed_action_prev.to(device)
        valid = segment.evidence_valid.to(device)
        if self.scan_local_memory is None:
            tokens, candidate, present = self.core.scan_segment_masked_encoded_many(
                self.encoder, visual, action, valid, state
            )
        else:
            tokens, candidate, present = self.scan_local_memory(visual, action, valid, state)
            self.core.validate_state(candidate, 1)
            if tokens.shape != (*visual.shape[:2], self.core.k_local, self.core.local_dim):
                raise ValueError("model-owned Local scan 返回的 token 形状不匹配")
        payloads, locals_, identities = segment.gather_consumers(tokens, present)
        result = SegmentScanResult(tokens, present, candidate, payloads, locals_, identities)
        transaction.prepare(identity, result, count)
        self._pending = identity, transaction, result, provenance
        return result

    def _require_exact(
        self,
        identity: SegmentIdentity,
        result: SegmentScanResult,
        transaction: LocalMemoryTransaction,
    ) -> SegmentProvenance:
        pending = self._pending
        if pending is None or pending[0] is not identity or pending[1] is not transaction or pending[2] is not result:
            raise RuntimeError("adapter 要求精确的 identity/transaction/result capability")
        return pending[3]

    def commit(
        self,
        identity: SegmentIdentity,
        result: SegmentScanResult,
        *,
        transaction: LocalMemoryTransaction,
    ) -> None:
        provenance = self._require_exact(identity, result, transaction)
        transaction.validate_commit(identity, result)
        # 先检查 live frontier，再复制 candidate；此前无任何 live 写入。
        self.sidecar.read(identity, provenance)
        self.core.validate_state(result.state_out, 1)
        state = self.core.detach_state(result.state_out)
        if identity.training_stream_end:
            self.sidecar._records.pop(identity.slot_id, None)
        else:
            self.sidecar._records[identity.slot_id] = identity, provenance, state
        transaction._publish(identity)
        self._pending = None

    def discard_pending(
        self,
        identity: SegmentIdentity,
        result: SegmentScanResult,
        *,
        transaction: LocalMemoryTransaction,
    ) -> None:
        self._require_exact(identity, result, transaction)
        transaction.require_pending(identity, result)
        transaction.terminal_failure(transaction.failure_code or "discard")
        transaction._pending = None
        self._pending = None

    def reset(self, slot_id: int, *, scheduler: Any) -> None:
        """显式 episode reset；有待处理事务时拒绝改变 frontier。"""
        if self._pending is not None:
            raise RuntimeError("pending scan 期间不得 reset")
        self.sidecar._records.pop(slot_id, None)
        scheduler._committed.pop(slot_id, None)


class BatchedLocalMemorySegmentAdapter:
    """Keep row capabilities separate while using one model-owned Local scan."""

    def __init__(
        self,
        encoder: LocalEvidenceEncoder,
        core: ContinualTTTLocalMemoryCore,
        sidecar: LocalMemorySegmentSidecar,
        scan_local_memory: Callable[..., tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]],
    ) -> None:
        if encoder.evidence_dim != core.evidence_dim or scan_local_memory is None:
            raise ValueError("batched adapter 需要匹配的 encoder/core 和 model-owned scan")
        self.encoder, self.core, self.sidecar = encoder, core, sidecar
        self.scan_local_memory = scan_local_memory
        self._pending: dict[
            int, tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult, SegmentProvenance]
        ] = {}

    def scan(
        self,
        segments: tuple[SegmentBatch, ...],
        *,
        identities: tuple[SegmentIdentity, ...],
        transactions: tuple[LocalMemoryTransaction, ...],
    ) -> tuple[SegmentScanResult, ...]:
        if self._pending or not segments or len(segments) != len(identities) or len(segments) != len(transactions):
            raise ValueError("batched scan 需要对齐且无 pending 的 per-row 输入")
        states: list[ContinualTTTFastState | None] = []
        device = self.encoder.visual_proj.weight.device
        for segment, identity, transaction in zip(segments, identities, transactions, strict=True):
            segment.validate(self.core.ttt_tbptt_steps)
            if segment.consumer_valid.shape != (1, self.core.ttt_tbptt_steps):
                raise ValueError("batched row 必须为完整 TBPTT 宽度")
            if segment.evidence_executed_action_prev.shape[-1] != self.encoder.action_proj.in_features:
                raise ValueError("segment action 维度不匹配")
            provenance = segment.segment_provenance
            if (
                int(segment.slot_id[0]) != identity.slot_id
                or segment.episode_id[0] != identity.episode_id
                or segment.category[0] != identity.category
                or provenance.segment_id != identity.segment_id
                or provenance.source_digest != identity.source_digest
            ):
                raise ValueError("segment 与 identity 不匹配")
            count = int(segment.consumer_valid.sum())
            if count <= 0 or (count != self.core.ttt_tbptt_steps and not identity.training_stream_end):
                raise ValueError("非 terminal segment 必须完整")
            if int(segment.consumer_step[0, 0]) != identity.cursor * self.core.ttt_tbptt_steps:
                raise ValueError("consumer 起点不匹配 cursor*TBPTT")
            transaction.scheduler.validate(identity)
            states.append(self.sidecar.read(identity, provenance))
        continuation = tuple(state is not None for state in states)
        state_in = None
        mask = None
        if any(continuation):
            template = next(state for state in states if state is not None)
            assert template is not None
            state_in = ContinualTTTFastState(
                *(
                    torch.cat(
                        [state[index] if state is not None else value.new_zeros(value.shape) for state in states],
                        dim=0,
                    )
                    for index, value in enumerate(template)
                )
            )
            if not all(continuation):
                mask = torch.tensor(continuation, dtype=torch.bool, device=device)
        visual = torch.cat([segment.evidence_visual_summary_prev.to(device) for segment in segments])
        action = torch.cat([segment.evidence_executed_action_prev.to(device) for segment in segments])
        valid = torch.cat([segment.evidence_valid.to(device) for segment in segments])
        tokens, candidate, present = self.scan_local_memory(visual, action, valid, state_in, continuation_mask=mask)
        self.core.validate_state(candidate, len(segments))
        if tokens.shape != (len(segments), self.core.ttt_tbptt_steps, self.core.k_local, self.core.local_dim):
            raise ValueError("model-owned batched scan token 形状不匹配")
        if present.shape != valid.shape or present.dtype != torch.bool:
            raise ValueError("model-owned batched scan present 形状不匹配")
        results = []
        for row, (segment, identity, transaction) in enumerate(zip(segments, identities, transactions, strict=True)):
            row_tokens, row_present = tokens[row : row + 1], present[row : row + 1]
            row_state = ContinualTTTFastState(*(value[row : row + 1] for value in candidate))
            payloads, locals_, row_identities = segment.gather_consumers(row_tokens, row_present)
            result = SegmentScanResult(row_tokens, row_present, row_state, payloads, locals_, row_identities)
            transaction.prepare(identity, result, int(segment.consumer_valid.sum()))
            self._pending[row] = identity, transaction, result, segment.segment_provenance
            results.append(result)
        return tuple(results)

    def _require_exact(
        self, row: int, identity: SegmentIdentity, result: SegmentScanResult, transaction: LocalMemoryTransaction
    ) -> SegmentProvenance:
        pending = self._pending.get(row)
        if pending is None or pending[0] is not identity or pending[1] is not transaction or pending[2] is not result:
            raise RuntimeError("batched adapter 要求每 row 精确 capability")
        return pending[3]

    def commit(
        self,
        row: int,
        identity: SegmentIdentity,
        result: SegmentScanResult,
        *,
        transaction: LocalMemoryTransaction,
    ) -> None:
        provenance = self._require_exact(row, identity, result, transaction)
        transaction.validate_commit(identity, result)
        self.sidecar.read(identity, provenance)
        self.core.validate_state(result.state_out, 1)
        state = self.core.detach_state(result.state_out)
        if identity.training_stream_end:
            self.sidecar._records.pop(identity.slot_id, None)
        else:
            self.sidecar._records[identity.slot_id] = identity, provenance, state
        transaction._publish(identity)
        del self._pending[row]

    def discard_pending(
        self,
        row: int,
        identity: SegmentIdentity,
        result: SegmentScanResult,
        *,
        transaction: LocalMemoryTransaction,
    ) -> None:
        self._require_exact(row, identity, result, transaction)
        transaction.require_pending(identity, result)
        transaction.terminal_failure(transaction.failure_code or "discard")
        transaction._pending = None
        del self._pending[row]

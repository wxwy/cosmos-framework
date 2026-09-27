# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""单 slot 原生 consumer 梯度接力，不接 trainer 或 GPU。"""

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


@dataclass(frozen=True)
class NativeConsumerResult:
    loss: torch.Tensor
    output: Any = None
    sample_coupled_auxiliary_loss: torch.Tensor | None = None


NativeForward = Callable[[Any, torch.Tensor | None, int], NativeConsumerResult]


def _local_params_are_dtensors(runtime: nn.Module) -> bool:
    return any(isinstance(parameter, DTensor) for parameter in runtime.parameters())


def _detach_metadata(value: Any) -> Any:
    """保留回调诊断数据，同时释放已完成的原生计算图。"""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_metadata(item) for item in value)
    if isinstance(value, list):
        return [_detach_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Native callback output must be tensor-free metadata or detachable containers")


@dataclass(frozen=True)
class PreparedNativeSegment:
    identity: SegmentIdentity
    transaction: LocalMemoryTransaction
    scan_result: SegmentScanResult
    valid_count: int


@dataclass(frozen=True)
class NativeSegmentOutcome:
    mean_loss: torch.Tensor
    outputs: tuple[Any, ...]
    valid_count: int


class SingleSegmentNativeGradientRelay:
    """对模型持有的 Local 慢参数执行单 member 串行事务。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        sidecar: LocalMemorySegmentSidecar | None = None,
        scheduler: RankLocalSegmentScheduler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.sidecar = sidecar if sidecar is not None else LocalMemorySegmentSidecar()
        self.scheduler = scheduler if scheduler is not None else RankLocalSegmentScheduler()
        runtime = model.net.local_memory_runtime
        scan = getattr(model.net, "scan_local_memory", None)
        if _local_params_are_dtensors(runtime) and (
            scan is None or not getattr(model.net, "_local_memory_scan_fsdp_registered", False)
        ):
            raise RuntimeError("FSDP-owned Local parameters require registered model scan_local_memory")
        self.adapter = CanonicalLocalMemorySegmentAdapter(
            runtime.encoder, runtime.core, self.sidecar, scan_local_memory=scan
        )
        if self.adapter.encoder is not runtime.encoder or self.adapter.core is not runtime.core:
            raise RuntimeError("Local adapter must reuse model-owned encoder/core instances")
        named = dict(model.net.named_parameters())
        selected = {name: parameter for name, parameter in named.items() if "local_memory" in name}
        if not selected or any(not parameter.requires_grad for parameter in selected.values()):
            raise ValueError("All model-owned Local slow parameters must be trainable")
        if any(parameter.requires_grad for name, parameter in named.items() if "local_memory" not in name):
            raise ValueError("Single-segment relay requires every host parameter frozen")
        optimizer_params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        selected_ids = {id(parameter) for parameter in selected.values()}
        if (
            len(optimizer_params) != len(selected_ids)
            or {id(parameter) for parameter in optimizer_params} != selected_ids
        ):
            raise ValueError("Optimizer must contain exactly the model-owned Local slow parameters")
        self._selected_params = tuple(selected.values())
        self._prepared: PreparedNativeSegment | None = None
        self._fatal_after_step = False

    def prepare(
        self,
        segment: SegmentBatch,
        identity: SegmentIdentity,
        plan: GAWindowPlan,
    ) -> PreparedNativeSegment:
        """在扫描或原生 consumer 工作前拒绝不支持的几何配置。"""
        if self._fatal_after_step:
            raise RuntimeError("Optimizer may be partially mutated; restart before another Local segment")
        if self._prepared is not None or self.adapter._pending is not None:
            raise RuntimeError("Only one pending Local segment is supported")
        if plan.ga_effective != 1 or plan.members[0] != identity.member:
            raise ValueError("B2-B requires one exact GA plan member")
        if segment.consumer_valid.shape != (1, self.adapter.core.ttt_tbptt_steps):
            raise ValueError("B2-B requires one slot and the full T=16 segment geometry")
        segment.validate(self.adapter.core.ttt_tbptt_steps)
        count = int(segment.consumer_valid.sum())
        if count != plan.planned_n_valid[0]:
            raise ValueError("Planned and produced valid consumer counts must match")
        self.optimizer.zero_grad(set_to_none=True)
        transaction = LocalMemoryTransaction(plan, self.scheduler)
        scan_result = self.adapter.scan(segment, identity=identity, transaction=transaction)
        if len(scan_result.payloads) != count or len(scan_result.locals) != count:
            self.adapter.discard_pending(identity, scan_result, transaction=transaction)
            raise RuntimeError("B0 scan cardinality differs from the exact plan")
        prepared = PreparedNativeSegment(identity, transaction, scan_result, count)
        self._prepared = prepared
        return prepared

    def _discard(self) -> None:
        prepared = self._prepared
        self.optimizer.zero_grad(set_to_none=True)
        if prepared is not None and self.adapter._pending is not None:
            self.adapter.discard_pending(
                prepared.identity,
                prepared.scan_result,
                transaction=prepared.transaction,
            )
        self._prepared = None

    @staticmethod
    def _relay_gradients(prefixes: list[torch.Tensor], gradients: list[torch.Tensor]) -> None:
        if prefixes:
            torch.autograd.backward(prefixes, gradients)

    def execute_prepared(
        self,
        prepared: PreparedNativeSegment,
        native_forward: NativeForward,
    ) -> NativeSegmentOutcome:
        """执行精确串行反传，仅在优化步骤成功后发布 fast state。"""
        live = self._prepared
        if live is None:
            raise RuntimeError("No pending Local segment")
        step_started = False
        try:
            if (
                prepared is not live
                or prepared.identity is not live.identity
                or prepared.transaction is not live.transaction
                or prepared.scan_result is not live.scan_result
            ):
                raise RuntimeError("Stale or copied Local segment capability")
            identity, transaction, result, count = (
                live.identity,
                live.transaction,
                live.scan_result,
                live.valid_count,
            )
            self.adapter._require_exact(identity, result, transaction)
            if len(result.payloads) != count or len(result.locals) != count or len(result.identities) != count:
                raise RuntimeError("Scan result cardinality changed after prepare")
            prefixes: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []
            losses: list[torch.Tensor] = []
            outputs: list[Any] = []
            for index, (payload, prefix, consumer_identity) in enumerate(
                zip(result.payloads, result.locals, result.identities, strict=True)
            ):
                slot, episode, step = consumer_identity
                if (
                    slot != identity.slot_id
                    or episode != identity.episode_id
                    or step != identity.cursor * self.adapter.core.ttt_tbptt_steps + index
                    or (prefix is None) != (step == 0)
                ):
                    raise ValueError("S0/Local mapping or stream order changed after scan")
                leaf = None if prefix is None else prefix.detach().requires_grad_(True)
                native = native_forward(payload, leaf, index)
                if not isinstance(native, NativeConsumerResult):
                    raise TypeError("Native callback must return NativeConsumerResult")
                if native.sample_coupled_auxiliary_loss is not None:
                    raise ValueError("Sample-coupled auxiliary loss is unsupported by serial relay")
                loss = native.loss
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("Native consumer loss must be a finite scalar")
                if leaf is None and loss.requires_grad:
                    raise ValueError("S0 native loss must not depend on Local trainable parameters")
                losses.append(loss.detach())
                outputs.append(_detach_metadata(native.output))
                if leaf is not None:
                    (loss / count).backward()
                    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
                        raise ValueError("Native consumer did not provide a finite Local leaf gradient")
                    prefixes.append(prefix)
                    gradients.append(leaf.grad.detach().clone())
                del native, loss, leaf
            self._relay_gradients(prefixes, gradients)
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in self._selected_params
            ):
                raise ValueError("Local slow gradient is nonfinite")
            transaction.successful_backward(identity, result)
            transaction.validate_commit(identity, result)
            step_started = True
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self._selected_params):
                raise ValueError("Local slow parameter is nonfinite after optimizer step")
            self.adapter.commit(identity, result, transaction=transaction)
            self._prepared = None
            return NativeSegmentOutcome(torch.stack(losses).mean(), tuple(outputs), count)
        except Exception:
            if step_started:
                self._fatal_after_step = True
            self._discard()
            raise

    def run(
        self,
        segment: SegmentBatch,
        identity: SegmentIdentity,
        plan: GAWindowPlan,
        native_forward: NativeForward,
    ) -> NativeSegmentOutcome:
        return self.execute_prepared(self.prepare(segment, identity, plan), native_forward)

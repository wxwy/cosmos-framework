# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected exact-window Local trainer 接线；resume 由后续 Gate 处理。"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.data.generator.joint_dataloader import JointDataLoader, custom_collate_fn
from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.trainer.local_memory_grouped_resume import (
    GroupedLocalMemoryStateCallback,
    require_dcp_grouped_resume_component,
    restore_grouped_local_state,
)
from cosmos_framework.utils import misc
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


def collate_grouped_native_batch(payloads: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """复用 JointDataLoader 的 per-sample ABI，把同一 index 的有效 slot 封装为原生 batch。"""
    if not payloads or not all(isinstance(payload, dict) for payload in payloads):
        raise ValueError("grouped native batch 要求非空 dict payload tuple")
    loader = object.__new__(JointDataLoader)
    loader.buffers = [deque()]
    loader.dataloaders = [iter((custom_collate_fn(list(payloads)),))]
    batch: dict[str, Any] = {}
    for _ in payloads:
        loader._update_output_batch(batch, loader._get_next_sample(0))
    if len(batch.get("sequence_plan", ())) != len(payloads):
        raise ValueError("grouped native batch 的 SequencePlan 数量不匹配")
    return batch


class GroupedLocalMemoryTrainer(ImaginaireTrainer):
    """保留上游 train loop；active_ga 次 training_step 是一个 grouped optimizer window。"""

    def _observe_grouped(
        self,
        phase: str,
        *,
        iteration: int,
        member: int,
        index: int | None = None,
        loss: torch.Tensor | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        observer = getattr(self, "grouped_observer", None)
        if observer is not None:
            observer(
                phase=phase,
                iteration=iteration,
                member=member,
                index=index,
                loss=loss,
                metrics=metrics,
                trainer=self,
            )

    def _telemetry_stage(self, name: str):
        observer = getattr(self, "grouped_observer", None)
        return observer.time_stage(name) if observer is not None and hasattr(observer, "time_stage") else nullcontext()

    @staticmethod
    def _optimizer_lr_metrics(optimizer: Any) -> dict[str, float]:
        optimizers = optimizer.optimizers if isinstance(optimizer, OptimizersContainer) else [optimizer]
        lrs = [
            float(group["lr"]) for inner in optimizers for group in getattr(inner, "param_groups", ()) if "lr" in group
        ]
        if not lrs:
            return {}
        return {"lr_min": min(lrs), "lr_max": max(lrs)}

    def bind_grouped_stream(
        self,
        planner: ExactWindowRankPlanner,
        producer: ExactWindowSegmentProducer,
        *,
        config_digest: str,
    ) -> None:
        if (
            hasattr(self, "_grouped_planner")
            or getattr(self, "_grouped_window", None) is not None
            or not isinstance(planner, ExactWindowRankPlanner)
            or not isinstance(producer, ExactWindowSegmentProducer)
            or not config_digest
        ):
            raise RuntimeError("grouped stream 已绑定或 planner/producer/config_digest 不合法")
        if producer.catalog is not planner.catalog:
            raise ValueError("grouped planner/producer catalog 不一致")
        if (
            not isinstance(planner.catalog, ExactWindowLocalCatalog)
            or planner.b_stream <= 0
            or planner.active_ga != self.config.trainer.grad_accum_iter
            or producer.config_digest != config_digest
        ):
            raise ValueError("grouped catalog/B/GA/config_digest 与 trainer 不匹配")
        if not hasattr(self.callbacks, "_callbacks") or any(
            getattr(callback, "checkpoint_component", None) == "dataloader" for callback in self.callbacks._callbacks
        ):
            raise ValueError("H3-D 要求独占 DCP dataloader state callback")
        self._grouped_planner, self._grouped_producer = planner, producer
        self._grouped_config_digest = config_digest
        self._grouped_completed_iteration = 0
        self._pending_grouped_resume = None
        self._resume_required = False
        self._grouped_restore_failed = False
        self.callbacks._callbacks.insert(0, GroupedLocalMemoryStateCallback(self))

    def train(self, model, dataloader_train, dataloader_val) -> None:
        if not hasattr(self, "_grouped_planner"):
            raise RuntimeError("H3-D train 前必须绑定 grouped stream")
        if self.config.trainer.save_zero_checkpoint or not self.config.checkpoint.strict_resume:
            raise ValueError("H3-D 禁止零步 checkpoint，并要求 strict_resume")
        self._resume_required = require_dcp_grouped_resume_component(self.checkpointer)
        super().train(model, dataloader_train, dataloader_val)

    @staticmethod
    def _optimizer_parameters(optimizer: Any) -> list[torch.Tensor]:
        optimizers = optimizer.optimizers if isinstance(optimizer, OptimizersContainer) else [optimizer]
        if not optimizers:
            raise ValueError("H3-C optimizer container 为空")
        parameters: list[torch.Tensor] = []
        seen: set[int] = set()
        for inner_optimizer in optimizers:
            if not hasattr(inner_optimizer, "param_groups"):
                raise TypeError("H3-C inner optimizer 缺少 param_groups")
            for group in inner_optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
        if not parameters:
            raise ValueError("H3-C optimizer 选中参数为空")
        return parameters

    @staticmethod
    def _require_finite_gradients(optimizer: Any) -> None:
        parameters = GroupedLocalMemoryTrainer._optimizer_parameters(optimizer)
        local_bad = 0
        local_present = 0
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            local_present = 1
            if isinstance(gradient, DTensor):
                gradient = gradient.to_local()
            if gradient.is_sparse or not bool(torch.isfinite(gradient).all()):
                local_bad = 1
        if dist.is_available() and dist.is_initialized():
            flags = torch.tensor([local_bad, local_present], device=parameters[0].device, dtype=torch.int32)
            dist.all_reduce(flags[:1], op=dist.ReduceOp.MAX)
            dist.all_reduce(flags[1:], op=dist.ReduceOp.SUM)
            local_bad, local_present = flags.tolist()
        if local_bad or not local_present:
            raise FloatingPointError("H3-C 选中参数梯度缺失或非有限；optimizer/fast state 均不发布")

    @staticmethod
    def _optimizer_step_success(optimizer, scheduler, grad_scaler) -> bool:
        old_scale = grad_scaler.get_scale() if grad_scaler.is_enabled() else 1.0
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if grad_scaler.is_enabled() and grad_scaler.get_scale() < old_scale:
            return False
        scheduler.step()
        return True

    def training_step(
        self,
        model_ddp: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, Any],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, Any], torch.Tensor, int]:
        del data  # dataloader 驱动 GA 调用；raw/RGB 由 exact-window producer 提供。
        if not hasattr(self, "_grouped_planner"):
            raise RuntimeError("必须先 bind_grouped_stream")
        active_ga = self._grouped_planner.active_ga
        if self.config.trainer.grad_accum_iter != active_ga or not 0 <= grad_accum_iter < active_ga:
            raise ValueError("trainer GA 与 corrected planner active_ga 不匹配")
        if getattr(getattr(self.config, "model_parallel", None), "context_parallel_size", 1) != 1:
            raise ValueError("Phase5A grouped trainer 要求 context_parallel_size=1")
        if self._grouped_restore_failed:
            raise RuntimeError("Local 恢复失败后不得继续使用该 trainer 实例")
        model = model_ddp.module if self.config.trainer.distributed_parallelism == "ddp" else model_ddp
        if not hasattr(self, "_grouped_window"):
            self._grouped_window = GroupedLocalMemoryWindow(model, self._grouped_planner)
            pending = self._pending_grouped_resume
            if self._resume_required and pending is None:
                self._grouped_restore_failed = True
                raise RuntimeError("同 job DCP 未向 grouped callback 恢复 Local 状态")
            if pending is not None:
                try:
                    restore_grouped_local_state(
                        self._grouped_window,
                        pending,
                        iteration=iteration,
                        config_digest=self._grouped_config_digest,
                    )
                    self._grouped_completed_iteration = iteration
                except Exception:
                    self._grouped_restore_failed = True
                    raise
                self._pending_grouped_resume = None
        window = self._grouped_window
        # Observational context manager only; not part of DCP or Local live state.
        window._telemetry_stage = self._telemetry_stage
        if grad_accum_iter == 0:
            window.begin()
        elif window.plan is None:
            raise RuntimeError("后续 GA 调用缺少 pending grouped window")
        assert window.plan is not None
        output: dict[str, Any] = {}
        backward_index = 0

        def native_loss(payloads, prefixes, index):
            nonlocal output
            with self._telemetry_stage("batch_collate"):
                batch = collate_grouped_native_batch(payloads)
            with self._telemetry_stage("batch_transfer"):
                batch = misc.to(batch, device=next(model.parameters()).device)
            self.callbacks.on_before_forward(iteration=iteration)
            with self._telemetry_stage("forward"):
                with self.training_timer("forward"):
                    output, loss = model_ddp.training_step(batch, iteration, _local_memory_prefixes=prefixes)
            if "_backward_loss" in output:
                raise ValueError("H3-C 禁止额外 surrogate backward loss")
            scalar_metrics = {
                key: value.detach()
                for key, value in output.items()
                if isinstance(value, torch.Tensor) and value.ndim == 0 and "loss" in key
            }
            self._observe_grouped(
                "native_forward",
                iteration=iteration,
                member=grad_accum_iter,
                index=index,
                loss=loss,
                metrics=scalar_metrics,
            )
            self.callbacks.on_after_forward(iteration=iteration)
            return loss

        def backward(weighted_loss: torch.Tensor, retain_graph: bool) -> None:
            nonlocal backward_index
            self.callbacks.on_before_backward(model, weighted_loss, iteration=iteration)
            with self._telemetry_stage("backward"):
                with self.training_timer("backward"):
                    grad_scaler.scale(weighted_loss).backward(retain_graph=retain_graph)
                    model.on_after_backward()
            self._observe_grouped(
                "native_backward",
                iteration=iteration,
                member=grad_accum_iter,
                index=backward_index,
                loss=weighted_loss.detach(),
            )
            backward_index += 1
            self.callbacks.on_after_backward(model, iteration=iteration)

        try:
            with self._telemetry_stage("data_prepare"):
                segments = tuple(
                    self._grouped_producer.produce(request) for request in window.plan.members[grad_accum_iter]
                )
            member_loss = window.run_member(segments, native_loss, backward)
            if grad_accum_iter + 1 < active_ga:
                return output, member_loss, grad_accum_iter + 1
            if grad_scaler.is_enabled():
                grad_scaler.unscale_(optimizer)
            self._require_finite_gradients(optimizer)
            self.callbacks.on_before_optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=iteration)
            model.on_before_optimizer_step(optimizer, scheduler, iteration=iteration)
            self._require_finite_gradients(optimizer)
            self._observe_grouped(
                "pre_optimizer",
                iteration=iteration,
                member=grad_accum_iter,
                metrics=self._optimizer_lr_metrics(optimizer),
            )
            def observed_optimizer_step() -> bool:
                with self._telemetry_stage("optimizer"):
                    return self._optimizer_step_success(optimizer, scheduler, grad_scaler)

            window.finish(observed_optimizer_step)
            self._observe_grouped("post_commit", iteration=iteration, member=grad_accum_iter)
            self._grouped_completed_iteration = iteration + 1
            self.callbacks.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
            model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
            self._zero_grad(model, optimizer, iteration)
            return output, member_loss, 0
        except Exception:
            window.abort()
            raise

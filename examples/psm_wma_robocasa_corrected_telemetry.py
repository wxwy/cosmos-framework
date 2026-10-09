# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only, rank-local telemetry for the corrected grouped Local-TTT trainer.

This observer never owns checkpoint state, optimizer calls, or autograd tensors.
All losses shown are rank-local. FSDP gradient norms are rank-local shard
norms, not globally reduced norms.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

import torch
from torch.distributed.tensor import DTensor


class GroupedPlanObserver:
    """Validate grouped call counts and report successful iterations from rank 0."""

    def __init__(
        self,
        *,
        rank: int = 0,
        parameter_group: Callable[[str], str | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.rank = rank
        self._parameter_group = parameter_group
        self._clock = clock
        self._emit = emit
        self.forward = 0
        self.backward = 0
        self.completed = 0
        self._started_at: float | None = None
        self._started_iteration: int | None = None
        self._outer_loss = 0.0
        self._metric_sums: dict[str, float] = {}
        self._metric_weights: dict[str, float] = {}
        self._lr: dict[str, float] = {}
        self._gradient_stats: dict[str, float | int | None] = {}
        self._telemetry_errors: list[str] = []
        self.last_record: dict[str, Any] | None = None

    def start_iteration(self, iteration: int) -> None:
        """Called at the first trigger fetch; does not change yielded batches."""
        if self.rank != 0:
            return
        self._started_iteration = iteration
        self._started_at = self._clock()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as exc:
                self._telemetry_errors.append(f"peak_memory_reset: {type(exc).__name__}")

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().float().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _weight_for_index(plan: Any, member: int, index: int) -> float:
        if member < 0 or member >= len(plan.members) or index < 0:
            raise ValueError("invalid grouped telemetry member/index")
        denominator = sum(request.valid_count for group in plan.members for request in group)
        if denominator <= 0:
            raise ValueError("zero grouped consumer denominator")
        active = sum(request.valid_count > index for request in plan.members[member])
        if not active:
            raise ValueError("forward telemetry index has no valid consumers")
        return active / denominator

    def _on_forward(self, *, trainer: Any, member: int | None, index: int | None, metrics: Any) -> None:
        if self.rank != 0 or not metrics or member is None or index is None:
            return
        plan = trainer._grouped_window.plan
        weight = self._weight_for_index(plan, member, index)
        for source_key, output_key in (
            ("flow_matching_loss_action", "action_loss"),
            ("flow_matching_loss_vision", "vision_loss"),
        ):
            scalar = self._scalar(metrics.get(source_key))
            if scalar is None:
                continue
            self._metric_sums[output_key] = self._metric_sums.get(output_key, 0.0) + scalar * weight
            self._metric_weights[output_key] = self._metric_weights.get(output_key, 0.0) + weight

    def _grad_norms(self, model: Any) -> dict[str, float | int | None]:
        if self._parameter_group is None or model is None:
            return {}
        names = ("total", "generation", "action", "local")
        sum_sq: dict[str, torch.Tensor | None] = dict.fromkeys(names)
        counts = dict.fromkeys(names, 0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                group = self._parameter_group(name)
                if group is None or parameter.grad is None:
                    continue
                if group not in ("generation", "action", "local"):
                    raise ValueError(f"unknown telemetry parameter group: {group}")
                grad = parameter.grad
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                square = grad.detach().float().square().sum()
                keys = ("total", "local") if group == "local" else (
                    ("total", "generation", "action") if group == "action" else ("total", "generation")
                )
                for key in keys:
                    sum_sq[key] = square if sum_sq[key] is None else sum_sq[key] + square
                    counts[key] += 1
        result: dict[str, float | int | None] = {}
        for name in names:
            squared = sum_sq[name]
            result[f"{name}_grad_norm_rank_local"] = (
                self._scalar(squared.sqrt()) if squared is not None else None
            )
            result[f"{name}_grad_tensors_rank_local"] = counts[name]
        return result

    def _on_pre_optimizer(self, *, trainer: Any, metrics: Any) -> None:
        if self.rank != 0:
            return
        self._lr = {
            name: scalar
            for name in ("lr_min", "lr_max")
            if (scalar := self._scalar((metrics or {}).get(name))) is not None
        }
        try:
            model = trainer._grouped_window.model
            self._gradient_stats = self._grad_norms(model)
        except Exception as exc:  # Observation failure must never modify the training transaction.
            self._gradient_stats = {}
            self._telemetry_errors.append(f"gradient_observer: {type(exc).__name__}: {exc}")

    def _on_post_commit(self, *, trainer: Any, iteration: int | None) -> None:
        if self.rank != 0:
            return
        try:
            window = trainer._grouped_window
            core = window.model.net.local_memory_runtime.core
            summaries = core.drain_telemetry()
            frontier = window.live.frontier
            local: dict[str, float | None] = {}
            for key in ("inner_loss", "fast_state_norm", "fast_update_norm"):
                total = self._scalar(summaries.get(f"ttt_{key}_sum"))
                count = self._scalar(summaries.get(f"ttt_{key}_count"))
                local[f"{key}_mean"] = total / count if total is not None and count and count > 0 else None
                local[f"{key}_count"] = count
            epoch = frontier.epoch
            active_slots = sum(slot.uid is not None for slot in frontier.slots)
        except Exception as exc:
            local = {name: None for name in (
                "inner_loss_mean", "inner_loss_count", "fast_state_norm_mean",
                "fast_state_norm_count", "fast_update_norm_mean", "fast_update_norm_count",
            )}
            epoch, active_slots = None, None
            self._telemetry_errors.append(f"local_observer: {type(exc).__name__}: {exc}")

        step_wall = self._clock() - self._started_at if self._started_at is not None else None
        allocated_gb = reserved_gb = None
        if torch.cuda.is_available():
            try:
                allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
                reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            except RuntimeError as exc:
                self._telemetry_errors.append(f"peak_memory_read: {type(exc).__name__}")

        record: dict[str, Any] = {
            "status": "optimizer_committed",
            "iteration": iteration + 1 if iteration is not None else None,
            "rank": self.rank,
            "loss_scope": "rank0_local_consumer_weighted",
            "grad_norm_scope": "rank0_local_shards_not_global_reduced",
            "lr_scope": "pre_optimizer_before_scheduler_step",
            "step_wall_scope": "first_trigger_fetch_to_post_commit_excludes_checkpoint",
            "native_forward_calls": self.forward,
            "native_backward_calls": self.backward,
            "outer_loss": self._outer_loss if self.backward else None,
            "epoch": epoch,
            "active_slots": active_slots,
            "step_wall_s": step_wall,
            "peak_allocated_gib": allocated_gb,
            "peak_reserved_gib": reserved_gb,
            **self._lr,
            **self._gradient_stats,
            **local,
        }
        for key in ("action_loss", "vision_loss"):
            coverage = self._metric_weights.get(key, 0.0)
            record[key] = self._metric_sums.get(key) if math.isclose(coverage, 1.0, abs_tol=1e-6) else None
            record[f"{key}_consumer_weight_coverage"] = coverage
        if self._started_iteration is not None and iteration != self._started_iteration:
            self._telemetry_errors.append("trigger/commit iteration mismatch")
        record["telemetry_errors"] = list(self._telemetry_errors)
        self.last_record = record
        try:
            self._emit("[CorrectedV3][train] " + json.dumps(record, allow_nan=False, sort_keys=True))
        except Exception:  # A completed optimizer transaction must not be rolled back by logging.
            pass

    def __call__(self, *, phase: str, trainer: Any, **data: Any) -> None:
        if phase == "native_forward":
            self.forward += 1
            if self.rank == 0:
                try:
                    self._on_forward(
                        trainer=trainer, member=data.get("member"), index=data.get("index"),
                        metrics=data.get("metrics"),
                    )
                except Exception as exc:
                    self._telemetry_errors.append(f"loss_observer: {type(exc).__name__}: {exc}")
        elif phase == "native_backward":
            self.backward += 1
            if self.rank == 0:
                scalar = self._scalar(data.get("loss"))
                if scalar is None:
                    self._telemetry_errors.append("missing/nonfinite weighted backward loss")
                else:
                    self._outer_loss += scalar
        elif phase == "pre_optimizer":
            plan = trainer._grouped_window.plan
            if plan is None:
                raise RuntimeError("observer 缺少 pending grouped plan")
            expected = sum(max(request.valid_count for request in member) for member in plan.members)
            if (self.forward, self.backward) != (expected, expected):
                raise RuntimeError(
                    f"native 调用次数不匹配：forward={self.forward}, backward={self.backward}, expected={expected}"
                )
            self._on_pre_optimizer(trainer=trainer, metrics=data.get("metrics"))
        elif phase == "post_commit":
            self.completed += 1
            # Observation must not break a transaction after optimizer success.
            try:
                self._on_post_commit(trainer=trainer, iteration=data.get("iteration"))
            except Exception:
                pass
            self.forward = self.backward = 0
            self._outer_loss = 0.0
            self._metric_sums.clear()
            self._metric_weights.clear()
            self._lr.clear()
            self._gradient_stats.clear()
            self._telemetry_errors.clear()
            self._started_at = None
            self._started_iteration = None

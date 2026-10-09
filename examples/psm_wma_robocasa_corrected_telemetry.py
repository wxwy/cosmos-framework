# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Passive per-optimizer-step telemetry for corrected grouped Local-TTT.

Losses are detached tensors accumulated on-device and materialized once,
after a successful optimizer/Local commit. CPU stage timings measure host
wall time (often CUDA dispatch), not GPU kernel occupancy. CUDA-event phase
estimates are sampled every 100 steps on rank 0 only. All FSDP gradient
and parameter norms are rank-local shard norms, never global reductions.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed.tensor import DTensor

LOSS_KEYS = (
    "flow_matching_loss_action",
    "flow_matching_loss_vision",
    "aux_loss_gen",
    "aux_loss_und",
)
GRAD_GROUPS = (
    "total",
    "generation",
    "action",
    "local",
    "local_encoder",
    "local_core",
    "local2llm",
    "local_modality_embed",
)
CPU_TIMING_STAGES = (
    "data_prepare",
    "batch_collate",
    "batch_transfer",
    "local_scan",
    "forward",
    "backward",
    "optimizer",
)
GPU_TIMING_STAGES = frozenset(("batch_transfer", "local_scan", "forward", "backward", "optimizer"))


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _add(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else current + value


class GroupedPlanObserver:
    """Keep the existing call-count guard; report only successfully committed steps."""

    def __init__(
        self,
        *,
        rank: int = 0,
        parameter_group: Callable[[str], str | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        emit: Callable[[str], None] = print,
        cuda_sample_interval: int = 100,
        parameter_norm_interval: int = 100,
    ) -> None:
        if cuda_sample_interval < 0 or parameter_norm_interval < 0:
            raise ValueError("telemetry intervals cannot be negative")
        self.rank = rank
        self._parameter_group = parameter_group
        self._clock = clock
        self._emit = emit
        self.cuda_sample_interval = cuda_sample_interval
        self.parameter_norm_interval = parameter_norm_interval
        self.forward = 0
        self.backward = 0
        self.completed = 0
        self.last_record: dict[str, Any] | None = None
        self._reset_step()

    def _reset_step(self) -> None:
        self._started_at: float | None = None
        self._started_iteration: int | None = None
        self._last_plan: Any = None
        self._losses: dict[str, torch.Tensor] = {}
        self._loss_coverage: dict[str, float] = {}
        self._outer: torch.Tensor | None = None
        self._pre_optimizer_lr: dict[str, float | None] = {}
        self._gradient_sums: dict[str, torch.Tensor | None] = {}
        self._gradient_counts: dict[str, int] = {}
        self._missing_gradient_tensors = 0
        self._nonfinite_gradient_tensors: torch.Tensor | None = None
        self._parameter_sums: dict[str, torch.Tensor | None] = {}
        self._stage_ms: dict[str, float] = {}
        self._cuda_events: dict[str, list[tuple[Any, Any]]] = {}
        self._sample_cuda = False
        self._sample_params = False
        self._errors: list[str] = []
        self.forward = self.backward = 0

    def start_iteration(self, iteration: int) -> None:
        """Called once per GA window; does not mutate the batch or training state."""
        if self.rank != 0:
            return
        self._reset_step()
        self._started_iteration = iteration
        self._started_at = self._clock()
        self._sample_cuda = bool(
            self.cuda_sample_interval
            and (iteration + 1) % self.cuda_sample_interval == 0
            and torch.cuda.is_available()
        )
        self._sample_params = bool(self.parameter_norm_interval and (iteration + 1) % self.parameter_norm_interval == 0)
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as exc:
                self._errors.append(f"memory_reset:{type(exc).__name__}")

    @contextmanager
    def time_stage(self, name: str) -> Iterator[None]:
        """Time the CPU dispatch interval and optional current-stream CUDA events."""
        if self.rank != 0:
            yield
            return
        before = self._clock()
        first_event = None
        if self._sample_cuda and name in GPU_TIMING_STAGES:
            try:
                first_event = torch.cuda.Event(enable_timing=True)
                first_event.record()
            except RuntimeError as exc:
                self._errors.append(f"cuda_event_start:{name}:{type(exc).__name__}")
                first_event = None
        try:
            yield
        finally:
            self._stage_ms[name] = self._stage_ms.get(name, 0.0) + (self._clock() - before) * 1000.0
            if first_event is not None:
                try:
                    last_event = torch.cuda.Event(enable_timing=True)
                    last_event.record()
                    self._cuda_events.setdefault(name, []).append((first_event, last_event))
                except RuntimeError as exc:
                    self._errors.append(f"cuda_event_end:{name}:{type(exc).__name__}")

    @staticmethod
    def _weight_for_index(plan: Any, member: int, index: int) -> float:
        if member < 0 or member >= len(plan.members) or index < 0:
            raise ValueError("invalid grouped member/index")
        denominator = plan.n_window if hasattr(plan, "n_window") else sum(
            request.valid_count for group in plan.members for request in group
        )
        if denominator <= 0:
            raise ValueError("empty grouped optimizer window")
        active = sum(request.valid_count > index for request in plan.members[member])
        if active <= 0:
            raise ValueError("native forward has no valid consumers")
        return active / denominator

    def _on_forward(self, trainer: Any, data: dict[str, Any]) -> None:
        member, index = data.get("member"), data.get("index")
        metrics = data.get("metrics")
        if member is None or index is None or not isinstance(metrics, dict):
            return
        plan = trainer._grouped_window.plan
        weight = self._weight_for_index(plan, member, index)
        with torch.no_grad():
            for name in LOSS_KEYS:
                value = metrics.get(name)
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    continue
                # Never .item() here: the native forward loop must not introduce
                # per-consumer GPU/CPU synchronization for logging.
                term = value.detach().float().reshape(()) * weight
                self._losses[name] = _add(self._losses.get(name), term)
                self._loss_coverage[name] = self._loss_coverage.get(name, 0.0) + weight

    @staticmethod
    def _gradient_group_names(name: str, group: str) -> tuple[str, ...]:
        if group == "generation":
            return ("total", "generation")
        if group == "action":
            return ("total", "generation", "action")
        if group == "local":
            if name.startswith("net.local_memory_runtime.encoder."):
                sub = "local_encoder"
            elif name.startswith("net.local_memory_runtime.core."):
                sub = "local_core"
            elif name.startswith("net.local_memory2llm."):
                sub = "local2llm"
            elif name.startswith("net.local_memory_modality_embed"):
                sub = "local_modality_embed"
            else:
                raise ValueError(f"unrecognized selected Local parameter: {name}")
            return ("total", "local", sub)
        raise ValueError(f"invalid selected gradient group: {group}")

    def _observe_gradients(self, model: Any) -> None:
        if self._parameter_group is None:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                group = self._parameter_group(name)
                if group is None:
                    continue
                groups = self._gradient_group_names(name, group)
                if self._sample_params:
                    # Parameter norms cover *all* trainable tensors, even when
                    # a particular step did not produce their gradients.
                    value = parameter.detach()
                    if isinstance(value, DTensor):
                        value = value.to_local()
                    p_sq = value.float().square().sum()
                    for key in groups:
                        self._parameter_sums[key] = _add(self._parameter_sums.get(key), p_sq)
                grad = parameter.grad
                if grad is None:
                    self._missing_gradient_tensors += 1
                    continue
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                grad = grad.detach()
                squared = grad.float().square().sum()
                invalid = (~torch.isfinite(grad).all()).to(dtype=torch.int32)
                self._nonfinite_gradient_tensors = _add(self._nonfinite_gradient_tensors, invalid)
                for key in groups:
                    self._gradient_sums[key] = _add(self._gradient_sums.get(key), squared)
                    self._gradient_counts[key] = self._gradient_counts.get(key, 0) + 1

    def _pre_optimizer(self, trainer: Any, metrics: Any) -> None:
        self._pre_optimizer_lr = {key: _scalar((metrics or {}).get(key)) for key in ("lr_min", "lr_max")}
        try:
            self._observe_gradients(trainer._grouped_window.model)
        except Exception as exc:
            self._errors.append(f"grad_read:{type(exc).__name__}:{exc}")
            self._gradient_sums.clear()
            self._parameter_sums.clear()

    def _materialize_norms(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {}
        for name in GRAD_GROUPS:
            grad_sq = self._gradient_sums.get(name)
            result[f"{name}_grad_norm_rank_local"] = _scalar(grad_sq.sqrt()) if grad_sq is not None else None
            result[f"{name}_grad_tensor_count_rank_local"] = self._gradient_counts.get(name, 0)
            if self._sample_params:
                param_sq = self._parameter_sums.get(name)
                result[f"{name}_param_norm_rank_local"] = _scalar(param_sq.sqrt()) if param_sq is not None else None
        result["missing_grad_tensors_rank_local"] = self._missing_gradient_tensors
        result["nonfinite_grad_tensors_rank_local"] = (
            int(_scalar(self._nonfinite_gradient_tensors) or 0)
            if self._nonfinite_gradient_tensors is not None
            else None
        )
        return result

    def _materialize_core(self, trainer: Any) -> dict[str, float | None]:
        core = trainer._grouped_window.model.net.local_memory_runtime.core
        summaries = core.drain_telemetry()
        result: dict[str, float | None] = {}
        for name in ("inner_loss", "fast_state_norm", "fast_update_norm"):
            count = _scalar(summaries.get(f"ttt_{name}_count"))
            summed = _scalar(summaries.get(f"ttt_{name}_sum"))
            result[f"{name}_count"] = count
            result[f"{name}_mean"] = summed / count if summed is not None and count and count > 0 else None
            result[f"{name}_max"] = _scalar(summaries.get(f"ttt_{name}_max"))
        return result

    def _sample_cuda_events(self) -> dict[str, float | None]:
        if not self._sample_cuda:
            return {}
        if not self._cuda_events:
            return {}
        result: dict[str, float | None] = {}
        try:
            # Once every N steps, rank0 only; never synchronize on each consumer.
            torch.cuda.synchronize()
            for name, pairs in self._cuda_events.items():
                result[f"{name}_cuda_event_ms"] = sum(start.elapsed_time(end) for start, end in pairs)
            compute = ("local_scan", "forward", "backward", "optimizer")
            if all(f"{name}_cuda_event_ms" in result for name in compute):
                result["gpu_compute_cuda_event_ms"] = sum(result[f"{name}_cuda_event_ms"] or 0.0 for name in compute)
        except RuntimeError as exc:
            self._errors.append(f"cuda_event_read:{type(exc).__name__}")
        return result

    def _post_commit(self, trainer: Any, iteration: int | None) -> None:
        if self.rank != 0:
            return
        # Wall time is captured before scalar materialization (which synchronizes).
        wall_s = self._clock() - self._started_at if self._started_at is not None else None
        window = trainer._grouped_window
        frontier = window.live.frontier
        plan = getattr(self, "_last_plan", None)
        valid_consumers = plan.n_window if plan is not None and hasattr(plan, "n_window") else None
        if self._started_iteration is not None and iteration != self._started_iteration:
            self._errors.append("trigger/commit_iteration_mismatch")
        gpu = self._sample_cuda_events()
        local = {}
        try:
            local = self._materialize_core(trainer)
        except Exception as exc:
            self._errors.append(f"local_read:{type(exc).__name__}:{exc}")
        loss_values = {
            key: (
                _scalar(self._losses[key])
                if math.isclose(self._loss_coverage.get(key, 0.0), 1.0, abs_tol=1e-6)
                and key in self._losses
                else None
            )
            for key in LOSS_KEYS
        }
        rf = getattr(getattr(window.model, "config", None), "rectified_flow_training_config", None)
        sample_level = getattr(rf, "sample_level_loss_averaging", False)
        action_coeff = _scalar(getattr(rf, "action_loss_weight", None))
        vision_coeff = _scalar(getattr(rf, "loss_scale", None))
        # An image-only branch can have image_loss_scale; under mixed image/video
        # streams the exact coefficient is not globally constant.
        image_coeff = getattr(rf, "image_loss_scale", None)
        if image_coeff is not None:
            vision_coeff = None
        weighted = {}
        for name, loss_key, coeff in (
            ("action_contribution", "flow_matching_loss_action", action_coeff),
            ("vision_contribution", "flow_matching_loss_vision", vision_coeff),
        ):
            scalar = loss_values[loss_key]
            weighted[name] = scalar * coeff if scalar is not None and coeff is not None and not sample_level else None
        if sample_level:
            self._errors.append("dynamic_sample_level_loss_scale:weighted_contributions_unavailable")
        norms = self._materialize_norms()
        mem_allocated = mem_reserved = None
        if torch.cuda.is_available():
            try:
                mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            except RuntimeError as exc:
                self._errors.append(f"peak_memory:{type(exc).__name__}")
        stage = {f"{key}_ms": self._stage_ms.get(key) for key in CPU_TIMING_STAGES}
        total_measured = sum(self._stage_ms.values())
        step_wall_ms = wall_s * 1000.0 if wall_s is not None else None
        record: dict[str, Any] = {
            "status": "optimizer_committed",
            "iteration": iteration + 1 if iteration is not None else None,
            "rank": self.rank,
            "rank_scope": "rank0_local_not_global_reduced",
            "loss_scope": "valid_consumer_weighted",
            "grad_norm_scope": "rank0_local_fsdp_shard",
            "cpu_timing_scope": "host_wall_cuda_dispatch_not_gpu_kernel_time",
            "cuda_timing_scope": "current_stream_events_every_100_steps_rank0_only",
            "data_wait_ms": None,
            "data_wait_reason": "synchronous_producer_no_background_dataloader",
            "valid_consumers": valid_consumers,
            "valid_consumers_per_second_rank_local": (
                valid_consumers / wall_s if valid_consumers is not None and wall_s and wall_s > 0 else None
            ),
            "native_forward_calls": self.forward,
            "native_backward_calls": self.backward,
            "outer_loss": _scalar(self._outer),
            **loss_values,
            **weighted,
            "action_loss_weight": action_coeff,
            "vision_loss_weight": vision_coeff,
            "epoch": getattr(frontier, "epoch", None),
            "active_slots": sum(slot.uid is not None for slot in frontier.slots),
            "lr_min": self._pre_optimizer_lr.get("lr_min"),
            "lr_max": self._pre_optimizer_lr.get("lr_max"),
            "lr_scope": "pre_optimizer_before_scheduler_step",
            "step_wall_ms": step_wall_ms,
            "step_wall_s": wall_s,
            "action_loss": loss_values.get("flow_matching_loss_action"),
            "vision_loss": loss_values.get("flow_matching_loss_vision"),
            "action_loss_consumer_weight_coverage": self._loss_coverage.get("flow_matching_loss_action", 0.0),
            "vision_loss_consumer_weight_coverage": self._loss_coverage.get("flow_matching_loss_vision", 0.0),
            "step_wall_scope": "first_trigger_to_post_commit_excludes_checkpoint",
            "other_host_overhead_ms": max(0.0, step_wall_ms - total_measured) if step_wall_ms is not None else None,
            "peak_allocated_gib": mem_allocated,
            "peak_reserved_gib": mem_reserved,
            "param_norm_sampled": self._sample_params,
            "cuda_sampled": self._sample_cuda,
            **norms,
            **stage,
            **local,
            **gpu,
        }
        for key in LOSS_KEYS:
            record[f"{key}_coverage"] = self._loss_coverage.get(key, 0.0)
        record["telemetry_errors"] = list(self._errors)
        self.last_record = record
        try:
            self._emit("[CorrectedV3][train] " + json.dumps(record, sort_keys=True, allow_nan=False))
        except Exception:
            # A post-commit logging failure must never roll back a committed optimizer step.
            pass

    def log_checkpoint(self, iteration: int, elapsed_ms: float) -> None:
        """Emit one separate save-event line; checkpoint is outside the step wall."""
        if self.rank != 0:
            return
        try:
            self._emit(
                "[CorrectedV3][checkpoint] "
                + json.dumps(
                    {"iteration": iteration, "checkpoint_save_ms": elapsed_ms}, sort_keys=True, allow_nan=False
                )
            )
        except Exception:
            pass

    def __call__(self, *, phase: str, trainer: Any, **data: Any) -> None:
        if phase == "native_forward":
            self.forward += 1
            if self.rank == 0:
                try:
                    self._on_forward(trainer, data)
                except Exception as exc:
                    self._errors.append(f"loss_observer:{type(exc).__name__}:{exc}")
        elif phase == "native_backward":
            self.backward += 1
            if self.rank == 0 and isinstance((loss := data.get("loss")), torch.Tensor):
                try:
                    self._outer = _add(self._outer, loss.detach().float().reshape(()))
                except Exception as exc:
                    self._errors.append(f"outer_observer:{type(exc).__name__}:{exc}")
        elif phase == "pre_optimizer":
            plan = trainer._grouped_window.plan
            if plan is None:
                raise RuntimeError("observer 缺少 pending grouped plan")
            expected = sum(max(request.valid_count for request in member) for member in plan.members)
            if (self.forward, self.backward) != (expected, expected):
                raise RuntimeError(
                    f"native 调用次数不匹配：forward={self.forward}, backward={self.backward}, expected={expected}"
                )
            if self.rank == 0:
                self._last_plan = plan
                self._pre_optimizer(trainer, data.get("metrics"))
        elif phase == "post_commit":
            self.completed += 1
            try:
                self._post_commit(trainer, data.get("iteration"))
            except Exception:
                # Metrics must never change post-success training semantics.
                pass
            finally:
                self._reset_step()
                self._last_plan = None

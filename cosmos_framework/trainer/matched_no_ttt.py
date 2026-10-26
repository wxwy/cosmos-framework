# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Matched Packed128 training without a Local module, prefix, scan or fast state.

Only source planning/materialization and trainer safety utilities are shared with
G1. Native Cosmos computes the unchanged action/vision/auxiliary outer loss.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import move_cached_native_batch
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowCatalogFrontier,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
from cosmos_framework.trainer.local_memory_grouped_resume import require_dcp_grouped_resume_component
from cosmos_framework.utils import misc
from cosmos_framework.utils.callback import Callback

FORMAT = "psm_v3_matched_no_ttt_frontier_v1"
LOADER_FORMAT = "psm_v3_matched_no_ttt_loader_frontier_v1"  # Old opt-in candidate; not written anymore.


def require_no_local_model(model: Any) -> None:
    if getattr(model.config, "local_memory_enabled", None) is not False:
        raise ValueError("Matched No-TTT requires local_memory_enabled=False")
    if any("local_memory" in name for name, _ in model.named_parameters()) or any(
        hasattr(model.net, name) for name in ("local_memory_runtime", "local_memory2llm", "local_memory_modality_embed")
    ):
        raise ValueError("Matched No-TTT must not instantiate Local modules/parameters")


@dataclass(frozen=True)
class FrontierState:
    frontier: ExactWindowCatalogFrontier


class MatchedNoTTTWindow:
    def __init__(self, model: Any, planner: ExactWindowRankPlanner):
        require_no_local_model(model)
        self.model, self.planner = model, planner
        self.live = FrontierState(planner.initial_frontier())
        self.plan = None
        self.member = 0

    def begin(self) -> None:
        if self.plan is not None:
            raise RuntimeError("No-TTT window already pending")
        self.plan = self.planner.plan_window(self.live.frontier)
        self.member = 0

    def pack(self, segments) -> tuple[tuple[dict, ...], tuple[tuple[int, int], ...]]:
        if self.plan is None or self.member >= self.planner.active_ga:
            raise RuntimeError("No-TTT has no pending member")
        requests = self.plan.members[self.member]
        if len(segments) != len(requests):
            raise ValueError("No-TTT member row count mismatch")
        payloads, positions = [], []
        t = self.planner.catalog.ttt_tbptt_steps
        for segment, request in zip(segments, requests, strict=True):
            segment.validate(t)
            identity = request.identity
            if (
                tuple(segment.slot_id.tolist()) != (identity.slot_id,)
                or segment.episode_id != (identity.episode_id,)
                or segment.category != (identity.category,)
                or segment.segment_provenance.source_digest != identity.source_digest
                or segment.segment_provenance.segment_id != identity.segment_id
                or segment.segment_provenance.manifest_digest != self.planner.catalog.manifest_digest
                or segment.consumer_valid[0].tolist() != [index < request.valid_count for index in range(t)]
            ):
                raise ValueError("No-TTT segment identity/slot/order/valid mask mismatch")
            for index in range(request.valid_count):
                payload = segment.consumer_payload[0][index]
                if (
                    int(segment.consumer_step[0, index]) != identity.cursor * t + index
                    or not isinstance(payload, dict)
                    or any(key in payload for key in ("local_memory", "x0_tokens_local_memory"))
                ):
                    raise ValueError("No-TTT consumer chronology/payload/prefix mismatch")
                payloads.append(payload)
                positions.append((identity.slot_id, index))
        if len(payloads) != self.plan.member_counts[self.member] or len(set(positions)) != len(payloads):
            raise ValueError("No-TTT packed consumer count mismatch")
        return tuple(payloads), tuple(positions)

    def finish(self) -> None:
        if self.plan is None or self.member != self.planner.active_ga:
            raise RuntimeError("No-TTT cannot commit before every GA backward")
        self.live = FrontierState(self.plan.candidate_frontier)
        self.abort()

    def abort(self) -> None:
        self.plan, self.member = None, 0


def _binding(trainer) -> dict[str, Any]:
    planner = trainer._grouped_planner
    # Worker pool configuration never changes the committed consumer frontier.
    return {
        "config_digest": trainer._grouped_config_digest,
        "cache_manifest_sha256": planner.catalog.manifest_digest,
        "cache_corpus_digest": planner.catalog.cache_corpus_digest,
        "source_binding_digest": planner.catalog.source_binding_digest,
        "profile": (
            planner.rank,
            planner.world_size,
            planner.seed,
            planner.b_stream,
            planner.active_ga,
            planner.catalog.ttt_tbptt_steps,
        ),
    }


class NoTTTFrontierCallback(Callback):
    """Per-rank dataloader state; deliberately incompatible with G1 Local state."""

    checkpoint_component = "dataloader"

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def has_checkpoint_state(self) -> bool:
        return hasattr(self.trainer, "_grouped_planner")

    def state_dict(self):
        trainer = self.trainer
        window = getattr(trainer, "_grouped_window", None)
        loader = getattr(trainer, "_no_ttt_loader", None)
        if (
            trainer._no_ttt_failed
            or window is None
            or window.plan is not None
            or trainer._grouped_completed_iteration < 1
            or (loader is not None and loader.pending is not None)
        ):
            raise RuntimeError("No-TTT checkpoint requires a committed optimizer boundary")
        trainer._grouped_planner.validate_frontier(window.live.frontier)
        return {
            "format": FORMAT,  # Transport-independent; compatible with existing workers=0 checkpoints.
            **_binding(trainer),
            "iteration": trainer._grouped_completed_iteration,
            "frontier": window.live.frontier,
        }

    def load_state_dict(self, state_dict):
        if not self.has_checkpoint_state() or hasattr(self.trainer, "_grouped_window"):
            raise RuntimeError("No-TTT restore must precede first training step")
        if self.trainer._pending_grouped_resume is not None:
            raise RuntimeError("No-TTT duplicate dataloader restore")
        self.trainer._pending_grouped_resume = deepcopy(state_dict)


def restore_no_ttt_frontier(trainer, iteration: int) -> None:
    state = trainer._pending_grouped_resume
    if state is None:
        if trainer._resume_required or iteration != 0:
            raise RuntimeError("No-TTT same-job Resume requires its rank frontier")
        return
    binding = _binding(trainer)
    if (
        not isinstance(state, dict)
        or set(state) != {"format", "iteration", "frontier", *binding}
        or state["format"] != FORMAT
        or type(state["iteration"]) is not int
        or state["iteration"] != iteration
        or iteration < 1
        or any(state[key] != value for key, value in binding.items())
        or type(state["frontier"]) is not ExactWindowCatalogFrontier
    ):
        raise ValueError("No-TTT frontier schema/iteration/source/config/profile mismatch")
    planner = trainer._grouped_planner
    planner.validate_frontier(state["frontier"])
    if any(slot.next_segment_id != iteration * planner.active_ga for slot in state["frontier"].slots):
        raise ValueError("No-TTT frontier segment counters disagree with committed iteration")
    trainer._grouped_window.live = FrontierState(state["frontier"])
    trainer._grouped_completed_iteration = iteration
    trainer._pending_grouped_resume = None


class MatchedNoTTTTrainer(ImaginaireTrainer):
    """Reuse the native outer lifecycle; no GroupedLocalMemoryTrainer instance."""

    # Shared safety and passive telemetry utilities do not access Local state.
    _observe_grouped = GroupedLocalMemoryTrainer._observe_grouped
    _telemetry_stage = GroupedLocalMemoryTrainer._telemetry_stage
    _optimizer_lr_metrics = staticmethod(GroupedLocalMemoryTrainer._optimizer_lr_metrics)
    _require_finite_gradients = staticmethod(GroupedLocalMemoryTrainer._require_finite_gradients)
    _optimizer_step_success = staticmethod(GroupedLocalMemoryTrainer._optimizer_step_success)

    def bind_grouped_stream(
        self, planner, producer, *, config_digest, num_workers=0, full_segment_prefetch=False, loader_profile=None
    ):
        if num_workers != 0 or full_segment_prefetch:
            raise ValueError("Matched No-TTT initially requires the matched synchronous G1 data path")
        if hasattr(self, "_grouped_planner"):
            raise RuntimeError("No-TTT stream already bound")
        if (
            not isinstance(planner, ExactWindowRankPlanner)
            or not isinstance(producer, ExactWindowSegmentProducer)
            or producer.catalog is not planner.catalog
            or not config_digest
            or producer.config_digest != config_digest
            or planner.active_ga != self.config.trainer.grad_accum_iter
        ):
            raise ValueError("No-TTT planner/producer/catalog/GA/config mismatch")
        if any(getattr(cb, "checkpoint_component", None) == "dataloader" for cb in self.callbacks._callbacks):
            raise ValueError("No-TTT requires exclusive dataloader checkpoint state")
        self._grouped_planner, self._grouped_producer = planner, producer
        self._grouped_config_digest, self._grouped_completed_iteration = config_digest, 0
        self._grouped_num_workers = 0
        self._pending_grouped_resume = None
        self._resume_required = self._no_ttt_failed = False
        self._no_ttt_loader = None
        if loader_profile is not None and loader_profile.workers:
            from cosmos_framework.trainer.no_ttt_cpu_loader import NoTTTCpuLoader

            self._no_ttt_loader = NoTTTCpuLoader(producer, loader_profile)
        self.callbacks._callbacks.insert(0, NoTTTFrontierCallback(self))

    def train(self, model, dataloader_train, dataloader_val):
        if not hasattr(self, "_grouped_planner"):
            raise RuntimeError("No-TTT stream not bound")
        require_no_local_model(model)
        if self.config.trainer.save_zero_checkpoint or not self.config.checkpoint.strict_resume:
            raise ValueError("No-TTT requires strict Resume and forbids zero-step checkpoints")
        self._resume_required = require_dcp_grouped_resume_component(self.checkpointer)
        try:
            return super().train(model, dataloader_train, dataloader_val)
        finally:
            if self._no_ttt_loader is not None:
                self._no_ttt_loader.close()

    def training_step(self, model_ddp, optimizer, scheduler, grad_scaler, data, iteration=0, grad_accum_iter=0):
        del data
        planner = self._grouped_planner
        if (
            self._no_ttt_failed
            or self.config.trainer.grad_accum_iter != planner.active_ga
            or not 0 <= grad_accum_iter < planner.active_ga
            or getattr(getattr(self.config, "model_parallel", None), "context_parallel_size", 1) != 1
        ):
            raise RuntimeError("No-TTT failed instance or GA/CP mismatch")
        model = model_ddp.module if self.config.trainer.distributed_parallelism == "ddp" else model_ddp
        try:
            if not hasattr(self, "_grouped_window"):
                self._grouped_window = MatchedNoTTTWindow(model, planner)
                restore_no_ttt_frontier(self, iteration)
            window = self._grouped_window
            if iteration != self._grouped_completed_iteration or grad_accum_iter != window.member:
                raise RuntimeError("No-TTT iteration/member chronology mismatch")
            if grad_accum_iter == 0:
                window.begin()
                if self._no_ttt_loader is not None:
                    self._no_ttt_loader.begin(window.plan)
            with self._telemetry_stage("data_prepare"):
                segments = tuple(
                    (
                        self._grouped_producer.produce(request)
                        if self._no_ttt_loader is None
                        else self._no_ttt_loader.materialize(request)
                    )
                    for request in window.plan.members[window.member]
                )
                if any(segment.segment_provenance.config_digest != self._grouped_config_digest for segment in segments):
                    raise ValueError("No-TTT segment config provenance mismatch")
                payloads, positions = window.pack(segments)
            with self._telemetry_stage("batch_collate"):
                started = time.perf_counter() if self._no_ttt_loader is not None else None
                batch = collate_grouped_native_batch(payloads)
                if started is not None:
                    self._no_ttt_loader.metrics["collate_host_ms"] += (time.perf_counter() - started) * 1000
            with self._telemetry_stage("batch_transfer"):
                batch = move_cached_native_batch(batch, device=next(model.parameters()).device, move=misc.to)
            self.callbacks.on_before_forward(iteration=iteration)
            with self._telemetry_stage("forward"), self.training_timer("forward"):
                # Omit the prefix override entirely, including an all-None tuple.
                output, loss = model_ddp.training_step(batch, iteration)
            if (
                not isinstance(loss, torch.Tensor)
                or loss.ndim != 0
                or not loss.requires_grad
                or not bool(torch.isfinite(loss))
                or "_backward_loss" in output
            ):
                raise ValueError("No-TTT requires finite differentiable native outer loss")
            metrics = {
                key: value.detach()
                for key, value in output.items()
                if isinstance(value, torch.Tensor) and value.ndim == 0 and "loss" in key
            }
            self._observe_grouped(
                "native_forward",
                iteration=iteration,
                member=grad_accum_iter,
                loss=loss,
                metrics=metrics,
                payloads=payloads,
                packed_positions=positions,
            )
            self.callbacks.on_after_forward(iteration=iteration)
            # Same G1 rank-local valid-consumer objective, including terminal partials.
            weighted = loss * (len(payloads) / window.plan.n_window)
            self.callbacks.on_before_backward(model, weighted, iteration=iteration)
            with self._telemetry_stage("backward"), self.training_timer("backward"):
                grad_scaler.scale(weighted).backward()
                model.on_after_backward()
            self._observe_grouped(
                "native_backward", iteration=iteration, member=grad_accum_iter, index=0, loss=weighted.detach()
            )
            self.callbacks.on_after_backward(model, iteration=iteration)
            window.member += 1
            if window.member < planner.active_ga:
                return output, weighted.detach(), window.member
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
                optimizer=optimizer,
            )
            with self._telemetry_stage("optimizer"):
                if self._optimizer_step_success(optimizer, scheduler, grad_scaler) is not True:
                    raise RuntimeError("No-TTT optimizer skipped; frontier not committed")
            window.finish()
            self._grouped_completed_iteration = iteration + 1
            if self._no_ttt_loader is not None:
                loader_metrics = self._no_ttt_loader.finish()
                loader_metrics.update(iteration=iteration + 1, rank=planner.rank)
                try:
                    emit = getattr(self, "cpu_loader_observer", None)
                    if emit is not None:
                        emit(loader_metrics)
                    else:
                        print("[NoTTT][cpu_loader] " + json.dumps(loader_metrics, sort_keys=True), flush=True)
                except Exception:
                    pass  # Passive post-commit telemetry cannot change optimizer semantics.
            self._observe_grouped("post_commit", iteration=iteration, member=grad_accum_iter)
            self.callbacks.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
            model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
            self._zero_grad(model, optimizer, iteration)
            return output, weighted.detach(), 0
        except BaseException:
            self._no_ttt_failed = True
            if hasattr(self, "_grouped_window"):
                self._grouped_window.abort()
            if self._no_ttt_loader is not None:
                self._no_ttt_loader.close()
            raise

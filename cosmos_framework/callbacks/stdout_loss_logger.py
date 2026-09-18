# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stdout-only training loss logger.

A lightweight alternative to the ``basic`` callback group when W&B must stay
disabled. Logs the total loss and key sub-losses to stdout every
``logging_iter`` iterations so that the training log file captures the loss
curve without requiring ``wandb`` initialization.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.misc import get_data_batch_size


class StdoutLossLogger(Callback):
    """Log training loss and sub-losses to stdout (rank 0 only).

    Args:
        every_n: Log every ``every_n`` optimizer steps. Defaults to 1.
        sub_loss_keys: Sub-loss keys from ``output_batch`` to print alongside the
            total loss. Defaults to vision and action flow-matching losses for
            action-policy SFT.
    """

    def __init__(
        self,
        every_n: int = 1,
        sub_loss_keys: tuple[str, ...] = (
            "flow_matching_loss_vision",
            "flow_matching_loss_action",
        ),
    ) -> None:
        super().__init__()
        self.every_n = every_n
        self.sub_loss_keys = sub_loss_keys
        self._active_window_objectives: list[float] = []
        self._active_raw_losses: list[float] = []



    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        active = output_batch.get("psm_local_memory_active_forward")
        if active is None:
            return
        prepared = active.prepared
        objective = prepared.transaction.plan.objective(
            prepared.member_index,
            active.result.primary_consumer_mean,
            active.result.auxiliary_loss,
            prepared.actual_n_valid,
        )
        self._active_window_objectives.append(float(objective.detach()))
        self._active_raw_losses.append(float(loss.detach()))

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if iteration % self.every_n != 0:
            return

        active_window = bool(self._active_window_objectives)
        sample_size = torch.tensor(get_data_batch_size(data_batch), device="cuda")
        if active_window:
            # Active Local Memory already owns exact GA/window normalization through
            # GAWindowPlan.objective().  The final microbatch's raw forward loss is not
            # an optimizer-step loss and varies with grouped member size.
            loss_sum = torch.tensor(sum(self._active_window_objectives), device="cuda", dtype=torch.float32)
            sample_size = torch.tensor(1.0, device="cuda")
        else:
            loss_sum = loss.detach().float() * sample_size

        sub_losses: dict[str, torch.Tensor] = {}
        for key in self.sub_loss_keys:
            value = output_batch.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                sub_losses[key] = value.detach().float() * sample_size
            else:
                sub_losses[key] = torch.tensor(float(value), device="cuda") * sample_size

        dist_available = dist.is_available() and dist.is_initialized()
        if dist_available:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
            for v in sub_losses.values():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)

        if not distributed.is_rank0():
            return

        avg_loss = loss_sum.item() / sample_size.item() if sample_size.item() > 0 else float("nan")
        parts = [f"iteration={iteration}", f"train/loss={avg_loss:.6f}"]
        if active_window:
            parts.append(f"raw_last_group_loss={self._active_raw_losses[-1]:.6f}")
            parts.append(f"active_groups={len(self._active_window_objectives)}")
        for key in self.sub_loss_keys:
            if key not in sub_losses:
                continue
            avg = sub_losses[key].item() / sample_size.item() if sample_size.item() > 0 else float("nan")
            parts.append(f"{key}={avg:.6f}")

        timing = self.trainer.last_optimizer_step_timing
        if timing is not None:
            parts.extend(
                (
                    f"perf/dataloader_wait_s={timing['dataloader_wait_seconds']:.6f}",
                    f"perf/dataloader_wait_mean_s={timing['dataloader_wait_mean_seconds']:.6f}",
                    f"perf/model_compute_s={timing['model_compute_seconds']:.6f}",
                    f"perf/model_compute_mean_s={timing['model_compute_mean_seconds']:.6f}",
                    f"perf/other_s={timing['other_seconds']:.6f}",
                    f"perf/step_wall_s={timing['step_wall_seconds']:.6f}",
                    f"perf/dataloader_wait_pct={100.0 * timing['dataloader_wait_fraction']:.2f}",
                    f"perf/model_compute_pct={100.0 * timing['model_compute_fraction']:.2f}",
                    f"perf/microbatches={timing['microbatch_count']}",
                )
            )

        log.info(" | ".join(parts))
        if active_window:
            self._active_window_objectives.clear()
            self._active_raw_losses.clear()

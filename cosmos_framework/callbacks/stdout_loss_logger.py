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

        sample_size = torch.tensor(get_data_batch_size(data_batch), device="cuda")
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
        for key in self.sub_loss_keys:
            if key not in sub_losses:
                continue
            avg = sub_losses[key].item() / sample_size.item() if sample_size.item() > 0 else float("nan")
            parts.append(f"{key}={avg:.6f}")

        log.info(" | ".join(parts))

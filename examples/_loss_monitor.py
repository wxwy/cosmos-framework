# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal per-part loss monitor for action-policy SFT.

Motivation: ``OmniMoTModel.training_step`` already computes the vision and action
flow-matching losses separately and exposes them in ``output_batch``
(``flow_matching_loss_vision`` / ``flow_matching_loss_action``, both RAW / unscaled).
But the edge-warmstart recipe strips the official ``wandb_log`` callback (offline)
and the ``grad_clip`` callback logs only clip norms — so neither the total nor the
per-part loss reaches stdout during a run. That hides exactly the signal needed to
judge "is the action head learning" independently of the video head.

This callback accumulates those two keys (plus the scalar total loss) from
``output_batch`` in ``on_training_step_end`` — which fires once per outer training
iteration with the last grad-accum micro-batch's output — and prints them at
INFO level (rank 0) every ``log_interval`` iterations. Raw values are printed;
for cross-check the weighted relation is:
    total ≈ vision * rectified_flow.loss_scale + action * rectified_flow.action_loss_weight
(both weights are 10.0 in the edge recipe).

No cosmos source is touched: it is an independent callback file under examples/,
injected at launch via the Hydra override
    trainer.callbacks.loss_monitor._target_=examples._loss_monitor.LossMonitor
(see examples/launch_sft_action_policy_libero_edge_task0_overfit.sh + EXTRA_TAIL_OVERRIDES).
"""

from __future__ import annotations

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback

_LOSS_KEYS = ("flow_matching_loss_vision", "flow_matching_loss_action")


class LossMonitor(Callback):
    """Accumulate and print the vision / action loss components separately."""

    def __init__(self, log_interval: int | None = None) -> None:
        """``log_interval=None`` falls back to ``config.trainer.logging_iter`` (>=1)."""
        super().__init__()
        self.log_interval = log_interval
        self._acc = {"total": 0.0, "vision": 0.0, "action": 0.0}
        self._n = 0
        self._seen = 0

    def on_training_step_end(
        self,
        model,
        data_batch,
        output_batch,
        loss,
        iteration: int = 0,
    ) -> None:
        interval = self.log_interval
        if interval is None:
            cfg = getattr(self, "config", None)
            interval = getattr(getattr(cfg, "trainer", None), "logging_iter", 1) if cfg is not None else 1
        if interval is None or interval < 1:
            interval = 1

        self._acc["total"] += float(loss.detach().float())
        for key in _LOSS_KEYS:
            val = output_batch.get(key)
            if val is not None:
                self._acc[key.split("flow_matching_loss_")[-1]] += float(val.detach().float())
        self._n += 1
        self._seen += 1

        if self._seen >= interval:
            n = max(self._n, 1)
            log.info(
                f"[loss] iter={iteration}  vision={self._acc['vision'] / n:.5f}  "
                f"action={self._acc['action'] / n:.5f}  total={self._acc['total'] / n:.5f}"
            )
            self._acc = {"total": 0.0, "vision": 0.0, "action": 0.0}
            self._n = 0
            self._seen = 0

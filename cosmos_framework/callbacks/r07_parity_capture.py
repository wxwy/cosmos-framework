# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""R07 Local-disabled parity 的最小输出摘要采集器。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback


class R07ParityCaptureCallback(Callback):
    """将一次真实训练步的原生输出与 mRoPE 摘要写入 JSON。"""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self._written = False

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        value = tensor.detach().contiguous().cpu()
        raw_bytes = value.view(torch.uint16).numpy().tobytes() if value.dtype is torch.bfloat16 else value.numpy().tobytes()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "max_abs": float(value.float().abs().max().item()) if value.numel() else 0.0,
        }

    @classmethod
    def _tensor_list_summary(cls, tensors: list[torch.Tensor]) -> list[dict[str, object]]:
        return [cls._tensor_summary(tensor) for tensor in tensors]

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch
        if self._written or not distributed.is_rank0():
            return
        payload = {
            "schema_version": "r07_no_memory_parity_v1",
            "iteration": iteration,
            "loss": float(loss.detach().item()),
            "flow_matching_loss_vision": float(output_batch["flow_matching_loss_vision"].detach().item()),
            "flow_matching_loss_action": float(output_batch["flow_matching_loss_action"].detach().item()),
            "preds_vision": self._tensor_list_summary(output_batch["model_pred"]),
            "preds_action": self._tensor_list_summary(output_batch["r07_parity_preds_action"]),
            "position_ids": self._tensor_summary(output_batch["r07_parity_position_ids"]),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_path.replace(self.output_path)
        self._written = True

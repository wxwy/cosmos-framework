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

    def __init__(self, output_path: str, tensor_output_path: str | None = None) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.tensor_output_path = Path(tensor_output_path) if tensor_output_path else None
        self._written = False

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        value = tensor.detach().contiguous().cpu()
        raw_bytes = (
            value.view(torch.uint16).numpy().tobytes()
            if value.dtype is torch.bfloat16
            else value.numpy().tobytes()
        )
        value_float = value.float()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "max_abs": float(value_float.abs().max().item()) if value.numel() else 0.0,
            "mean": float(value_float.mean().item()) if value.numel() else 0.0,
            "l2_norm": float(torch.linalg.vector_norm(value_float).item()) if value.numel() else 0.0,
        }

    @classmethod
    def _tensor_list_summary(cls, tensors: list[torch.Tensor]) -> list[dict[str, object]]:
        return [cls._tensor_summary(tensor) for tensor in tensors]

    @classmethod
    def _tensor_or_list_summary(cls, value: torch.Tensor | list[torch.Tensor | list[torch.Tensor]]) -> dict[str, object] | list[dict[str, object]]:
        if isinstance(value, torch.Tensor):
            return cls._tensor_summary(value)
        tensors = [item[0] if isinstance(item, list) and len(item) == 1 else item for item in value]
        if not all(isinstance(item, torch.Tensor) for item in tensors):
            raise TypeError("history_mask list entries must be tensors or singleton tensor lists.")
        return cls._tensor_list_summary(tensors)

    @staticmethod
    def _cpu_tensor_list(tensors: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        return [tensor.detach().cpu() if tensor is not None else None for tensor in tensors]

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model
        if self._written or not distributed.is_rank0():
            return
        payload = {
            "schema_version": "r07_no_memory_parity_v1",
            "iteration": iteration,
            "loss": float(loss.detach().item()),
            "flow_matching_loss_vision": float(output_batch["flow_matching_loss_vision"].detach().item()),
            "flow_matching_loss_action": float(output_batch["flow_matching_loss_action"].detach().item()),
            "x0_vision": self._tensor_list_summary(output_batch["x0"]),
            "xt_vision": self._tensor_list_summary(output_batch["xt"]),
            "sigma_vision_schedule": self._tensor_summary(output_batch["sigma"]),
            "sigma_vision_effective": self._tensor_list_summary(
                output_batch["r07_parity_sigma_vision_effective"]
            ),
            "x0_action": self._tensor_list_summary(output_batch["r07_parity_x0_action"]),
            "xt_action": self._tensor_list_summary(output_batch["r07_parity_xt_action"]),
            "sigma_action_effective": self._tensor_list_summary(
                output_batch["r07_parity_sigma_action_effective"]
            ),
            "text_ids": self._tensor_summary(output_batch["r07_parity_text_ids"]),
            "text_indexes": self._tensor_summary(output_batch["r07_parity_text_indexes"]),
            "vision_indexes": self._tensor_summary(output_batch["r07_parity_vision_indexes"]),
            "action_indexes": self._tensor_summary(output_batch["r07_parity_action_indexes"]),
            "split_lens": output_batch["split_lens"],
            "attn_modes": output_batch["attn_modes"],
            # Vision has already been assigned to output_batch["model_pred"]; action
            # is exposed explicitly by the parity-only model branch above.
            "preds_vision": self._tensor_list_summary(output_batch["model_pred"]),
            "preds_action": self._tensor_list_summary(output_batch["r07_parity_preds_action"]),
            "position_ids": self._tensor_summary(output_batch["r07_parity_position_ids"]),
            "history_mask": self._tensor_or_list_summary(data_batch["history_mask"]),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_path.replace(self.output_path)
        if self.tensor_output_path is not None:
            tensor_payload = {
                "schema_version": "r07_sensitivity_tensors_v1",
                "iteration": iteration,
                "preds_vision": self._cpu_tensor_list(output_batch["model_pred"]),
                "preds_action": self._cpu_tensor_list(output_batch["r07_parity_preds_action"]),
                "local_memory": self._cpu_tensor_list(data_batch.get("local_memory", [])),
            }
            self.tensor_output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_tensor_path = self.tensor_output_path.with_suffix(self.tensor_output_path.suffix + ".tmp")
            torch.save(tensor_payload, temporary_tensor_path)
            temporary_tensor_path.replace(self.tensor_output_path)
        self._written = True

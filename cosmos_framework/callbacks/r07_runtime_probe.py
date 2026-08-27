# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""R07 运行时证据写入器，仅由 smoke 配方启用。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback


class R07RuntimeProbeCallback(Callback):
    """在真实优化步骤中记录 Local 参数和梯度证据。"""

    def __init__(self, output_path: str, every_n: int = 1) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.every_n = every_n
        self._gradient_summaries: dict[int, dict[str, dict[str, float | bool | None]]] = {}
        self._hook_gradient_summaries: dict[int, dict[str, dict[str, float | bool | None]]] = {}
        self._records: list[dict[str, object]] = []

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, float | bool | None]:
        if tensor is None:
            return {"present": False, "finite": None, "max_abs": None}
        detached = tensor.detach()
        return {
            "present": True,
            "finite": bool(torch.isfinite(detached).all().item()),
            "max_abs": float(detached.abs().max().item()),
        }

    @staticmethod
    def _local_parameters(model: ImaginaireModel) -> tuple[torch.nn.Linear, torch.nn.Parameter]:
        net = getattr(model, "net", None)
        adapter = getattr(net, "local_memory2llm", None)
        embed = getattr(net, "local_memory_modality_embed", None)
        if adapter is None or embed is None:
            raise RuntimeError("R07 runtime probe requires Local-enabled Cosmos3VFMNetwork")
        return adapter, embed

    def on_training_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if iteration % self.every_n != 0 or not distributed.is_rank0():
            return
        adapter, embed = self._local_parameters(model)
        step = iteration + 1
        summaries: dict[str, dict[str, float | bool | None]] = {}
        self._hook_gradient_summaries[step] = summaries

        def register(name: str, tensor: torch.Tensor) -> None:
            if tensor.requires_grad:
                tensor.register_hook(lambda gradient: summaries.__setitem__(name, self._tensor_summary(gradient)))

        register("local_memory2llm_weight_grad", adapter.weight)
        register("local_memory2llm_bias_grad", adapter.bias)
        register("local_memory_modality_embed_grad", embed)

    def _fsdp_gradient_summaries(self, model: ImaginaireModel) -> dict[str, dict[str, float | bool | None]]:
        targets = {
            "local_memory2llm.weight": "local_memory2llm_weight_grad",
            "local_memory2llm.bias": "local_memory2llm_bias_grad",
            "local_memory_modality_embed": "local_memory_modality_embed_grad",
        }
        summaries: dict[str, dict[str, float | bool | None]] = {}
        for handle in _get_fsdp_handles(model):
            flat_parameter = handle.flat_param
            for fqn, shard_info in zip(flat_parameter._fqns, flat_parameter._shard_param_infos):
                matched_name = next((name for suffix, name in targets.items() if fqn.endswith(suffix)), None)
                if matched_name is None:
                    continue
                if not shard_info.in_shard or flat_parameter.grad is None:
                    summaries[matched_name] = self._tensor_summary(None)
                    continue
                start = shard_info.offset_in_shard
                length = shard_info.numel_in_shard
                if start is None or length is None:
                    raise RuntimeError(f"R07 FSDP Local gradient metadata missing for {fqn}")
                summaries[matched_name] = self._tensor_summary(flat_parameter.grad.narrow(0, start, length))
        return summaries

    def on_after_backward(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if iteration % self.every_n != 0 or not distributed.is_rank0():
            return
        adapter, embed = self._local_parameters(model)
        # Trainer 在反向阶段传入零基 iteration，而 optimizer step 结束回调传入一基 iteration。
        summaries = self._hook_gradient_summaries.pop(iteration + 1, None)
        if not summaries:
            summaries = self._fsdp_gradient_summaries(model)
        self._gradient_summaries[iteration + 1] = summaries or {
            "local_memory2llm_weight_grad": self._tensor_summary(adapter.weight.grad),
            "local_memory2llm_bias_grad": self._tensor_summary(adapter.bias.grad),
            "local_memory_modality_embed_grad": self._tensor_summary(embed.grad),
        }

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if iteration % self.every_n != 0 or not distributed.is_rank0():
            return
        adapter, embed = self._local_parameters(model)
        payload = {
            "iteration": iteration,
            "loss": float(loss.detach().item()),
            "flow_matching_loss_vision": float(output_batch["flow_matching_loss_vision"].detach().item()),
            "flow_matching_loss_action": float(output_batch["flow_matching_loss_action"].detach().item()),
            "local_memory2llm_weight": self._tensor_summary(adapter.weight),
            "local_memory2llm_bias": self._tensor_summary(adapter.bias),
            "local_memory_modality_embed": self._tensor_summary(embed),
        }
        gradient_summaries = self._gradient_summaries.pop(iteration, None)
        if gradient_summaries is None:
            raise RuntimeError(f"R07 runtime probe did not capture gradients for iteration {iteration}")
        payload.update(gradient_summaries)
        self._records.append(payload)
        document = {"schema_version": "r07_runtime_probe_v2", "records": self._records}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_path.replace(self.output_path)

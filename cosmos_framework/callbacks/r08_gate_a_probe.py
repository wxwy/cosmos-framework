"""Machine-readable evidence for the controlled R08 Gate A optimizer step."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback


class R08GateAProbeCallback(Callback):
    """Record real R08/R07 Local gradients, optimizer membership, and updates."""

    _PREFIXES = ("local_history_runtime.", "local_memory2llm", "local_memory_modality_embed")

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.before: dict[str, torch.Tensor] = {}
        self.grads: dict[str, dict[str, object]] = {}
        self.optimizer_membership: dict[str, bool] = {}

    @staticmethod
    def _summary(value: torch.Tensor | None) -> dict[str, object]:
        if value is None:
            return {"present": False, "finite": None, "max_abs": None}
        value = value.detach()
        return {"present": True, "finite": bool(torch.isfinite(value).all()), "max_abs": float(value.abs().max())}

    def _targets(self, model: ImaginaireModel):
        return {name: value for name, value in model.net.named_parameters() if name.startswith(self._PREFIXES)}

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        if iteration != 0 or not distributed.is_rank0():
            return
        for name, value in self._targets(model).items():
            self.before[name] = value.detach().clone()
            value.register_hook(lambda grad, name=name: self.grads.__setitem__(name, self._summary(grad)))

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        if iteration != 0 or not distributed.is_rank0():
            return
        params = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.optimizer_membership = {name: id(value) in params for name, value in self._targets(model).items()}

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if iteration != 1 or not distributed.is_rank0():
            return
        targets = self._targets(model)
        updates = {name: self._summary(value.detach() - self.before[name]) for name, value in targets.items()}
        payload = {"schema_version": "r08_gate_a_probe_v1", "iteration": iteration, "loss": float(loss.detach()),
                   "optimizer_membership": self.optimizer_membership, "gradients": self.grads, "updates": updates}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

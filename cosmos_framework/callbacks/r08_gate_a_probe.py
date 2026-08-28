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
        self.before: dict[int, dict[str, torch.Tensor]] = {}
        self.grads: dict[int, dict[str, dict[str, object]]] = {}
        self.optimizer_membership: dict[int, dict[str, bool]] = {}
        self.steps: list[dict[str, object]] = []

    @staticmethod
    def _summary(value: torch.Tensor | None) -> dict[str, object]:
        if value is None:
            return {"present": False, "finite": None, "max_abs": None}
        value = value.detach()
        return {"present": True, "finite": bool(torch.isfinite(value).all()), "max_abs": float(value.abs().max())}

    def _targets(self, model: ImaginaireModel):
        return {name: value for name, value in model.net.named_parameters() if name.startswith(self._PREFIXES)}

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        self.before[iteration] = {}
        self.grads[iteration] = {}
        for name, value in self._targets(model).items():
            self.before[iteration][name] = value.detach().clone()
            value.register_hook(
                lambda grad, name=name, iteration=iteration: self.grads[iteration].__setitem__(name, self._summary(grad))
            )

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        optimizers = getattr(optimizer, "optimizers", [optimizer])
        params = {id(param) for inner in optimizers for group in inner.param_groups for param in group["params"]}
        self.optimizer_membership[iteration] = {name: id(value) in params for name, value in self._targets(model).items()}

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        start_iteration = iteration - 1
        if start_iteration not in self.before:
            return
        targets = self._targets(model)
        updates = {
            name: self._summary(value.detach() - self.before[start_iteration][name]) for name, value in targets.items()
        }
        self.steps.append(
            {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "optimizer_membership": self.optimizer_membership.get(start_iteration, {}),
                "gradients": self.grads[start_iteration],
                "updates": updates,
            }
        )
        payload = {"schema_version": "r08_gate_a_probe_v2", "steps": self.steps}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

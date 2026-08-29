"""Machine-readable runtime evidence for the bounded R09-A1 smoke."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


class R09A1RuntimeProbeCallback(Callback):
    """Capture A1 optimizer, gradient, CUDA, and attached-backend state evidence."""

    _PREFIXES = (
        "local_history_runtime.encoder.",
        "local_history_runtime.recurrent_backend.",
        "local_memory2llm",
        "local_memory_modality_embed",
    )

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.payload: dict[str, object] = {"schema_version": "r09_a1_runtime_probe_v1"}

    @classmethod
    def _targets(cls, model):
        return {name: value for name, value in model.net.named_parameters() if name.startswith(cls._PREFIXES)}

    @staticmethod
    def _summary(value: torch.Tensor | None) -> dict[str, object]:
        if value is None:
            return {"present": False, "finite": None, "max_abs": None}
        return {
            "present": True,
            "finite": bool(torch.isfinite(value).all()),
            "max_abs": float(value.detach().abs().max()),
        }

    @staticmethod
    def _optimizer_parameters(optimizer) -> set[int]:
        children = getattr(optimizer, "optimizers", [optimizer])
        return {
            id(parameter)
            for child in children
            for group in child.param_groups
            for parameter in group["params"]
        }

    @torch.no_grad()
    def _state_contract(self, model) -> dict[str, object]:
        backend = model.net.local_history_runtime.recurrent_backend
        if backend is None:
            raise RuntimeError("R09-A1 probe requires an attached recurrent backend.")
        device = backend.cell.weight_ih.device
        dtype = backend.cell.weight_ih.dtype
        evidence = torch.arange(2 * 4 * backend.evidence_dim, device=device, dtype=dtype).view(2, 4, -1)
        evidence = evidence / max(evidence.numel(), 1)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]], device=device)
        full_token, full_state, full_present = backend.replay(evidence, mask)
        first_token, first_state, first_present = backend.replay(evidence[:, :2], mask[:, :2])
        del first_token, first_present
        detached_state = (first_state[0].detach(), first_state[1].detach())
        segment_token, segment_state, segment_present = backend.replay(evidence[:, 2:], mask[:, 2:], detached_state)
        reset = backend.reset_mask(full_state, torch.tensor([True, False], device=device))
        masked_token, masked_state, masked_present = backend.replay(
            torch.zeros_like(evidence[:, :1]), torch.zeros_like(mask[:, :1]), reset
        )
        del masked_state
        state_bytes = full_state[0].numel() * full_state[0].element_size() + full_state[1].numel() * full_state[1].element_size()
        return {
            "shape": list(full_state[0].shape),
            "dtype": str(full_state[0].dtype),
            "bytes": state_bytes,
            "segment_token_max_abs_diff": float((full_token - segment_token).abs().max()),
            "segment_state_max_abs_diff": float((full_state[0] - segment_state[0]).abs().max()),
            "segment_present_equal": bool(torch.equal(full_present, segment_present)),
            "reset_selected_latent_zero": bool(torch.equal(reset[0][0], torch.zeros_like(reset[0][0]))),
            "reset_selected_initialized_false": not bool(reset[1][0]),
            "reset_unselected_latent_exact": bool(torch.equal(reset[0][1], full_state[0][1])),
            "reset_unselected_initialized_exact": bool(torch.equal(reset[1][1], full_state[1][1])),
            "reset_all_mask_selected_absent": not bool(masked_present[0]),
            "reset_all_mask_selected_token_zero": bool(torch.equal(masked_token[0], torch.zeros_like(masked_token[0]))),
        }

    def on_train_start(self, model, iteration: int = 0) -> None:
        targets = self._targets(model)
        self.payload["target_names"] = sorted(targets)
        self.payload["target_tensor_count"] = len(targets)
        self.payload["target_element_count"] = sum(value.numel() for value in targets.values())
        self.payload["state_contract"] = self._state_contract(model)

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        del scheduler, grad_scaler
        if "representative_step" in self.payload:
            return
        targets = self._targets(model)
        optimizer_parameters = self._optimizer_parameters(optimizer)
        selected_names = sorted(name for name, value in targets.items() if id(value) in optimizer_parameters)
        unexpected_names = sorted(
            name for name, value in model.net.named_parameters() if id(value) in optimizer_parameters and name not in targets
        )
        gradients = {name: self._summary(value.grad) for name, value in targets.items()}
        self.payload["representative_step"] = {
            "iteration": iteration,
            "optimizer_names": selected_names,
            "optimizer_matches_targets": selected_names == sorted(targets),
            "unexpected_optimizer_names": unexpected_names,
            "gradients": gradients,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch, output_batch, loss
        if "representative_step" not in self.payload:
            return
        self.payload["last_iteration"] = iteration
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

"""Machine-readable runtime evidence for the bounded R09-B1 TTT smoke."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cosmos_framework.model.generator.mot.local_evidence import TTTLocalMemoryBackend
from cosmos_framework.utils.callback import Callback


class R09B1RuntimeProbeCallback(Callback):
    """Capture TTT state, optimizer membership, gradients, and CUDA peak memory."""

    _PREFIXES = (
        "local_history_runtime.encoder.",
        "local_memory2llm",
        "local_memory_modality_embed",
    )

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.payload: dict[str, object] = {"schema_version": "r09_b1_runtime_probe_v1"}

    @classmethod
    def _targets(cls, model):
        return {name: value for name, value in model.net.named_parameters() if name.startswith(cls._PREFIXES)}

    @staticmethod
    def _summary(value: torch.Tensor | None) -> dict[str, object]:
        if value is None:
            return {"present": False, "finite": None, "max_abs": None}
        value = value.detach()
        return {"present": True, "finite": bool(torch.isfinite(value).all()), "max_abs": float(value.abs().max())}

    @staticmethod
    def _optimizer_parameters(optimizer) -> set[int]:
        children = getattr(optimizer, "optimizers", [optimizer])
        return {id(parameter) for child in children for group in child.param_groups for parameter in group["params"]}

    @staticmethod
    def _state_members(state: tuple[torch.Tensor, ...]) -> dict[str, dict[str, object]]:
        names = ("W", "pending_evidence", "last_evidence", "initialized", "segment_progress")
        return {
            name: {
                "shape_per_sample": list(value.shape[1:]),
                "dtype": str(value.dtype).removeprefix("torch."),
                "bytes_per_sample": value[0].numel() * value.element_size(),
            }
            for name, value in zip(names, state, strict=True)
        }

    def _state_contract(self, model) -> dict[str, object]:
        backend = model.net.local_history_runtime.recurrent_backend
        if not isinstance(backend, TTTLocalMemoryBackend):
            raise RuntimeError("R09-B1 probe requires TTTLocalMemoryBackend.")
        device = next(model.net.local_memory2llm.parameters()).device
        evidence = torch.arange(2 * 7 * backend.evidence_dim, device=device, dtype=torch.float32).view(2, 7, -1)
        evidence = (evidence / max(evidence.numel(), 1)).requires_grad_()
        mask = torch.tensor([[True] * 7, [True, True, True, True, True, False, False]], device=device)
        full_token, full_state, full_present = backend.replay(evidence, mask)
        _, first_state, _ = backend.replay(evidence[:, :4], mask[:, :4])
        segment_token, segment_state, segment_present = backend.replay(evidence[:, 4:], mask[:, 4:], first_state)
        repeated_token, repeated_state, repeated_present = backend.replay(evidence, mask)
        reset = backend.reset_mask(full_state, torch.tensor([True, False], device=device))
        masked_token, masked_state, masked_present = backend.replay(
            torch.zeros_like(evidence[:, :1]), torch.zeros_like(mask[:, :1]), reset
        )
        per_sample_bytes = sum(value[0].numel() * value.element_size() for value in full_state)
        return {
            "members": self._state_members(full_state),
            "bytes_per_sample": per_sample_bytes,
            "segment_token_max_abs_diff": float((full_token - segment_token).abs().max()),
            "segment_members_exact": [bool(torch.equal(left, right)) for left, right in zip(full_state, segment_state, strict=True)],
            "segment_present_equal": bool(torch.equal(full_present, segment_present)),
            "fresh_token_exact": bool(torch.equal(full_token, repeated_token)),
            "fresh_members_exact": [bool(torch.equal(left, right)) for left, right in zip(full_state, repeated_state, strict=True)],
            "fresh_present_equal": bool(torch.equal(full_present, repeated_present)),
            "token_detached": not full_token.requires_grad and full_token.grad_fn is None,
            "members_detached": [not value.requires_grad and value.grad_fn is None for value in full_state],
            "reset_selected_members_zero": [bool(torch.equal(value[0], torch.zeros_like(value[0]))) for value in reset[:3]],
            "reset_selected_initialized_false": not bool(reset[3][0]),
            "reset_selected_progress_zero": int(reset[4][0]) == 0,
            "reset_unselected_members_exact": [bool(torch.equal(left[1], right[1])) for left, right in zip(reset, full_state, strict=True)],
            "reset_all_mask_selected_absent": not bool(masked_present[0]),
            "reset_all_mask_selected_token_zero": bool(torch.equal(masked_token[0], torch.zeros_like(masked_token[0]))),
            "reset_all_mask_members_detached": [not value.requires_grad and value.grad_fn is None for value in masked_state],
        }

    def on_train_start(self, model, iteration: int = 0) -> None:
        del iteration
        targets = self._targets(model)
        self.payload["target_names"] = sorted(targets)
        self.payload["state_contract"] = self._state_contract(model)

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        del scheduler, grad_scaler
        if "representative_step" in self.payload:
            return
        targets = self._targets(model)
        optimizer_parameters = self._optimizer_parameters(optimizer)
        selected = {name: value for name, value in targets.items() if id(value) in optimizer_parameters}
        unexpected = sorted(name for name, value in model.net.named_parameters() if id(value) in optimizer_parameters and name not in targets)
        gradients = {name: self._summary(value.grad) for name, value in targets.items()}
        groups = {
            "encoder": "local_history_runtime.encoder.",
            "local_memory2llm": "local_memory2llm",
            "local_memory_modality_embed": "local_memory_modality_embed",
        }
        self.payload["representative_step"] = {
            "iteration": iteration,
            "optimizer_names": sorted(selected),
            "optimizer_matches_targets": set(selected) == set(targets),
            "missing_optimizer_names": sorted(set(targets) - set(selected)),
            "unexpected_optimizer_names": unexpected,
            "gradients": gradients,
            "gradient_groups": {
                group: [gradients[name] for name in gradients if name.startswith(prefix)] for group, prefix in groups.items()
            },
        }

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch, output_batch, loss
        if "representative_step" not in self.payload:
            return
        self.payload["last_iteration"] = iteration
        self.payload["full_run_cuda_peak"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

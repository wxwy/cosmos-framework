"""CPU/static contract for Local Memory config, selectors and slow checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn


SELECTORS = (
    "local_history_runtime.encoder",
    "local_history_runtime.recurrent_backend",
    "local_memory2llm",
    "local_memory_modality_embed",
)


@dataclass(frozen=True)
class LocalMemoryConfig:
    ttt_tbptt_steps: int = 16
    inner_lr: float = 0.1
    k_local: int = 1
    runtime_evidence_steps: int = 1

    def validate(self) -> None:
        if isinstance(self.ttt_tbptt_steps, bool) or not isinstance(self.ttt_tbptt_steps, int) or self.ttt_tbptt_steps <= 0:
            raise ValueError("ttt_tbptt_steps must be a positive integer")
        if isinstance(self.inner_lr, bool) or not isinstance(self.inner_lr, (int, float)) or not math.isfinite(float(self.inner_lr)) or self.inner_lr <= 0:
            raise ValueError("inner_lr must be finite and positive")
        if isinstance(self.k_local, bool) or self.k_local not in (1, 4, 8):
            raise ValueError("k_local must be a positive integer")
        if isinstance(self.runtime_evidence_steps, bool) or not isinstance(self.runtime_evidence_steps, int) or self.runtime_evidence_steps != 1:
            raise ValueError("runtime_evidence_steps is fixed at 1")


def validate_slow_inventory(module: nn.Module, *, runtime_encoder: nn.Module, runtime_backend: nn.Module, external: Mapping[str, nn.Module] | None = None) -> tuple[str, ...]:
    if not isinstance(module, nn.Module) or module.encoder is not runtime_encoder or module.recurrent_backend is not runtime_backend:
        raise ValueError("runtime must reference the registered Local module objects")
    names = tuple(f"local_history_runtime.{name}" for name, _ in module.named_parameters() if not name.startswith("readout."))
    return names


def validate_optimizer_membership(named_parameters: Mapping[str, torch.Tensor], external: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    groups = {"local_history_runtime.encoder", "local_history_runtime.recurrent_backend"}
    groups |= {"local_memory2llm", "local_memory_modality_embed"}
    all_names = tuple(named_parameters) + tuple(external)
    if any(not any(name.startswith(group + ".") or name == group for group in groups) for name in all_names):
        raise ValueError("unexpected optimizer parameter group")
    if not all(any(name.startswith(group + ".") or name == group for name in all_names) for group in groups):
        raise ValueError("missing optimizer parameter group")
    return all_names


def canonical_slow_inventory(module: nn.Module, local_memory2llm: nn.Module, modality: nn.Parameter) -> dict[str, torch.Tensor]:
    values = {f"local_history_runtime.{name}": value for name, value in module.named_parameters() if not name.startswith("readout.")}
    values.update({f"local_memory2llm.{name}": value for name, value in local_memory2llm.named_parameters()})
    values["local_memory_modality_embed"] = modality
    return values


def validate_exact_optimizer_membership(candidate: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]) -> None:
    if set(candidate) != set(expected) or any(candidate[name] is not expected[name] for name in expected) or len({id(value) for value in candidate.values()}) != len(candidate):
        raise ValueError("optimizer membership is not exact")


def slow_checkpoint_payload(named_parameters: Mapping[str, torch.Tensor], config: LocalMemoryConfig) -> dict[str, object]:
    config.validate()
    if any(name.startswith(("fast_", "_state", "_pending", "_epoch", "_replay")) for name in named_parameters):
        raise ValueError("fast runtime state cannot enter checkpoint")
    return {"config": config, "parameters": {name: value.detach().clone() for name, value in named_parameters.items()}}


def strict_restore(payload: Mapping[str, object], expected: Mapping[str, torch.Tensor], config: LocalMemoryConfig) -> dict[str, torch.Tensor]:
    config.validate()
    if payload.get("config") != config or set(payload.get("parameters", {})) != set(expected):
        raise ValueError("checkpoint identity mismatch")
    values = payload["parameters"]
    if not isinstance(values, Mapping):
        raise ValueError("invalid checkpoint parameters")
    restored: dict[str, torch.Tensor] = {}
    for name, target in expected.items():
        value = values[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("checkpoint tensor mismatch")
        restored[name] = value.detach().clone()
    return restored


def strict_restore_into(module: nn.Module, payload: Mapping[str, object], expected: Mapping[str, torch.Tensor], config: LocalMemoryConfig, *, runtime_encoder: nn.Module, runtime_backend: nn.Module) -> nn.Module:
    restored = strict_restore(payload, expected, config)
    prefix = "local_history_runtime."
    state = {name[len(prefix):] if name.startswith(prefix) else name: value for name, value in restored.items()}
    module.load_state_dict(state, strict=True)
    validate_slow_inventory(module, runtime_encoder=runtime_encoder, runtime_backend=runtime_backend)
    return module

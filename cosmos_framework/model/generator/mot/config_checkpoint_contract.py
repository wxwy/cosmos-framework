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
        if not isinstance(self.inner_lr, (int, float)) or not math.isfinite(float(self.inner_lr)) or self.inner_lr <= 0:
            raise ValueError("inner_lr must be finite and positive")
        if isinstance(self.k_local, bool) or self.k_local not in (1, 4, 8):
            raise ValueError("k_local must be a positive integer")
        if self.runtime_evidence_steps != 1:
            raise ValueError("runtime_evidence_steps is fixed at 1")


def validate_slow_inventory(module: nn.Module, *, runtime_encoder: nn.Module, runtime_backend: nn.Module, external: Mapping[str, nn.Module] | None = None) -> tuple[str, ...]:
    if not isinstance(module, nn.Module) or module.encoder is not runtime_encoder or module.recurrent_backend is not runtime_backend:
        raise ValueError("runtime must reference the registered Local module objects")
    names = tuple(f"local_history_runtime.{name}" for name, _ in module.named_parameters() if not name.startswith("readout."))
    if any(name.startswith("readout.") for name in names):
        raise ValueError("dormant readout must not enter the Local optimizer inventory")
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

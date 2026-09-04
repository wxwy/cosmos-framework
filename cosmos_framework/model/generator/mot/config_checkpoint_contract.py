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
        if not isinstance(self.ttt_tbptt_steps, int) or self.ttt_tbptt_steps <= 0:
            raise ValueError("ttt_tbptt_steps must be a positive integer")
        if not isinstance(self.inner_lr, (int, float)) or not math.isfinite(float(self.inner_lr)) or self.inner_lr <= 0:
            raise ValueError("inner_lr must be finite and positive")
        if not isinstance(self.k_local, int) or self.k_local <= 0:
            raise ValueError("k_local must be a positive integer")
        if self.runtime_evidence_steps != 1:
            raise ValueError("runtime_evidence_steps is fixed at 1")


def validate_slow_inventory(module: nn.Module, *, runtime_encoder: nn.Module, runtime_backend: nn.Module) -> tuple[str, ...]:
    if not isinstance(module, nn.Module) or module.encoder is not runtime_encoder or module.recurrent_backend is not runtime_backend:
        raise ValueError("runtime must reference the registered Local module objects")
    names = tuple(name for name, _ in module.named_parameters())
    if any(name.startswith("readout.") for name in names):
        raise ValueError("dormant readout must not enter the Local optimizer inventory")
    return names


def slow_checkpoint_payload(named_parameters: Mapping[str, torch.Tensor], config: LocalMemoryConfig) -> dict[str, object]:
    config.validate()
    if any(name.startswith(("fast_", "_state", "_pending", "_epoch", "_replay")) for name in named_parameters):
        raise ValueError("fast runtime state cannot enter checkpoint")
    return {"config": config, "parameters": {name: value.detach().clone() for name, value in named_parameters.items()}}


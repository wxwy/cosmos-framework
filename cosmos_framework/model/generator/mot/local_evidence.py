# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stateless R08 per-step Local evidence encoding.

This module deliberately stops at ``E_history``. Readout, packed Local tokens,
Cosmos wiring, and persistent-memory mechanisms belong to later R08/R09 steps.
"""

from __future__ import annotations

import torch
from torch import nn


class LocalEvidenceEncoder(nn.Module):
    """Project causal per-step evidence into a mask-preserving shared space."""

    def __init__(
        self,
        evidence_dim: int = 256,
        visual_dim: int = 96,
        action_dim: int = 10,
        state_dim: int = 8,
        max_age_steps: int = 64,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        state_std_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if evidence_dim <= 0 or visual_dim <= 0 or action_dim <= 0 or state_dim <= 0 or max_age_steps < 0:
            raise ValueError("Local evidence dimensions must be positive and max_age_steps non-negative.")
        if (state_mean is None) != (state_std is None):
            raise ValueError("state_mean and state_std must be provided together.")
        if state_std_floor <= 0:
            raise ValueError("state_std_floor must be positive.")

        self.visual_proj = nn.Linear(visual_dim, evidence_dim)
        self.action_proj = nn.Linear(action_dim, evidence_dim)
        self.age_embedding = nn.Embedding(max_age_steps + 1, evidence_dim)
        self.dt_proj = nn.Linear(1, evidence_dim)
        self.norm = nn.LayerNorm(evidence_dim)
        self.max_age_steps = max_age_steps
        self.state_proj: nn.Linear | None = None
        if state_mean is not None and state_std is not None:
            if tuple(state_mean.shape) != (state_dim,) or tuple(state_std.shape) != (state_dim,):
                raise ValueError(f"state_mean/state_std must have shape [{state_dim}].")
            if not torch.isfinite(state_mean).all() or not torch.isfinite(state_std).all():
                raise ValueError("state_mean/state_std must be finite.")
            self.register_buffer("state_mean", state_mean.detach().float().clone())
            self.register_buffer("state_std", state_std.detach().float().clamp_min(state_std_floor).clone())
            self.state_proj = nn.Linear(state_dim, evidence_dim)
        else:
            self.register_buffer("state_mean", None, persistent=False)
            self.register_buffer("state_std", None, persistent=False)

    @staticmethod
    def _check_shape(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")

    def forward(
        self,
        *,
        history_visual_summary: torch.Tensor,
        local_history_action: torch.Tensor,
        history_age_steps: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_mask: torch.Tensor,
        history_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B,H,D_e]`` with every masked position exactly zero."""
        if history_visual_summary.ndim != 3:
            raise ValueError("history_visual_summary must have shape [B,H,D_v].")
        batch, horizon, _ = history_visual_summary.shape
        self._check_shape(
            history_visual_summary, (batch, horizon, self.visual_proj.in_features), "history_visual_summary"
        )
        self._check_shape(local_history_action, (batch, horizon, self.action_proj.in_features), "local_history_action")
        self._check_shape(history_age_steps, (batch, horizon), "history_age_steps")
        self._check_shape(history_dt_s, (batch, horizon, 1), "history_dt_s")
        self._check_shape(history_mask, (batch, horizon), "history_mask")
        if not torch.isfinite(history_visual_summary).all() or not torch.isfinite(local_history_action).all():
            raise ValueError("Local evidence inputs must be finite.")

        encoded = self.visual_proj(history_visual_summary)
        encoded = encoded + self.action_proj(local_history_action)
        encoded = encoded + self.age_embedding(history_age_steps.long().clamp(0, self.max_age_steps))
        encoded = encoded + self.dt_proj(history_dt_s.to(dtype=encoded.dtype))

        if history_state is not None:
            if self.state_proj is None or self.state_mean is None or self.state_std is None:
                raise ValueError("history_state requires explicit training-split state_mean/state_std.")
            self._check_shape(history_state, (batch, horizon, self.state_proj.in_features), "history_state")
            if not torch.isfinite(history_state).all():
                raise ValueError("history_state must be finite.")
            normalized_state = (history_state.to(dtype=encoded.dtype) - self.state_mean) / self.state_std
            encoded = encoded + self.state_proj(normalized_state)
        elif self.state_proj is not None:
            raise ValueError("history_state is required when the state adapter is enabled.")

        return self.norm(encoded) * history_mask.to(dtype=encoded.dtype).unsqueeze(-1)

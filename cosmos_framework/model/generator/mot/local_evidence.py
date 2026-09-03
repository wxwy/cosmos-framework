# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stateless R08 per-step Local evidence encoding.

This module deliberately stops at ``E_history``. Readout, packed Local tokens,
Cosmos wiring, and persistent-memory mechanisms belong to later R08/R09 steps.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


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
        self.evidence_dim = evidence_dim
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

        encoded = self.visual_proj(history_visual_summary.to(dtype=self.visual_proj.weight.dtype))
        encoded = encoded + self.action_proj(local_history_action.to(dtype=self.action_proj.weight.dtype))
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


class StatelessLocalReplayReadout(nn.Module):
    """Read one temporary Local token from causal evidence without temporal state."""

    def __init__(self, evidence_dim: int = 256, local_dim: int = 32, hidden_dim: int | None = None) -> None:
        super().__init__()
        if evidence_dim <= 0 or local_dim <= 0:
            raise ValueError("evidence_dim and local_dim must be positive.")
        hidden_dim = hidden_dim or evidence_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.evidence_dim = evidence_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * evidence_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, local_dim),
        )

    @staticmethod
    def summarize(evidence_history: torch.Tensor, history_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked mean, latest valid evidence, and per-sample valid flags."""
        if evidence_history.ndim != 3:
            raise ValueError("evidence_history must have shape [B,H,D_e].")
        batch, horizon, _ = evidence_history.shape
        if tuple(history_mask.shape) != (batch, horizon):
            raise ValueError(f"history_mask must have shape {(batch, horizon)}, got {tuple(history_mask.shape)}.")
        if not torch.isfinite(evidence_history).all():
            raise ValueError("evidence_history must be finite.")
        mask = history_mask.bool()
        has_valid = mask.any(dim=1)
        mask_f = mask.to(dtype=evidence_history.dtype).unsqueeze(-1)
        mean = (evidence_history * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)
        positions = torch.arange(horizon, device=evidence_history.device).expand(batch, -1)
        latest_positions = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values.clamp_min(0)
        latest = evidence_history.gather(
            dim=1, index=latest_positions[:, None, None].expand(-1, 1, evidence_history.shape[-1])
        ).squeeze(1)
        valid_f = has_valid.to(dtype=evidence_history.dtype).unsqueeze(-1)
        return mean * valid_f, latest * valid_f, has_valid

    def forward(self, evidence_history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """Return one ``[B,1,D_local]`` token; all-mask samples are exact zero."""
        if evidence_history.shape[-1] != self.evidence_dim:
            raise ValueError(
                f"evidence_history last dimension must be {self.evidence_dim}, got {evidence_history.shape[-1]}."
            )
        mean, latest, has_valid = self.summarize(evidence_history, history_mask)
        token = self.mlp(torch.cat([mean, latest], dim=-1)).unsqueeze(1)
        return token * has_valid.to(dtype=token.dtype).view(-1, 1, 1)


class RecurrentLocalMemoryBackend(nn.Module):
    """R09-A CPU-contract recurrent backend; it does not wire Cosmos runtime yet."""

    def __init__(self, evidence_dim: int = 256, local_dim: int = 32) -> None:
        super().__init__()
        if evidence_dim <= 0 or local_dim <= 0:
            raise ValueError("evidence_dim and local_dim must be positive.")
        self.evidence_dim = evidence_dim
        self.local_dim = local_dim
        self.cell = nn.GRUCell(evidence_dim, local_dim)

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.local_dim, device=device, dtype=dtype)

    def step(self, evidence_t: torch.Tensor, state_in: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(evidence_t.shape) != (state_in.shape[0], self.evidence_dim):
            raise ValueError("evidence_t/state_in shapes are incompatible.")
        if tuple(valid.shape) != (state_in.shape[0],):
            raise ValueError("valid must have shape [B].")
        proposed = self.cell(evidence_t.to(dtype=self.cell.weight_ih.dtype), state_in)
        state_out = torch.where(valid[:, None], proposed, state_in)
        return state_out, state_out[:, None], valid.bool()

    @staticmethod
    def reset_mask(state: tuple[torch.Tensor, torch.Tensor], done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, initialized = state
        if tuple(done.shape) != (latent.shape[0],):
            raise ValueError("done must have shape [B].")
        return torch.where(done[:, None], torch.zeros_like(latent), latent), initialized & ~done.bool()

    def replay(self, evidence: torch.Tensor, mask: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch, horizon, width = evidence.shape
        if width != self.evidence_dim or tuple(mask.shape) != (batch, horizon):
            raise ValueError("evidence/mask shapes are incompatible.")
        if state is None:
            latent = self.initial_state(batch, device=evidence.device, dtype=self.cell.weight_ih.dtype)
            initialized = torch.zeros(batch, device=evidence.device, dtype=torch.bool)
        else:
            latent, initialized = state
        present = initialized | mask.bool().any(dim=1)
        for index in range(horizon):
            latent, _, _ = self.step(evidence[:, index], latent, mask[:, index])
        initialized = initialized | mask.bool().any(dim=1)
        return latent[:, None] * present[:, None, None], (latent, initialized), present


class TTTLocalMemoryBackend(nn.Module):
    """B0 CPU-only per-sample fast-weight backend; not wired into production."""

    def __init__(self, evidence_dim: int = 256, local_dim: int = 32, segment_steps: int = 4) -> None:
        super().__init__()
        if evidence_dim <= 0 or local_dim <= 0 or segment_steps <= 0:
            raise ValueError("TTT dimensions and segment_steps must be positive.")
        self.evidence_dim, self.local_dim, self.segment_steps = evidence_dim, local_dim, segment_steps

    def initial_state(self, batch: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch, self.local_dim, self.evidence_dim, device=device, dtype=torch.bfloat16), torch.zeros(batch, self.segment_steps, self.evidence_dim, device=device, dtype=torch.bfloat16), torch.zeros(batch, self.evidence_dim, device=device, dtype=torch.bfloat16), torch.zeros(batch, device=device, dtype=torch.bool), torch.zeros(batch, device=device, dtype=torch.int64))

    @staticmethod
    def reset_mask(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], done: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset only completed samples without sharing fast state across boundaries."""
        W, pending, last, initialized, progress = state
        if tuple(done.shape) != (W.shape[0],):
            raise ValueError("done must have shape [B].")
        done = done.bool()
        return (
            torch.where(done[:, None, None], torch.zeros_like(W), W).detach(),
            torch.where(done[:, None, None], torch.zeros_like(pending), pending).detach(),
            torch.where(done[:, None], torch.zeros_like(last), last).detach(),
            initialized & ~done,
            torch.where(done, torch.zeros_like(progress), progress),
        )

    def replay(self, evidence: torch.Tensor, mask: torch.Tensor, state=None):
        if not torch.is_grad_enabled():
            raise RuntimeError("TTTLocalMemoryBackend supports training grad-mode only.")
        batch, horizon, width = evidence.shape
        if width != self.evidence_dim or tuple(mask.shape) != (batch, horizon):
            raise ValueError("evidence/mask shapes are incompatible.")
        W, pending, last, initialized, progress = self.initial_state(batch, device=evidence.device) if state is None else state
        for index in range(horizon):
            valid = mask[:, index].bool()
            for row in valid.nonzero(as_tuple=False).flatten().tolist():
                value = evidence[row, index].detach().to(torch.bfloat16)
                pending[row, progress[row]] = value
                last[row], initialized[row], progress[row] = value, True, progress[row] + 1
                if progress[row] == self.segment_steps:
                    work = W[row].detach().float().requires_grad_(True)
                    values = pending[row].detach().float()
                    loss = ((work @ values.T).T - values[:, : self.local_dim]).square().mean()
                    grad = torch.autograd.grad(loss, work, create_graph=False)[0]
                    W[row] = (work - 0.1 * grad).to(torch.bfloat16).detach()
                    pending[row].zero_(); progress[row] = 0
        token = torch.einsum("bde,be->bd", W, last).detach()[:, None] * initialized[:, None, None]
        return token, (W.detach(), pending.detach(), last.detach(), initialized, progress), initialized


class ContinualTTTFastState(NamedTuple):
    """Per-sample fast MLP parameters carried across timesteps."""

    fast_in_weight: torch.Tensor
    fast_in_bias: torch.Tensor
    fast_out_weight: torch.Tensor
    fast_out_bias: torch.Tensor


class ContinualTTTLocalMemoryCore(nn.Module):
    """Functional CPU reference core for continual per-timestep KVB updates.

    Each step reads one ``[B,1,D_local]`` value. The time axis returned by
    :meth:`scan_segment` exists for outer-loss/TBPTT tests; it is not a claim
    that production attention receives ``T`` simultaneous Memory tokens.
    """

    def __init__(
        self,
        evidence_dim: int = 256,
        local_dim: int = 32,
        ttt_dim: int = 64,
        fast_hidden_dim: int = 128,
        inner_lr: float = 0.1,
        ttt_tbptt_steps: int = 16,
        k_local: int = 1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("evidence_dim", evidence_dim),
            ("local_dim", local_dim),
            ("ttt_dim", ttt_dim),
            ("fast_hidden_dim", fast_hidden_dim),
            ("ttt_tbptt_steps", ttt_tbptt_steps),
            ("k_local", k_local),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        try:
            inner_lr = float(inner_lr)
        except (TypeError, ValueError) as error:
            raise ValueError("inner_lr must be finite and positive.") from error
        if not math.isfinite(inner_lr) or inner_lr <= 0:
            raise ValueError("inner_lr must be finite and positive.")

        self.evidence_dim = evidence_dim
        self.local_dim = local_dim
        self.ttt_dim = ttt_dim
        self.fast_hidden_dim = fast_hidden_dim
        self.inner_lr = inner_lr
        self.ttt_tbptt_steps = ttt_tbptt_steps
        self.k_local = k_local
        self.key_proj = nn.Linear(evidence_dim, ttt_dim)
        self.query_proj = nn.Linear(evidence_dim, ttt_dim)
        self.value_proj = nn.Linear(evidence_dim, local_dim)
        self.slot_queries = nn.Parameter(torch.empty(k_local, ttt_dim))
        self.w0_fast_in_weight = nn.Parameter(torch.empty(fast_hidden_dim, ttt_dim))
        self.w0_fast_in_bias = nn.Parameter(torch.empty(fast_hidden_dim))
        self.w0_fast_out_weight = nn.Parameter(torch.empty(local_dim, fast_hidden_dim))
        self.w0_fast_out_bias = nn.Parameter(torch.empty(local_dim))
        self._reset_w0_parameters()
        if k_local == 1:
            nn.init.zeros_(self.slot_queries)
        else:
            nn.init.normal_(self.slot_queries, mean=0.0, std=1 / math.sqrt(ttt_dim))

    def _reset_w0_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w0_fast_in_weight, a=math.sqrt(5))
        nn.init.uniform_(self.w0_fast_in_bias, -1 / math.sqrt(self.ttt_dim), 1 / math.sqrt(self.ttt_dim))
        nn.init.kaiming_uniform_(self.w0_fast_out_weight, a=math.sqrt(5))
        nn.init.uniform_(
            self.w0_fast_out_bias,
            -1 / math.sqrt(self.fast_hidden_dim),
            1 / math.sqrt(self.fast_hidden_dim),
        )

    @property
    def _w0(self) -> ContinualTTTFastState:
        return ContinualTTTFastState(
            self.w0_fast_in_weight,
            self.w0_fast_in_bias,
            self.w0_fast_out_weight,
            self.w0_fast_out_bias,
        )

    def initial_state(
        self,
        batch: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContinualTTTFastState:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError("batch must be a positive integer.")
        device = self.w0_fast_in_weight.device if device is None else device
        dtype = self.w0_fast_in_weight.dtype if dtype is None else dtype
        return ContinualTTTFastState(
            *(
                value.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, *value.shape).clone()
                for value in self._w0
            )
        )

    def project_evidence(self, evidence_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if evidence_t.ndim != 2 or tuple(evidence_t.shape) != (evidence_t.shape[0], self.evidence_dim):
            raise ValueError(f"evidence_t must have shape [B,{self.evidence_dim}].")
        if not evidence_t.is_floating_point() or not torch.isfinite(evidence_t).all():
            raise ValueError("evidence_t must be floating point and finite.")
        if evidence_t.device != self.key_proj.weight.device:
            raise ValueError("evidence_t device must match projection parameters.")
        evidence_fp32 = evidence_t.float()
        return (
            F.linear(evidence_fp32, self.key_proj.weight.float(), self.key_proj.bias.float()),
            F.linear(evidence_fp32, self.query_proj.weight.float(), self.query_proj.bias.float()),
            F.linear(evidence_fp32, self.value_proj.weight.float(), self.value_proj.bias.float()),
        )

    def project_queries(self, query_base_t: torch.Tensor) -> torch.Tensor:
        if query_base_t.ndim != 2:
            raise ValueError("query_base_t must have shape [B,D_ttt].")
        batch = query_base_t.shape[0]
        if (
            tuple(query_base_t.shape) != (batch, self.ttt_dim)
            or query_base_t.dtype != torch.float32
            or not torch.isfinite(query_base_t).all()
        ):
            raise ValueError("query_base_t must have shape [B,D_ttt], dtype float32, and finite values.")
        if query_base_t.device != self.slot_queries.device:
            raise ValueError("query_base_t device must match slot_queries.")
        return query_base_t.unsqueeze(1) + self.slot_queries.float().unsqueeze(0)

    def _validate_state(self, state: ContinualTTTFastState, batch: int) -> None:
        if not isinstance(state, ContinualTTTFastState):
            raise ValueError("state must be ContinualTTTFastState.")
        expected = (
            (batch, self.fast_hidden_dim, self.ttt_dim),
            (batch, self.fast_hidden_dim),
            (batch, self.local_dim, self.fast_hidden_dim),
            (batch, self.local_dim),
        )
        first = state.fast_in_weight
        for name, value, shape in zip(ContinualTTTFastState._fields, state, expected, strict=True):
            if tuple(value.shape) != shape:
                raise ValueError(f"state.{name} must have shape {shape}.")
            if value.device != first.device or value.dtype != first.dtype:
                raise ValueError("state members must share device and dtype.")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"state.{name} must be floating point and finite.")

    @staticmethod
    def _fast_mlp(value: torch.Tensor, state: ContinualTTTFastState) -> torch.Tensor:
        hidden = F.silu(F.linear(value, state.fast_in_weight, state.fast_in_bias))
        return F.linear(hidden, state.fast_out_weight, state.fast_out_bias)

    def read_many(self, query_t: torch.Tensor, state: ContinualTTTFastState) -> torch.Tensor:
        if query_t.ndim != 3:
            raise ValueError("query_t must have shape [B,K_local,D_ttt].")
        batch = query_t.shape[0]
        expected = (batch, self.k_local, self.ttt_dim)
        if tuple(query_t.shape) != expected or query_t.dtype != torch.float32 or not torch.isfinite(query_t).all():
            raise ValueError(f"query_t must have shape {expected}, dtype float32, and finite values.")
        self._validate_state(state, batch)
        if query_t.device != state.fast_in_weight.device:
            raise ValueError("state and query_t must share a device.")
        return torch.stack(
            [
                self._fast_mlp(
                    query_t[row],
                    ContinualTTTFastState(*(member[row].float() for member in state)),
                )
                for row in range(batch)
            ]
        )

    def step_projected_many(
        self,
        *,
        key_t: torch.Tensor,
        query_base_t: torch.Tensor,
        value_t: torch.Tensor,
        state_in: ContinualTTTFastState,
        valid: torch.Tensor,
        create_graph: bool,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise RuntimeError("Continual TTT update requires ordinary grad mode.")
        if not isinstance(create_graph, bool):
            raise ValueError("create_graph must be bool.")
        if key_t.ndim != 2:
            raise ValueError("key_t must have shape [B,D_ttt].")
        batch = key_t.shape[0]
        for name, value, shape in (
            ("key_t", key_t, (batch, self.ttt_dim)),
            ("query_base_t", query_base_t, (batch, self.ttt_dim)),
            ("value_t", value_t, (batch, self.local_dim)),
        ):
            if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must have shape {shape}, dtype float32, and finite values.")
            if value.device != key_t.device:
                raise ValueError("K/Q/V must share a device.")
        if tuple(valid.shape) != (batch,) or valid.device != key_t.device:
            raise ValueError("valid must have shape [B] on the K/Q/V device.")
        self._validate_state(state_in, batch)
        if state_in.fast_in_weight.device != key_t.device:
            raise ValueError("state and K/Q/V must share a device.")
        queries = self.project_queries(query_base_t)

        valid = valid.bool()
        state_rows: list[list[torch.Tensor]] = [[], [], [], []]
        for row in range(batch):
            if not valid[row]:
                for output, member in zip(state_rows, state_in, strict=True):
                    output.append(member[row])
                continue

            work = []
            for member in state_in:
                member_row = member[row].float()
                if not member_row.requires_grad:
                    member_row = member_row.detach().requires_grad_(True)
                work.append(member_row)
            work_state = ContinualTTTFastState(*work)
            prediction = self._fast_mlp(key_t[row], work_state)
            inner_loss = (prediction - value_t[row]).square().mean()
            gradients = torch.autograd.grad(inner_loss, work_state, create_graph=create_graph)
            updated = ContinualTTTFastState(
                *(member - self.inner_lr * gradient for member, gradient in zip(work_state, gradients, strict=True))
            )
            for output, member, reference in zip(state_rows, updated, state_in, strict=True):
                output.append(member.to(dtype=reference.dtype))

        state_out = ContinualTTTFastState(*(torch.stack(rows) for rows in state_rows))
        tokens = self.read_many(queries, state_out)
        return tokens * valid[:, None, None], state_out, valid

    def step_projected(
        self,
        *,
        key_t: torch.Tensor,
        query_t: torch.Tensor,
        value_t: torch.Tensor,
        state_in: ContinualTTTFastState,
        valid: torch.Tensor,
        create_graph: bool,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if self.k_local != 1:
            raise ValueError("step_projected is only valid when k_local=1; use step_projected_many.")
        return self.step_projected_many(
            key_t=key_t,
            query_base_t=query_t,
            value_t=value_t,
            state_in=state_in,
            valid=valid,
            create_graph=create_graph,
        )

    def step_many(
        self,
        evidence_t: torch.Tensor,
        state_in: ContinualTTTFastState,
        valid: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise RuntimeError("Continual TTT update requires ordinary grad mode.")
        key_t, query_base_t, value_t = self.project_evidence(evidence_t)
        return self.step_projected_many(
            key_t=key_t,
            query_base_t=query_base_t,
            value_t=value_t,
            state_in=state_in,
            valid=valid,
            create_graph=create_graph,
        )

    def step(
        self,
        evidence_t: torch.Tensor,
        state_in: ContinualTTTFastState,
        valid: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if self.k_local != 1:
            raise ValueError("step is only valid when k_local=1; use step_many.")
        return self.step_many(evidence_t, state_in, valid, create_graph=create_graph)

    def scan_segment_many(
        self,
        evidence: torch.Tensor,
        valid: torch.Tensor,
        state_in: ContinualTTTFastState | None = None,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise RuntimeError("Continual TTT update requires ordinary grad mode.")
        if evidence.ndim != 3:
            raise ValueError("evidence must have shape [B,T,D_e].")
        batch, steps, width = evidence.shape
        if width != self.evidence_dim or tuple(valid.shape) != (batch, steps):
            raise ValueError("evidence/valid shapes are incompatible.")
        if steps <= 0 or steps > self.ttt_tbptt_steps:
            raise ValueError("segment length must be in [1, ttt_tbptt_steps].")
        state = self.initial_state(batch, device=evidence.device) if state_in is None else state_in
        tokens, present = [], []
        for index in range(steps):
            token, state, step_present = self.step_many(
                evidence[:, index], state, valid[:, index], create_graph=create_graph
            )
            tokens.append(token)
            present.append(step_present)
        return torch.stack(tokens, dim=1), state, torch.stack(present, dim=1)

    def scan_segment(
        self,
        evidence: torch.Tensor,
        valid: torch.Tensor,
        state_in: ContinualTTTFastState | None = None,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        """Scan a training segment and retain every timestep readout for its outer loss."""
        if self.k_local != 1:
            raise ValueError("scan_segment is only valid when k_local=1; use scan_segment_many.")
        tokens, state, present = self.scan_segment_many(
            evidence, valid, state_in, create_graph=create_graph
        )
        return tokens[:, :, 0], state, present

    def reset_mask(self, state: ContinualTTTFastState, done: torch.Tensor) -> ContinualTTTFastState:
        if not isinstance(state, ContinualTTTFastState):
            raise ValueError("state must be ContinualTTTFastState.")
        batch = state.fast_in_weight.shape[0]
        self._validate_state(state, batch)
        if tuple(done.shape) != (batch,) or done.device != state.fast_in_weight.device:
            raise ValueError("done must have shape [B] on the state device.")
        done = done.bool()
        reset = self.initial_state(batch, device=state.fast_in_weight.device, dtype=state.fast_in_weight.dtype)
        return ContinualTTTFastState(
            *(
                torch.where(done.view(batch, *([1] * (value.ndim - 1))), replacement, value)
                for value, replacement in zip(state, reset, strict=True)
            )
        )

    def detach_state(self, state: ContinualTTTFastState) -> ContinualTTTFastState:
        batch = state.fast_in_weight.shape[0] if isinstance(state, ContinualTTTFastState) else 0
        self._validate_state(state, batch)
        return ContinualTTTFastState(*(value.detach() for value in state))


class LocalHistoryRuntime(nn.Module):
    """Encode a batched causal history and gate absent samples out of Local."""

    def __init__(self, encoder: LocalEvidenceEncoder, readout: StatelessLocalReplayReadout, recurrent_backend: RecurrentLocalMemoryBackend | None = None) -> None:
        super().__init__()
        if encoder.evidence_dim != readout.evidence_dim:
            raise ValueError("Local history encoder/readout evidence dimensions must match.")
        self.encoder = encoder
        self.readout = readout
        self.recurrent_backend = recurrent_backend

    def reset_parameters(self) -> None:
        """Initialize R08 parameters after meta-device materialization."""
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        *,
        history_visual_summary: torch.Tensor,
        local_history_action: torch.Tensor,
        history_age_steps: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_mask: torch.Tensor,
        history_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(tokens [B,1,D], present [B], evidence [B,H,D])``.

        A sample with no valid history is marked absent; callers must not pack its
        zero readout token because downstream R07 projections may have bias.
        """
        if not torch.isfinite(history_dt_s).all():
            raise ValueError("history_dt_s must be finite before Local runtime wiring.")
        evidence = self.encoder(
            history_visual_summary=history_visual_summary,
            local_history_action=local_history_action,
            history_age_steps=history_age_steps,
            history_dt_s=history_dt_s,
            history_mask=history_mask,
            history_state=history_state,
        )
        if self.recurrent_backend is None:
            tokens = self.readout(evidence, history_mask)
            present = history_mask.bool().any(dim=1)
        else:
            tokens, _, present = self.recurrent_backend.replay(evidence, history_mask)
        return tokens, present, evidence

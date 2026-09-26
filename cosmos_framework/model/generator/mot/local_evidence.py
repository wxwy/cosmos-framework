# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B0 的因果 evidence 编码与函数式 fast-state core；不接模型或训练循环。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceFeatureConfig:
    state: bool = False
    dt: bool = False
    age: bool = False


CANONICAL_EVIDENCE_FEATURE_CONFIG = EvidenceFeatureConfig()


def _positive_dim(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


class LocalEvidenceEncoder(nn.Module):
    def __init__(
        self,
        evidence_dim: int = 256,
        visual_dim: int = 96,
        action_dim: int = 15,
        feature_config: EvidenceFeatureConfig = CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ) -> None:
        super().__init__()
        for name, value in (("evidence_dim", evidence_dim), ("visual_dim", visual_dim), ("action_dim", action_dim)):
            _positive_dim(name, value)
        if feature_config != CANONICAL_EVIDENCE_FEATURE_CONFIG:
            raise ValueError("B0 禁止 state/dt/age 特征")
        self.feature_config = feature_config
        self.evidence_dim = evidence_dim
        self.visual_proj = nn.Linear(visual_dim, evidence_dim)
        self.action_proj = nn.Linear(action_dim, evidence_dim)
        self.norm = nn.LayerNorm(evidence_dim)

    def encode_segment(self, visual_summary: torch.Tensor, executed_action: torch.Tensor) -> torch.Tensor:
        if visual_summary.ndim != 3 or visual_summary.shape[-1] != self.visual_proj.in_features:
            raise ValueError("visual_summary 维度必须匹配 [B,T,visual_dim]")
        if executed_action.shape != (*visual_summary.shape[:2], self.action_proj.in_features):
            raise ValueError("executed_action 维度必须匹配 [B,T,action_dim]")
        for value in (visual_summary, executed_action):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("有效 evidence 必须为有限浮点数")
            if value.device != self.visual_proj.weight.device:
                raise ValueError("evidence 与 encoder 必须同设备")
        encoded = self.visual_proj(visual_summary.to(self.visual_proj.weight.dtype))
        encoded = encoded + self.action_proj(executed_action.to(self.action_proj.weight.dtype))
        return self.norm(encoded)


class ContinualTTTFastState(NamedTuple):
    """每个 row 的四个普通 fp32 tensor，不注册到 nn.Module。"""

    fast_in_weight: torch.Tensor
    fast_in_bias: torch.Tensor
    fast_out_weight: torch.Tensor
    fast_out_bias: torch.Tensor


class ContinualTTTLocalMemoryCore(nn.Module):
    def __init__(
        self,
        evidence_dim: int = 256,
        local_dim: int = 32,
        ttt_dim: int = 64,
        fast_hidden_dim: int = 256,
        inner_lr: float = 0.1,
        ttt_tbptt_steps: int = 16,
        k_local: int = 4,
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
            _positive_dim(name, value)
        if isinstance(inner_lr, bool) or not math.isfinite(inner_lr) or inner_lr <= 0:
            raise ValueError("inner_lr 必须是有限正数")
        self.evidence_dim, self.local_dim, self.ttt_dim = evidence_dim, local_dim, ttt_dim
        self.fast_hidden_dim, self.inner_lr = fast_hidden_dim, inner_lr
        self.ttt_tbptt_steps, self.k_local = ttt_tbptt_steps, k_local
        self.key_proj = nn.Linear(evidence_dim, ttt_dim)
        self.query_proj = nn.Linear(evidence_dim, ttt_dim)
        self.value_proj = nn.Linear(evidence_dim, local_dim)
        self.slot_queries = nn.Parameter(torch.empty(k_local, ttt_dim))
        self.w0_fast_in_weight = nn.Parameter(torch.empty(fast_hidden_dim, ttt_dim))
        self.w0_fast_in_bias = nn.Parameter(torch.empty(fast_hidden_dim))
        self.w0_fast_out_weight = nn.Parameter(torch.empty(local_dim, fast_hidden_dim))
        self.w0_fast_out_bias = nn.Parameter(torch.empty(local_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.key_proj, self.query_proj, self.value_proj):
            projection.reset_parameters()
        nn.init.kaiming_uniform_(self.w0_fast_in_weight, a=math.sqrt(5))
        nn.init.uniform_(self.w0_fast_in_bias, -1 / math.sqrt(self.ttt_dim), 1 / math.sqrt(self.ttt_dim))
        nn.init.kaiming_uniform_(self.w0_fast_out_weight, a=math.sqrt(5))
        nn.init.uniform_(
            self.w0_fast_out_bias, -1 / math.sqrt(self.fast_hidden_dim), 1 / math.sqrt(self.fast_hidden_dim)
        )
        nn.init.normal_(self.slot_queries, std=1 / math.sqrt(self.ttt_dim))

    def initial_state(self, batch: int) -> ContinualTTTFastState:
        _positive_dim("batch", batch)
        return ContinualTTTFastState(
            *(
                parameter.float().unsqueeze(0).expand(batch, *parameter.shape).clone()
                for parameter in (
                    self.w0_fast_in_weight,
                    self.w0_fast_in_bias,
                    self.w0_fast_out_weight,
                    self.w0_fast_out_bias,
                )
            )
        )

    def validate_state(self, state: ContinualTTTFastState, batch: int) -> None:
        if not isinstance(state, ContinualTTTFastState):
            raise ValueError("state 类型必须是 ContinualTTTFastState")
        shapes = (
            (batch, self.fast_hidden_dim, self.ttt_dim),
            (batch, self.fast_hidden_dim),
            (batch, self.local_dim, self.fast_hidden_dim),
            (batch, self.local_dim),
        )
        for value, shape in zip(state, shapes, strict=True):
            if value.shape != shape or value.dtype != torch.float32 or isinstance(value, nn.Parameter):
                raise ValueError("fast state 必须是指定形状的普通 fp32 tensor")
            if value.device != self.slot_queries.device or not torch.isfinite(value).all():
                raise ValueError("fast state 必须有限且与 core 同设备")

    @staticmethod
    def detach_state(state: ContinualTTTFastState) -> ContinualTTTFastState:
        return ContinualTTTFastState(*(value.detach().clone() for value in state))

    @staticmethod
    def _fast_mlp(value: torch.Tensor, state: ContinualTTTFastState) -> torch.Tensor:
        hidden = F.silu(torch.bmm(value, state.fast_in_weight.transpose(1, 2)) + state.fast_in_bias[:, None])
        return torch.bmm(hidden, state.fast_out_weight.transpose(1, 2)) + state.fast_out_bias[:, None]

    def step_many(
        self,
        evidence: torch.Tensor,
        state: ContinualTTTFastState,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState]:
        """只计算 candidate；inner 标量只服务 fast 更新，不返回给 outer objective。"""
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("TTT inner update 需要普通 grad mode")
        if evidence.ndim != 2 or evidence.shape[-1] != self.evidence_dim or evidence.shape[0] <= 0:
            raise ValueError("evidence 必须为 [B,evidence_dim]")
        if not evidence.is_floating_point() or not torch.isfinite(evidence).all():
            raise ValueError("evidence 必须为有限浮点数")
        self.validate_state(state, evidence.shape[0])
        if evidence.device != self.slot_queries.device:
            raise ValueError("evidence 与 core 必须同设备")
        # 显式关闭 autocast，保证 fast 运算及 candidate 始终 fp32。
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            key, query, target = (
                F.linear(evidence.float(), proj.weight.float(), proj.bias.float())
                for proj in (self.key_proj, self.query_proj, self.value_proj)
            )
            work = ContinualTTTFastState(
                *(value if value.requires_grad else value.detach().requires_grad_(True) for value in state)
            )
            prediction = self._fast_mlp(key[:, None], work).squeeze(1)
            # 每 row 独立 MSE 均值后求和，绝不除 B，保持原 inner_lr。
            inner_loss = (prediction - target).square().mean(dim=-1).sum()
            gradients = torch.autograd.grad(inner_loss, work, create_graph=create_graph)
            candidate = ContinualTTTFastState(
                *(value - self.inner_lr * grad for value, grad in zip(work, gradients, strict=True))
            )
            tokens = self._fast_mlp(query[:, None] + self.slot_queries.float()[None], candidate)
        self.validate_state(candidate, evidence.shape[0])
        return tokens, candidate

    def scan_segment_masked_encoded_many(
        self,
        encoder: LocalEvidenceEncoder,
        visual_summary: torch.Tensor,
        executed_action: torch.Tensor,
        valid: torch.Tensor,
        state_in: ContinualTTTFastState | None = None,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        """只编码有效 evidence；S0/PAD 不读占位内容，不更新 fast state。"""
        if visual_summary.ndim != 3 or executed_action.ndim != 3:
            raise ValueError("scan 输入必须是 [B,T,D]")
        batch, steps, visual_dim = visual_summary.shape
        if batch <= 0 or not 1 <= steps <= self.ttt_tbptt_steps:
            raise ValueError("scan 的 B 必须为正，T 必须在 TBPTT 范围内")
        if (
            visual_dim != encoder.visual_proj.in_features
            or executed_action.shape != (batch, steps, encoder.action_proj.in_features)
            or encoder.evidence_dim != self.evidence_dim
        ):
            raise ValueError("scan 维度与 encoder/core 不匹配")
        if valid.shape != (batch, steps) or valid.dtype != torch.bool:
            raise ValueError("valid 必须为 [B,T] bool")
        if any(value.device != self.slot_queries.device for value in (visual_summary, executed_action, valid)):
            raise ValueError("scan 输入与 core 必须同设备")
        state = self.initial_state(batch) if state_in is None else state_in
        self.validate_state(state, batch)
        tokens = []
        for index in range(steps):
            rows = valid[:, index].nonzero(as_tuple=False).flatten()
            token = state.fast_out_bias.new_zeros(batch, self.k_local, self.local_dim)
            if rows.numel():
                evidence = encoder.encode_segment(
                    visual_summary[:, index].index_select(0, rows)[:, None],
                    executed_action[:, index].index_select(0, rows)[:, None],
                ).squeeze(1)
                compact = ContinualTTTFastState(*(value.index_select(0, rows) for value in state))
                result, updated = self.step_many(evidence, compact, create_graph=create_graph)
                state = ContinualTTTFastState(
                    *(value.index_copy(0, rows, change) for value, change in zip(state, updated, strict=True))
                )
                token = token.index_copy(0, rows, result)
            tokens.append(token)
        return torch.stack(tokens, dim=1), state, valid.clone()

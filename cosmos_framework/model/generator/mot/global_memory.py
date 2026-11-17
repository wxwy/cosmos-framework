# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Task queries read UND once and current spatial patches twice (G2 route A).

No UND/VAE/model construction or persistent inference state. B owns model/FSDP
registration; D owns caching of prepare_task_queries under fixed weights.
"""

from __future__ import annotations

import torch
from torch import nn

from cosmos_framework.model.generator.mot.g2_contracts import (
    GlobalMemoryConfig,
    ReaderOutput,
    SpatialView,
    UNDTextFeatures,
)
from cosmos_framework.model.generator.mot.global_spatial import SpatialFeatureEncoder

READER_VERSION = "g2_reader_prenorm_silu_xavier_queries002_v1"
PROJECTOR_VERSION = "g2_global_linear_modality002_v1"


def _dispatch_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # The repository dispatcher owns backend selection. Tests replace only this
    # boundary with a CPU reference; there is no second production backend.
    from cosmos_framework.model.attention import attention

    return attention(query, key, value)


def _reset_affine_modules(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()


class _ReaderAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # K bias would cancel in softmax. Use bias-free attention projections.
        self.query = nn.Linear(512, 512, bias=False)
        self.key = nn.Linear(512, 512, bias=False)
        self.value = nn.Linear(512, 512, bias=False)
        self.output = nn.Linear(512, 512, bias=False)

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        q = self.query(query).unflatten(-1, (8, 64))
        k = self.key(source).unflatten(-1, (8, 64))
        v = self.value(source).unflatten(-1, (8, 64))
        return self.output(_dispatch_attention(q, k, v).flatten(-2))


class _FeedForward(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Linear(512, 2048), nn.SiLU(), nn.Linear(2048, 512))


class _LanguageBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(512, eps=1e-5)
        self.source_norm = nn.LayerNorm(512, eps=1e-5)
        self.cross_attention = _ReaderAttention()
        self.ffn_norm = nn.LayerNorm(512, eps=1e-5)
        self.ffn = _FeedForward()

    def forward(self, queries: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if text.shape[1]:
            queries = queries + self.cross_attention(self.query_norm(queries), self.source_norm(text))
        return queries + self.ffn(self.ffn_norm(queries))


class _VisualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(512, eps=1e-5)
        self.source_norm = nn.LayerNorm(512, eps=1e-5)
        self.cross_attention = _ReaderAttention()
        self.memory_norm = nn.LayerNorm(512, eps=1e-5)
        self.self_attention = _ReaderAttention()
        self.ffn_norm = nn.LayerNorm(512, eps=1e-5)
        self.ffn = _FeedForward()

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        queries = queries + self.cross_attention(self.query_norm(queries), self.source_norm(patches))
        memory = self.memory_norm(queries)
        queries = queries + self.self_attention(memory, memory)
        return queries + self.ffn(self.ffn_norm(queries))


class GlobalMemoryReader(nn.Module):
    """C0 Reader API. No language dropout, previous Global or Local input.

    Ragged views/patches are compacted per sample. Attention is dispatched on
    each sample independently; batching/packing optimization is not claimed.
    """

    reader_version = READER_VERSION

    def __init__(self, text_dim: int, config: GlobalMemoryConfig, *, latent_channels: int = 48) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("disabled Global must not construct a Reader")
        if type(text_dim) is not int or text_dim <= 0:
            raise ValueError("text_dim must be the actual positive UND hidden width")
        self.config = config
        self.queries = nn.Parameter(torch.empty(64, 512))
        self.spatial = SpatialFeatureEncoder(latent_channels)
        self.text_adapter = nn.Sequential(nn.LayerNorm(text_dim, eps=1e-5), nn.Linear(text_dim, 512, bias=False))
        self.language = _LanguageBlock()
        self.visual = nn.ModuleList([_VisualBlock(), _VisualBlock()])
        self.final_norm = nn.LayerNorm(512, eps=1e-5)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.spatial.reset_parameters()
        for root in (self.text_adapter, self.language, self.visual, self.final_norm):
            _reset_affine_modules(root)
        nn.init.normal_(self.queries, std=0.02)

    def prepare_task_queries(self, text: UNDTextFeatures) -> torch.Tensor:
        """Trainable task preparation; D may cache only under fixed weights.

        The supplied effective text is already the shared dropout result. Empty
        conditions with native special tokens read those tokens; length zero
        skips the CA contribution and retains the language block's FFN.
        """
        hidden, mask = text.hidden, text.valid_mask
        if hidden.device != self.queries.device or not hidden.is_floating_point():
            raise ValueError("UND hidden must be floating and on the Reader device")
        if hidden.shape[-1] != self.text_adapter[1].in_features:
            raise ValueError("UND hidden width disagrees with text_dim")
        lengths = torch.tensor(text.text.lengths, device=mask.device)
        expected = torch.arange(hidden.shape[1], device=mask.device)[None] < lengths[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("UND mask must match C0 effective tokens and right padding")
        queries = []
        for row, valid in zip(hidden, mask, strict=True):
            tokens = row[valid]
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("valid UND hidden must be finite")
            adapted = self.text_adapter(tokens.to(self.queries.dtype))
            queries.append(self.language(self.queries[None], adapted[None]))
        return torch.cat(queries, dim=0)

    def read_visual(self, task_queries: torch.Tensor, views: tuple[tuple[SpatialView, ...], ...]) -> ReaderOutput:
        """Read current scene afresh; does not update or persist task_queries."""
        if task_queries.ndim != 3 or task_queries.shape[1:] != (64, 512) or task_queries.shape[0] != len(views):
            raise ValueError("task queries and views must have matching [B,64,512] sample order")
        if not views or task_queries.device != self.queries.device or not task_queries.is_floating_point():
            raise ValueError("task queries must be a nonempty floating batch on the Reader device")
        if not bool(torch.isfinite(task_queries).all()):
            raise ValueError("task queries must be finite")
        result = []
        for query, sample_views in zip(task_queries, views, strict=True):
            if not sample_views:
                raise ValueError("selected views must include the configured mandatory top")
            ids = [view.identity.view_id for view in sample_views]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate selected logical view identity")
            if len({view.projection.world_frame for view in sample_views}) != 1:
                raise ValueError("selected views must use a common world frame")
            tops = [view for view in sample_views if view.direction == "floor_top"]
            if not tops or any(view.top_variant != self.config.top_view_variant for view in tops):
                raise ValueError("exactly one configured top variant must be used, without fallback")
            if any(view.top_variant is not None for view in sample_views if view.direction != "floor_top"):
                raise ValueError("only floor_top has a top variant")
            patches = torch.cat([self.spatial(view) for view in sample_views], dim=0)[None]
            value = query.to(self.queries.dtype)[None]
            for block in self.visual:
                value = block(value, patches)
            result.append(self.final_norm(value))
        return ReaderOutput(tokens=torch.cat(result, dim=0))

    def forward(self, text: UNDTextFeatures, views: tuple[tuple[SpatialView, ...], ...]) -> ReaderOutput:
        return self.read_visual(self.prepare_task_queries(text), views)


class GlobalMemoryProjector(nn.Module):
    """A's sole Global projector and modality; B owns registration/calls."""

    projector_version = PROJECTOR_VERSION

    def __init__(self, model_dim: int, config: GlobalMemoryConfig) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("disabled Global must not construct a projector")
        if type(model_dim) is not int or model_dim <= 0:
            raise ValueError("model_dim must be the actual positive backbone width")
        self.config = config
        # A separate affine bias would duplicate the modality vector exactly.
        self.projection = nn.Linear(512, model_dim, bias=False)
        self.modality = nn.Parameter(torch.empty(model_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_affine_modules(self.projection)
        nn.init.normal_(self.modality, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (64, 512):
            raise ValueError("Global projector input must be [B,64,512]")
        if tokens.device != self.modality.device or not tokens.is_floating_point():
            raise ValueError("Global tokens and projector must share device and be floating")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Global tokens must be finite")
        return self.projection(tokens.to(self.projection.weight.dtype)) + self.modality

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""G2 C0 public tensor boundary, shared by model, training and inference.

No service, dataset, renderer or model construction imports. Geometry is native
producer evidence, not an encoded 14-vector or a reconstructed scene. All tensor
dimensions except the Owner's Reader starting architecture come from actual data.
Frozen dataclasses do not make tensor storage immutable: callers must own a
request snapshot; training tensors retain their autograd graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import torch
from torch.nn import functional as F

G2_ABI_VERSION = "psm_g2_c0_v1"
G2_MODEL_ABI_VERSION = "psm_g2_b1_v1"


class TrainingStage(StrEnum):
    TRAIN_A = "TRAIN_A"
    TRAIN_B = "TRAIN_B"
    R1 = "R1"


class ParameterGroup(StrEnum):
    GLOBAL = "global"
    GEN = "gen"
    ACTION = "action_adapter"


@dataclass(frozen=True, kw_only=True)
class GlobalMemoryConfig:
    """Owner-confirmed experimental starting architecture; no measured optimum."""

    enabled: bool = False
    k_global: int = 64
    hidden_dim: int = 512
    num_attention_heads: int = 8
    ffn_hidden_dim: int = 2048
    num_language_blocks: int = 1
    num_visual_blocks: int = 2
    top_view_variant: str = "with_robot"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be bool")
        if (
            self.k_global,
            self.hidden_dim,
            self.num_attention_heads,
            self.ffn_hidden_dim,
            self.num_language_blocks,
            self.num_visual_blocks,
        ) != (64, 512, 8, 2048, 1, 2):
            raise ValueError("C0 supports the confirmed 64/512/8/2048/1/2 Reader only")
        if self.top_view_variant not in ("with_robot", "without_robot"):
            raise ValueError("select exactly one explicit top variant")


def _identity(*values: str):
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("source/version identities must be nonempty strings")


@dataclass(frozen=True, kw_only=True)
class SpatialProjection:
    """Image-owner projection, metres; axes are producer's actual world basis.

    uv_min_room and reference origin are preserved independently. The image plane
    has undetermined depth; none of these fields assert per-patch object XYZ.
    """

    projection_id: str
    world_frame: str
    origin_world: tuple[float, float, float]
    axes_world: tuple[tuple[float, float, float], ...]
    right_world: tuple[float, float, float]
    up_world: tuple[float, float, float]
    forward_world: tuple[float, float, float]
    uv_min_room: tuple[float, float]
    width_px: int
    height_px: int
    meters_per_pixel: float
    physical_width_m: float
    physical_height_m: float
    rows_flipped: bool
    pixel_center_offset: float

    def __post_init__(self):
        _identity(self.projection_id, self.world_frame)
        vectors = (self.origin_world, self.right_world, self.up_world, self.forward_world, *self.axes_world)
        if len(self.axes_world) != 3 or any(len(v) != 3 for v in vectors) or len(self.uv_min_room) != 2:
            raise ValueError("native projection requires explicit 3D frame/basis and 2D uv_min")
        values = (
            *sum((tuple(v) for v in vectors), ()),
            *self.uv_min_room,
            self.meters_per_pixel,
            self.physical_width_m,
            self.physical_height_m,
            self.pixel_center_offset,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("projection must be finite")
        if (
            min(self.width_px, self.height_px, self.meters_per_pixel, self.physical_width_m, self.physical_height_m)
            <= 0
        ):
            raise ValueError("projection dimensions must be positive")
        if type(self.rows_flipped) is not bool:
            raise ValueError("row convention must be explicit")


@dataclass(frozen=True, kw_only=True)
class RasterMapping:
    """preprocess_rgb fields. content_box is half-open (left,top,right,bottom)."""

    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    content_box: tuple[int, int, int, int]

    def __post_init__(self):
        if any(
            len(hw) != 2 or any(type(v) is not int or v <= 0 for v in hw)
            for hw in (self.original_hw, self.resized_hw, self.canvas_hw)
        ):
            raise ValueError("raster hw must be positive integer pairs")
        left, top, right, bottom = self.content_box
        if not (0 <= left < right <= self.canvas_hw[1] and 0 <= top < bottom <= self.canvas_hw[0]):
            raise ValueError("content_box is outside canvas")
        if (bottom - top, right - left) != self.resized_hw:
            raise ValueError("content_box and actual resized raster disagree")


@dataclass(frozen=True, kw_only=True)
class ViewIdentity:
    view_id: str
    content_version: str
    image_owner: str
    image_owner_frame: int
    geometry_version: str
    membership_version: str
    recipe_id: str

    def __post_init__(self):
        _identity(
            self.view_id,
            self.content_version,
            self.image_owner,
            self.geometry_version,
            self.membership_version,
            self.recipe_id,
        )
        if type(self.image_owner_frame) is not int or self.image_owner_frame < 0:
            raise ValueError("image owner frame must be explicit and nonnegative")


@dataclass(frozen=True, kw_only=True)
class SpatialView:
    identity: ViewIdentity
    latent: torch.Tensor  # [C,H_z,W_z], complete single-frame grid, no temporal axis
    valid_mask: torch.Tensor  # bool [H_z,W_z]; derived from mapping, never RGB colour
    projection: SpatialProjection
    raster: RasterMapping
    scope: str  # floor/room/island, categorical rather than list position
    direction: str  # actual producer direction (floor_top, plus_X, ...)
    room_id: str
    island_id: str | None
    top_variant: str | None  # with_robot/without_robot for top, None for other views

    def __post_init__(self):
        if self.latent.ndim != 3 or min(self.latent.shape) < 1 or not self.latent.is_floating_point():
            raise ValueError("view latent must be a nonempty floating [C,H_z,W_z] grid")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != self.latent.shape[1:]:
            raise ValueError("valid_mask must be bool [H_z,W_z]")
        if self.valid_mask.device != self.latent.device:
            raise ValueError("view mask and latent must share device")
        if self.scope not in ("floor", "room", "island"):
            raise ValueError("unknown spatial scope")
        _identity(self.direction, self.room_id)
        if self.scope == "island" and not self.island_id:
            raise ValueError("Island identity is required")
        if self.top_variant not in (None, "with_robot", "without_robot"):
            raise ValueError("unknown top variant")
        if (self.projection.height_px, self.projection.width_px) != self.raster.original_hw:
            raise ValueError("projection must describe the actual source image")


@dataclass(frozen=True, kw_only=True)
class G2SampleIdentity:
    episode_id: str
    scene_id: str
    subtask_id: str
    decision_frame: int
    logical_state_frame: int
    data_version: str
    alignment_version: str
    state15_recipe_id: str
    task_binding_id: str

    def __post_init__(self):
        _identity(
            self.episode_id,
            self.scene_id,
            self.subtask_id,
            self.data_version,
            self.alignment_version,
            self.state15_recipe_id,
            self.task_binding_id,
        )
        if not 0 <= self.logical_state_frame <= self.decision_frame:
            raise ValueError("logical spatial state cannot be from the future")


@dataclass(frozen=True, kw_only=True)
class EffectiveTextBatch:
    """Post-formatter/dropout tokens, unpadded per sample, including native specials.

    Dropout is sampled by C once, upstream of this boundary. Empty means actual
    empty condition, not multiplied hidden. Positions are native mRoPE [3,L].
    Hashable identity includes tokens, positions, formatter AND dropout decision.
    """

    token_ids: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[tuple[int | float, ...], ...], ...]
    dropped: tuple[bool, ...]
    formatter_id: str
    tokenizer_id: str
    dropout_id: str

    def __post_init__(self):
        _identity(self.formatter_id, self.tokenizer_id, self.dropout_id)
        if not self.token_ids or not len(self.token_ids) == len(self.position_ids) == len(self.dropped):
            raise ValueError("one effective text/dropout/position entry per sample is required")
        for tokens, positions, dropped in zip(self.token_ids, self.position_ids, self.dropped, strict=True):
            if any(type(t) is not int or t < 0 for t in tokens) or type(dropped) is not bool:
                raise ValueError("invalid tokens or dropout decision")
            if len(positions) != 3 or any(len(axis) != len(tokens) for axis in positions):
                raise ValueError("native text mRoPE positions must be [3,L]")

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(map(len, self.token_ids))


@dataclass(frozen=True, kw_only=True)
class UNDLayerKV:
    """Frozen UND projections BEFORE GEN-specific norm and RoPE.

    key is after UND k_norm (Identity for Edge), value is raw UND v_proj.
    GEN must apply current k_norm_und_for_gen then captured RoPE: the resulting
    GEN keys may retain trainable norm gradients and must not be cached across
    optimizer steps. Packed rows exclude padding, sample order is text.lengths.
    """

    key: torch.Tensor  # [sum(lengths),H_kv,D_head]
    value: torch.Tensor
    cos: torch.Tensor  # [sum(lengths),D_head]
    sin: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class UNDTextFeatures:
    text: EffectiveTextBatch
    hidden: torch.Tensor  # post-norm [B,L_max,D_text], right padded
    valid_mask: torch.Tensor  # bool [B,L_max]
    layers: tuple[UNDLayerKV, ...]

    def __post_init__(self):
        if self.hidden.ndim != 3 or self.hidden.shape[:2] != self.valid_mask.shape:
            raise ValueError("UND hidden/mask must be [B,L_max,D] / [B,L_max]")
        if self.valid_mask.dtype != torch.bool or self.hidden.shape[0] != len(self.text.lengths):
            raise ValueError("UND sample/mask identity mismatch")
        if self.hidden.shape[1] != max(self.text.lengths):
            raise ValueError("UND padding length disagrees with effective tokens")
        if self.hidden.device != self.valid_mask.device:
            raise ValueError("UND hidden and mask must share device")
        for layer in self.layers:
            if (
                layer.key.ndim != 3
                or layer.key.shape != layer.value.shape
                or layer.key.shape[0] != sum(self.text.lengths)
            ):
                raise ValueError("UND layer K/V must contain only effective packed text rows")
            if layer.cos.shape != layer.sin.shape or layer.cos.shape != (layer.key.shape[0], layer.key.shape[2]):
                raise ValueError("UND layer RoPE shape mismatch")


@dataclass(frozen=True, kw_only=True)
class GENLayerKV:
    """Current-weight inference K/V, already normalized/rotated for GEN.

    Text rows follow text.lengths; Global rows follow present samples (64 each).
    These are inference snapshots, never substitutes for a trainable projection.
    """

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        if self.key.ndim != 3 or self.key.shape != self.value.shape:
            raise ValueError("GEN K/V must share [N,H_kv,D_head] shape")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("GEN K/V device/dtype mismatch")


@dataclass(frozen=True, kw_only=True)
class G2ConditionBatch:
    """One condition authority for native training and generation.

    C supplies the actual post-dropout native text. D may additionally supply
    prepared snapshots under its checkpoint/session signature. A cached projected
    row establishes Global presence without reloading views. With no projected
    row, None views means absent; it never reserves zero-valued prefix tokens.
    """

    text: EffectiveTextBatch
    views: tuple[tuple[SpatialView, ...] | None, ...]
    text_features: UNDTextFeatures | None = None
    global_tokens: tuple[torch.Tensor | None, ...] | None = None
    gen_text_kv: tuple[GENLayerKV, ...] | None = None
    global_kv: tuple[GENLayerKV, ...] | None = None

    def __post_init__(self):
        if len(self.views) != len(self.text.lengths):
            raise ValueError("G2 views require one entry per text sample")
        if any(v is not None and not v for v in self.views):
            raise ValueError("absent Global uses None, not an empty view collection")
        if self.text_features is not None and self.text_features.text != self.text:
            raise ValueError("prepared UND text identity mismatch")
        if self.global_tokens is not None:
            if len(self.global_tokens) != len(self.views):
                raise ValueError("projected Global sample count mismatch")
            for views, tokens in zip(self.views, self.global_tokens, strict=True):
                if views is not None and tokens is None:
                    raise ValueError("prepared Global tokens omitted a present view sample")
                if tokens is not None and (tokens.ndim != 2 or tokens.shape[0] != 64):
                    raise ValueError("projected Global must be [64,D_model]")
        for layers, rows in (
            (self.gen_text_kv, sum(self.text.lengths)),
            (self.global_kv, 64 * sum(self.global_present)),
        ):
            if layers is not None and any(layer.key.shape[0] != rows for layer in layers):
                raise ValueError("prepared GEN K/V row count mismatch")
        if self.global_kv is not None and self.global_tokens is None:
            raise ValueError("prepared Global K/V requires its projected tokens")

    @property
    def global_present(self) -> tuple[bool, ...]:
        source = self.views if self.global_tokens is None else self.global_tokens
        return tuple(value is not None for value in source)


def move_g2_condition(condition: G2ConditionBatch, device) -> G2ConditionBatch:
    """Move only tensor carriers; preserve text, geometry and autograd identity."""

    def layer_to(layer):
        fields = {"key": layer.key.to(device), "value": layer.value.to(device)}
        if isinstance(layer, UNDLayerKV):
            fields.update(cos=layer.cos.to(device), sin=layer.sin.to(device))
        return replace(layer, **fields)

    features = condition.text_features
    if features is not None:
        features = replace(
            features,
            hidden=features.hidden.to(device),
            valid_mask=features.valid_mask.to(device),
            layers=tuple(layer_to(l) for l in features.layers),
        )
    return replace(
        condition,
        text_features=features,
        views=tuple(
            None
            if views is None
            else tuple(replace(v, latent=v.latent.to(device), valid_mask=v.valid_mask.to(device)) for v in views)
            for views in condition.views
        ),
        global_tokens=None
        if condition.global_tokens is None
        else tuple(None if t is None else t.to(device) for t in condition.global_tokens),
        gen_text_kv=None if condition.gen_text_kv is None else tuple(layer_to(l) for l in condition.gen_text_kv),
        global_kv=None if condition.global_kv is None else tuple(layer_to(l) for l in condition.global_kv),
    )


def stack_g2_prepared(*, text_features, text_kv, global_tokens, global_kv, device=None) -> G2ConditionBatch:
    """Combine D's ordered B1 snapshots into one native mixed batch.

    Missing converted KV falls back to current model projection for the batch.
    Projected Global presence survives a view-payload cache hit. No service or
    cache implementation is imported into the model ABI.
    """
    count = len(text_features)
    if not count or not count == len(text_kv) == len(global_tokens) == len(global_kv):
        raise ValueError("one ordered prepared condition per native sample is required")
    first = text_features[0]
    texts = [f.text for f in text_features]
    identity = (first.text.formatter_id, first.text.tokenizer_id, first.text.dropout_id)
    if any(len(t.lengths) != 1 or (t.formatter_id, t.tokenizer_id, t.dropout_id) != identity for t in texts):
        raise ValueError("prepared samples need B1 text and a common formatter/tokenizer/dropout protocol")
    text = replace(
        first.text,
        token_ids=tuple(t.token_ids[0] for t in texts),
        position_ids=tuple(t.position_ids[0] for t in texts),
        dropped=tuple(t.dropped[0] for t in texts),
    )
    device = first.hidden.device if device is None else device
    layer_count = len(first.layers)
    if any(len(f.layers) != layer_count for f in text_features):
        raise ValueError("prepared UND layer counts differ")
    layers = tuple(
        UNDLayerKV(
            **{
                field: torch.cat([getattr(f.layers[i], field).to(device) for f in text_features])
                for field in ("key", "value", "cos", "sin")
            }
        )
        for i in range(layer_count)
    )
    longest = max(text.lengths)
    hidden = torch.cat([F.pad(f.hidden.to(device), (0, 0, 0, longest - f.hidden.shape[1])) for f in text_features])
    mask = torch.arange(longest, device=device)[None] < torch.tensor(text.lengths, device=device)[:, None]
    features = UNDTextFeatures(text=text, hidden=hidden, valid_mask=mask, layers=layers)
    projected = tuple(
        None if t is None else (t[0] if t.ndim == 3 and t.shape[0] == 1 else t).to(device) for t in global_tokens
    )

    def combine(values):
        if not values or any(v is None for v in values):
            return None
        if any(len(v) != layer_count for v in values):
            raise ValueError("prepared GEN K/V layer counts differ")
        return tuple(
            GENLayerKV(
                key=torch.cat([v[i].key.to(device) for v in values]),
                value=torch.cat([v[i].value.to(device) for v in values]),
            )
            for i in range(layer_count)
        )

    return G2ConditionBatch(
        text=text,
        views=(None,) * count,
        text_features=features,
        global_tokens=projected,
        gen_text_kv=combine(text_kv),
        global_kv=combine(tuple(v for t, v in zip(projected, global_kv, strict=True) if t is not None)),
    )


@dataclass(frozen=True, kw_only=True)
class ReaderOutput:
    tokens: torch.Tensor  # [B,64,512], before A's independent projector

    def __post_init__(self):
        if self.tokens.ndim != 3 or self.tokens.shape[1:] != (64, 512):
            raise ValueError("Reader output must be [B,64,512]")


@dataclass(frozen=True, kw_only=True)
class PrefixLayout:
    """Independent branches. Off means length zero; layout is [G;L;text;native]."""

    global_lengths: tuple[int, ...]
    local_lengths: tuple[int, ...]

    def __post_init__(self):
        if not self.global_lengths or len(self.global_lengths) != len(self.local_lengths):
            raise ValueError("branch lengths must have one entry per sample")
        if any(type(n) is not int or n < 0 for n in (*self.global_lengths, *self.local_lengths)):
            raise ValueError("branch lengths must be nonnegative integers")

    @property
    def sample_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for g, loc in zip(self.global_lengths, self.local_lengths, strict=True):
            offsets.append(offsets[-1] + g + loc)
        return tuple(offsets)

    @property
    def global_present(self) -> tuple[bool, ...]:
        return tuple(n > 0 for n in self.global_lengths)

    @property
    def local_present(self) -> tuple[bool, ...]:
        return tuple(n > 0 for n in self.local_lengths)


@dataclass(frozen=True, kw_only=True)
class TextCacheSignature:
    session_id: str
    checkpoint_id: str
    und_version: str
    gen_version: str
    position_rule: str
    dtype: str
    text: EffectiveTextBatch

    def __post_init__(self):
        _identity(
            self.session_id, self.checkpoint_id, self.und_version, self.gen_version, self.position_rule, self.dtype
        )


@dataclass(frozen=True, kw_only=True)
class GlobalCacheSignature:
    """Only selected views. No unconditional per-replan/frame invalidation."""

    text: TextCacheSignature
    reader_version: str
    projector_version: str
    geometry_encoding_version: str
    top_variant: str
    views: tuple[ViewIdentity, ...]  # canonical sorted view_id order
    projections: tuple[SpatialProjection, ...]
    rasters: tuple[RasterMapping, ...]
    view_types: tuple[tuple[str, str, str, str | None, str | None], ...]  # scope,direction,room,island,variant
    mask_version: str

    def __post_init__(self):
        _identity(self.reader_version, self.projector_version, self.geometry_encoding_version, self.mask_version)
        if self.top_variant not in ("with_robot", "without_robot"):
            raise ValueError("invalid cache top variant")
        ids = tuple(v.view_id for v in self.views)
        if ids != tuple(sorted(set(ids))) or not len(ids) == len(self.projections) == len(self.rasters) == len(
            self.view_types
        ):
            raise ValueError("cache must bind unique sorted selected views and all input metadata")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared training/serving text preparation, before B's single UND forward.

The offline provider owns enhanced instruction construction and initial GT/SG
facts. This module only serializes them with the existing action formatter and
the fixed generation recipe. No window, action labels, or current state enters
this interface. It neither runs UND nor approximates an empty hidden state.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch

from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transform_random import transform_draw

TEXT_PROTOCOL = "psm_g2_static_initial_text_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class StaticTaskText:
    """One provider-owned asset per atomic subtask, shared by every window."""

    original_instruction: str
    enhanced_instruction: str
    episode_id: str
    subtask_id: str
    template_identity: str
    source_identity: str
    initial_context_identity: str
    context_scope: str = "initial_spatial_context"

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("static task text requires nonempty instructions and provider identities")
        if self.context_scope != "initial_spatial_context":
            raise ValueError("enhanced instruction must describe initial spatial context")

    @property
    def identity(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class StaticTaskTextRecipe:
    """Explicit planned video length; never infer it from clean state/action rows.

    Dropout 0.1 is the Owner/G1 starting recipe [S14, O/V], not a new optimum.
    The provider/template and tokenizer are separately versioned assets.
    """

    planned_video_frames: int
    fps: float
    height: int
    width: int
    tokenizer_identity: str
    viewpoint: str = "concat_view"
    additional_view_description: str = (
        "The left half is a third-person view of the scene. The right half is from the wrist-mounted camera."
    )
    float_seconds: bool = False
    use_system_prompt: bool = False
    max_tokens: int = 4096
    cfg_dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.planned_video_frames, self.height, self.width, self.max_tokens)
        ):
            raise ValueError("planned video frames, resolution and token limit must be positive integers")
        if self.max_tokens > 4096 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("invalid fixed text fps/token limit")
        if not self.tokenizer_identity or not self.viewpoint:
            raise ValueError("text recipe requires tokenizer and viewpoint identities")
        if not math.isfinite(self.cfg_dropout_rate) or not 0 <= self.cfg_dropout_rate <= 1:
            raise ValueError("invalid text dropout probability")

    @property
    def identity(self) -> str:
        return _digest({"protocol": TEXT_PROTOCOL, **asdict(self)})


def build_static_task_prompt(task: StaticTaskText, recipe: StaticTaskTextRecipe) -> dict[str, Any]:
    """Use the one action JSON formatter on a whitelist of static fields only."""
    sample = {
        "ai_caption": task.enhanced_instruction,
        "viewpoint": recipe.viewpoint,
        "additional_view_description": recipe.additional_view_description,
        "conditioning_fps": recipe.fps,
        "image_size": (recipe.height, recipe.width),
        # Formatter checks a video tensor, but only reads its logical T. Zero
        # storage avoids allocating fake RGB and never consults action/state.
        "video": torch.empty(0).reshape(1, 1, 1, 0),
        "video_num_frames": recipe.planned_video_frames,
    }
    return ActionPromptJsonFormatter(
        append_idle_frame_text=False,
        float_seconds=recipe.float_seconds,
    )(sample)["ai_caption"]


def serialize_static_task_prompt(task: StaticTaskText, recipe: StaticTaskTextRecipe) -> str:
    # Match TextTokenizerTransform's existing JSON serialization exactly.
    return json.dumps(build_static_task_prompt(task, recipe))


def resolve_consumer_text_dropout(sample: dict[str, Any], rate: float) -> bool:
    """Draw once in consumer materialization; recompute uses the stored bool.

    The existing scoped transform RNG is respected. Model/Reader/GEN callers
    consume the resulting tokens and must never draw another dropout decision.
    """
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("invalid text dropout probability")
    if "text_dropout_applied" not in sample:
        sample["text_dropout_applied"] = rate > 0 and transform_draw(random.random) < rate
    decision = sample["text_dropout_applied"]
    if type(decision) is not bool:
        raise TypeError("text_dropout_applied must be the consumer's bool decision")
    return decision


def prepare_static_task_text(
    task: StaticTaskText,
    recipe: StaticTaskTextRecipe,
    processor: Any,
    *,
    dropout_applied: bool,
) -> dict[str, Any]:
    """Final effective tokens/mask/positions, callable by both train and server.

    ``processor`` is the existing Cosmos tokenizer instance, not another model.
    An empty condition is tokenizer.tokenize_text("") with the same flags; its
    special tokens are real inputs. If the tokenizer emits no tokens, masks and
    positions are empty as well. No hidden-state zeroing is performed.
    """
    if type(dropout_applied) is not bool:
        raise TypeError("dropout_applied must be explicitly decided by the consumer")
    serialized = "" if dropout_applied else serialize_static_task_prompt(task, recipe)
    ids = processor.tokenize_text(
        serialized,
        is_video=False,
        use_system_prompt=recipe.use_system_prompt,
    )[: recipe.max_tokens]
    tokens = torch.tensor(ids, dtype=torch.long)
    if tokens.ndim != 1:
        raise ValueError("Cosmos tokenizer must emit one token sequence per consumer")
    condition_identity = _digest(
        {
            "protocol": TEXT_PROTOCOL,
            "recipe": recipe.identity,
            "token_ids": tokens.tolist(),
            "empty": dropout_applied,
        }
    )
    return {
        "ai_caption": serialized,
        "text_token_ids": tokens,
        "text_attention_mask": torch.ones_like(tokens, dtype=torch.bool),
        "text_position_ids": torch.arange(tokens.numel(), dtype=torch.long),
        "text_condition_identity": condition_identity,
        "text_task_identity": task.identity,
        "text_recipe_identity": recipe.identity,
        "text_dropout_applied": dropout_applied,
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical Local-Memory segment producer for the active production route.

Builds exactly one ``SegmentBatch`` per (stream slot, ``T``-block) from the LIBERO
frame rows plus the ``exact_window_v1`` latent cache.

One consumer is one model sample: the dataset's 17-frame window anchored at the
consumer's episode-local frame, i.e. exactly the per-sample payload
``custom_collate_fn`` expects and ``ActiveNativeBatchInputs`` requires.  Every
evidence tensor reuses the dataset's own口径 (``_build_local_history``): the 96-d
visual summary is ``adaptive_avg_pool2d(latent[0], (1, 2))`` and the executed
action is ``_build_frame_wise_action`` under the dataset's normalization.

``consumer_step`` is the stream-global step, not a block-relative index: a
continuation block keeps counting from where its predecessor stopped, so every
consumer carries a unique ``(slot_id, episode_id, step)`` identity and only a
fresh episode's step 0 is the Local-neutral S0 consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_normalization import normalize_action
from cosmos_framework.model.generator.mot.local_memory_segment import (
    SegmentBatch,
    SegmentProvenance,
)

VISUAL_SUMMARY_DIM = 96

# Keys the joint loader keeps as per-sample single-element lists so that a packed
# sample may carry several items.  ``IterativeJointDataLoader._get_next_sample``
# wraps them (``v[i : i + 1]``) on the loader route; payloads produced here reach
# ``custom_collate_fn`` directly, which only collects per-sample values, so the
# wrapping has to happen here.  A bare tensor is not merely cosmetic:
# ``get_data_and_condition`` requires ``len(cached_video_latents[i]) == 1``, and its
# multi-vision probe reads a bare tensor's first dim as an item count.
MULTI_ITEM_KEYS = ("text_token_ids", "images", "video", "video_latent", "action", "action_raw", "sound")


@dataclass(frozen=True)
class CanonicalSegmentStream:
    """One stream slot's fixed position inside the dataset.

    ``episode_position`` indexes the dataset's kept-episode arrays; all frame
    arithmetic below is episode-local so a stream never crosses an episode.
    """

    slot_id: int
    episode_index: int
    episode_position: int
    category: str

    def __post_init__(self) -> None:
        if self.slot_id < 0 or self.episode_index < 0 or self.episode_position < 0 or not self.category:
            raise ValueError("segment stream fields are invalid")


class CanonicalLocalMemorySegmentProducer:
    """Turn (stream, cursor) into the canonical ``SegmentBatch`` ABI.

    ``wrapped_dataset`` is the production ``ActionSFTDataset``: per-sample payloads
    must carry the transform pipeline's output (``sequence_plan``, ``text_token_ids``,
    ...), while the evidence口径 lives on the wrapped ``LIBEROLeRobotDataset``.  The
    producer calls ``_build_item`` directly instead of ``__getitem__`` so a decode
    failure propagates with its exact frame identity instead of being silently
    resampled onto a different frame.
    """

    def __init__(
        self,
        wrapped_dataset: Any,
        *,
        category: str,
        ttt_tbptt_steps: int,
        manifest_digest: str,
        config_digest: str,
        source_digest: str,
    ) -> None:
        if ttt_tbptt_steps <= 0:
            raise ValueError("producer requires a positive ttt_tbptt_steps")
        if not manifest_digest or not config_digest or not source_digest:
            raise ValueError("producer digests must be non-empty")
        frame_source = getattr(wrapped_dataset, "_dataset", None)
        if frame_source is None or not hasattr(frame_source, "_load_cached_latent"):
            raise TypeError("segment producer requires an ActionSFTDataset over a latent-cached frame source")
        if getattr(frame_source, "_latent_cache_root", None) is None:
            raise ValueError("canonical Local-Memory production requires an exact-window latent cache")
        self.wrapped_dataset = wrapped_dataset
        self.frame_source = frame_source
        self.category = category
        self.ttt_tbptt_steps = ttt_tbptt_steps
        self.manifest_digest = manifest_digest
        self.config_digest = config_digest
        self.source_digest = source_digest

        if hasattr(wrapped_dataset, "get_shuffle_blocks"):
            blocks = wrapped_dataset.get_shuffle_blocks()
        elif hasattr(frame_source, "get_shuffle_blocks"):
            blocks = frame_source.get_shuffle_blocks()
        elif hasattr(frame_source, "_valid_cum"):
            blocks = []
            previous = 0
            for end in frame_source._valid_cum:
                end = int(end)
                if end > previous:
                    blocks.append((previous, end - previous))
                previous = end
        else:
            raise TypeError("segment producer requires episode-aligned shuffle blocks")
        self._blocks = tuple((int(start), int(length)) for start, length in blocks if int(length) > 0)
        if not self._blocks:
            raise ValueError("segment producer found no valid episode blocks")

        if hasattr(frame_source, "_ep_vals") and len(frame_source._ep_vals) == len(self._blocks):
            episode_ids = [int(value) for value in frame_source._ep_vals]
        elif hasattr(frame_source, "_episode_records") and len(frame_source._episode_records) == len(self._blocks):
            episode_ids = [int(record[3]) for record in frame_source._episode_records]
        elif hasattr(frame_source, "_resolve_index"):
            episode_ids = [int(frame_source._resolve_index(start)[2]) for start, _ in self._blocks]
        else:
            raise TypeError("segment producer cannot resolve episode identities")
        self._episode_catalog = tuple(enumerate(episode_ids))

        evidence_action_dim = getattr(frame_source, "local_memory_action_dim", None)
        if evidence_action_dim is None:
            evidence_action_dim = getattr(frame_source, "action_dim", None)
        self.evidence_action_dim = int(evidence_action_dim)
        if self.evidence_action_dim <= 0:
            raise ValueError("segment producer requires a positive evidence action dimension")

    # ---- geometry ----------------------------------------------------------

    def episode_catalog(self) -> tuple[tuple[int, int], ...]:
        """Return deterministic episode-position / episode-id pairs."""
        return self._episode_catalog

    def valid_start_count(self, stream: CanonicalSegmentStream) -> int:
        """Number of dataset samples (window anchors) this episode can serve."""
        position = self._position(stream)
        return int(self._blocks[position][1])

    def block_count(self, stream: CanonicalSegmentStream) -> int:
        """Number of whole ``T``-blocks this episode can still serve."""
        return self.valid_start_count(stream) // self.ttt_tbptt_steps

    def _position(self, stream: CanonicalSegmentStream) -> int:
        position = stream.episode_position
        if not 0 <= position < len(self._episode_catalog):
            raise ValueError("segment stream episode position is out of range")
        if int(self._episode_catalog[position][1]) != stream.episode_index:
            raise ValueError("segment stream episode index disagrees with its dataset position")
        return position

    def _flat_index(self, stream: CanonicalSegmentStream, local_frame: int) -> int:
        """Inverse of ``LIBEROLeRobotDataset._build_item``'s idx -> start mapping."""
        position = self._position(stream)
        start, length = self._blocks[position]
        if not 0 <= local_frame < length:
            raise ValueError("segment local frame is outside the episode block")
        return int(start) + int(local_frame)

    # ---- evidence口径 (mirrors _build_local_history) -----------------------

    def _visual_summary(self, episode_index: int, local_frame: int) -> torch.Tensor:
        latent = self.frame_source._load_cached_latent(episode_index, local_frame)
        if latent is None:
            raise AssertionError("canonical Local-Memory evidence requires exact-window latent cache")
        return F.adaptive_avg_pool2d(latent[0].unsqueeze(0), output_size=(1, 2)).flatten()

    def _executed_action(self, stream: CanonicalSegmentStream, local_frame: int) -> torch.Tensor:
        flat_index = self._flat_index(stream, local_frame)
        hook = getattr(self.frame_source, "local_memory_executed_action", None)
        if callable(hook):
            action = hook(flat_index)
        else:
            position = self._position(stream)
            row = int(self.frame_source._ep_starts[position]) + local_frame
            raw = self.frame_source._row_action[row : row + 1]
            action = self.frame_source._build_frame_wise_action(raw)[0]
            if self.frame_source.action_normalization is not None:
                action = normalize_action(
                    action,
                    self.frame_source.action_normalization,
                    self.frame_source._load_norm_stats(),
                )
        action = action.detach().float().reshape(-1)
        if tuple(action.shape) != (self.evidence_action_dim,) or not torch.isfinite(action).all():
            raise ValueError(
                "canonical Local evidence action mismatch: "
                f"expected [{self.evidence_action_dim}], got {tuple(action.shape)}"
            )
        return action

    def _payload(self, stream: CanonicalSegmentStream, local_frame: int) -> Any:
        flat_index = self._flat_index(stream, local_frame)
        builder = getattr(self.frame_source, "_build_item", None)
        item = builder(flat_index) if callable(builder) else self.frame_source[flat_index]
        payload = self.wrapped_dataset._transform(item, self.wrapped_dataset._resolution)
        if not isinstance(payload, dict):
            return payload
        for key in MULTI_ITEM_KEYS:
            value = payload.get(key)
            # Mirror ``_get_next_sample``: an existing list is kept as-is, a bare
            # value is wrapped.  ``None`` stays ``None`` so ``custom_collate_fn``'s
            # sparse-key path, which keys off a per-sample ``None``, still sees it.
            if value is not None and not isinstance(value, list):
                payload[key] = [value]
        return payload

    # ---- production --------------------------------------------------------

    def produce(
        self, stream: CanonicalSegmentStream, *, cursor: int, steps: int | None = None
    ) -> SegmentBatch:
        """Build the exact ``SegmentBatch`` for ``stream`` at block ``cursor``.

        Consumer ``t`` is the dataset sample anchored at episode-local frame
        ``cursor * T + t``; its evidence is the frame immediately before it, and is
        absent only at global step 0.
        """
        if cursor < 0:
            raise ValueError("segment cursor must be non-negative")
        if stream.category != self.category:
            raise ValueError("segment stream category differs from the producer's")
        width = self.ttt_tbptt_steps if steps is None else steps
        if not 1 <= width <= self.ttt_tbptt_steps:
            raise ValueError("segment block width must be in [1, ttt_tbptt_steps]")
        base = cursor * self.ttt_tbptt_steps
        available = self.valid_start_count(stream)
        if base + width > available:
            raise ValueError("segment block runs past the end of its episode")

        payloads: list[Any] = []
        consumer_visual_summary = torch.zeros((1, width, VISUAL_SUMMARY_DIM), dtype=torch.float32)
        evidence_visual_summary_prev = torch.zeros((1, width, VISUAL_SUMMARY_DIM), dtype=torch.float32)
        evidence_executed_action_prev = torch.zeros((1, width, self.evidence_action_dim), dtype=torch.float32)
        evidence_valid = torch.zeros((1, width), dtype=torch.bool)
        evidence_source_step = torch.full((1, width), -1, dtype=torch.long)
        consumer_step = torch.arange(base, base + width, dtype=torch.long).unsqueeze(0)
        for index in range(width):
            local_frame = base + index
            payloads.append(self._payload(stream, local_frame))
            consumer_visual_summary[0, index] = self._visual_summary(stream.episode_index, local_frame)
            if local_frame == 0:
                continue
            evidence_valid[0, index] = True
            evidence_source_step[0, index] = local_frame - 1
            evidence_visual_summary_prev[0, index] = self._visual_summary(stream.episode_index, local_frame - 1)
            evidence_executed_action_prev[0, index] = self._executed_action(stream, local_frame - 1)

        segment = SegmentBatch(
            consumer_visual_summary=consumer_visual_summary,
            consumer_payload=(tuple(payloads),),
            consumer_valid=torch.ones((1, width), dtype=torch.bool),
            consumer_step=consumer_step,
            evidence_visual_summary_prev=evidence_visual_summary_prev,
            evidence_executed_action_prev=evidence_executed_action_prev,
            evidence_valid=evidence_valid,
            evidence_source_step=evidence_source_step,
            slot_id=torch.tensor([stream.slot_id], dtype=torch.long),
            episode_id=(str(stream.episode_index),),
            category=(stream.category,),
            segment_provenance=SegmentProvenance(
                manifest_digest=self.manifest_digest,
                config_digest=self.config_digest,
                source_digest=self.source_digest,
                segment_id=cursor,
            ),
        )
        segment.validate(self.ttt_tbptt_steps)
        return segment

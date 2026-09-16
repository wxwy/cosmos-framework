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

    # ---- geometry ----------------------------------------------------------

    def valid_start_count(self, stream: CanonicalSegmentStream) -> int:
        """Number of dataset samples (window anchors) this episode can serve."""
        position = self._position(stream)
        cum = self.frame_source._valid_cum
        previous = int(cum[position - 1]) if position > 0 else 0
        return int(cum[position]) - previous

    def block_count(self, stream: CanonicalSegmentStream) -> int:
        """Number of whole ``T``-blocks this episode can still serve."""
        return self.valid_start_count(stream) // self.ttt_tbptt_steps

    def _position(self, stream: CanonicalSegmentStream) -> int:
        position = stream.episode_position
        ep_vals = self.frame_source._ep_vals
        if not 0 <= position < len(ep_vals):
            raise ValueError("segment stream episode position is out of range")
        if int(ep_vals[position]) != stream.episode_index:
            raise ValueError("segment stream episode index disagrees with its dataset position")
        return position

    def _flat_index(self, stream: CanonicalSegmentStream, local_frame: int) -> int:
        """Inverse of ``LIBEROLeRobotDataset._build_item``'s idx -> start mapping."""
        position = self._position(stream)
        base = int(self.frame_source._valid_cum[position - 1]) if position > 0 else 0
        return base + local_frame

    def _row_index(self, stream: CanonicalSegmentStream, local_frame: int) -> int:
        return int(self.frame_source._ep_starts[self._position(stream)]) + local_frame

    # ---- evidence口径 (mirrors _build_local_history) -----------------------

    def _visual_summary(self, episode_index: int, local_frame: int) -> torch.Tensor:
        latent = self.frame_source._load_cached_latent(episode_index, local_frame)
        if latent is None:
            raise AssertionError("canonical Local-Memory evidence requires exact-window latent cache")
        return F.adaptive_avg_pool2d(latent[0].unsqueeze(0), output_size=(1, 2)).flatten()

    def _executed_action(self, stream: CanonicalSegmentStream, local_frame: int) -> torch.Tensor:
        row = self._row_index(stream, local_frame)
        raw = self.frame_source._row_action[row : row + 1]  # [1, 7]
        action = self.frame_source._build_frame_wise_action(raw)[0]
        if self.frame_source.action_normalization is None:
            return action
        return normalize_action(
            action, self.frame_source.action_normalization, self.frame_source._load_norm_stats()
        )

    def _payload(self, stream: CanonicalSegmentStream, local_frame: int) -> Any:
        item = self.frame_source._build_item(self._flat_index(stream, local_frame))
        return self.wrapped_dataset._transform(item, self.wrapped_dataset._resolution)

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
        evidence_executed_action_prev = torch.zeros((1, width, self.frame_source.action_dim), dtype=torch.float32)
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

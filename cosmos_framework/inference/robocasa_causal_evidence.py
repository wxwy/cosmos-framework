# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transactional online materialization of the frozen RoboCasa B1 Local visual evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model.generator.mot.robocasa_latent_evidence import (
    TEMPORAL_COMPRESSION_FACTOR,
    latent_to_visual96,
    stream_endpoint_step,
)
from cosmos_framework.inference.robocasa_local_memory_contract import CAMERA_HEIGHT, CAMERA_WIDTH


@dataclass(frozen=True)
class RoboCasaVisualStreamState:
    """Committed/candidate visual frontier for one Local-TTT episode."""

    next_source_step: int
    last_endpoint_step: int | None
    current_visual96: torch.Tensor | None
    pending_left: tuple[torch.Tensor, ...]
    pending_wrist: tuple[torch.Tensor, ...]
    encoder_state: Any


def validate_b1_rgb_frame(frame: torch.Tensor, name: str) -> torch.Tensor:
    value = frame.detach().to(device="cpu").contiguous()
    if value.shape != (3, CAMERA_HEIGHT, CAMERA_WIDTH) or value.dtype != torch.uint8:
        raise ValueError(f"{name} must be uint8 [3,{CAMERA_HEIGHT},{CAMERA_WIDTH}]")
    return value.clone()


def completed_visual_digest(
    source_steps: tuple[int, ...],
    left_frames: tuple[torch.Tensor, ...],
    wrist_frames: tuple[torch.Tensor, ...],
) -> str:
    if len(source_steps) != len(left_frames) or len(source_steps) != len(wrist_frames):
        raise ValueError("Local-TTT visual evidence lengths must match")
    digest = hashlib.sha256()
    for step, left, wrist in zip(source_steps, left_frames, wrist_frames, strict=True):
        digest.update(f"{step}:".encode("ascii"))
        digest.update(left.numpy().tobytes(order="C"))
        digest.update(wrist.numpy().tobytes(order="C"))
    return digest.hexdigest()


class RoboCasaCausalEvidenceStream:
    """Reproduce the episode-level B1 causal-endpoint evidence with bounded RGB state."""

    def __init__(self, model: Any) -> None:
        self.model = model
        tokenizer = getattr(model, "tokenizer_vision_gen", None)
        if tokenizer is None or not getattr(tokenizer, "is_causal", False):
            raise ValueError("RoboCasa Local evidence requires the loaded causal Wan vision tokenizer")
        for name in (
            "new_encoder_stream_state",
            "snapshot_encoder_stream_state",
            "restore_encoder_stream_state",
        ):
            if not callable(getattr(tokenizer, name, None)):
                raise ValueError(f"RoboCasa Local evidence requires tokenizer.{name}()")
        if not callable(getattr(model, "_encode_vision_item_streaming", None)):
            raise ValueError("RoboCasa Local evidence requires model._encode_vision_item_streaming()")
        self.tokenizer = tokenizer

    def fresh(self) -> RoboCasaVisualStreamState:
        return RoboCasaVisualStreamState(
            next_source_step=0,
            last_endpoint_step=None,
            current_visual96=None,
            pending_left=(),
            pending_wrist=(),
            encoder_state=self.tokenizer.new_encoder_stream_state(),
        )

    @staticmethod
    def _pixel_chunk(
        left_frames: tuple[torch.Tensor, ...],
        wrist_frames: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if not left_frames or len(left_frames) != len(wrist_frames):
            raise ValueError("left/wrist causal chunks must be non-empty and aligned")
        left = torch.stack(left_frames, dim=1)  # [3,T,H,W]
        wrist = torch.stack(wrist_frames, dim=1)
        return torch.stack((left, wrist), dim=0)  # [2,3,T,H,W]

    def _encode_endpoint(
        self,
        left_frames: tuple[torch.Tensor, ...],
        wrist_frames: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        pixels = self._pixel_chunk(left_frames, wrist_frames)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            latent = self.model._encode_vision_item_streaming(pixels.to(device), num_views=1)
        expected = (2, 48, 1, 16, 16)
        if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
            raise ValueError(f"RoboCasa streaming endpoint latent must be finite {expected}, got {tuple(latent.shape)}")
        # Training B1 persists fp16 before constructing visual96. Preserve that quantization ABI.
        current = latent[:, :, 0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        return latent_to_visual96(current[0], current[1])

    def advance(
        self,
        state: RoboCasaVisualStreamState,
        source_steps: tuple[int, ...],
        left_frames: tuple[torch.Tensor, ...],
        wrist_frames: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, RoboCasaVisualStreamState]:
        if not isinstance(state, RoboCasaVisualStreamState):
            raise ValueError("invalid RoboCasa visual stream state")
        if len(source_steps) != len(left_frames) or len(source_steps) != len(wrist_frames):
            raise ValueError("RoboCasa visual stream evidence lengths must match")
        if not source_steps:
            return torch.empty(0, 96, dtype=torch.float32), state
        expected_steps = tuple(range(state.next_source_step, state.next_source_step + len(source_steps)))
        if source_steps != expected_steps:
            raise ValueError(f"RoboCasa visual stream must be contiguous {expected_steps}, got {source_steps}")
        if len(state.pending_left) != len(state.pending_wrist) or len(state.pending_left) >= TEMPORAL_COMPRESSION_FACTOR:
            raise ValueError("invalid committed RoboCasa visual tail")
        if state.next_source_step == 0:
            if state.last_endpoint_step is not None or state.current_visual96 is not None or state.pending_left:
                raise ValueError("fresh RoboCasa visual stream carries stale state")
        elif (
            state.last_endpoint_step != stream_endpoint_step(state.next_source_step - 1)
            or state.current_visual96 is None
        ):
            raise ValueError("committed RoboCasa visual frontier is inconsistent")

        original_tokenizer_state = self.tokenizer.snapshot_encoder_stream_state()
        self.tokenizer.restore_encoder_stream_state(state.encoder_state)
        pending_left = list(state.pending_left)
        pending_wrist = list(state.pending_wrist)
        current = None if state.current_visual96 is None else state.current_visual96.detach().clone()
        last_endpoint = state.last_endpoint_step
        outputs: list[torch.Tensor] = []
        try:
            for source_step, left_frame, wrist_frame in zip(source_steps, left_frames, wrist_frames, strict=True):
                left = validate_b1_rgb_frame(left_frame, "left_frame")
                wrist = validate_b1_rgb_frame(wrist_frame, "wrist_frame")
                if source_step == 0:
                    if current is not None or pending_left or pending_wrist:
                        raise ValueError("source step0 must prime a fresh RoboCasa visual stream")
                    current = self._encode_endpoint((left,), (wrist,))
                    last_endpoint = 0
                else:
                    pending_left.append(left)
                    pending_wrist.append(wrist)
                    if len(pending_left) == TEMPORAL_COMPRESSION_FACTOR:
                        if source_step % TEMPORAL_COMPRESSION_FACTOR != 0:
                            raise ValueError("RoboCasa causal endpoint chunk ended on a non-4-grid source step")
                        current = self._encode_endpoint(tuple(pending_left), tuple(pending_wrist))
                        last_endpoint = source_step
                        pending_left.clear()
                        pending_wrist.clear()
                if current is None:
                    raise RuntimeError("RoboCasa visual stream has no causal endpoint for completed evidence")
                if last_endpoint != stream_endpoint_step(source_step):
                    raise RuntimeError("RoboCasa visual endpoint/source-step mapping drift")
                outputs.append(current.detach().clone())

            candidate_encoder_state = self.tokenizer.snapshot_encoder_stream_state()
        finally:
            # The shared tokenizer never publishes speculative Local evidence state.
            self.tokenizer.restore_encoder_stream_state(original_tokenizer_state)

        candidate = RoboCasaVisualStreamState(
            next_source_step=state.next_source_step + len(source_steps),
            last_endpoint_step=last_endpoint,
            current_visual96=current.detach().clone() if current is not None else None,
            pending_left=tuple(pending_left),
            pending_wrist=tuple(pending_wrist),
            encoder_state=candidate_encoder_state,
        )
        if len(candidate.pending_left) >= TEMPORAL_COMPRESSION_FACTOR:
            raise RuntimeError("RoboCasa visual stream retained too many RGB tail frames")
        return torch.stack(outputs), candidate

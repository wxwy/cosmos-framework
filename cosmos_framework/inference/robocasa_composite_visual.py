# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected RoboCasa composite spatial prep and independent single-frame Encode1."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.utils.transforms import find_closest_target_size, reflection_pad_to_target
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import robocasa_current_latent_to_visual96
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


@dataclass(frozen=True)
class PreparedCompositeFrame:
    source_uint8: torch.Tensor
    padded_single_frame: torch.Tensor
    padded_image_size: torch.Tensor


def composite_frame_digest(frame: torch.Tensor) -> str:
    """Canonical CHW uint8 digest, including dimensions and a format domain."""
    if (
        not isinstance(frame, torch.Tensor)
        or frame.dtype != torch.uint8
        or frame.ndim != 3
        or frame.shape[0] != 3
        or min(frame.shape[1:]) <= 0
    ):
        raise ValueError("composite digest requires nonempty uint8 [3,H,W]")
    digest = hashlib.sha256(b"PSM-WMA/robocasa-composite-chw-uint8/v1\0")
    digest.update(struct.pack(">QQQ", *frame.shape))
    digest.update(frame.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CapturedCurrentVisual:
    source_digest: str
    visual96: torch.Tensor
    image_size: int
    preprocess_profile: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.source_digest)
            or type(self.image_size) is not int
            or self.image_size <= 0
            or self.preprocess_profile != PREPROCESS_PROFILE
        ):
            raise ValueError("invalid captured current visual identity")
        if (
            not isinstance(self.visual96, torch.Tensor)
            or self.visual96.device.type != "cpu"
            or self.visual96.dtype != torch.float32
            or self.visual96.shape != (96,)
            or not bool(torch.isfinite(self.visual96).all())
        ):
            raise ValueError("captured current visual requires finite CPU fp32 [96]")
        object.__setattr__(self, "visual96", self.visual96.detach().clone())


class _PolicyCurrentVisualCapture:
    """One model-lock-owned scope; no encoder calls or fallback of its own."""

    def __init__(
        self, tokenizer: object, prepared_frames: Sequence[PreparedCompositeFrame], image_sizes: Sequence[int]
    ):
        self._tokenizer = tokenizer
        self._frames = tuple(prepared_frames)
        self._sizes = tuple(image_sizes)
        self._entered = False
        self._active = False
        self._complete = False
        self._failure: BaseException | None = None
        self._captured: list[CapturedCurrentVisual] = []
        self._calls = 0

    @property
    def current(self) -> tuple[CapturedCurrentVisual, ...]:
        if not self._complete:
            raise RuntimeError("Policy current visuals require a successfully completed capture scope")
        return tuple(self._captured)

    def _validate_tokenizer(self) -> None:
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
            Wan2pt2VAEInterface,
            WanVAE,
            WanVAE_,
        )

        tokenizer = self._tokenizer
        if (
            not isinstance(tokenizer, Wan2pt2VAEInterface)
            or not isinstance(getattr(tokenizer, "model", None), WanVAE)
            or not isinstance(getattr(tokenizer.model, "model", None), WanVAE_)
            or not tokenizer.is_causal
            or tokenizer.spatial_compression_factor != 16
            or tokenizer.temporal_compression_factor != 4
            or getattr(tokenizer, "use_streaming_encode", None) is not False
            or getattr(tokenizer, "_keep_encoder_cache", None) is not False
            or tokenizer.model.model.training
        ):
            raise RuntimeError("Policy current capture requires a frozen causal non-streaming Wan2.2 tokenizer")

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Policy current capture scope cannot be reentered")
        self._entered = True
        self._validate_tokenizer()
        if not self._frames or len(self._frames) != len(self._sizes):
            raise ValueError("Policy current capture requires one image_size per nonempty prepared frame")
        original = self._tokenizer.encode
        if getattr(original, "_robocasa_current_capture_scope", None) is not None:
            raise RuntimeError("nested Policy current capture on the same tokenizer is forbidden")
        snapshots = []
        for prepared, image_size in zip(self._frames, self._sizes, strict=True):
            if not isinstance(prepared, PreparedCompositeFrame) or type(image_size) is not int or image_size <= 0:
                raise ValueError("invalid Policy current capture prepared frame/image_size")
            composite_frame_digest(prepared.source_uint8)
            if prepared.source_uint8.shape[1] != image_size:
                raise ValueError("Policy current source height must match image_size")
            # Bind source identity to the exact pre-pad/resize contract, including
            # its canvas metadata. This is spatial processing, never another VAE call.
            checked = prepare_robocasa_composite_frame(prepared.source_uint8, image_size)
            if (
                not isinstance(prepared.padded_single_frame, torch.Tensor)
                or prepared.padded_single_frame.dtype != torch.uint8
                or not torch.equal(prepared.padded_single_frame, checked.padded_single_frame)
                or not isinstance(prepared.padded_image_size, torch.Tensor)
                or not torch.equal(prepared.padded_image_size, checked.padded_image_size)
            ):
                raise ValueError("Policy current prepared source/padded canvas binding differs")
            snapshots.append(
                (composite_frame_digest(checked.source_uint8), checked.padded_single_frame.clone(), image_size)
            )
        self._snapshots = tuple(snapshots)
        self._loaded = self._tokenizer.model
        self._module = self._loaded.model
        first = next(self._module.parameters(), None)
        if first is None or any(parameter.requires_grad for parameter in self._module.parameters()):
            raise RuntimeError("Policy current capture requires loaded frozen Wan parameters")
        self._device = first.device
        self._had_encode = "encode" in vars(self._tokenizer)
        self._old_encode = vars(self._tokenizer).get("encode")
        self._original = original

        def encode(state: torch.Tensor, *args, **kwargs):
            if not self._active:
                raise RuntimeError("Policy current capture encoder wrapper is no longer active")
            try:
                self._validate_tokenizer()
                if self._tokenizer.model is not self._loaded or self._loaded.model is not self._module:
                    raise RuntimeError("Policy current capture tokenizer/model identity changed")
                if self._failure is not None:
                    raise RuntimeError("Policy current capture already failed") from self._failure
                if args or kwargs or self._calls >= len(self._snapshots):
                    raise ValueError("Policy current capture received unexpected encoder call/count")
                source_digest, frame, image_size = self._snapshots[self._calls]
                expected = normalize_uint8_item(frame.unsqueeze(0), {"device": self._device, "dtype": torch.float32})
                if (
                    not isinstance(state, torch.Tensor)
                    or state.dtype != torch.float32
                    or state.device != expected.device
                    or state.shape != expected.shape
                    or not torch.equal(state, expected)
                ):
                    raise ValueError("Policy current capture requires the matching full fp32 T=1 input in order")
                if torch.is_grad_enabled():
                    raise RuntimeError("Policy current capture is inference-only")
                self._calls += 1
                encoded = self._original(state)
                if (
                    not isinstance(encoded, torch.Tensor)
                    or encoded.dtype != torch.float32
                    or encoded.device != self._device
                    or encoded.shape != (1, 48, 1, frame.shape[-2] // 16, frame.shape[-1] // 16)
                    or not bool(torch.isfinite(encoded).all())
                ):
                    raise ValueError("Policy current capture requires finite complete pre-crop fp32 [1,48,1,H_z,W_z]")
                z0 = encoded[0, :, 0].float().detach()
                visual96 = robocasa_current_latent_to_visual96(z0).detach().to(device="cpu", dtype=torch.float32)
                self._captured.append(CapturedCurrentVisual(source_digest, visual96, image_size, PREPROCESS_PROFILE))
                return encoded
            except BaseException as error:
                self._failure = error
                self._captured.clear()
                raise

        encode._robocasa_current_capture_scope = self
        self._wrapper = encode
        self._tokenizer.encode = encode
        self._active = True
        return self

    def __exit__(self, exception_type, exception, traceback):
        replaced = vars(self._tokenizer).get("encode") is not self._wrapper
        self._active = False
        # Restore both the exact object and the presence/absence of an instance
        # override, even when encoding/generation or a caller inside the scope fails.
        if self._had_encode:
            self._tokenizer.encode = self._old_encode
        elif "encode" in vars(self._tokenizer):
            delattr(self._tokenizer, "encode")
        if exception_type is not None:
            self._captured.clear()
            return False
        if replaced:
            self._captured.clear()
            raise RuntimeError("Policy current capture encoder was replaced during its scope")
        try:
            self._validate_tokenizer()
            if self._tokenizer.model is not self._loaded or self._loaded.model is not self._module:
                raise RuntimeError("Policy current capture tokenizer/model identity changed")
        except BaseException:
            self._captured.clear()
            raise
        if self._failure is not None:
            raise RuntimeError("Policy current capture failed; no current visuals may be reused") from self._failure
        if self._calls != len(self._snapshots) or len(self._captured) != len(self._snapshots):
            self._captured.clear()
            raise RuntimeError("Policy current capture did not observe exactly one encoding per prepared frame")
        self._complete = True
        return False


def capture_policy_current_visuals(
    tokenizer: object,
    prepared_frames: Sequence[PreparedCompositeFrame],
    *,
    image_sizes: Sequence[int],
) -> _PolicyCurrentVisualCapture:
    """Capture existing Policy Encode1 outputs before crop, under the caller's model lock.

    Required-mode RoboCasa callers may read ``scope.current`` only after successful
    exit. No-TTT callers must not enter this scope. It never calls an encoder except
    when forwarding the existing, strictly matched Policy call, and never falls back.
    ``source_digest`` identifies the resized ``PreparedCompositeFrame.source_uint8``.
    """
    return _PolicyCurrentVisualCapture(tokenizer, prepared_frames, image_sizes)


def prepare_robocasa_composite_frame(composite: torch.Tensor, image_size: int) -> PreparedCompositeFrame:
    """复用原 Policy 的 bilinear resize 和 reflection pad 空间合同。"""
    if composite.dtype != torch.uint8 or composite.ndim != 3 or composite.shape[0] != 3:
        raise ValueError("composite 必须为 uint8 [3,H,W]")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size 必须为正整数")
    source = composite.contiguous()
    height, width = source.shape[-2:]
    if height != image_size:
        new_width = int(round(width * image_size / height))
        resized = Image.fromarray(source.permute(1, 2, 0).cpu().numpy()).resize(
            (new_width, image_size), resample=Image.Resampling.BILINEAR
        )
        source = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()
    _, height, width = source.shape
    resolution = get_vision_data_resolution((height, width))
    target_w, target_h = find_closest_target_size(height, width, resolution)
    padded: dict[str, torch.Tensor] = {"video": source.unsqueeze(1)}
    reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
    return PreparedCompositeFrame(source, padded["video"], padded["image_size"])


def encode_current_composite_visual96(
    tokenizer: object, prepared: PreparedCompositeFrame, *, timing: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """单帧、无 stream state 的 Wan 编码；真实数值 parity 属于后续 Gate。"""
    frame = prepared.padded_single_frame
    if frame.dtype != torch.uint8 or frame.ndim != 4 or frame.shape[:2] != (3, 1):
        raise ValueError("Encode1 要求 uint8 [3,1,H,W]")
    if getattr(tokenizer, "use_streaming_encode", False) or getattr(tokenizer, "_keep_encoder_cache", False):
        raise RuntimeError("Encode1 禁止 streaming 或 retained encoder cache")
    wan = getattr(tokenizer, "model", None)
    module = getattr(wan, "model", None)
    parameters = getattr(module, "parameters", None)
    first = next(parameters(), None) if callable(parameters) else None
    device = first.device if first is not None else getattr(wan, "device", None)
    if device is None:
        raise ValueError("Encode1 无法解析 loaded Wan tokenizer device")
    normalized = normalize_uint8_item(frame.unsqueeze(0), {"device": device, "dtype": torch.float32})
    with torch.no_grad():
        with timing.measure("independent_encode1") if timing is not None else nullcontext():
            encoded = tokenizer.encode(normalized)
    if encoded.ndim != 5 or encoded.shape[:3] != (1, 48, 1):
        raise ValueError("Encode1 必须返回 [1,48,1,H_z,W_z]")
    z0 = encoded[0, :, 0].float().detach().clone()
    if not bool(torch.isfinite(z0).all()):
        raise FloatingPointError("Encode1 latent 非有限")
    with timing.measure("visual96") if timing is not None else nullcontext():
        visual96 = robocasa_current_latent_to_visual96(z0)
    return z0, visual96

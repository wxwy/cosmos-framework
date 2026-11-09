# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone, injected Encode1 adapter; no model loading or Policy integration.

Importing this module needs only the standard library. Numerical dependencies
are imported when preprocessing or using an already-loaded tokenizer. The small
inference math below preserves the official composite/Visual96 operations while
avoiding their transitive training-dataset imports. Actual Wan BF16/hardware
parity is a separate Gate; a supplied identity is an Owner attestation, not a
claim that this adapter has loaded or independently authenticated model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

PREPROCESS_PROFILE = "left_wrist_reflection_pad_v1"
RGB_SHAPE = (256, 512, 3)
RGB_BYTES = 256 * 512 * 3
_IDENTITY_KEYS = frozenset(
    (
        "vae_weights_sha256",
        "vae_config_sha256",
        "preprocess_profile_sha256",
        "visual96_contract_sha256",
        "encoder_backend",
        "encoder_dtype",
    )
)


def _configuration_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def preprocess_profile_digest(image_size: int = 256) -> str:
    if type(image_size) is not int or not 1 <= image_size <= 2048:
        raise ValueError("Worker image_size must be an integer in 1..2048")
    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO

    return _configuration_digest(
        {
            "version": "PSM23-preprocess-v1",
            "profile": PREPROCESS_PROFILE,
            "image_size": image_size,
            "input": "HWC_UINT8_RGB_256x512x3",
            "initial_resize": "PIL_BILINEAR_height_round_width",
            "canvas": "VIDEO_RES_SIZE_INFO_aspect_nearest_first_tie",
            "fit": "no_upscale_round_half_up",
            "fit_resize": "torchvision_BICUBIC_antialias",
            "pad": "bottom_right_reflect_else_edge",
            "normalize": "float32_div127.5_sub1",
            "time": "independent_T1",
            "canvas_table": VIDEO_RES_SIZE_INFO,
        }
    )


def visual96_contract_digest() -> str:
    return _configuration_digest(
        {
            "version": "PSM23-visual96-v1",
            "latent": "float32_48xHzxWz_complete_padded_canvas",
            "time_index": 0,
            "crop": False,
            "pool": "adaptive_avg_pool2d_1x2_channel_major_flatten",
            "output": "finite_float32_little_endian_96",
        }
    )


@dataclass(frozen=True)
class PreparedCompositeFrame:
    source_uint8: Any
    padded_single_frame: Any
    padded_image_size: Any


def prepare_composite_frame(composite: Any, image_size: int) -> PreparedCompositeFrame:
    """The existing Policy's spatial math, with no dataset or training imports."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as transforms_f

    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
    from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution

    if (
        not isinstance(composite, torch.Tensor)
        or composite.dtype != torch.uint8
        or composite.ndim != 3
        or composite.shape[0] != 3
        or min(composite.shape[-2:]) <= 0
    ):
        raise ValueError("composite must be nonempty uint8 [3,H,W]")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    source = composite.contiguous()
    height, width = source.shape[-2:]
    if height != image_size:
        new_width = int(round(width * image_size / height))
        resized = Image.fromarray(source.permute(1, 2, 0).cpu().numpy()).resize(
            (new_width, image_size), resample=Image.Resampling.BILINEAR
        )
        source = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()
    _, height, width = source.shape
    candidates = VIDEO_RES_SIZE_INFO[get_vision_data_resolution((height, width))]
    # min preserves the table's first-entry tie behavior.
    target_w, target_h = min(candidates.values(), key=lambda size: abs(height / width - size[1] / size[0]))
    scaling_ratio = min(target_w / width, target_h / height, 1.0)
    content_h, content_w = int(scaling_ratio * height + 0.5), int(scaling_ratio * width + 0.5)
    padded = source.unsqueeze(1)
    if (content_h, content_w) != (height, width):
        padded = transforms_f.resize(
            padded, size=[content_h, content_w], interpolation=transforms_f.InterpolationMode.BICUBIC, antialias=True
        )
    if (content_h, content_w) != (target_h, target_w):
        right, bottom = target_w - content_w, target_h - content_h
        mode = "edge" if right >= content_w or bottom >= content_h else "reflect"
        padded = transforms_f.pad(padded, [0, 0, right, bottom], padding_mode=mode)
    image_dimensions = torch.tensor([target_h, target_w, content_h, content_w], dtype=torch.float)
    return PreparedCompositeFrame(source, padded, image_dimensions)


def current_latent_to_visual96(z0: Any) -> Any:
    """Preserve the full-canvas fp32 Visual96 reduction, without latent cropping."""
    import torch
    from torch.nn import functional as functional

    if not isinstance(z0, torch.Tensor) or z0.dtype != torch.float32 or z0.ndim != 3 or z0.shape[0] != 48:
        raise ValueError("current z0 must be fp32 [48,H,W]")
    if min(z0.shape[1:]) <= 0 or not bool(torch.isfinite(z0).all()):
        raise ValueError("current z0 must be finite and nonempty")
    return functional.adaptive_avg_pool2d(z0.unsqueeze(0), output_size=(1, 2)).flatten()


class LoadedWanEncode1:
    """Callable ``list[Frame] -> list[384-byte Visual96]`` for one loaded encoder.

    The Worker owns serialization; callers must not invoke the same adapter
    concurrently. Eager identity requires no installed AOT functions. ``aot_b1``
    requires a matching existing B1 artifact and never accepts a larger batch.
    This adapter neither compiles nor removes the tokenizer's B1 AOT guard.
    """

    def __init__(self, tokenizer: Any, *, identity: Mapping[str, str], image_size: int = 256):
        if set(identity) != _IDENTITY_KEYS:
            raise ValueError("exact six-field loaded VAE identity required")
        self.identity = MappingProxyType(dict(identity))
        for key in _IDENTITY_KEYS - {"encoder_backend", "encoder_dtype"}:
            value = self.identity[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("invalid loaded VAE identity digest")
        if self.identity["encoder_dtype"] != "bfloat16" or self.identity["encoder_backend"] not in ("eager", "aot_b1"):
            raise ValueError("only declared Wan bf16 eager/aot_b1 encoding is supported")
        if self.identity["preprocess_profile_sha256"] != preprocess_profile_digest(image_size):
            raise ValueError("loaded preprocessing identity differs from image_size/profile")
        if self.identity["visual96_contract_sha256"] != visual96_contract_digest():
            raise ValueError("loaded Visual96 identity differs from complete pre-crop contract")
        self.tokenizer, self.image_size = tokenizer, image_size
        self._loaded = getattr(tokenizer, "model", None)
        self._module = getattr(self._loaded, "model", None)
        self._validate_tokenizer()

    def _validate_tokenizer(self) -> None:
        import torch

        tokenizer = self.tokenizer
        if (
            self.identity["preprocess_profile_sha256"] != preprocess_profile_digest(self.image_size)
            or self.identity["visual96_contract_sha256"] != visual96_contract_digest()
        ):
            raise RuntimeError("encoder preprocessing configuration changed after attestation")
        if (
            not callable(getattr(tokenizer, "encode", None))
            or getattr(tokenizer, "model", None) is not self._loaded
            or getattr(self._loaded, "model", None) is not self._module
            or getattr(tokenizer, "is_causal", None) is not True
            or getattr(tokenizer, "use_streaming_encode", None) is not False
            or getattr(tokenizer, "_keep_encoder_cache", None) is not False
            or getattr(tokenizer, "spatial_compression_factor", None) != 16
            or getattr(tokenizer, "temporal_compression_factor", None) != 4
            or getattr(self._loaded, "dtype", None) != torch.bfloat16
            or getattr(self._module, "training", None) is not False
        ):
            raise RuntimeError("already-loaded frozen causal non-streaming Wan bf16 tokenizer required")
        parameters = tuple(self._module.parameters())
        if not parameters or any(
            parameter.requires_grad or parameter.dtype != torch.bfloat16 for parameter in parameters
        ):
            raise RuntimeError("Wan encoder must have frozen bf16 parameters")
        self._device = parameters[0].device
        if any(parameter.device != self._device for parameter in parameters):
            raise RuntimeError("one Worker encoder device required")

    def __call__(self, frames: Sequence[Any]) -> list[bytes]:
        from cosmos_framework.inference.robocasa_vae_protocol import EncoderDrainUnproven

        # Returning CPU bytes proves the normal-path transfer completed. A CUDA
        # failure does not prove outstanding device work ended: do not turn a
        # Python exception into a positive End/drain receipt.
        try:
            return self._encode_batch(frames)
        except BaseException as error:
            if getattr(self._device, "type", None) != "cpu":
                raise EncoderDrainUnproven(type(error).__name__) from error
            raise

    def _encode_batch(self, frames: Sequence[Any]) -> list[bytes]:
        import numpy as np
        import torch

        from cosmos_framework.inference.robocasa_feature_trace import get_trace
        from cosmos_framework.inference.robocasa_vae_protocol import Frame, source_digest, validate_frame

        trace, prepare_started = get_trace(), time.monotonic_ns()
        self._validate_tokenizer()
        if not isinstance(frames, Sequence) or not 1 <= len(frames) <= 8:
            raise ValueError("independent Encode1 microbatch must contain 1..8 frames")
        frames = tuple(frames)
        if self.identity["encoder_backend"] == "aot_b1" and len(frames) != 1:
            raise ValueError("B4/B8 and partial multi-sample batches require eager identity")
        prepared = []
        for frame in frames:
            if not isinstance(frame, Frame):
                raise TypeError("encoder accepts only validated Worker Frame objects")
            frame = validate_frame(frame)
            header = frame.header
            if any(header.get(key) != value for key, value in self.identity.items()):
                raise ValueError("Frame VAE/preprocessing identity differs from loaded encoder")
            if (
                type(frame.rgb) is not bytes
                or len(frame.rgb) != RGB_BYTES
                or header.get("rgb_shape") != list(RGB_SHAPE)
                or header.get("rgb_layout") != "HWC_UINT8_RGB"
                or header.get("source_rgb_contract") != "PSM23-RGB-v1_HWC_UINT8_RGB"
            ):
                raise ValueError("Frame requires immutable canonical HWC uint8 RGB 256x512x3")
            if header.get("source_rgb_sha256") != source_digest(frame.rgb):
                raise ValueError("Frame RGB digest differs before preprocessing")
            array = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(RGB_SHAPE).copy()
            composite = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            prepared.append(prepare_composite_frame(composite, self.image_size))
        shape = prepared[0].padded_single_frame.shape
        if any(item.padded_single_frame.shape != shape for item in prepared):
            raise ValueError("only matching prepared H/W may share an Encode1 microbatch")
        aot = getattr(self._module, "_aot_chunk_fns", None)
        if self.identity["encoder_backend"] == "eager" and aot:
            raise RuntimeError("eager identity cannot silently use an installed B1 AOT artifact")
        key = (1, shape[-2] // 2, shape[-1] // 2, 0)
        if self.identity["encoder_backend"] == "aot_b1" and (not isinstance(aot, dict) or key not in aot):
            raise RuntimeError("declared aot_b1 requires the actual matching prime artifact")
        # Stacking introduces only B; time remains exactly one for every Session.
        pixels = torch.stack([item.padded_single_frame for item in prepared])
        normalized = pixels.to(device=self._device, dtype=torch.float32)
        normalized.div_(127.5).sub_(1.0)
        if trace:
            trace.record("history_vae_preprocess_copy", prepare_started, time.monotonic_ns(), producer="worker")
        with (
            torch.no_grad(),
            trace.span("history_vae", producer="worker") if trace else nullcontext(),
            trace.cuda_span("history_vae", device=self._device, producer="worker") if trace else nullcontext(),
        ):
            latent = self.tokenizer.encode(normalized)
        expected = (len(frames), 48, 1, shape[-2] // 16, shape[-1] // 16)
        if (
            not isinstance(latent, torch.Tensor)
            or latent.shape != expected
            or latent.dtype != torch.float32
            or latent.device != self._device
            or not bool(torch.isfinite(latent).all())
        ):
            raise ValueError("Wan Encode1 must return the complete finite fp32 [B,48,1,H_z,W_z] latent")
        self._validate_tokenizer()
        result = []
        for index in range(len(frames)):
            visual = current_latent_to_visual96(latent[index, :, 0].float().detach()).detach().cpu()
            values = visual.tolist()
            if len(values) != 96 or not all(math.isfinite(value) for value in values):
                raise FloatingPointError("Visual96 must remain finite")
            result.append(struct.pack("<96f", *values))
        return result

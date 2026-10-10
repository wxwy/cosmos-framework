# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One-byte shape proxies for the strict offline cached-vision training seam.

No pixel values enter VAE/vision normalization. The logical Tensor ABI remains
uint8 [3,17,H,W] for unchanged native geometry/temporal helpers; its storage is
one byte, it is never resized and it stays on CPU at the native batch boundary.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import torch

SOURCE_SHAPE = (3, 17, 256, 512)


def compact_cached_video_enabled() -> bool:
    value = os.environ.get("PSM_V3_COMPACT_CACHED_VIDEO", "1")
    if value not in {"0", "1"}:
        raise ValueError("PSM_V3_COMPACT_CACHED_VIDEO must be 0 or 1")
    return value == "1"


def pixel_shape_proxy(shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) != 4 or shape[:2] != (3, 17) or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("cached proxy requires positive [3,17,H,W]")
    # Each sample owns its scalar; never share writable storage across samples.
    return torch.zeros((), dtype=torch.uint8).expand(shape)


def is_pixel_shape_proxy(value: object) -> bool:
    return (
        isinstance(value, torch.Tensor) and value.device.type == "cpu"
        and value.dtype == torch.uint8 and value.ndim == 4
        and tuple(value.shape[:2]) == (3, 17) and value.storage_offset() == 0
        and all(v > 0 for v in value.shape) and value.stride() == (0, 0, 0, 0)
        and value.untyped_storage().nbytes() == 1
        and int(value[0, 0, 0, 0]) == 0
    )


def cached_pixel_placeholder(*, compact: bool) -> torch.Tensor:
    return pixel_shape_proxy(SOURCE_SHAPE) if compact else torch.zeros(SOURCE_SHAPE, dtype=torch.uint8)


class CachedGeometryResize:
    """Replace only pixel work, using canvas metadata computed by official VideoResize.

    The supplied canvas is checked by RoboCasaExactWindowCachedDataset against
    the cache/tokenizer. Prompt formatting, tokenization, action processing and
    SequencePlan still run through the original ActionTransformPipeline.
    """

    def __init__(self, original: Callable[..., dict], canvas: tuple[int, int, int, int]) -> None:
        if len(canvas) != 4 or any(type(v) is not int or v <= 0 for v in canvas):
            raise ValueError("invalid official canvas")
        th, tw, ch, cw = canvas
        if ch > th or cw > tw:
            raise ValueError("content exceeds canvas")
        self.original, self.canvas = original, canvas

    def __call__(self, sample: dict[str, Any], resolution: object) -> dict[str, Any]:
        video = sample.get("video")
        if not is_pixel_shape_proxy(video):
            # Original dense/noncache samples retain exactly the original path.
            if isinstance(video, torch.Tensor) and video.ndim == 4 and video.stride() == (0, 0, 0, 0):
                raise ValueError("invalid cached pixel proxy")
            return self.original(sample, resolution)
        if sample.get("cached_latent_required") is not True or "video_latent" not in sample:
            raise ValueError("shape-only pixels require verified cached latent; no online fallback")
        if resolution is not None or tuple(video.shape) != SOURCE_SHAPE:
            raise ValueError("compact cached input disagrees with frozen source geometry/resolution")
        sample["video"] = pixel_shape_proxy((3, 17, self.canvas[0], self.canvas[1]))
        sample["image_size"] = torch.tensor(self.canvas, dtype=torch.float32)
        return sample


def move_cached_native_batch(batch: Mapping[str, Any], *, device: object, move: Callable[..., Any]) -> dict:
    """Avoid H2D for validated shape-only proxies, without changing misc.to globally."""
    samples = batch.get("video")
    items = [s[0] for s in samples] if (
        isinstance(samples, (list, tuple)) and samples
        and all(isinstance(s, (list, tuple)) and len(s) == 1 for s in samples)
    ) else []
    proxies = [is_pixel_shape_proxy(v) for v in items]
    if any(isinstance(v, torch.Tensor) and v.ndim == 4 and v.stride() == (0, 0, 0, 0)
           and not valid for v, valid in zip(items, proxies, strict=True)):
        raise ValueError("invalid shape-only pixel tensor")
    if not any(proxies):
        return move(batch, device=device)
    if not all(proxies):
        raise ValueError("mixed compact/dense cached video batches are not supported")
    markers, latents, sizes = (batch.get(k) for k in ("cached_latent_required", "video_latent", "image_size"))
    if any(not isinstance(v, (list, tuple)) or len(v) != len(items) for v in (markers, latents, sizes)):
        raise ValueError("compact cached batch metadata/latent count mismatch")
    for video, marker, latent, size in zip(items, markers, latents, sizes, strict=True):
        if not isinstance(marker, torch.Tensor) or marker.dtype != torch.bool or marker.shape != (1,) or not bool(marker.item()):
            raise ValueError("compact pixels require a true cache marker per sample")
        if not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or latent.ndim != 5 or tuple(latent.shape[:3]) != (1, 5, 48):
            raise ValueError("compact cached latent ABI mismatch")
        if not isinstance(size, torch.Tensor) or size.numel() != 4 or not torch.isfinite(size).all():
            raise ValueError("compact cached image_size missing/nonfinite")
        flat = size.reshape(-1)
        if not torch.equal(flat, flat.trunc()) or tuple(int(v) for v in flat[:2]) != tuple(video.shape[-2:]):
            raise ValueError("compact cached canvas mismatch")
    result = move({k: v for k, v in batch.items() if k != "video"}, device=device)
    result["video"] = samples
    return result

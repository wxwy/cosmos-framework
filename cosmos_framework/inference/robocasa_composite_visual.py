# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected RoboCasa composite spatial prep and independent single-frame Encode1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.utils.transforms import find_closest_target_size, reflection_pad_to_target
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import robocasa_current_latent_to_visual96
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


@dataclass(frozen=True)
class PreparedCompositeFrame:
    source_uint8: torch.Tensor
    padded_single_frame: torch.Tensor
    padded_image_size: torch.Tensor


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
    tokenizer: object, prepared: PreparedCompositeFrame
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
        encoded = tokenizer.encode(normalized)
    if encoded.ndim != 5 or encoded.shape[:3] != (1, 48, 1):
        raise ValueError("Encode1 必须返回 [1,48,1,H_z,W_z]")
    z0 = encoded[0, :, 0].float().detach().clone()
    if not bool(torch.isfinite(z0).all()):
        raise FloatingPointError("Encode1 latent 非有限")
    return z0, robocasa_current_latent_to_visual96(z0)

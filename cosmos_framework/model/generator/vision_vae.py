"""训练与离线 LIBERO latent cache 共用的 VAE 编码契约。"""

from __future__ import annotations

from typing import Protocol

import torch


# 与 action_policy_libero_edge_all.py 的 tokenizer 配置一致。exact-window
# cache 的构建、probe 和 parity 必须使用这一组值，因果 VAE 的分块不同会改变结果。
LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS = [17, 61, 73]
LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 8, "768": 8}


class VisionVAEEncoder(Protocol):
    """公共编码器最小协议，避免离线工具依赖完整训练模型。"""

    tensor_kwargs_fp32: dict[str, object]

    def encode(self, state: torch.Tensor) -> torch.Tensor: ...


def normalize_uint8_vision_item(state: torch.Tensor, *, tensor_kwargs_fp32: dict[str, object]) -> torch.Tensor:
    """将 uint8 ``[...,C,T,H,W]`` 转为 fp32 ``[-1, 1]``。"""
    if state.dtype != torch.uint8:
        raise ValueError(f"Per-camera VAE encoding requires uint8 pixels, got {state.dtype}.")
    normalized_state = state.to(**tensor_kwargs_fp32)
    normalized_state.div_(127.5).sub_(1.0)
    return normalized_state


def encode_uint8_vision_item(
    encoder: VisionVAEEncoder,
    state: torch.Tensor,
    *,
    num_views: int = 1,
    frames_per_view: int | None = None,
) -> torch.Tensor:
    """编码原始 uint8 RGB，返回 fp32 ``[...,C_latent,T_latent,H,W]``。

    ``frames_per_view`` 为 ``None`` 时视为一个已空间拼接的单视图视频；否则
    输入沿时间维按 camera-major 排列，每个相机独立归一化、编码后再拼回。
    """
    if frames_per_view is None:
        if num_views != 1:
            raise ValueError("frames_per_view is required when num_views is greater than one.")
        normalized = normalize_uint8_vision_item(state, tensor_kwargs_fp32=encoder.tensor_kwargs_fp32)
        return encoder.encode(normalized).contiguous().float()

    temporal_dim = state.ndim - 3
    expected_frames = num_views * frames_per_view
    actual_frames = int(state.shape[temporal_dim])
    if actual_frames != expected_frames:
        raise ValueError(
            "Multiview vision length must equal num_views * frames_per_view: "
            f"got T={actual_frames}, num_views={num_views}, frames_per_view={frames_per_view}."
        )

    encoded_views: list[torch.Tensor] = []
    for view_idx in range(num_views):
        view_state = state.narrow(temporal_dim, view_idx * frames_per_view, frames_per_view)
        normalized_view = normalize_uint8_vision_item(view_state, tensor_kwargs_fp32=encoder.tensor_kwargs_fp32)
        encoded_views.append(encoder.encode(normalized_view).contiguous().float())
    return torch.cat(encoded_views, dim=temporal_dim)

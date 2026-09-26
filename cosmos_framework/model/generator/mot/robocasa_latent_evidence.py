# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa B1 的离线 latent evidence；不导入 VAE 或 policy。"""

from __future__ import annotations

import bisect
from pathlib import Path

import h5py
import numpy as np
import torch


def causal_endpoint_index(endpoint_indices: tuple[int, ...], source_step: int) -> int:
    """返回不晚于 source_step 的最新 endpoint 下标；绝不使用 nearest/ceil。"""
    if type(source_step) is not int or source_step < 0:
        raise ValueError("source_step 必须是非负 episode-frame 整数")
    if not endpoint_indices or endpoint_indices[0] != 0:
        raise ValueError("endpoint 必须从0开始")
    if any(type(value) is not int or value != index * 4 for index, value in enumerate(endpoint_indices)):
        raise ValueError("endpoint 必须严格为 0,4,8,...")
    return bisect.bisect_right(endpoint_indices, source_step) - 1


def latent_to_visual96(latent: torch.Tensor) -> torch.Tensor:
    """fp16 [48,16,16] → fp32 mean48+RMS48；无参数、无时间混合。"""
    if latent.shape != (48, 16, 16) or latent.dtype != torch.float16 or latent.device.type != "cpu":
        raise ValueError("缓存 latent 必须是 CPU fp16 [48,16,16]")
    if not torch.isfinite(latent).all():
        raise ValueError("缓存 latent 含非有限值")
    value = latent.detach().float()
    mean = value.mean(dim=(1, 2))
    rms = (value.square().mean(dim=(1, 2)) + 1e-6).sqrt()
    summary = torch.cat((mean, rms))
    if not torch.isfinite(summary).all():
        raise ValueError("visual96 含非有限值")
    return summary


class RoboCasaLatentReader:
    """显式 H5 keys，严格读取 group attrs；不猜布局，不执行 VAE fallback。

    每次绑定一个完整 episode。分帧校验全部 latent，仅保留 fp32 visual96，
    不提供可接入 policy 的 cached video latent payload。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_episode_id: str,
        expected_source_frames: int,
        latent_key: str,
        endpoint_key: str,
        metadata_group: str,
    ) -> None:
        if not isinstance(expected_episode_id, str) or not expected_episode_id:
            raise ValueError("expected_episode_id 必须是非空的源 episode 身份")
        if type(expected_source_frames) is not int or expected_source_frames <= 0:
            raise ValueError("expected_source_frames 必须来自源视频的正整数帧数")
        if not all(isinstance(key, str) and key for key in (latent_key, endpoint_key, metadata_group)):
            raise ValueError("必须显式提供 H5 latent/endpoint 路径及 metadata group")
        self.path = Path(path)
        with h5py.File(self.path, "r") as cache:
            metadata = cache[metadata_group]
            if not isinstance(metadata, h5py.Group):
                raise ValueError("metadata_group 必须是带权威 attrs 的 H5 group")
            attrs = metadata.attrs
            episode_id = attrs["episode_id"]
            policy = attrs["source_frame_to_latent_policy"]
            if isinstance(episode_id, bytes):
                episode_id = episode_id.decode("utf-8")
            if isinstance(policy, bytes):
                policy = policy.decode("utf-8")
            frames = attrs["source_video_frames"]
            compression = attrs["temporal_compression_factor"]
            if episode_id != expected_episode_id:
                raise ValueError("cache/episode identity 不匹配")
            if (
                not isinstance(frames, (int, np.integer))
                or isinstance(frames, (bool, np.bool_))
                or frames != expected_source_frames
            ):
                raise ValueError("cache/source video 帧数不匹配")
            if not isinstance(compression, (int, np.integer)) or compression != 4 or policy != "causal_endpoint":
                raise ValueError("缓存必须声明 temporal_compression_factor=4 和 causal_endpoint")
            latent, endpoints = cache[latent_key], cache[endpoint_key]
            if not isinstance(latent, h5py.Dataset) or not isinstance(endpoints, h5py.Dataset):
                raise ValueError("latent/endpoint keys 必须指向 H5 datasets")
            if latent.ndim != 4 or latent.shape[1:] != (48, 16, 16) or latent.dtype != np.dtype("float16"):
                raise ValueError("episode latent 必须是 fp16 [N,48,16,16]")
            if endpoints.ndim != 1 or endpoints.dtype.kind not in "iu" or len(endpoints) != len(latent):
                raise ValueError("endpoint 必须为与 latent 等长的一维整数数组")
            indices = tuple(int(value) for value in endpoints[:])
            if indices != tuple(range(0, expected_source_frames, 4)):
                raise ValueError("endpoint 必须覆盖本 episode 的 0,4,8,... 且不可越界、缺失或乱序")
            # 完整校验包括未被当前 consumer 选中的帧，非有限缓存立即拒绝。
            summaries = torch.stack(
                [latent_to_visual96(torch.from_numpy(latent[index])) for index in range(len(latent))]
            )
        self.episode_id = expected_episode_id
        self.source_frames = expected_source_frames
        self.endpoint_indices = indices
        self._summaries = summaries

    def visual_summary(self, source_step: int) -> tuple[int, torch.Tensor]:
        if type(source_step) is not int or not 0 <= source_step < self.source_frames:
            raise ValueError("source_step 超出源 episode frame 范围")
        index = causal_endpoint_index(self.endpoint_indices, source_step)
        return self.endpoint_indices[index], self._summaries[index].clone()

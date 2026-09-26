# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa B1 的离线 latent evidence；不导入 VAE 或 policy。"""

from __future__ import annotations

import bisect
from pathlib import Path

import h5py
import numpy as np
import torch

LEFT_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"


def causal_endpoint_index(endpoint_indices: tuple[int, ...], source_step: int) -> int:
    """返回不晚于 source_step 的最新 endpoint 下标；绝不使用 nearest/ceil。"""
    if type(source_step) is not int or source_step < 0:
        raise ValueError("source_step 必须是非负 episode-frame 整数")
    if not endpoint_indices or endpoint_indices[0] != 0:
        raise ValueError("endpoint 必须从0开始")
    if any(type(value) is not int for value in endpoint_indices) or any(
        left >= right for left, right in zip(endpoint_indices, endpoint_indices[1:])
    ):
        raise ValueError("endpoint 必须严格递增；完整4-grid加terminal合同由 reader 校验")
    return bisect.bisect_right(endpoint_indices, source_step) - 1


def latent_to_visual96(left: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    """left+wrist 按 width 融合为 fp32 [48,16,32] 后取 mean48+RMS48。"""
    for latent in (left, wrist):
        if latent.shape != (48, 16, 16) or latent.dtype != torch.float16 or latent.device.type != "cpu":
            raise ValueError("两路缓存 latent 必须是 CPU fp16 [48,16,16]")
        if not torch.isfinite(latent).all():
            raise ValueError("缓存 latent 含非有限值")
    value = torch.cat((left.detach().float(), wrist.detach().float()), dim=-1)
    mean = value.mean(dim=(1, 2))
    rms = (value.square().mean(dim=(1, 2)) + 1e-6).sqrt()
    summary = torch.cat((mean, rms))
    if not torch.isfinite(summary).all():
        raise ValueError("visual96 含非有限值")
    return summary


class RoboCasaLatentReader:
    """固定 root attrs 和 Stage-A left_wrist H5 schema，不允许调用方选 camera。

    每次绑定一个完整 episode。分帧校验全部 latent，仅保留 fp32 visual96，
    不提供可接入 policy 的 cached video latent payload。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_episode_id: str,
        expected_source_frames: int,
    ) -> None:
        if not isinstance(expected_episode_id, str) or not expected_episode_id:
            raise ValueError("expected_episode_id 必须是非空的源 episode 身份")
        if type(expected_source_frames) is not int or expected_source_frames <= 0:
            raise ValueError("expected_source_frames 必须来自源视频的正整数帧数")
        self.path = Path(path)
        with h5py.File(self.path, "r") as cache:
            attrs = cache.attrs
            episode_id = attrs["episode_id"]
            policy = attrs["source_frame_to_latent_policy"]
            if isinstance(episode_id, bytes):
                episode_id = episode_id.decode("utf-8")
            if isinstance(policy, bytes):
                policy = policy.decode("utf-8")
            frames = attrs["frame_count"]
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
            indices = tuple(range(0, expected_source_frames, 4))
            if indices[-1] != expected_source_frames - 1:
                indices += (expected_source_frames - 1,)
            latents = []
            for camera in (LEFT_CAMERA, WRIST_CAMERA):
                latent = cache[f"latents/{camera}"]
                endpoints = cache[f"indices/latent_source_frame_indices/{camera}"]
                valid = cache[f"valid/{camera}"]
                if not all(isinstance(value, h5py.Dataset) for value in (latent, endpoints, valid)):
                    raise ValueError(f"{camera} 的 latent/endpoint/valid 必须是 datasets")
                if latent.shape != (len(indices), 48, 16, 16) or latent.dtype != np.dtype("float16"):
                    raise ValueError(f"{camera} latent 必须是 fp16 [N,48,16,16]")
                if endpoints.shape != (len(indices),) or endpoints.dtype.kind not in "iu":
                    raise ValueError(f"{camera} endpoint 必须为等长的一维整数数组")
                if tuple(int(value) for value in endpoints[:]) != indices:
                    raise ValueError(f"{camera} endpoint 必须精确等于4-grid加必要的terminal F-1")
                if valid.shape != (len(indices),) or valid.dtype != np.dtype("bool") or not valid[:].all():
                    raise ValueError(f"{camera} valid 必须为 bool [N] 且全部为真")
                latents.append(latent)
            # 完整校验包括未被当前 consumer 选中的帧，非有限缓存立即拒绝。
            summaries = torch.stack(
                [
                    latent_to_visual96(torch.from_numpy(latents[0][index]), torch.from_numpy(latents[1][index]))
                    for index in range(len(indices))
                ]
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

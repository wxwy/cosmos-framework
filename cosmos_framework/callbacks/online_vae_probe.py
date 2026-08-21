# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""训练时采集 concat-view 在线 VAE 的 uint8 输入与 latent 输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils import distributed, log


class OnlineVAEProbeCallback(Callback):
    """在真实训练 batch 上记录离线 cache 的在线 VAE 基准。

    仅支持当前 LIBERO concat-view 的单视频项输入。采样在 batch 开始时复制
    归一化前的 uint8 像素，随后包装 ``_encode_vision_item`` 保存同一次调用的
    online latent；因此不改变模型的编码契约。
    """

    def __init__(self, output_dir: str, max_samples: int = 200) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self._pending: list[tuple[torch.Tensor, dict[str, Any]]] = []
        self._saved = 0

    @staticmethod
    def _unwrap_one(value: Any) -> torch.Tensor | None:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
        return value if isinstance(value, torch.Tensor) else None

    @staticmethod
    def _metadata_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1)
            return flat.item() if flat.numel() == 1 else flat.tolist()
        if isinstance(value, (list, tuple)):
            return [OnlineVAEProbeCallback._metadata_value(item) for item in value]
        return value

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self.max_samples <= 0:
            return
        rank = distributed.get_rank()
        self.output_dir = self.output_dir / f"rank_{rank:02d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        original_encode = model._encode_vision_item

        def wrapped_encode(state: torch.Tensor, *, num_views: int, frames_per_view: int | None) -> torch.Tensor:
            latent = original_encode(state, num_views=num_views, frames_per_view=frames_per_view)
            if self._pending:
                raw_uint8, metadata = self._pending.pop(0)
                self._save(raw_uint8, latent, metadata, num_views=num_views, frames_per_view=frames_per_view)
            return latent

        model._encode_vision_item = wrapped_encode
        log.info(f"OnlineVAEProbe enabled: output={self.output_dir}, max_samples={self.max_samples}")

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        del model
        if self._saved + len(self._pending) >= self.max_samples:
            return
        videos = data.get("video")
        if not isinstance(videos, list):
            return
        metadata = {
            "iteration": iteration,
            "episode_index": self._metadata_value(data.get("episode_index")),
            "start_frame": self._metadata_value(data.get("start_frame")),
            "task_index": self._metadata_value(data.get("task_index")),
            "dataset_name": self._metadata_value(data.get("dataset_name")),
        }
        verify_flags = data.get("verify_cached_latent")
        for video_index, video in enumerate(videos):
            if self._saved + len(self._pending) >= self.max_samples:
                break
            if "video_latent" in data and isinstance(verify_flags, list) and video_index < len(verify_flags):
                verify_flag = verify_flags[video_index]
                if isinstance(verify_flag, torch.Tensor):
                    verify_flag = bool(verify_flag.detach().reshape(-1)[0].item())
                if not bool(verify_flag):
                    continue
            raw_uint8 = self._unwrap_one(video)
            if raw_uint8 is None or raw_uint8.dtype != torch.uint8:
                continue
            self._pending.append((raw_uint8.detach().cpu().clone(), metadata.copy()))

    def _save(
        self,
        raw_uint8: torch.Tensor,
        latent: torch.Tensor,
        metadata: dict[str, Any],
        *,
        num_views: int,
        frames_per_view: int | None,
    ) -> None:
        if self._saved >= self.max_samples:
            return
        sample_dir = self.output_dir / f"sample_{self._saved:06d}"
        sample_dir.mkdir()
        torch.save(raw_uint8, sample_dir / "raw_uint8.pt")
        torch.save(latent.detach().float().cpu(), sample_dir / "online_latent.pt")
        metadata.update(
            num_views=num_views,
            frames_per_view=frames_per_view,
            raw_shape=list(raw_uint8.shape),
            raw_dtype=str(raw_uint8.dtype),
            latent_shape=list(latent.shape),
            latent_dtype=str(latent.dtype),
        )
        (sample_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._saved += 1

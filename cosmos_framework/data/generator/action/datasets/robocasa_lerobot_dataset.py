"""RoboCasa LeRobot v2.1 单任务视频/动作读取与 exact-window cache 读取。"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Reserved, Rot, build_action_spec
from cosmos_framework.model.generator.vision_vae import (
    LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES,
    LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS,
)

CameraMode = Literal["wrist_top_agentview_lr_bottom"]
_VIDEO_KEYS = (
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)
_CAMERA_MODE = "wrist_top_agentview_lr_bottom"


def robocasa_task_identity(root: Path) -> tuple[str, str]:
    """从 ``.../<atomic|composite>/<Task>/<date>/lerobot`` 生成稳定 task id/slug。"""
    root = root.resolve()
    if root.name != "lerobot" or len(root.parents) < 4:
        raise ValueError(f"Expected RoboCasa .../<category>/<task>/<date>/lerobot root, got {root}")
    category, task = root.parents[2].name, root.parents[1].name
    if category not in {"atomic", "composite"}:
        raise ValueError(f"Expected atomic/composite RoboCasa category in {root}, got {category!r}")
    return f"{category}/{task}", f"{category}__{task}"


class RoboCasaLeRobotDataset(ActionBaseDataset):
    """一个 RoboCasa 任务根；action/state/annotation 保持其 LeRobot 原始位置。"""

    def __init__(
        self,
        root: str,
        *,
        fps: float = 20.0,
        chunk_length: int = 16,
        mode: str = "wam",
        tolerance_s: float = 1e-4,
        camera_mode: CameraMode = _CAMERA_MODE,
        image_size: int = 256,
        sample_stride: int = 1,
        latent_cache_root: str | None = None,
        latent_cache_verify_ratio: float = 0.0,
    ) -> None:
        if camera_mode != _CAMERA_MODE:
            raise ValueError(f"Unsupported RoboCasa camera_mode={camera_mode!r}")
        if chunk_length != 16:
            raise ValueError(f"RoboCasa exact-window cache requires chunk_length=16, got {chunk_length}")
        super().__init__(
            root=root,
            domain_name="robocasa_panda_omron",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",
            tolerance_s=tolerance_s,
            viewpoint=_CAMERA_MODE,
            action_normalization=None,
            sample_stride=sample_stride,
        )
        if int(self._info.get("fps", fps)) != int(fps):
            raise ValueError(f"RoboCasa fps mismatch: requested={fps}, source={self._info.get('fps')}")
        self._camera_mode = camera_mode
        self._image_size = int(image_size)
        self.task_id, self.task_slug = robocasa_task_identity(self._root)
        self._latent_cache_root = Path(latent_cache_root) if latent_cache_root else None
        self._latent_cache_verify_ratio = float(latent_cache_verify_ratio)
        if not 0.0 <= self._latent_cache_verify_ratio <= 1.0:
            raise ValueError("latent_cache_verify_ratio must be in [0,1]")
        self._latent_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        index_parts, episode_parts, task_parts, timestamp_parts, action_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/episode_*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", "action"])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            timestamp_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No RoboCasa episode parquet files under {self._root / 'data'}")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(timestamp_parts).astype(np.float64)[order]
        self._row_action = np.concatenate(action_parts, axis=0).astype(np.float32)[order]
        if self._row_action.ndim != 2 or self._row_action.shape[1] != 12:
            raise ValueError(f"Expected RoboCasa action [N,12], got {self._row_action.shape}")
        self._ep_vals, self._ep_starts, counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._valid_cum = np.cumsum(np.maximum(0, counts - self._chunk_length)).astype(np.int64)
        if self._latent_cache_root is not None:
            self._validate_latent_cache_manifest()

    @property
    def action_dim(self) -> int:
        return 12

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Reserved(5, "base_and_mode"), Pos(prefix="eef"), Rot("axisangle", "eef"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        raise RuntimeError("RoboCasa action_normalization=None; no action stats are loaded")

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def _compose_video(self, frames: dict[str, torch.Tensor]) -> torch.Tensor:
        wrist, left, right = (frames[key] for key in _VIDEO_KEYS)
        if any(frame.dtype != torch.float32 for frame in (wrist, left, right)):
            raise ValueError("RoboCasa decoder must compose views in float32 [0,1]")
        half = self._image_size // 2
        left = F.interpolate(left, size=(half, half), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half, half), mode="bilinear", align_corners=False)
        return torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        frames: dict[str, torch.Tensor] = {}
        for key in _VIDEO_KEYS:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            value = decode_video_frames(self._video_path(episode, key), [from_ts + ts for ts in timestamps], self._tolerance_s)
            frames[key] = F.interpolate(value, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
        return self._compose_video(frames)

    def _validate_latent_cache_manifest(self) -> None:
        assert self._latent_cache_root is not None
        path = self._latent_cache_root / "dataset_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "exact_window_v1",
            "suite": f"robocasa365_{self.task_id.split('/', 1)[0]}",
            "chunk_length": self._chunk_length,
            "camera_mode": self._camera_mode,
            "sample_stride": self._sample_stride,
            "fps": self._fps,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"RoboCasa latent cache {key} mismatch: cache={manifest.get(key)!r}, dataset={value!r}")
        contract = {"compute_dtype": "torch.bfloat16", "encode_exact_durations": LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS, "encode_chunk_frames": LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES}
        if manifest.get("vae_encode_contract") != contract:
            raise ValueError("RoboCasa latent cache VAE encoding contract mismatch")
        self._latent_cache_expected_shape = tuple(manifest.get("latent_shape", []))
        if len(self._latent_cache_expected_shape) != 4 or self._latent_cache_expected_shape[:2] != (5, 48):
            raise ValueError(f"Invalid RoboCasa cache latent_shape={self._latent_cache_expected_shape}")

    def _load_cached_latent(self, episode_index: int, start_frame: int) -> torch.Tensor | None:
        if self._latent_cache_root is None:
            return None
        item = self._latent_cache.get(episode_index)
        if item is None:
            path = self._latent_cache_root / "tasks" / self.task_slug / "episodes" / f"episode_{episode_index:06d}.pt"
            item = torch.load(path, map_location="cpu", weights_only=True)
            if item.get("format") != "exact_window_v1" or item.get("task_id") != self.task_id:
                raise ValueError(f"Invalid RoboCasa cache task payload: {path}")
            self._latent_cache[episode_index] = item
            if len(self._latent_cache) > 8:
                self._latent_cache.popitem(last=False)
        window = item.get("windows", {}).get(str(start_frame))
        if not isinstance(window, dict) or not isinstance(window.get("latent"), torch.Tensor):
            raise KeyError(f"Missing RoboCasa cache key {(self.task_id, episode_index, start_frame)}")
        latent = window["latent"].contiguous()
        expected_window = torch.arange(start_frame, start_frame + 17, dtype=torch.long)
        expected_anchors = torch.arange(start_frame, start_frame + 17, 4, dtype=torch.long)
        if latent.dtype != torch.float32 or tuple(latent.shape) != self._latent_cache_expected_shape or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid latent for RoboCasa cache key {(self.task_id, episode_index, start_frame)}")
        if not torch.equal(window.get("window_frame_indices"), expected_window) or not torch.equal(window.get("latent_source_frame_indices"), expected_anchors):
            raise ValueError(f"Invalid window metadata for RoboCasa cache key {(self.task_id, episode_index, start_frame)}")
        return latent

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep = int(np.searchsorted(self._valid_cum, int(idx), side="right"))
        previous = int(self._valid_cum[ep - 1]) if ep else 0
        start = int(self._ep_starts[ep]) + (int(idx) - previous)
        episode_index = int(self._ep_vals[ep])
        local_start = start - int(self._ep_starts[ep])
        latent = self._load_cached_latent(episode_index, local_start)
        verify = latent is not None and random.random() < self._latent_cache_verify_ratio
        if latent is None or verify:
            timestamps = [float(v) for v in self._row_timestamp[start : start + self._chunk_length + 1]]
            video = self._load_video(self._episodes[episode_index], timestamps)
        else:
            video = torch.zeros((17, 3, self._image_size * 3 // 2, self._image_size), dtype=torch.float32)
        task = self._tasks[int(self._row_task[start])]
        result = self._build_result(mode=self._choose_mode(), video=video, action=torch.from_numpy(self._row_action[start : start + 16].copy()), ai_caption=task)
        result.update({
            "video_latent": latent,
            "verify_cached_latent": torch.tensor(verify, dtype=torch.bool),
            "suite": f"robocasa365_{self.task_id.split('/', 1)[0]}",
            "task_id": self.task_id,
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "start_frame": torch.tensor(local_start, dtype=torch.long),
            "task_index": torch.tensor(int(self._row_task[start]), dtype=torch.long),
            "window_frame_indices": torch.arange(local_start, local_start + 17),
            "latent_source_frame_indices": torch.arange(local_start, local_start + 17, 4),
            "additional_view_description": "The top view is the eye-in-hand camera. The bottom row is agentview left then agentview right.",
        })
        return result

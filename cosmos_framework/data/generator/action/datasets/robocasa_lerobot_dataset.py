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

CameraSet = Literal["left_wrist", "wrist_lr", "left_wrist_right"]

_WRIST_KEY = "observation.images.robot0_eye_in_hand"
_LEFT_KEY = "observation.images.robot0_agentview_left"
_RIGHT_KEY = "observation.images.robot0_agentview_right"
_DEFAULT_CAMERA_SET: CameraSet = "left_wrist"

_CAMERA_SET_KEYS: dict[CameraSet, tuple[str, ...]] = {
    "left_wrist": (_LEFT_KEY, _WRIST_KEY),
    "wrist_lr": (_WRIST_KEY, _LEFT_KEY, _RIGHT_KEY),
    "left_wrist_right": (_LEFT_KEY, _WRIST_KEY, _RIGHT_KEY),
}
_CAMERA_SET_VIEWPOINT: dict[CameraSet, str] = {
    "left_wrist": "concat_view",
    "wrist_lr": "wrist_top_agentview_lr_bottom",
    "left_wrist_right": "concat_view",
}
_CAMERA_SET_DESCRIPTION: dict[CameraSet, str] = {
    "left_wrist": "The left half is agentview left. The right half is the wrist-mounted camera.",
    "wrist_lr": "The top view is the wrist-mounted camera. The bottom row is agentview left then agentview right.",
    "left_wrist_right": (
        "The three horizontal panels are agentview left, the wrist-mounted camera, "
        "and agentview right, from left to right."
    ),
}
_CAMERA_SET_ALIASES = {
    "wrist_top_agentview_lr_bottom": "wrist_lr",
}


def normalize_robocasa_camera_set(value: str) -> CameraSet:
    value = _CAMERA_SET_ALIASES.get(value, value)
    if value not in _CAMERA_SET_KEYS:
        raise ValueError(f"Unsupported RoboCasa camera_set={value!r}; expected one of {sorted(_CAMERA_SET_KEYS)}")
    return value  # type: ignore[return-value]


def robocasa_camera_keys(camera_set: str) -> tuple[str, ...]:
    return _CAMERA_SET_KEYS[normalize_robocasa_camera_set(camera_set)]


def robocasa_composed_size(camera_set: str, image_size: int = 256) -> tuple[int, int]:
    camera_set = normalize_robocasa_camera_set(camera_set)
    size = int(image_size)
    if camera_set == "left_wrist":
        return size, size * 2
    if camera_set == "wrist_lr":
        return size * 3 // 2, size
    return size, size * 3


def robocasa_view_description(camera_set: str) -> str:
    return _CAMERA_SET_DESCRIPTION[normalize_robocasa_camera_set(camera_set)]


def compose_robocasa_video(
    frames: dict[str, torch.Tensor],
    *,
    camera_set: str,
    image_size: int = 256,
) -> torch.Tensor:
    """Compose decoded RoboCasa views before the Cosmos resolution bucket.

    Inputs are float32 tensors shaped [T,C,H,W]. The result keeps source-view
    resolution; VideoResize(resolution=None) remains the only Cosmos canvas map.
    """
    camera_set = normalize_robocasa_camera_set(camera_set)
    required = robocasa_camera_keys(camera_set)
    missing = [key for key in required if key not in frames]
    if missing:
        raise KeyError(f"Missing RoboCasa camera tensors for {camera_set}: {missing}")

    selected = [frames[key] for key in required]
    if any(frame.dtype != torch.float32 for frame in selected):
        raise ValueError("RoboCasa decoder must compose views in float32 [0,1]")

    size = int(image_size)
    selected = [
        frame
        if tuple(frame.shape[-2:]) == (size, size)
        else F.interpolate(frame, size=(size, size), mode="bilinear", align_corners=False)
        for frame in selected
    ]

    if camera_set == "left_wrist":
        left, wrist = selected
        return torch.cat([left, wrist], dim=-1)

    if camera_set == "wrist_lr":
        wrist, left, right = selected
        half = size // 2
        left = F.interpolate(left, size=(half, half), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half, half), mode="bilinear", align_corners=False)
        return torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)

    left, wrist, right = selected
    return torch.cat([left, wrist, right], dim=-1)


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
        camera_set: CameraSet = _DEFAULT_CAMERA_SET,
        camera_mode: str | None = None,
        image_size: int = 256,
        sample_stride: int = 1,
        latent_cache_root: str | None = None,
        latent_cache_verify_ratio: float = 0.0,
    ) -> None:
        if camera_mode is not None:
            legacy_camera_set = normalize_robocasa_camera_set(camera_mode)
            if camera_set != _DEFAULT_CAMERA_SET and normalize_robocasa_camera_set(camera_set) != legacy_camera_set:
                raise ValueError(
                    f"Conflicting RoboCasa camera_set={camera_set!r} and legacy camera_mode={camera_mode!r}"
                )
            camera_set = legacy_camera_set
        camera_set = normalize_robocasa_camera_set(camera_set)
        if chunk_length != 16:
            raise ValueError(f"RoboCasa exact-window cache requires chunk_length=16, got {chunk_length}")
        super().__init__(
            root=root,
            domain_name="robocasa",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",
            tolerance_s=tolerance_s,
            viewpoint=_CAMERA_SET_VIEWPOINT[camera_set],
            action_normalization=None,
            sample_stride=sample_stride,
        )
        if int(self._info.get("fps", fps)) != int(fps):
            raise ValueError(f"RoboCasa fps mismatch: requested={fps}, source={self._info.get('fps')}")
        self._camera_set = camera_set
        self._camera_mode = camera_set
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

    def _load_episodes(self) -> dict[int, dict[str, Any]]:
        """Read RoboCasa LeRobot v2.1 meta/episodes.jsonl."""
        path = self._root / "meta" / "episodes.jsonl"
        if not path.is_file():
            return super()._load_episodes()
        episodes: dict[int, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "episode_index" not in row:
                    raise ValueError(f"Missing episode_index in {path}:{line_number}")
                episode_index = int(row["episode_index"])
                if episode_index in episodes:
                    raise ValueError(f"Duplicate episode_index={episode_index} in {path}")
                episodes[episode_index] = row
        if not episodes:
            raise ValueError(f"RoboCasa episode metadata is empty: {path}")
        return episodes

    def _load_tasks(self) -> dict[int, str]:
        """Read RoboCasa LeRobot v2.1 meta/tasks.jsonl."""
        path = self._root / "meta" / "tasks.jsonl"
        if not path.is_file():
            return super()._load_tasks()
        tasks: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "task_index" not in row or "task" not in row:
                    raise ValueError(f"Missing task_index/task in {path}:{line_number}")
                task_index = int(row["task_index"])
                if task_index in tasks:
                    raise ValueError(f"Duplicate task_index={task_index} in {path}")
                tasks[task_index] = str(row["task"])
        if not tasks:
            raise ValueError(f"RoboCasa task metadata is empty: {path}")
        return tasks
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

    @property
    def camera_set(self) -> CameraSet:
        return self._camera_set

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return robocasa_camera_keys(self._camera_set)

    @property
    def composed_output_size(self) -> tuple[int, int]:
        return robocasa_composed_size(self._camera_set, self._image_size)

    def _compose_video(self, frames: dict[str, torch.Tensor]) -> torch.Tensor:
        return compose_robocasa_video(frames, camera_set=self._camera_set, image_size=self._image_size)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        frames: dict[str, torch.Tensor] = {}
        for key in self.camera_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            value = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )
            frames[key] = F.interpolate(
                value,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self._compose_video(frames)

    def _validate_latent_cache_manifest(self) -> None:
        assert self._latent_cache_root is not None
        path = self._latent_cache_root / "dataset_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "exact_window_v1",
            "suite": f"robocasa365_{self.task_id.split('/', 1)[0]}",
            "chunk_length": self._chunk_length,
            "sample_stride": self._sample_stride,
            "fps": self._fps,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"RoboCasa latent cache {key} mismatch: cache={manifest.get(key)!r}, dataset={value!r}")
        manifest_camera_set = manifest.get("camera_set", manifest.get("camera_mode"))
        if not isinstance(manifest_camera_set, str):
            raise ValueError("RoboCasa latent cache is missing camera_set/camera_mode")
        if normalize_robocasa_camera_set(manifest_camera_set) != self._camera_set:
            raise ValueError(
                f"RoboCasa latent cache camera_set mismatch: cache={manifest_camera_set!r}, "
                f"dataset={self._camera_set!r}"
            )
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
            composed_h, composed_w = self.composed_output_size
            video = torch.zeros((17, 3, composed_h, composed_w), dtype=torch.float32)
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
            "camera_set": self._camera_set,
            "additional_view_description": robocasa_view_description(self._camera_set),
        })
        return result

"""RoboCasa365 LeRobot v3 action-policy dataset.

This loader intentionally targets the flat LeRobot v3 mirrors used by the
RoboCasa365 target/pretrain releases.  Storage/indexing is delegated to the
shared BaseActionLeRobotDataset + LeRobotDataset implementation; this module
owns only RoboCasa semantics: actions, task-class recovery, camera composition,
and exact-window latent-cache consumption.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, convert_rotation
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint
from cosmos_framework.model.generator.vision_vae import (
    LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES,
    LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS,
)

_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_TASK_CLASS_FEATURE = "annotation.human.task_name"

_WRIST_KEY = "observation.images.robot0_eye_in_hand"
_LEFT_KEY = "observation.images.robot0_agentview_left"
_RIGHT_KEY = "observation.images.robot0_agentview_right"

CameraSet = Literal["left_wrist", "wrist_lr", "left_wrist_right"]
_DEFAULT_CAMERA_SET: CameraSet = "left_wrist"

_CAMERA_SET_KEYS: dict[CameraSet, tuple[str, ...]] = {
    "left_wrist": (_LEFT_KEY, _WRIST_KEY),
    "wrist_lr": (_WRIST_KEY, _LEFT_KEY, _RIGHT_KEY),
    "left_wrist_right": (_LEFT_KEY, _WRIST_KEY, _RIGHT_KEY),
}
_CAMERA_SET_VIEWPOINT: dict[CameraSet, Viewpoint] = {
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

_EEF_POS = slice(5, 8)
_EEF_ROT = slice(8, 11)
_GRIPPER = slice(11, 12)
_BASE_MOTION = slice(0, 4)
_CONTROL_MODE = slice(4, 5)

_STATE_BASE_POS = slice(0, 3)
_STATE_BASE_ROT = slice(3, 7)
_STATE_EEF_POS = slice(7, 10)
_STATE_EEF_ROT = slice(10, 14)


def normalize_robocasa_camera_set(value: str) -> CameraSet:
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
    """Compose float32 [T,C,H,W] RoboCasa views before Cosmos resizing."""
    camera_set = normalize_robocasa_camera_set(camera_set)
    required = robocasa_camera_keys(camera_set)
    missing = [key for key in required if key not in frames]
    if missing:
        raise KeyError(f"Missing RoboCasa camera tensors for {camera_set}: {missing}")

    selected = [frames[key] for key in required]
    if any(frame.dtype != torch.float32 for frame in selected):
        raise ValueError("RoboCasa camera tensors must be float32 in [0,1]")

    size = int(image_size)
    selected = [
        frame if tuple(frame.shape[-2:]) == (size, size)
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


def _scalar_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape={tuple(value.shape)}")
        return int(value.item())
    return int(value)


def robocasa_task_class_from_index(tasks: Any, task_index: int) -> str:
    """Resolve annotation.human.task_name's global task-table index to text."""
    if tasks is None:
        raise ValueError("LeRobot v3 metadata has no tasks table")
    if "task_index" not in tasks.columns:
        raise ValueError("LeRobot v3 tasks table is missing task_index")
    matches = tasks[tasks["task_index"] == int(task_index)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one task-table row for annotation task index={task_index}, got {len(matches)}"
        )
    return str(matches.index[0])


class RoboCasaLeRobotDataset(BaseActionLeRobotDataset):
    """Flat LeRobot v3 RoboCasa365 dataset with optional exact-window latent cache."""

    def __init__(
        self,
        root: str,
        *,
        suite: str,
        fps: float = 20.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.01,
        split: str = "train",
        mode: str = "wam",
        pose_convention: PoseConvention = "backward_framewise",
        rotation_format: str = "rot6d",
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 1e-4,
        image_size: int = 256,
        sample_stride: int = 1,
        camera_set: CameraSet = _DEFAULT_CAMERA_SET,
        use_state: bool = False,
        use_base_action: bool = True,
        base_encoding: str = "raw",
        latent_cache_root: str | None = None,
    ) -> None:
        if rotation_format != "rot6d":
            raise NotImplementedError("RoboCasa loader only supports rotation_format='rot6d'")
        if chunk_length != 16:
            raise ValueError(f"RoboCasa exact-window cache requires chunk_length=16, got {chunk_length}")

        camera_set = normalize_robocasa_camera_set(camera_set)
        root_path = Path(root)
        info_path = root_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Expected flat LeRobot v3 root with meta/info.json, got {root}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        codebase_version = str(info.get("codebase_version", ""))
        if not codebase_version.startswith("v3"):
            raise ValueError(
                f"RoboCasa loader is v3-only; meta/info.json codebase_version={codebase_version!r}"
            )

        self._suite = str(suite)
        self._root = root_path
        self._camera_set = camera_set
        self._image_size = int(image_size)
        self._use_state = bool(use_state)
        self._use_base_action = bool(use_base_action)
        if base_encoding not in {"raw", "ego"}:
            raise ValueError(f"Unsupported base_encoding={base_encoding!r}")
        self._base_encoding = base_encoding

        self._latent_cache_root = Path(latent_cache_root) if latent_cache_root else None
        self._latent_cache_expected_shape: tuple[int, ...] | None = None
        self._cache_episode_task: dict[int, tuple[str, str]] = {}
        self._latent_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        if self._latent_cache_root is not None:
            self._load_latent_cache_manifest()

        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="robocasa",
            viewpoint=_CAMERA_SET_VIEWPOINT[camera_set],
            pose_convention=pose_convention,
            rotation_format=rotation_format,
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            sample_stride=sample_stride,
            skip_video_loading=self._latent_cache_root is not None,
        )

        observation_ts = [i * self._dt for i in range(self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(self._chunk_length)]
        self._delta_timestamps = {_ACTION_FEATURE: action_ts}
        if self._use_state or (self._use_base_action and self._base_encoding == "ego"):
            self._delta_timestamps[_STATE_FEATURE] = observation_ts
        for key in robocasa_camera_keys(self._camera_set):
            self._delta_timestamps[key] = observation_ts

        self._all_shard_roots = [str(root_path)]
        self._register_sources()

    @property
    def suite(self) -> str:
        return self._suite

    @property
    def camera_set(self) -> CameraSet:
        return self._camera_set

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return robocasa_camera_keys(self._camera_set)

    @property
    def composed_output_size(self) -> tuple[int, int]:
        return robocasa_composed_size(self._camera_set, self._image_size)

    @property
    def action_dim(self) -> int:
        if not self._use_base_action:
            return 10
        return 15 if self._base_encoding == "raw" else 20

    @property
    def local_memory_action_dim(self) -> int:
        """Action evidence width consumed by Local-TTT."""
        return self.action_dim

    def local_memory_executed_action(self, idx: int) -> torch.Tensor:
        """Return the first executable action for one flat window anchor.

        With ego base encoding this is the full 20D mobile-manipulation
        action: ego base delta(9) + control mode(1) + EEF/gripper(10).
        """
        dataset_idx, row_idx, _, _ = self._resolve_index(int(idx))
        sample = self._get_dataset(dataset_idx)[row_idx]
        raw = sample[_ACTION_FEATURE]
        arm = self._build_frame_wise_action(raw)
        if self._use_base_action and self._base_encoding == "ego":
            base_delta = self._build_base_delta(sample[_STATE_FEATURE])
            control_mode = raw.float()[:, _CONTROL_MODE]
            action = torch.cat([base_delta, control_mode, arm], dim=-1)
        else:
            action = arm
        if self._action_normalizer is not None:
            action = self._action_normalizer.normalize_action(action)
        result = action[0].detach().float()
        if tuple(result.shape) != (self.action_dim,):
            raise ValueError(
                f"RoboCasa Local evidence action must have shape [{self.action_dim}], got {tuple(result.shape)}"
            )
        return result

    def _build_action_spec(self) -> ActionSpec:
        if not self._use_base_action:
            return build_action_spec(Pos(), Rot("rot6d"), Gripper())
        if self._base_encoding == "raw":
            return build_action_spec(Pos(dim=4), Gripper(), Pos(), Rot("rot6d"), Gripper())
        return build_action_spec(Pos(), Rot("rot6d"), Gripper(), Pos(), Rot("rot6d"), Gripper())

    def _build_frame_wise_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        raw = raw_action.float()
        if raw.ndim != 2 or raw.shape[-1] != 12:
            raise ValueError(f"Expected RoboCasa raw action [T,12], got {tuple(raw.shape)}")
        translation = raw[:, _EEF_POS]
        rotation_matrix = convert_rotation(raw[:, _EEF_ROT], input_format="axisangle", output_format="matrix")
        rotation = convert_rotation(rotation_matrix, input_format="matrix", output_format="rot6d")
        gripper = raw[:, _GRIPPER]
        arm = torch.cat([translation, rotation, gripper], dim=-1)
        if not self._use_base_action:
            return arm
        control_mode = raw[:, _CONTROL_MODE]
        if self._base_encoding == "raw":
            return torch.cat([raw[:, _BASE_MOTION], control_mode, arm], dim=-1)
        # Ego base motion requires observation.state, so this low-level helper
        # returns only the 10D arm/gripper primitive. __getitem__ and the
        # Local-TTT executed-action hook prepend ego base delta(9) + control(1)
        # to form the canonical 20D mobile-manipulation action.
        return arm

    def _build_base_delta(self, state_seq: torch.Tensor) -> torch.Tensor:
        state = state_seq[-self._chunk_length - 1 :].float()
        pos = state[:, _STATE_BASE_POS]
        quat = state[:, _STATE_BASE_ROT]
        rot = convert_rotation(quat, input_format="quat_xyzw", output_format="matrix")
        r_prev, r_next = rot[:-1], rot[1:]
        d_world = (pos[1:] - pos[:-1]).unsqueeze(-1)
        d_ego = torch.matmul(r_prev.transpose(-1, -2), d_world).squeeze(-1)
        r_rel = torch.matmul(r_prev.transpose(-1, -2), r_next)
        rot6d = convert_rotation(r_rel, input_format="matrix", output_format="rot6d")
        return torch.cat([d_ego, rot6d], dim=-1)

    def _build_initial_state(self, state_seq: torch.Tensor) -> torch.Tensor:
        state = state_seq[-self._chunk_length - 1].float()
        pos = state[_STATE_EEF_POS]
        quat = state[_STATE_EEF_ROT].unsqueeze(0)
        matrix = convert_rotation(quat, input_format="quat_xyzw", output_format="matrix")
        rot6d = convert_rotation(matrix, input_format="matrix", output_format="rot6d").reshape(6)
        grip = state[14:15] - state[15:16]
        return torch.cat([pos, rot6d, grip], dim=-1)

    def _compose_sample_video(self, sample: dict[str, Any]) -> torch.Tensor:
        frames = {key: sample[key] for key in self.camera_keys}
        return compose_robocasa_video(frames, camera_set=self._camera_set, image_size=self._image_size)

    def _task_class(self, dataset_idx: int, sample: dict[str, Any]) -> str:
        if _TASK_CLASS_FEATURE not in sample:
            raise KeyError(
                f"RoboCasa365 v3 sample is missing {_TASK_CLASS_FEATURE!r}; "
                "do not fall back to task_index because it represents natural-language phrasing"
            )
        task_index = _scalar_int(sample[_TASK_CLASS_FEATURE], name=_TASK_CLASS_FEATURE)
        return robocasa_task_class_from_index(self._get_dataset(dataset_idx).meta.tasks, task_index)

    def _load_latent_cache_manifest(self) -> None:
        assert self._latent_cache_root is not None
        path = self._latent_cache_root / "dataset_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "exact_window_v1",
            "suite": self._suite,
            "source_format": "lerobot_v3",
            "chunk_length": 16,
            "camera_set": self._camera_set,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"RoboCasa latent cache {key} mismatch: cache={manifest.get(key)!r}, expected={value!r}"
                )
        contract = {
            "compute_dtype": "torch.bfloat16",
            "encode_exact_durations": LIBERO_EXACT_WINDOW_ENCODE_EXACT_DURATIONS,
            "encode_chunk_frames": LIBERO_EXACT_WINDOW_ENCODE_CHUNK_FRAMES,
        }
        if manifest.get("vae_encode_contract") != contract:
            raise ValueError("RoboCasa latent cache VAE encoding contract mismatch")
        shape = tuple(manifest.get("latent_shape", []))
        if len(shape) != 4 or shape[:2] != (5, 48):
            raise ValueError(f"Invalid RoboCasa cache latent_shape={shape}")
        self._latent_cache_expected_shape = shape

        for task in manifest.get("tasks", []):
            task_class = str(task["task_class"])
            task_slug = str(task["task_slug"])
            for row in task.get("episodes", []):
                episode_index = int(row["episode_index"])
                if episode_index in self._cache_episode_task:
                    raise ValueError(f"Duplicate cached episode_index={episode_index}")
                self._cache_episode_task[episode_index] = (task_class, task_slug)

    def _filter_valid_episodes(self, _meta: Any, episode_ids: list[int]) -> list[int]:
        if self._latent_cache_root is None:
            return episode_ids
        available = set(self._cache_episode_task)
        return [episode for episode in episode_ids if int(episode) in available]

    def _load_cached_latent(self, episode_index: int, start_frame: int) -> torch.Tensor | None:
        if self._latent_cache_root is None:
            return None
        task_info = self._cache_episode_task.get(int(episode_index))
        if task_info is None:
            raise KeyError(f"Episode {episode_index} is not present in the RoboCasa latent-cache manifest")
        _, task_slug = task_info
        item = self._latent_cache.get(int(episode_index))
        if item is None:
            path = self._latent_cache_root / "tasks" / task_slug / "episodes" / f"episode_{episode_index:06d}.pt"
            item = torch.load(path, map_location="cpu", weights_only=True)
            if item.get("format") != "exact_window_v1" or int(item.get("episode_index", -1)) != int(episode_index):
                raise ValueError(f"Invalid RoboCasa cache payload: {path}")
            self._latent_cache[int(episode_index)] = item
            if len(self._latent_cache) > 8:
                self._latent_cache.popitem(last=False)

        window = item.get("windows", {}).get(str(start_frame))
        if not isinstance(window, dict) or not isinstance(window.get("latent"), torch.Tensor):
            raise KeyError(f"Missing RoboCasa cache key {(episode_index, start_frame)}")
        latent = window["latent"].contiguous()
        expected_window = torch.arange(start_frame, start_frame + 17, dtype=torch.long)
        expected_anchors = torch.arange(start_frame, start_frame + 17, 4, dtype=torch.long)
        if (
            self._latent_cache_expected_shape is None
            or latent.dtype != torch.float32
            or tuple(latent.shape) != self._latent_cache_expected_shape
            or not torch.isfinite(latent).all()
        ):
            raise ValueError(f"Invalid latent for RoboCasa cache key {(episode_index, start_frame)}")
        if not torch.equal(window.get("window_frame_indices"), expected_window):
            raise ValueError(f"Invalid window_frame_indices for {(episode_index, start_frame)}")
        if not torch.equal(window.get("latent_source_frame_indices"), expected_anchors):
            raise ValueError(f"Invalid latent_source_frame_indices for {(episode_index, start_frame)}")
        return latent

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        dataset_idx, row_idx, episode_index, frame_offset = self._resolve_index(idx)
        sample = self._get_dataset(dataset_idx)[row_idx]

        raw = sample[_ACTION_FEATURE]
        action = self._build_frame_wise_action(raw)
        if self._use_base_action and self._base_encoding == "ego":
            state_seq = sample[_STATE_FEATURE]
            arm = action
            control_mode = raw.float()[:, _CONTROL_MODE]
            action = torch.cat([self._build_base_delta(state_seq), control_mode, arm], dim=-1)

        state_extras: dict[str, Any] = {}
        if self._use_state:
            initial_state = self._build_initial_state(sample[_STATE_FEATURE])
            if self._use_base_action:
                initial_state = torch.cat(
                    [torch.zeros(self.action_dim - 10, dtype=initial_state.dtype), initial_state],
                    dim=-1,
                )
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)
            state_extras["idle_frames"] = self._compute_idle_frames(action[1:])

        task_class = self._task_class(dataset_idx, sample)
        ai_caption = str(sample["task"])
        latent = self._load_cached_latent(int(episode_index), int(frame_offset))

        if self._latent_cache_root is None:
            video = self._compose_sample_video(sample)
            result = self._build_result(
                mode=mode,
                video=video,
                action=action,
                ai_caption=ai_caption,
                **state_extras,
                additional_view_description=robocasa_view_description(self._camera_set),
            )
        else:
            # The latent-cache path deliberately does not decode source video.
            # Keep a zero uint8 placeholder with the correct pre-resize geometry
            # so ActionTransformPipeline can build image_size/prompt metadata while
            # the model consumes video_latent instead of running the VAE.
            if "idle_frames" not in state_extras:
                idle_frames = self._compute_idle_frames(action)
                if idle_frames is not None:
                    state_extras["idle_frames"] = idle_frames
            if self._action_normalizer is not None:
                action = self._action_normalizer.normalize_action(action)
            h, w = self.composed_output_size
            result = {
                "ai_caption": ai_caption,
                "video": torch.zeros((3, self._chunk_length + 1, h, w), dtype=torch.uint8),
                "action": action,
                "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
                "mode": mode,
                "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
                "viewpoint": self._viewpoint,
                "additional_view_description": robocasa_view_description(self._camera_set),
                **state_extras,
            }

        result.update(
            {
                "video_latent": latent,
                "suite": self._suite,
                "task_class": task_class,
                "episode_index": torch.tensor(int(episode_index), dtype=torch.long),
                "start_frame": torch.tensor(int(frame_offset), dtype=torch.long),
                "window_frame_indices": torch.arange(int(frame_offset), int(frame_offset) + 17),
                "latent_source_frame_indices": torch.arange(int(frame_offset), int(frame_offset) + 17, 4),
                "camera_set": self._camera_set,
            }
        )
        return result

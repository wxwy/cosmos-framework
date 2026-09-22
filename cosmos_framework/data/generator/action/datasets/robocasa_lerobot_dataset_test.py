# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    RoboCasaLeRobotDataset,
    compose_robocasa_video,
    normalize_robocasa_camera_set,
    robocasa_camera_keys,
    robocasa_composed_size,
)
from cosmos_framework.data.generator.action.utils.transforms import VideoResize
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id


_WRIST = "observation.images.robot0_eye_in_hand"
_LEFT = "observation.images.robot0_agentview_left"
_RIGHT = "observation.images.robot0_agentview_right"


def _frames(size: int = 256) -> dict[str, torch.Tensor]:
    return {
        _LEFT: torch.ones(2, 3, size, size, dtype=torch.float32),
        _WRIST: torch.full((2, 3, size, size), 2.0, dtype=torch.float32),
        _RIGHT: torch.full((2, 3, size, size), 3.0, dtype=torch.float32),
    }


@pytest.mark.L0
def test_robocasa_left_wrist_layout_is_libero_style_horizontal_pair() -> None:
    video = compose_robocasa_video(_frames(), camera_set="left_wrist")
    assert video.shape == (2, 3, 256, 512)
    assert robocasa_composed_size("left_wrist") == (256, 512)
    assert robocasa_camera_keys("left_wrist") == (_LEFT, _WRIST)
    torch.testing.assert_close(video[..., :256], torch.ones_like(video[..., :256]))
    torch.testing.assert_close(video[..., 256:], torch.full_like(video[..., 256:], 2.0))


@pytest.mark.L0
def test_robocasa_wrist_lr_layout_keeps_large_wrist_over_two_agent_views() -> None:
    video = compose_robocasa_video(_frames(), camera_set="wrist_lr")
    assert video.shape == (2, 3, 384, 256)
    assert robocasa_composed_size("wrist_lr") == (384, 256)
    assert robocasa_camera_keys("wrist_lr") == (_WRIST, _LEFT, _RIGHT)
    torch.testing.assert_close(video[..., :256, :], torch.full_like(video[..., :256, :], 2.0))
    torch.testing.assert_close(video[..., 256:, :128], torch.ones_like(video[..., 256:, :128]))
    torch.testing.assert_close(video[..., 256:, 128:], torch.full_like(video[..., 256:, 128:], 3.0))


@pytest.mark.L0
def test_robocasa_left_wrist_right_layout_is_three_full_views_horizontal() -> None:
    video = compose_robocasa_video(_frames(), camera_set="left_wrist_right")
    assert video.shape == (2, 3, 256, 768)
    assert robocasa_composed_size("left_wrist_right") == (256, 768)
    assert robocasa_camera_keys("left_wrist_right") == (_LEFT, _WRIST, _RIGHT)
    torch.testing.assert_close(video[..., :256], torch.ones_like(video[..., :256]))
    torch.testing.assert_close(video[..., 256:512], torch.full_like(video[..., 256:512], 2.0))
    torch.testing.assert_close(video[..., 512:], torch.full_like(video[..., 512:], 3.0))


@pytest.mark.L0
@pytest.mark.parametrize(
    ("camera_set", "expected_canvas"),
    [
        ("left_wrist", (192, 320)),
        ("wrist_lr", (320, 192)),
        ("left_wrist_right", (192, 320)),
    ],
)
def test_robocasa_camera_layouts_snap_to_expected_cosmos_tier256_canvas(
    camera_set: str,
    expected_canvas: tuple[int, int],
) -> None:
    video = compose_robocasa_video(_frames(), camera_set=camera_set)
    video = video.permute(1, 0, 2, 3).contiguous()
    resized = VideoResize(pad_keys=["video"], keep_aspect_ratio=True)(
        {"video": video},
        resolution=None,
    )["video"]
    assert tuple(resized.shape[-2:]) == expected_canvas


@pytest.mark.L0
def test_robocasa_legacy_three_view_camera_mode_alias_is_preserved() -> None:
    assert normalize_robocasa_camera_set("wrist_top_agentview_lr_bottom") == "wrist_lr"


@pytest.mark.L0
def test_robocasa_v21_jsonl_metadata_loaders(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["Close blender"],"length":17}\\n'
        '{"episode_index":2,"tasks":["Close blender"],"length":21}\\n',
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        '{"task_index":7,"task":"Close blender"}\\n',
        encoding="utf-8",
    )
    dataset = object.__new__(RoboCasaLeRobotDataset)
    dataset._root = tmp_path
    assert dataset._load_episodes() == {
        0: {"episode_index": 0, "tasks": ["Close blender"], "length": 17},
        2: {"episode_index": 2, "tasks": ["Close blender"], "length": 21},
    }
    assert dataset._load_tasks() == {7: "Close blender"}


@pytest.mark.L0
def test_robocasa_v21_video_path_formats_episode_index(tmp_path) -> None:
    dataset = object.__new__(RoboCasaLeRobotDataset)
    dataset._root = tmp_path
    dataset._info = {
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    }
    episode = {"episode_index": 23}
    path = dataset._video_path(episode, _LEFT)
    assert path == tmp_path / "videos/chunk-000" / _LEFT / "episode_000023.mp4"


@pytest.mark.L0
def test_robocasa_domain_uses_canonical_upstream_slot() -> None:
    assert get_domain_id("robocasa") == 30
    assert get_domain_id("robocasa_panda_omron") == 30


@pytest.mark.L0
def test_robocasa_metadata_loaders_fall_back_to_default_parquet_format(tmp_path) -> None:
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [4], "length": [21]}),
        episode_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table({"task_index": [9], "task": ["Open drawer"]}),
        tmp_path / "meta" / "tasks.parquet",
    )
    dataset = object.__new__(RoboCasaLeRobotDataset)
    dataset._root = tmp_path
    assert dataset._load_episodes() == {4: {"episode_index": 4, "length": 21}}
    assert dataset._load_tasks() == {9: "Open drawer"}

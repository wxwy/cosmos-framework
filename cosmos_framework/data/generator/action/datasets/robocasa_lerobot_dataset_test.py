from __future__ import annotations

import pandas as pd
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    RoboCasaLeRobotDataset,
    compose_robocasa_video,
    normalize_robocasa_camera_set,
    robocasa_camera_keys,
    robocasa_composed_size,
    robocasa_task_class_from_index,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import VideoResize


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
def test_robocasa_v3_only_camera_sets() -> None:
    assert normalize_robocasa_camera_set("left_wrist") == "left_wrist"
    with pytest.raises(ValueError):
        normalize_robocasa_camera_set("wrist_top_agentview_lr_bottom")


@pytest.mark.L0
def test_robocasa_left_wrist_layout() -> None:
    video = compose_robocasa_video(_frames(), camera_set="left_wrist")
    assert video.shape == (2, 3, 256, 512)
    assert robocasa_composed_size("left_wrist") == (256, 512)
    assert robocasa_camera_keys("left_wrist") == (_LEFT, _WRIST)


@pytest.mark.L0
def test_robocasa_wrist_lr_layout() -> None:
    video = compose_robocasa_video(_frames(), camera_set="wrist_lr")
    assert video.shape == (2, 3, 384, 256)
    assert robocasa_composed_size("wrist_lr") == (384, 256)


@pytest.mark.L0
def test_robocasa_three_horizontal_layout() -> None:
    video = compose_robocasa_video(_frames(), camera_set="left_wrist_right")
    assert video.shape == (2, 3, 256, 768)
    assert robocasa_composed_size("left_wrist_right") == (256, 768)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("camera_set", "canvas"),
    [
        ("left_wrist", (192, 320)),
        ("wrist_lr", (320, 192)),
        ("left_wrist_right", (192, 320)),
    ],
)
def test_robocasa_camera_sets_snap_to_expected_canvas(camera_set: str, canvas: tuple[int, int]) -> None:
    video = compose_robocasa_video(_frames(), camera_set=camera_set).permute(1, 0, 2, 3).contiguous()
    resized = VideoResize(pad_keys=["video"], keep_aspect_ratio=True)(
        {"video": video},
        resolution=None,
    )["video"]
    assert tuple(resized.shape[-2:]) == canvas


@pytest.mark.L0
def test_robocasa_task_class_uses_annotation_task_name_table_index() -> None:
    tasks = pd.DataFrame(
        {"task_index": [0, 1, 230]},
        index=pd.Index(["description phrasing", "CloseBlenderLid", "OpenDrawer"], name="task"),
    )
    assert robocasa_task_class_from_index(tasks, 1) == "CloseBlenderLid"
    assert robocasa_task_class_from_index(tasks, 230) == "OpenDrawer"


@pytest.mark.L0
def test_robocasa_domain_is_canonical_30() -> None:
    assert get_domain_id("robocasa") == 30
    with pytest.raises(KeyError):
        get_domain_id("robocasa_panda_omron")


@pytest.mark.L0
def test_robocasa_ego_mobile_action_is_20d_for_policy_and_local_memory() -> None:
    dataset = RoboCasaLeRobotDataset.__new__(RoboCasaLeRobotDataset)
    dataset._use_base_action = True
    dataset._base_encoding = "ego"
    assert dataset.action_dim == 20
    assert dataset.local_memory_action_dim == 20

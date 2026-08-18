# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Edge warmstart single-task overfit on LIBERO task 0.

Deliberate overfit: feed only task_index==0 episodes (turn on the stove and put
the moka pot on it) so the model quickly memorizes one behavior. Used as the
fast end-to-end verification gate before the full multi-task LIBERO-10 SFT:
warmstart → SFT → checkpoint → closed-loop eval on one task. Does NOT verify
multi-task generalization (that is the later 10-task run's job).

The dataset filter lives entirely in this config module (a LIBEROLeRobotDataset
subclass + a builder that mirrors ``get_action_libero_sft_dataset`` but accepts
``task_index``). No cosmos framework source is modified.
"""

import copy

import numpy as np
from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_warmstart import (
    action_policy_libero_edge_warmstart,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.utils.lazy_config import LazyCall as L


class SingleTaskLIBEROLeRobotDataset(LIBEROLeRobotDataset):
    """LIBEROLeRobotDataset restricted to episodes of one task.

    Filters after the base split (train/val) by masking ``_ep_vals``/``_ep_starts``
    to episodes whose task_index == ``task_index`` and recomputing ``_valid_cum``.
    Indexing/``__len__``/``get_shuffle_blocks`` all derive from those three arrays,
    so filtering them is sufficient; the task→language mapping in the base reads
    ``meta/tasks.parquet`` by task_index and keeps working unchanged.
    """

    def __init__(self, task_index: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # Rows are sorted by frame index with episodes contiguous
        # (``np.diff(self._row_episode) >= 0`` asserted in the base). Rebuild episode
        # boundaries from the row arrays directly — _ep_starts alone cannot be diffed
        # after masking, since the starts live in full-row space and would span the
        # dropped episodes' rows.
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(self._row_episode)) + 1])
        ep_ids = self._row_episode[bounds]
        ep_tasks = self._row_task[bounds]
        ep_counts = np.diff(np.append(bounds, len(self._row_episode)))
        # Keep episodes that survive the base train/val split (_ep_vals) AND match task.
        kept_ids = set(self._ep_vals.tolist())
        sel = (ep_tasks == task_index) & np.isin(ep_ids, list(kept_ids))
        if not sel.any():
            raise ValueError(
                f"task_index={task_index} has no episodes in LIBERO dataset root={self._root}"
            )
        self._ep_vals = ep_ids[sel].astype(np.int64)
        self._ep_starts = bounds[sel].astype(np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts[sel] - self._chunk_length)).astype(
            np.int64
        )
        if self._valid_cum.size:
            import logging

            logging.getLogger(__name__).info(
                f"SingleTaskLIBEROLeRobotDataset: task_index={task_index} "
                f"kept_episodes={len(self._ep_vals)} valid_indices={int(self._valid_cum[-1])}"
            )


def get_action_libero_sft_task_dataset(
    *,
    task_index: int = 0,
    root: str,
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "wam",
    camera_mode: str = "concat_view",
    action_space: str = "frame_wise_relative",
    rotation_space: str = "6d",
    pose_coordinate_frame: str = "native",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    split: str = "train",
    val_ratio: float = 0.01,
    seed: int = 0,
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
):
    """Mirror of ``get_action_libero_sft_dataset`` restricted to one task."""
    dataset = SingleTaskLIBEROLeRobotDataset(
        root=root,
        task_index=task_index,
        image_size=image_size,
        chunk_length=chunk_length,
        fps=fps,
        mode=mode,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        camera_mode=camera_mode,
        action_space=action_space,
        rotation_space=rotation_space,
        pose_coordinate_frame=pose_coordinate_frame,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


action_policy_libero_edge_task0_overfit = copy.deepcopy(action_policy_libero_edge_warmstart)
action_policy_libero_edge_task0_overfit["job"].update(
    name="action_policy_libero_edge_task0_overfit",
)

# Replace the multi-task dataset with the single-task (task_index=0) builder.
# Kwargs mirror the warmstart/nano dataset node exactly (plus task_index).
action_policy_libero_edge_task0_overfit["dataloader_train"]["dataloader"]["datasets"]["libero"]["dataset"] = L(
    get_action_libero_sft_task_dataset
)(
    task_index=0,
    root="${oc.env:LIBERO_ROOT}",
    fps=20,
    chunk_length=16,
    image_size=256,
    mode="wam",
    camera_mode="concat_view",
    action_space="frame_wise_relative",
    rotation_space="6d",
    pose_coordinate_frame="native",
    action_normalization="quantile_rot",
    val_ratio=0.01,
    iterable_shuffle=True,
    episode_shuffle_seed=42,
    resolution=None,
    max_action_dim="${model.config.max_action_dim}",
    cfg_dropout_rate=0.1,
    format_prompt_as_json=True,
    tokenizer_config="${model.config.vlm_config.tokenizer}",
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_task0_overfit",
    node=action_policy_libero_edge_task0_overfit,
)

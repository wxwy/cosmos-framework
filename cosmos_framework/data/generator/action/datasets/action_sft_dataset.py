# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: ``DROIDLeRobotDataset`` → ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.datasets.droid_merged_lerobot_dataset import DROIDMergedLeRobotDataset
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)

    def get_shuffle_blocks(self):
        """Delegate to the inner DROIDLeRobotDataset (per-episode/segment flat-index blocks)."""
        return self._dataset.get_shuffle_blocks()


ACTION_SHUFFLE_STATE_NAME = "action_shuffle_state_name"
ACTION_SHUFFLE_WORKER_ID = "action_shuffle_worker_id"
ACTION_SHUFFLE_EPOCH = "action_shuffle_epoch"
ACTION_SHUFFLE_BLOCK_CURSOR = "action_shuffle_block_cursor"
ACTION_SHUFFLE_WINDOW_CURSOR = "action_shuffle_window_cursor"
ACTION_SHUFFLE_GLOBAL_SHARD = "action_shuffle_global_shard"
ACTION_SHUFFLE_TOTAL_SHARDS = "action_shuffle_total_shards"


def action_shuffle_env_prefix(state_name: str, worker_id: int) -> str:
    """Stable per-dataset/per-worker environment namespace used by DCP resume."""

    token = re.sub(r"[^A-Za-z0-9]+", "_", str(state_name)).strip("_").upper() or "ACTION"
    return f"PSM_ACTION_SHUFFLE_{token}_WORKER_{int(worker_id)}_"


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming action dataset with deterministic epoch shuffles and exact resume.

    Each ``(rank, worker)`` owns a disjoint episode subsequence. Epoch ``e``
    deterministically uses ``seed + e``, so the permutation never needs to be
    checkpointed. Instead each yielded sample carries the worker's exact
    ``(epoch, block_cursor, window_cursor)``. A DCP dataloader callback persists
    the last *consumed* coordinate and writes it back through the environment before
    worker creation; the resumed iterator reconstructs the same permutation and
    continues at the next sample.

    ``shard_world_size`` / ``shard_rank`` remain externally owned. Resume fails
    closed if the saved shard geometry differs from the current worker geometry.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42, state_name: str = "action"):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.state_name = str(state_name)
        self.shard_world_size = 1
        self.shard_rank = 0
        self.shard_assignment_source = "default"

    def set_shard_assignment(self, world_size: int, rank: int, *, source: str) -> None:
        """Bind this stream to one rank shard before DataLoader workers start."""

        world_size = int(world_size)
        rank = int(rank)
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(
                f"invalid action-shuffle shard assignment world_size={world_size}, rank={rank}"
            )
        self.shard_world_size = world_size
        self.shard_rank = rank
        self.shard_assignment_source = str(source)

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    @staticmethod
    def _pop_resume_position(
        state_name: str, worker_id: int, *, global_shard: int, total_shards: int
    ) -> tuple[int, int, int] | None:
        prefix = action_shuffle_env_prefix(state_name, worker_id)
        keys = {
            "epoch": prefix + "EPOCH",
            "block_cursor": prefix + "BLOCK_CURSOR",
            "window_cursor": prefix + "WINDOW_CURSOR",
            "global_shard": prefix + "GLOBAL_SHARD",
            "total_shards": prefix + "TOTAL_SHARDS",
        }
        present = {name: key in os.environ for name, key in keys.items()}
        if not any(present.values()):
            return None
        if not all(present.values()):
            missing = sorted(name for name, exists in present.items() if not exists)
            raise RuntimeError(
                f"incomplete action-shuffle resume state for {state_name!r}/worker{worker_id}: missing {missing}"
            )
        values = {name: int(os.environ.pop(key)) for name, key in keys.items()}
        if values["epoch"] < 0 or values["block_cursor"] < 0 or values["window_cursor"] < 0:
            raise RuntimeError("action-shuffle resume coordinates must be non-negative")
        if values["global_shard"] != global_shard or values["total_shards"] != total_shards:
            raise RuntimeError(
                "action-shuffle resume shard geometry changed: "
                f"saved=({values['global_shard']},{values['total_shards']}) "
                f"current=({global_shard},{total_shards})"
            )
        return values["epoch"], values["block_cursor"], values["window_cursor"]

    def __iter__(self):
        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)

        resume = self._pop_resume_position(
            self.state_name,
            wid,
            global_shard=global_shard,
            total_shards=total_shards,
        )
        epoch = resume[0] if resume is not None else 0
        resume_epoch = epoch if resume is not None else None
        resume_block = resume[1] if resume is not None else None
        resume_window = resume[2] if resume is not None else None

        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=g).tolist()
            local_order = order[global_shard::total_shards]

            if resume_epoch == epoch and resume_block is not None and resume_block >= len(local_order):
                raise RuntimeError(
                    f"action-shuffle resume block_cursor={resume_block} exceeds worker block count={len(local_order)}"
                )

            for block_cursor, block_index in enumerate(local_order):
                if resume_epoch == epoch and resume_block is not None and block_cursor < resume_block:
                    continue

                start, length = blocks[block_index]
                first_window = 0
                if resume_epoch == epoch and resume_block == block_cursor and resume_window is not None:
                    if resume_window >= length:
                        raise RuntimeError(
                            f"action-shuffle resume window_cursor={resume_window} exceeds block length={length}"
                        )
                    first_window = resume_window + 1

                for window_cursor in range(first_window, length):
                    sample = self._dataset[start + window_cursor]
                    sample[ACTION_SHUFFLE_STATE_NAME] = self.state_name
                    sample[ACTION_SHUFFLE_WORKER_ID] = int(wid)
                    sample[ACTION_SHUFFLE_EPOCH] = int(epoch)
                    sample[ACTION_SHUFFLE_BLOCK_CURSOR] = int(block_cursor)
                    sample[ACTION_SHUFFLE_WINDOW_CURSOR] = int(window_cursor)
                    sample[ACTION_SHUFFLE_GLOBAL_SHARD] = int(global_shard)
                    sample[ACTION_SHUFFLE_TOTAL_SHARDS] = int(total_shards)
                    yield sample

            epoch += 1
            resume_epoch = resume_block = resume_window = None


class B2ManifestAwareIterableDataset(IterableDataset):
    """Fail-closed ordered view of an action SFT dataset for R09-B2 P1.

    Each record fixes one map-style flat index and its externally auditable
    LIBERO identity.  This deliberately does not reuse the infinite shuffled
    worker stream: B2 requires one ordered, reproducible consumption sequence.
    """

    _IDENTITY_KEYS = ("task_index", "episode_index", "start_frame")

    def __init__(self, dataset: ActionSFTDataset, records: Iterable[Mapping[str, Any]], expected_suite: str) -> None:
        super().__init__()
        self._dataset = dataset
        self._records = [dict(record) for record in records]
        if any(record.get("suite") != expected_suite for record in self._records):
            raise ValueError(f"B2 manifest records must all belong to suite={expected_suite!r}.")
        ordinals = [int(record["ordinal"]) for record in self._records]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("B2 manifest ordinals must be unique and strictly increasing.")

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _scalar(item: Mapping[str, Any], key: str) -> int:
        value = item.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"B2 manifest sample has no scalar {key!r}.")
        return int(value.item())

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("B2 manifest stream requires DataLoader num_workers=0.")
        for record in self._records:
            flat_index = int(record["dataset_flat_index"])
            item = self._dataset[flat_index]
            for key in self._IDENTITY_KEYS:
                expected = int(record[key])
                actual = self._scalar(item, key)
                if actual != expected:
                    raise ValueError(
                        f"B2 manifest identity mismatch ordinal={record['ordinal']} key={key}: "
                        f"expected={expected}, actual={actual}."
                    )
            item["b2_stream_ordinal"] = torch.tensor(int(record["ordinal"]), dtype=torch.long)
            item["b2_stream_epoch"] = torch.tensor(int(record["epoch"]), dtype=torch.long)
            item["b2_stream_microbatch"] = torch.tensor(int(record["microbatch"]), dtype=torch.long)
            item["b2_dataset_flat_index"] = torch.tensor(flat_index, dtype=torch.long)
            yield item


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "wam",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    use_success_only: bool = True,
) -> Dataset:
    """Build the DROID action SFT dataset: ``action_space='joint_pos'`` (8D) +
    ``use_state`` (raw/un-normalized), concat_view, chunk_length 32.

    Reads ``root`` (a merged/versioned DROID LeRobot root) as a single flat
    dataset; ``use_success_only=True`` filters to the ``success/`` split."""
    shard_kwargs = dict(
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,  # i4: bundles random-crop+resize+ColorJitter
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        use_success_only=use_success_only,
    )
    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)
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


def get_action_droid_merged_lerobot_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 16,
    action_space: str = "ee_pose",
    mode: str = "forward_dynamics",
    use_state: bool = False,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    split: str = "train",
    use_success_only: bool = False,
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    idle_frames_dropout: float = 0.05,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the DROID-Merged LeRobot SFT dataset for action FD recipes."""
    dataset = DROIDMergedLeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        split=split,
        use_success_only=use_success_only,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        idle_frames_dropout=idle_frames_dropout,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_robocasa_sft_dataset(
    *,
    root: str,
    suite: str,
    fps: float = 20.0,
    chunk_length: int = 16,
    mode: str = "wam",
    camera_set: str = "left_wrist",
    use_state: bool = False,
    use_base_action: bool = True,
    base_encoding: str = "raw",
    action_normalization: str | None = None,
    split: str = "train",
    split_val_ratio: float = 0.01,
    split_seed: int = 42,
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
    shuffle_state_name: str | None = None,
    sample_stride: int = 1,
    latent_cache_root: str | None = None,
) -> Dataset:
    """Build a flat LeRobot v3 RoboCasa365 action-policy SFT dataset.

    The input root is one downloaded v3 repository (for example target-atomic,
    target-composite-seen, or target-composite-unseen).  The dataset recovers
    the underlying RoboCasa task class from annotation.human.task_name instead
    of task_index, because task_index indexes natural-language phrasings.
    """
    dataset: Dataset = RoboCasaLeRobotDataset(
        root=root,
        suite=suite,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        camera_set=camera_set,
        use_state=use_state,
        use_base_action=use_base_action,
        base_encoding=base_encoding,
        action_normalization=action_normalization,
        split=split,
        split_val_ratio=split_val_ratio,
        split_seed=split_seed,
        sample_stride=sample_stride,
        latent_cache_root=latent_cache_root,
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
        return ActionIterableShuffleDataset(
            sft,
            seed=episode_shuffle_seed,
            state_name=shuffle_state_name or suite,
        )
    return sft


def get_action_libero_sft_dataset(
    *,
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
    shuffle_state_name: str | None = None,
    latent_cache_root: str | None = None,
    latent_cache_verify_ratio: float = 0.0,
    max_episodes: int | None = None,
    local_dummy_enabled: bool = False,
    local_dummy_tokens: int = 1,
    local_dummy_dim: int = 32,
    local_dummy_mode: str = "normal",
    history_mode: str = "none",
    local_history_horizon: int = 0,
    stream_manifest_path: str | None = None,
    stream_manifest_suite: str | None = None,
) -> Dataset:
    """Build the LIBERO action-policy SFT dataset (GA reproduction defaults).

    Feeds ``LIBEROLeRobotDataset`` (frame-wise-relative rot6d actions,
    ``quantile_rot``-normalized, concat_view third-person + wrist at 256x256 each
    → 256x512) through ``ActionTransformPipeline``. ``root`` is a LOCAL LeRobot dir
    (read parquet + video directly); pre-sync the HF dataset once, e.g.
    ``hf download lerobot/libero_10 --repo-type dataset --local-dir <root>``. Point
    ``root`` at libero_10 alone. The
    dataset is FPS-agnostic (decodes at real frame timestamps); ``fps`` is metadata
    for ``conditioning_fps`` / prompt duration.
    """
    dataset = LIBEROLeRobotDataset(
        root=root,
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
        latent_cache_root=latent_cache_root,
        latent_cache_verify_ratio=latent_cache_verify_ratio,
        max_episodes=max_episodes,
        history_mode=history_mode,
        local_history_horizon=local_history_horizon,
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
        local_dummy_enabled=local_dummy_enabled,
        local_dummy_tokens=local_dummy_tokens,
        local_dummy_dim=local_dummy_dim,
        local_dummy_mode=local_dummy_mode,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if stream_manifest_path is not None:
        if iterable_shuffle:
            raise ValueError("stream_manifest_path and iterable_shuffle cannot be enabled together.")
        if not stream_manifest_suite:
            raise ValueError("stream_manifest_path requires stream_manifest_suite.")
        records = [json.loads(line) for line in Path(stream_manifest_path).read_text().splitlines() if line]
        return B2ManifestAwareIterableDataset(sft, records, stream_manifest_suite)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(
            sft,
            seed=episode_shuffle_seed,
            state_name=shuffle_state_name or Path(root).name,
        )
    return sft

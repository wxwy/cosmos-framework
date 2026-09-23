# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os

from cosmos_framework.callbacks.action_dataloader_state import ActionIterableShuffleStateCallback
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ACTION_SHUFFLE_BLOCK_CURSOR,
    ACTION_SHUFFLE_EPOCH,
    ACTION_SHUFFLE_GLOBAL_SHARD,
    ACTION_SHUFFLE_STATE_NAME,
    ACTION_SHUFFLE_TOTAL_SHARDS,
    ACTION_SHUFFLE_WINDOW_CURSOR,
    ACTION_SHUFFLE_WORKER_ID,
    ActionIterableShuffleDataset,
    action_shuffle_env_prefix,
)


class _ToyDataset:
    def __init__(self) -> None:
        self.blocks = [(0, 3), (3, 2), (5, 4)]

    def get_shuffle_blocks(self):
        return self.blocks

    def __len__(self) -> int:
        return 9

    def __getitem__(self, idx: int):
        return {"sample_id": idx}


def _as_batch(sample: dict) -> dict:
    return {
        ACTION_SHUFFLE_STATE_NAME: [sample[ACTION_SHUFFLE_STATE_NAME]],
        ACTION_SHUFFLE_WORKER_ID: [sample[ACTION_SHUFFLE_WORKER_ID]],
        ACTION_SHUFFLE_EPOCH: [sample[ACTION_SHUFFLE_EPOCH]],
        ACTION_SHUFFLE_BLOCK_CURSOR: [sample[ACTION_SHUFFLE_BLOCK_CURSOR]],
        ACTION_SHUFFLE_WINDOW_CURSOR: [sample[ACTION_SHUFFLE_WINDOW_CURSOR]],
        ACTION_SHUFFLE_GLOBAL_SHARD: [sample[ACTION_SHUFFLE_GLOBAL_SHARD]],
        ACTION_SHUFFLE_TOTAL_SHARDS: [sample[ACTION_SHUFFLE_TOTAL_SHARDS]],
    }


def test_action_shuffle_resume_continues_at_next_sample(monkeypatch):
    state_name = "libero_spatial"
    prefix = action_shuffle_env_prefix(state_name, 0)
    for suffix in ("EPOCH", "BLOCK_CURSOR", "WINDOW_CURSOR", "GLOBAL_SHARD", "TOTAL_SHARDS"):
        monkeypatch.delenv(prefix + suffix, raising=False)

    uninterrupted = iter(ActionIterableShuffleDataset(_ToyDataset(), seed=7, state_name=state_name))
    consumed = [next(uninterrupted) for _ in range(6)]
    expected_next = next(uninterrupted)["sample_id"]

    saver = ActionIterableShuffleStateCallback()
    saver._update_state_from_batch(_as_batch(consumed[-1]))
    payload = saver.state_dict()

    loader = ActionIterableShuffleStateCallback()
    loader.load_state_dict(payload)
    resumed = iter(ActionIterableShuffleDataset(_ToyDataset(), seed=7, state_name=state_name))
    assert next(resumed)["sample_id"] == expected_next

    for suffix in ("EPOCH", "BLOCK_CURSOR", "WINDOW_CURSOR", "GLOBAL_SHARD", "TOTAL_SHARDS"):
        assert prefix + suffix not in os.environ


def test_action_shuffle_resume_rejects_geometry_change(monkeypatch):
    state_name = "target-atomic"
    prefix = action_shuffle_env_prefix(state_name, 0)
    monkeypatch.setenv(prefix + "EPOCH", "0")
    monkeypatch.setenv(prefix + "BLOCK_CURSOR", "0")
    monkeypatch.setenv(prefix + "WINDOW_CURSOR", "0")
    monkeypatch.setenv(prefix + "GLOBAL_SHARD", "0")
    monkeypatch.setenv(prefix + "TOTAL_SHARDS", "2")

    iterator = iter(ActionIterableShuffleDataset(_ToyDataset(), seed=3, state_name=state_name))
    try:
        next(iterator)
    except RuntimeError as error:
        assert "shard geometry changed" in str(error)
    else:
        raise AssertionError("resume must fail closed when worker geometry changes")

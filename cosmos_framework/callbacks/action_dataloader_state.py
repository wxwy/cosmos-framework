# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DCP checkpoint state for ActionIterableShuffleDataset.

This callback is intentionally generic across LIBERO and RoboCasa.  The dataset
emits the exact last-yielded worker coordinate; DCP stores one rank-local pickle
under the existing "dataloader" checkpoint component.  On resume the callback
writes those coordinates to environment variables before lazy child DataLoader
worker creation, letting each worker reconstruct its deterministic
seed+epoch permutation and continue at the next sample.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ACTION_SHUFFLE_BLOCK_CURSOR,
    ACTION_SHUFFLE_EPOCH,
    ACTION_SHUFFLE_GLOBAL_SHARD,
    ACTION_SHUFFLE_STATE_NAME,
    ACTION_SHUFFLE_TOTAL_SHARDS,
    ACTION_SHUFFLE_WINDOW_CURSOR,
    ACTION_SHUFFLE_WORKER_ID,
    action_shuffle_env_prefix,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


@dataclass(frozen=True)
class _WorkerState:
    epoch: int
    block_cursor: int
    window_cursor: int
    global_shard: int
    total_shards: int

    @property
    def position(self) -> tuple[int, int, int]:
        return self.epoch, self.block_cursor, self.window_cursor


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


class ActionIterableShuffleStateCallback(Callback):
    """Persist per-dataset/per-worker action-shuffle progress in DCP."""

    checkpoint_component: str = "dataloader"
    schema_version: int = 1

    def __init__(self) -> None:
        super().__init__()
        self.state: dict[str, dict[int, _WorkerState]] = {}

    def _update_state_from_batch(self, data_batch: dict[str, Any]) -> None:
        if ACTION_SHUFFLE_STATE_NAME not in data_batch:
            return

        fields = {
            "state_name": _flatten(data_batch[ACTION_SHUFFLE_STATE_NAME]),
            "worker_id": _flatten(data_batch[ACTION_SHUFFLE_WORKER_ID]),
            "epoch": _flatten(data_batch[ACTION_SHUFFLE_EPOCH]),
            "block_cursor": _flatten(data_batch[ACTION_SHUFFLE_BLOCK_CURSOR]),
            "window_cursor": _flatten(data_batch[ACTION_SHUFFLE_WINDOW_CURSOR]),
            "global_shard": _flatten(data_batch[ACTION_SHUFFLE_GLOBAL_SHARD]),
            "total_shards": _flatten(data_batch[ACTION_SHUFFLE_TOTAL_SHARDS]),
        }
        sizes = {name: len(values) for name, values in fields.items()}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(f"action-shuffle metadata cardinality mismatch: {sizes}")

        count = next(iter(sizes.values()), 0)
        for i in range(count):
            state_name = str(fields["state_name"][i])
            worker_id = int(fields["worker_id"][i])
            candidate = _WorkerState(
                epoch=int(fields["epoch"][i]),
                block_cursor=int(fields["block_cursor"][i]),
                window_cursor=int(fields["window_cursor"][i]),
                global_shard=int(fields["global_shard"][i]),
                total_shards=int(fields["total_shards"][i]),
            )
            if min(candidate.epoch, candidate.block_cursor, candidate.window_cursor, candidate.global_shard) < 0:
                raise RuntimeError("action-shuffle checkpoint metadata must be non-negative")
            if candidate.total_shards <= 0 or candidate.global_shard >= candidate.total_shards:
                raise RuntimeError("action-shuffle checkpoint shard geometry is invalid")

            workers = self.state.setdefault(state_name, {})
            current = workers.get(worker_id)
            if current is not None:
                if (current.global_shard, current.total_shards) != (
                    candidate.global_shard,
                    candidate.total_shards,
                ):
                    raise RuntimeError(
                        f"action-shuffle worker geometry changed inside run for {state_name!r}/worker{worker_id}"
                    )
                if candidate.position < current.position:
                    raise RuntimeError(
                        f"action-shuffle worker position moved backwards for {state_name!r}/worker{worker_id}: "
                        f"{current.position} -> {candidate.position}"
                    )
            if current is None or candidate.position > current.position:
                workers[worker_id] = candidate

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: Any,
        iteration: int = 0,
    ) -> None:
        del model, output_batch, loss, iteration
        self._update_state_from_batch(data_batch)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[str, Any]:
        datasets: dict[str, dict[int, dict[str, int]]] = {}
        for state_name, workers in self.state.items():
            datasets[state_name] = {
                worker_id: {
                    "epoch": state.epoch,
                    "block_cursor": state.block_cursor,
                    "window_cursor": state.window_cursor,
                    "global_shard": state.global_shard,
                    "total_shards": state.total_shards,
                }
                for worker_id, state in workers.items()
            }
        log.info(f"Saved ActionIterableShuffle state for datasets={sorted(datasets)}", rank0_only=False)
        return {"schema_version": self.schema_version, "datasets": datasets}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            log.info("No ActionIterableShuffle dataloader state found", rank0_only=False)
            return
        if int(state_dict.get("schema_version", -1)) != self.schema_version:
            raise RuntimeError(
                f"unsupported ActionIterableShuffle state schema={state_dict.get('schema_version')!r}"
            )

        datasets = state_dict.get("datasets")
        if not isinstance(datasets, dict):
            raise RuntimeError("ActionIterableShuffle checkpoint is missing datasets mapping")

        restored: dict[str, dict[int, _WorkerState]] = {}
        for state_name, raw_workers in datasets.items():
            if not isinstance(raw_workers, dict):
                raise RuntimeError(f"invalid ActionIterableShuffle worker state for {state_name!r}")
            workers: dict[int, _WorkerState] = {}
            for raw_worker_id, raw_state in raw_workers.items():
                worker_id = int(raw_worker_id)
                if not isinstance(raw_state, dict):
                    raise RuntimeError(f"invalid ActionIterableShuffle worker payload for {state_name!r}")
                state = _WorkerState(
                    epoch=int(raw_state["epoch"]),
                    block_cursor=int(raw_state["block_cursor"]),
                    window_cursor=int(raw_state["window_cursor"]),
                    global_shard=int(raw_state["global_shard"]),
                    total_shards=int(raw_state["total_shards"]),
                )
                if min(state.epoch, state.block_cursor, state.window_cursor, state.global_shard) < 0:
                    raise RuntimeError("ActionIterableShuffle restored coordinates must be non-negative")
                if state.total_shards <= 0 or state.global_shard >= state.total_shards:
                    raise RuntimeError("ActionIterableShuffle restored shard geometry is invalid")
                workers[worker_id] = state

                prefix = action_shuffle_env_prefix(str(state_name), worker_id)
                os.environ[prefix + "EPOCH"] = str(state.epoch)
                os.environ[prefix + "BLOCK_CURSOR"] = str(state.block_cursor)
                os.environ[prefix + "WINDOW_CURSOR"] = str(state.window_cursor)
                os.environ[prefix + "GLOBAL_SHARD"] = str(state.global_shard)
                os.environ[prefix + "TOTAL_SHARDS"] = str(state.total_shards)
            restored[str(state_name)] = workers

        self.state = restored
        log.info(
            f"Loaded ActionIterableShuffle state for datasets={sorted(restored)}; "
            "child iterators must be lazily created after checkpoint load",
            rank0_only=False,
        )

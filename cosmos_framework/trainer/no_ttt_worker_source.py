# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verified, lazy row storage for the unchanged official LeRobot read methods.

Only the async worker uses this factory. It does not call LeRobot's eager
constructor, load all parquet files, or rebuild the accepted absolute row map.
Delta indices, padding, tensor queries and task captions still run through the
installed LeRobotDataset methods. The original nw0 dataset is untouched.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    hf_transform_to_torch,
)

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import _local_path
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex


@dataclass(frozen=True)
class _EpisodeRows:
    episode: int
    absolute: int
    relative: int
    length: int
    path: str


class _Column:
    def __init__(self, rows, key):
        self.rows, self.key = rows, key

    def __getitem__(self, indices):
        return self.rows.read(indices, self.key)


class _VerifiedRows:
    """HF indexing facade; at most a few accepted Episodes are materialized."""

    def __init__(self, root, meta, index, episodes, *, max_episodes, max_bytes):
        self.root, self.meta = Path(root), meta
        self.lookup = index.open_rows()
        self.features = get_hf_features_from_features(meta.features)
        if "episode_index" not in self.features or "index" not in self.features:
            raise ValueError("worker source requires LeRobot row identity columns")
        if type(max_episodes) is not int or max_episodes <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("worker source cache limits must be positive integers")
        self.max_episodes, self.max_bytes = max_episodes, max_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        spans = []
        for binding in index.receipt["bindings"]:
            ep = binding["episode_index"]
            row = meta.episodes[ep]
            start, end = row["dataset_from_index"], row["dataset_to_index"]
            length = row["length"]
            path = str(meta.get_data_file_path(ep))
            if (
                any(type(value) is not int for value in (ep, start, end, length))
                or length <= 0
                or end - start != length
                or binding["dataset_from_index"] != start
                or binding["dataset_to_index"] != end
                or binding["length"] != length
                or binding["data_file"] != path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("worker source verified episode metadata mismatch")
            first, last = self.lookup[start], self.lookup[end - 1]
            if last - first != length - 1:
                raise ValueError("worker source verified episode row mapping is not contiguous")
            spans.append(_EpisodeRows(ep, start, first, length, path))
        if sorted(span.episode for span in spans) != episodes or len(set(episodes)) != len(episodes):
            raise ValueError("worker source selected episode set differs from verified index")
        self.spans = sorted(spans, key=lambda span: span.relative)
        offset = 0
        for span in self.spans:
            if span.relative != offset:
                raise ValueError("worker source verified relative row spans overlap or contain gaps")
            offset += span.length
        if offset != len(self.lookup):
            raise ValueError("worker source verified row count mismatch")
        self.starts = [span.relative for span in self.spans]

    def __len__(self):
        return len(self.lookup)

    def _locate(self, relative):
        if type(relative) is not int or not 0 <= relative < len(self):
            raise IndexError(relative)
        span = self.spans[bisect_right(self.starts, relative) - 1]
        local = relative - span.relative
        if self.lookup[span.absolute + local] != relative:
            raise ValueError("worker source verified per-read row mapping mismatch")
        return span, local

    def _evict(self):
        _, (_, size) = self.cache.popitem(last=False)
        self.cache_bytes -= size

    def _load(self, span):
        if span.episode in self.cache:
            dataset, _ = self.cache[span.episode]
            self.cache.move_to_end(span.episode)
            return dataset
        while len(self.cache) >= self.max_episodes:
            self._evict()
        path = _local_path(self.root, Path(span.path), "worker source parquet")
        # Column projection, row-group pruning and bounded batches avoid the
        # full-dataset HF filter/index construction. Shared episode files work.
        batches, size = [], 0
        with pq.ParquetFile(path, memory_map=True) as parquet:
            ep_column = parquet.schema.names.index("episode_index")
            groups = []
            for number in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(number).column(ep_column).statistics
                if stats is None or not stats.has_min_max or stats.min <= span.episode <= stats.max:
                    groups.append(number)
            for batch in parquet.iter_batches(
                batch_size=4096, row_groups=groups, columns=list(self.features), use_threads=False
            ):
                selected = batch.filter(pc.equal(batch.column("episode_index"), span.episode))
                if not selected.num_rows:
                    continue
                size += selected.nbytes
                if size > self.max_bytes:
                    raise MemoryError("worker source episode exceeds bounded Arrow cache")
                while self.cache and self.cache_bytes + size > self.max_bytes:
                    self._evict()
                batches.append(selected)
        if not batches or sum(batch.num_rows for batch in batches) != span.length:
            raise ValueError("worker source parquet episode length mismatch")
        table = pa.Table.from_batches(batches).cast(self.features.arrow_schema)
        # Cast exactly as the official HF loader does (notably float32 action
        # and state). Conversion/tensor and caption semantics are not rewritten.
        if table.nbytes > self.max_bytes:
            raise MemoryError("worker source cast episode exceeds bounded Arrow cache")
        while self.cache and self.cache_bytes + table.nbytes > self.max_bytes:
            self._evict()
        dataset = datasets.Dataset(table, info=datasets.DatasetInfo(features=self.features))
        dataset.set_transform(hf_transform_to_torch)
        self.cache[span.episode] = (dataset, table.nbytes)
        self.cache_bytes += table.nbytes
        return dataset

    def read(self, indices, column=None):
        if type(indices) is int:
            span, local = self._locate(indices)
            dataset = self._load(span)
            return dataset[local] if column is None else dataset.select_columns([column])[local][column]
        locations = [self._locate(index) for index in indices]
        if not locations:
            return [] if column is not None else {key: [] for key in self.features}
        span = locations[0][0]
        if any(other != span for other, _ in locations):
            # Official non-visual deltas must remain inside the same Episode.
            raise ValueError("worker source queried rows across episode boundaries")
        dataset = self._load(span)
        local = [offset for _, offset in locations]
        return dataset[local] if column is None else dataset.select_columns([column])[local][column]

    def __getitem__(self, key):
        return _Column(self, key) if isinstance(key, str) else self.read(key)


class VerifiedWorkerLeRobotDataset(LeRobotDataset):
    """Read-only, verified-only construction; inherit all official read methods."""

    def __init__(
        self,
        *,
        repo_id,
        root,
        episodes,
        delta_timestamps,
        revision,
        force_cache_sync,
        download_videos,
        verified_index,
        max_episodes,
        max_bytes,
    ):
        if (
            type(verified_index) is not VerifiedExactWindowIndex
            or repo_id != "local"
            or revision != "local"
            or force_cache_sync is not False
            or download_videos is not False
        ):
            raise ValueError("worker source requires accepted local non-visual verified data")
        self.root, self.repo_id, self.revision = Path(root), repo_id, revision
        self.writer = None
        self._lazy_loading = False
        self.image_transforms = None
        self.episodes = episodes
        self.tolerance_s = 1e-4
        self.delta_timestamps = delta_timestamps
        self.meta = LeRobotDatasetMetadata(repo_id, self.root, revision, force_cache_sync=False)
        self.meta.info["features"] = {
            key: value for key, value in self.meta.features.items() if value.get("dtype") != "video"
        }
        check_delta_timestamps(delta_timestamps, self.fps, self.tolerance_s)
        self.delta_indices = get_delta_indices(delta_timestamps, self.fps)
        self.hf_dataset = _VerifiedRows(
            self.root, self.meta, verified_index, episodes, max_episodes=max_episodes, max_bytes=max_bytes
        )
        self._absolute_to_relative_idx = self.hf_dataset.lookup

    def load_hf_dataset(self):
        raise RuntimeError("worker source must never rebuild the eager LeRobot dataset")

    def download(self, download_videos=True):
        raise FileNotFoundError(f"worker source is offline and read-only: {self.root}")

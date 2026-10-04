# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bind exact cache windows to local flat LeRobot v3 non-visual rows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import _ensure_hf_hub_offline
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    ExactWindowIdentity,
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)

_ANNOTATION = "annotation.human.task_name"
_IDENTITY_COLUMNS = ("index", "episode_index", "frame_index", _ANNOTATION)
_REQUIRED_FEATURES = frozenset((*_IDENTITY_COLUMNS, "action", "observation.state", "task_index"))
_DELTA_LENGTHS = {
    "action": 16,
    "observation.state": 17,
    "index": 17,
    "episode_index": 17,
    "frame_index": 17,
    _ANNOTATION: 17,
}


def _delta_timestamps(fps: float) -> dict[str, list[float]]:
    return {key: [offset / fps for offset in range(length)] for key, length in _DELTA_LENGTHS.items()}


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _integer(value: object, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{label} 非标量")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 非整数：{value!r}")
    return value


def _vector(value: object, length: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (length,)
        or value.dtype
        not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
    ):
        raise ValueError(f"{label} shape/dtype 无效")
    return tuple(int(item) for item in value.tolist())


def _float_payload(value: object, shape: tuple[int, int], label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
        raise ValueError(f"{label} shape/dtype 无效，要求 {shape} 浮点")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} 包含非有限值")
    return value.to(dtype=torch.float32).contiguous()


def _local_path(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} 必须是 root 内相对路径：{relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} 路径越界：{relative}")
    return path


class _LocalNonVisualLeRobotDataset(LeRobotDataset):
    """保留官方 payload/delta/caption 路径，只封闭本地非视觉读取。"""

    def _check_cached_episodes_sufficient(self) -> bool:
        self.meta.info["features"] = {
            key: feature for key, feature in self.meta.info["features"].items() if feature.get("dtype") != "video"
        }
        return super()._check_cached_episodes_sufficient()

    def download(self, download_videos: bool = True) -> None:
        raise FileNotFoundError(f"Phase1B 本地非视觉 source 不足，禁止下载：{self.root}")


@dataclass(frozen=True)
class ExactWindowRawSourceWindow:
    key: ExactWindowEpisodeKey
    start_frame: int
    global_row_indices: tuple[int, ...]
    action12: torch.Tensor
    state16: torch.Tensor
    ai_caption: str
    task_index: int
    task_class: str
    source_binding_digest: str
    source_data_file: str


@dataclass(frozen=True)
class _BoundEpisode:
    key: ExactWindowEpisodeKey
    annotation_index: int
    length: int
    data_file: str
    dataset_from_index: int
    dataset_to_index: int
    first_rows: tuple[int, ...]
    terminal_rows: tuple[int, ...]


class CacheDrivenFlatWindowIndex:
    """只以 cache accepted episodes 定义窗口顺序和 shuffle block。"""

    def __init__(self, catalog: RoboCasaExactWindowCacheCatalog) -> None:
        self.catalog = catalog
        self._blocks = tuple(
            tuple((record.key, start) for start in record.window_starts) for record in catalog.episodes
        )
        self._windows = tuple(window for block in self._blocks for window in block)
        if len(self._windows) != catalog.stats.exact_window_count:
            raise ValueError("cache window index 数量不一致")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[ExactWindowEpisodeKey, int]:
        return self._windows[index]

    def get_shuffle_blocks(self) -> tuple[tuple[tuple[ExactWindowEpisodeKey, int], ...], ...]:
        return self._blocks


def _scan_identity(path: Path, episode_index: int) -> dict[str, tuple[int, ...]]:
    table = pq.read_table(path, columns=list(_IDENTITY_COLUMNS), filters=[("episode_index", "=", episode_index)])
    if table.num_rows == 0:
        raise ValueError(f"source episode={episode_index} 无身份行：{path}")
    values: dict[str, tuple[int, ...]] = {}
    for column in _IDENTITY_COLUMNS:
        raw = table[column].to_pylist()
        values[column] = tuple(_integer(item, f"{column} episode={episode_index}") for item in raw)
    order = sorted(range(table.num_rows), key=lambda row: values["index"][row])
    return {column: tuple(items[row] for row in order) for column, items in values.items()}


class RoboCasaExactWindowSourceReader:
    """Bind manifest identities once; every read uses official LeRobot delta sampling."""

    def __init__(
        self,
        catalog: RoboCasaExactWindowCacheCatalog,
        source_root: str | Path,
        *,
        metadata_factory: Callable[..., LeRobotDatasetMetadata] = LeRobotDatasetMetadata,
        dataset_factory: Callable[..., LeRobotDataset] = _LocalNonVisualLeRobotDataset,
        identity_scanner: Callable[[Path, int], dict[str, tuple[int, ...]]] = _scan_identity,
    ) -> None:
        self.catalog = catalog
        self.source_root = Path(source_root)
        self.index = CacheDrivenFlatWindowIndex(catalog)
        self.cache_reader = RoboCasaExactWindowEpisodeReader(catalog)
        _ensure_hf_hub_offline()
        self._preflight_metadata()
        self.meta = metadata_factory(repo_id="local", root=self.source_root, revision="local", force_cache_sync=False)
        self._validate_source_contract()
        episode_ids = [record.key.episode_index for record in catalog.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("cache episode_index 跨 task 重复")
        selected_files: dict[ExactWindowEpisodeKey, Path] = {}
        for record in catalog.episodes:
            ep_idx = record.key.episode_index
            if ep_idx < 0 or ep_idx >= len(self.meta.episodes):
                raise ValueError(f"source 缺少 cache episode_index={ep_idx}")
            relative = self.meta.get_data_file_path(ep_idx)
            path = _local_path(self.source_root, relative, "selected data")
            if not path.is_file():
                raise FileNotFoundError(f"selected data parquet 缺失：{path}")
            selected_files[record.key] = path
        self._bound: dict[ExactWindowEpisodeKey, _BoundEpisode] = {}
        for record in catalog.episodes:
            self._bound[record.key] = self._bind_episode(record, selected_files[record.key], identity_scanner)
        self.dataset = dataset_factory(
            repo_id="local",
            root=self.source_root,
            episodes=sorted(episode_ids),
            delta_timestamps=_delta_timestamps(catalog.stats.fps),
            revision="local",
            force_cache_sync=False,
            download_videos=False,
        )
        if self.dataset.meta.video_keys != []:
            raise ValueError("non-visual LeRobot reader 仍暴露 video_keys")
        loaded_episodes = {
            _integer(value, "hf_dataset.episode_index") for value in self.dataset.hf_dataset["episode_index"]
        }
        if loaded_episodes != set(episode_ids):
            raise ValueError(f"filtered hf_dataset episode set 与 cache 不一致：{sorted(loaded_episodes)}")
        self.abs_to_relative: dict[int, int] = {}
        for relative, absolute in enumerate(self.dataset.hf_dataset["index"]):
            index = _integer(absolute, "hf_dataset.index")
            if index in self.abs_to_relative:
                raise ValueError(f"hf_dataset absolute index 重复：{index}")
            self.abs_to_relative[index] = relative
        for record in catalog.episodes:
            for window_start in record.window_starts:
                for absolute in self._identity(record.key, window_start).global_row_indices:
                    if absolute not in self.abs_to_relative:
                        raise ValueError(f"cache witness 未在 filtered hf_dataset：{absolute}")
        self.source_binding_digest = _digest(
            {
                "cache_corpus_digest": catalog.corpus_digest,
                "source_codebase_version": self.meta.info["codebase_version"],
                "source_fps": self.meta.fps,
                "task_table": self._task_table_semantics(),
                "bound_episodes": [
                    {
                        "task_class": bound.key.task_class,
                        "episode_index": bound.key.episode_index,
                        "annotation_index": bound.annotation_index,
                        "length": bound.length,
                        "data_file": bound.data_file,
                        "dataset_from_index": bound.dataset_from_index,
                        "dataset_to_index": bound.dataset_to_index,
                        "first_rows": bound.first_rows,
                        "terminal_rows": bound.terminal_rows,
                    }
                    for record in catalog.episodes
                    for bound in (self._bound[record.key],)
                ],
            }
        )

    def _preflight_metadata(self) -> None:
        root = self.source_root
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"flat LeRobot v3 meta/info.json 缺失；不支持 multi-shard root：{root}")
        if not (root / "meta" / "tasks.parquet").is_file():
            raise FileNotFoundError(f"flat LeRobot v3 meta/tasks.parquet 缺失：{root}")
        if not (root / "meta" / "episodes").is_dir() or not tuple((root / "meta" / "episodes").rglob("*.parquet")):
            raise FileNotFoundError(f"flat LeRobot v3 meta/episodes parquet 缺失：{root}")
        if not (root / "data").is_dir():
            raise FileNotFoundError(f"flat LeRobot v3 data/ 缺失：{root}")

    def _validate_source_contract(self) -> None:
        info = self.meta.info
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(f"source codebase_version 非 v3：{info.get('codebase_version')!r}")
        if (
            isinstance(self.meta.fps, bool)
            or not math.isfinite(self.meta.fps)
            or self.meta.fps != self.catalog.stats.fps
        ):
            raise ValueError(f"source FPS 与 cache 不同：{self.meta.fps}/{self.catalog.stats.fps}")
        features = info.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("source features 非 mapping")
        missing = sorted(_REQUIRED_FEATURES - features.keys())
        if missing:
            raise ValueError(f"source non-visual features 缺失：{missing}")
        if self.meta.tasks is None or "task_index" not in self.meta.tasks.columns:
            raise ValueError("source meta.tasks 缺少 task_index")
        if self.meta.episodes is None:
            raise ValueError("source 缺少 episodes metadata")

    def _task_table_semantics(self) -> list[dict[str, object]]:
        return [
            {"name": str(name), "task_index": _integer(value, "meta.tasks.task_index")}
            for name, value in zip(self.meta.tasks.index, self.meta.tasks["task_index"], strict=True)
        ]

    def _resolve_class(self, annotation_index: int) -> str:
        matches = self.meta.tasks[self.meta.tasks["task_index"] == annotation_index]
        if len(matches) != 1 or not isinstance(matches.index[0], str) or not matches.index[0].strip():
            raise ValueError(f"annotation.human.task_name={annotation_index} task class 命中数={len(matches)}")
        return matches.index[0]

    def _identity(self, key: ExactWindowEpisodeKey, start: int) -> ExactWindowIdentity:
        identity = self.cache_reader.read_identity(key, start)
        if identity.global_row_indices is None:
            raise ValueError(f"Phase1B cache global_row_indices 缺失：{key}/{start}")
        return identity

    def _bind_episode(self, record: Any, path: Path, scanner: Callable) -> _BoundEpisode:
        key = record.key
        ep = self.meta.episodes[key.episode_index]
        start = _integer(ep["dataset_from_index"], "dataset_from_index")
        end = _integer(ep["dataset_to_index"], "dataset_to_index")
        length = _integer(ep["length"], "episode.length")
        if length < 17 or end - start != length or length - 16 != record.window_count:
            raise ValueError(f"source/cache episode length/window_count 不匹配：{key}")
        if record.source_video_frames is not None and length != record.source_video_frames:
            raise ValueError(f"source_video_frames 不匹配：{key}")
        rows = scanner(path, key.episode_index)
        expected_indices = tuple(range(start, end))
        if len(rows["index"]) != length or rows["index"] != expected_indices:
            raise ValueError(f"source row bounds/index 不匹配：{key}")
        if rows["episode_index"] != (key.episode_index,) * length:
            raise ValueError(f"source episode_index 漂移：{key}")
        if rows["frame_index"] != tuple(range(length)):
            raise ValueError(f"source frame_index 漂移：{key}")
        annotations = set(rows[_ANNOTATION])
        if len(annotations) != 1:
            raise ValueError(f"source annotation 多 task class：{key}")
        annotation_index = _integer(annotations.pop(), f"{_ANNOTATION} episode={key.episode_index}")
        if self._resolve_class(annotation_index) != key.task_class:
            raise ValueError(f"source annotation task_class 与 cache 不匹配：{key}")
        for window_start in record.window_starts:
            identity = self._identity(key, window_start)
            if identity.global_row_indices != rows["index"][window_start : window_start + 17]:
                if window_start in (0, record.window_count - 1):
                    raise ValueError(f"startup first/terminal global index witness 不匹配：{key}/{window_start}")
                # 中间窗仍保留 runtime 逐窗校验；其 absolute witness 必须可映射。
        relative = str(path.relative_to(self.source_root))
        return _BoundEpisode(
            key,
            annotation_index,
            length,
            relative,
            start,
            end,
            self._identity(key, 0).global_row_indices,
            self._identity(key, record.window_count - 1).global_row_indices,
        )

    def read_window(self, key: ExactWindowEpisodeKey, start_frame: int) -> ExactWindowRawSourceWindow:
        bound = self._bound[key]
        identity = self._identity(key, start_frame)
        absolute = identity.global_row_indices[0]
        try:
            relative = self.abs_to_relative[absolute]
        except KeyError as exc:
            raise ValueError(f"cache anchor 未在 filtered hf_dataset：{absolute}") from exc
        item = self.dataset[relative]
        for feature, length in _DELTA_LENGTHS.items():
            pad = item.get(f"{feature}_is_pad")
            if not isinstance(pad, torch.Tensor) or pad.shape != (length,) or bool(pad.any()):
                raise ValueError(f"{feature} padding/缺失：{key}/{start_frame}")
        if _vector(item.get("index"), 17, "index") != identity.global_row_indices:
            raise ValueError(f"runtime global index 漂移：{key}/{start_frame}")
        if _vector(item.get("episode_index"), 17, "episode_index") != (key.episode_index,) * 17:
            raise ValueError(f"runtime episode_index 漂移：{key}/{start_frame}")
        if _vector(item.get("frame_index"), 17, "frame_index") != identity.window_frame_indices:
            raise ValueError(f"runtime frame_index 漂移：{key}/{start_frame}")
        annotations = _vector(item.get(_ANNOTATION), 17, _ANNOTATION)
        if (
            annotations != (bound.annotation_index,) * 17
            or self._resolve_class(bound.annotation_index) != key.task_class
        ):
            raise ValueError(f"runtime annotation/task_class 漂移：{key}/{start_frame}")
        caption = item.get("task")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"official ai_caption 缺失或为空：{key}/{start_frame}")
        task_index = _integer(item.get("task_index"), "task_index")
        return ExactWindowRawSourceWindow(
            key,
            start_frame,
            identity.global_row_indices,
            _float_payload(item.get("action"), (16, 12), "action12"),
            _float_payload(item.get("observation.state"), (17, 16), "state16"),
            caption,
            task_index,
            key.task_class,
            self.source_binding_digest,
            bound.data_file,
        )

    def read_at(self, index: int) -> ExactWindowRawSourceWindow:
        return self.read_window(*self.index[index])

    def summary(self) -> dict[str, object]:
        return {
            "cache_corpus_digest": self.catalog.corpus_digest,
            "source_root": str(self.source_root),
            "source_codebase_version": self.meta.info["codebase_version"],
            "source_fps": self.meta.fps,
            "source_total_episodes": len(self.meta.episodes),
            "selected_cache_episodes": len(self._bound),
            "bound_windows": len(self.index),
            "mismatch_counters": {},
            "source_binding_digest": self.source_binding_digest,
            "source_revision": "unknown",
            "runtime_window_validation": "every_read",
            "offline_only": True,
            "loaded_episode_policy": "exact_cache_episode_set",
        }

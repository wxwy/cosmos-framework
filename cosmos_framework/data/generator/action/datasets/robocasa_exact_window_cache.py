# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cache-first RoboCasa exact-window catalog and lazy episode reader.

The manifest defines membership.  Source RoboCasa data and VAE encoding are
deliberately absent here; source-field binding belongs to a later phase.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex

_WINDOW_FRAMES = 17
_ANCHOR_STRIDE = 4


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} 必须是{'非负' if allow_zero else '正'}整数：{value!r}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _task_slug(task_class: str) -> str:
    """Match the exact-window builder's portable task directory naming."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_class).strip("_")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True, order=True)
class ExactWindowEpisodeKey:
    task_class: str
    task_slug: str
    episode_index: int


@dataclass(frozen=True)
class ExactWindowEpisodeRecord:
    key: ExactWindowEpisodeKey
    window_count: int
    source_video_frames: int | None
    relative_path: Path

    @property
    def window_starts(self) -> range:
        """The builder writes every stride-one start from 0 through N-1."""
        return range(self.window_count)


@dataclass(frozen=True)
class ExactWindowIdentity:
    key: ExactWindowEpisodeKey
    start_frame: int
    global_row_indices: tuple[int, ...] | None
    window_frame_indices: tuple[int, ...]
    latent_source_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class ExactWindowTaskStats:
    task_class: str
    task_slug: str
    episode_count: int
    window_count: int


@dataclass(frozen=True)
class ExactWindowCorpusStats:
    cache_root: str  # Display only; excluded from semantic corpus_digest.
    manifest_sha256: str
    corpus_digest: str
    schema_version: str
    source_format: str
    camera_set: str
    fps: float
    chunk_length: int
    task_class_count: int
    episode_count: int
    exact_window_count: int
    effective_consumer_count: int  # H_pred=16: one accepted consumer per exact window.
    source_unique_frame_count: int | None
    source_unique_frame_count_reason: str | None
    declared_episode_count: int
    discovered_episode_count: int
    accepted_episode_count: int
    rejected_episode_count: int
    missing_episode_count: int
    extra_episode_count: int
    duplicate_episode_count: int
    invalid_episode_count: int
    per_task: tuple[ExactWindowTaskStats, ...]
    min_episodes_per_task: int
    max_episodes_per_task: int
    min_windows_per_task: int
    max_windows_per_task: int

    def as_dict(self) -> dict[str, Any]:
        """Rank0-ready, JSON-compatible internal-cache-completeness accounting."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "per_task"}
        values["per_task"] = [
            {
                "task_class": task.task_class,
                "task_slug": task.task_slug,
                "episode_count": task.episode_count,
                "window_count": task.window_count,
            }
            for task in self.per_task
        ]
        values["payload_validation"] = "lazy_on_episode_and_window_read"
        values["completeness_scope"] = "declared_cache_corpus_only"
        return values


class RoboCasaExactWindowCacheCatalog:
    """Validate manifest identity and file presence without loading episode tensors."""

    def __init__(
        self, cache_root: str | Path, *, strict: bool = True, verified_index: VerifiedExactWindowIndex | None = None
    ) -> None:
        if verified_index is not None and not isinstance(verified_index, VerifiedExactWindowIndex):
            raise TypeError("cache catalog verified_index 类型不合法")
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "dataset_manifest.json"
        raw = manifest_path.read_bytes()
        try:
            manifest = _mapping(json.loads(raw), "dataset_manifest.json")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"无效 cache manifest：{manifest_path}") from exc
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if verified_index is not None and verified_index.cache_manifest_sha256 != self.manifest_sha256:
            raise ValueError("verified index 的 manifest SHA 与 cache 不匹配")
        self._validate_contract(manifest)
        self._records, task_stats = self._parse_tasks(manifest)
        missing, extra, discovered = self._audit_files()
        if missing:
            raise ValueError(f"cache manifest 声明的 episode 文件缺失（{len(missing)}）：{missing[:3]}")
        if extra and strict:
            raise ValueError(f"cache 发现未声明的 episode payload（extra={len(extra)}）：{extra[:3]}")
        self.episodes = tuple(self._records[key] for key in sorted(self._records))
        window_count = sum(record.window_count for record in self._records.values())
        known_frames = [record.source_video_frames for record in self._records.values()]
        all_frames_known = all(frames is not None for frames in known_frames)
        per_task = tuple(sorted(task_stats, key=lambda task: (task.task_class, task.task_slug)))
        # Only a verified immutable file snapshot may skip the 2M-window
        # canonical corpus-hash walk. The cold path remains byte-identical.
        self.corpus_digest = (
            verified_index.cache_corpus_digest if verified_index is not None else self._corpus_digest(manifest)
        )
        self.stats = ExactWindowCorpusStats(
            cache_root=str(self.cache_root),
            manifest_sha256=self.manifest_sha256,
            corpus_digest=self.corpus_digest,
            schema_version="exact_window_v1",
            source_format="lerobot_v3",
            camera_set="left_wrist",
            fps=20.0,
            chunk_length=16,
            task_class_count=len(per_task),
            episode_count=len(self._records),
            exact_window_count=window_count,
            effective_consumer_count=window_count,
            source_unique_frame_count=sum(known_frames) if all_frames_known else None,
            source_unique_frame_count_reason=None if all_frames_known else "source_video_frames 缺失于至少一个 episode",
            declared_episode_count=len(self._records),
            discovered_episode_count=discovered,
            accepted_episode_count=len(self._records),
            rejected_episode_count=len(extra),
            missing_episode_count=0,
            extra_episode_count=len(extra),
            duplicate_episode_count=0,
            invalid_episode_count=0,  # Payload validation is intentionally lazy.
            per_task=per_task,
            min_episodes_per_task=min(task.episode_count for task in per_task),
            max_episodes_per_task=max(task.episode_count for task in per_task),
            min_windows_per_task=min(task.window_count for task in per_task),
            max_windows_per_task=max(task.window_count for task in per_task),
        )
        if verified_index is not None:
            verified_index.check_catalog(self)

    def _validate_contract(self, manifest: Mapping[str, Any]) -> None:
        for name, expected in (
            ("schema_version", "exact_window_v1"),
            ("source_format", "lerobot_v3"),
            ("chunk_length", 16),
            ("sample_stride", 1),
            ("camera_set", "left_wrist"),
        ):
            if type(manifest.get(name)) is not type(expected) or manifest.get(name) != expected:
                raise ValueError(f"cache manifest {name} 不匹配：{manifest.get(name)!r} != {expected!r}")
        fps = manifest.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != 20.0:
            raise ValueError(f"cache manifest fps 必须为 20.0：{fps!r}")
        self.suite = _text(manifest.get("suite"), "suite")
        shape = manifest.get("latent_shape")
        if not isinstance(shape, list) or len(shape) != 4 or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"cache manifest latent_shape 无效：{shape!r}")
        if shape[:2] != [5, 48]:
            raise ValueError(f"cache manifest latent_shape 前缀必须为 [5,48]：{shape!r}")
        self.latent_shape = tuple(shape)
        for name, expected in (("action_shape_source", [12]), ("state_shape", [16])):
            if name in manifest and manifest[name] != expected:
                raise ValueError(f"cache manifest {name} 不匹配：{manifest[name]!r}")
        contract = _mapping(manifest.get("vae_encode_contract"), "vae_encode_contract")
        _text(contract.get("compute_dtype"), "vae_encode_contract.compute_dtype")
        durations = contract.get("encode_exact_durations")
        if (
            not isinstance(durations, list)
            or not durations
            or any(type(value) is not int or value <= 0 for value in durations)
            or len(set(durations)) != len(durations)
            or _WINDOW_FRAMES not in durations
        ):
            raise ValueError("vae_encode_contract.encode_exact_durations 必须是含17的正整数列表")
        chunks = _mapping(contract.get("encode_chunk_frames"), "vae_encode_contract.encode_chunk_frames")
        if not chunks or any(
            not isinstance(key, str) or not key or type(value) is not int or value <= 0 for key, value in chunks.items()
        ):
            raise ValueError("vae_encode_contract.encode_chunk_frames 无效")
        self._vae_encode_contract = deepcopy(dict(contract))

    @property
    def vae_encode_contract(self) -> dict[str, Any]:
        """Return the full manifest contract without imposing a historical list."""
        return deepcopy(self._vae_encode_contract)

    def _parse_tasks(
        self, manifest: Mapping[str, Any]
    ) -> tuple[dict[ExactWindowEpisodeKey, ExactWindowEpisodeRecord], list[ExactWindowTaskStats]]:
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("cache manifest tasks 必须是非空列表")
        records: dict[ExactWindowEpisodeKey, ExactWindowEpisodeRecord] = {}
        task_stats: list[ExactWindowTaskStats] = []
        classes: set[str] = set()
        slugs: set[str] = set()
        for task_value in tasks:
            task = _mapping(task_value, "task entry")
            task_class = _text(task.get("task_class"), "task_class")
            task_slug = _text(task.get("task_slug"), "task_slug")
            if task_slug != _task_slug(task_class) or task_slug in {".", ".."}:
                raise ValueError(f"task slug 与 builder 规则不一致：{task_class!r}/{task_slug!r}")
            if task_class in classes or task_slug in slugs:
                raise ValueError(f"重复 task class/slug：{task_class!r}/{task_slug!r}")
            classes.add(task_class)
            slugs.add(task_slug)
            rows = task.get("episodes")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"task {task_class!r} 没有 accepted episodes")
            task_windows = 0
            for row_value in rows:
                row = _mapping(row_value, "episode entry")
                index = _positive_int(row.get("episode_index"), "episode_index", allow_zero=True)
                count = _positive_int(row.get("window_count"), "window_count")
                if row.get("task_class") != task_class or row.get("task_slug") != task_slug:
                    raise ValueError(f"episode {index} 的 task identity 与父 task 不一致")
                key = ExactWindowEpisodeKey(task_class, task_slug, index)
                if key in records:
                    raise ValueError(f"重复 episode identity：{key}")
                frames = row.get("source_video_frames")
                if frames is not None:
                    frames = _positive_int(frames, "source_video_frames")
                    if frames < _WINDOW_FRAMES or count != frames - 16:
                        raise ValueError(f"episode {key} 的 window_count/source_video_frames 不一致")
                if "camera_set" in row and row["camera_set"] != "left_wrist":
                    raise ValueError(f"episode {key} 的 camera_set 不匹配")
                if "latent_shape" in row and row["latent_shape"] != list(self.latent_shape):
                    raise ValueError(f"episode {key} 的 latent_shape 不匹配")
                if "episode_path" in row and not isinstance(row["episode_path"], str):
                    raise ValueError(f"episode {key} 的 provenance episode_path 无效")
                relative = Path("tasks") / task_slug / "episodes" / f"episode_{index:06d}.pt"
                records[key] = ExactWindowEpisodeRecord(key, count, frames, relative)
                task_windows += count
            task_stats.append(ExactWindowTaskStats(task_class, task_slug, len(rows), task_windows))
        if not records:
            raise ValueError("cache manifest 没有 accepted episodes")
        for name, actual in (
            ("task_class_count", len(task_stats)),
            ("episode_count", len(records)),
            ("window_count", sum(record.window_count for record in records.values())),
        ):
            if _positive_int(manifest.get(name), name) != actual:
                raise ValueError(f"cache manifest {name} 不匹配：声明={manifest[name]!r} 实际={actual}")
        return records, task_stats

    def _audit_files(self) -> tuple[list[str], list[str], int]:
        expected = {record.relative_path for record in self._records.values()}
        missing = [str(path) for path in sorted(expected) if not (self.cache_root / path).is_file()]
        discovered: set[Path] = set()
        for path in (self.cache_root / "tasks").glob("*/episodes/*.pt"):
            relative = path.relative_to(self.cache_root)
            if not path.resolve().is_relative_to(self.cache_root.resolve()) or not path.is_file():
                raise ValueError(f"cache episode 路径越界或非文件：{relative}")
            discovered.add(relative)
        extra = [str(path) for path in sorted(discovered - expected)]
        return missing, extra, len(discovered)

    def _corpus_digest(self, manifest: Mapping[str, Any]) -> str:
        core = {
            "schema_version": "exact_window_v1",
            "source_format": "lerobot_v3",
            "suite": self.suite,
            "camera_set": "left_wrist",
            "fps": 20.0,
            "chunk_length": 16,
            "sample_stride": 1,
            "latent_shape": self.latent_shape,
            "action_shape_source": manifest.get("action_shape_source"),
            "state_shape": manifest.get("state_shape"),
            "vae_encode_contract": self._vae_encode_contract,
            "source_revision": manifest.get("source_revision"),
        }
        digest = hashlib.sha256(_canonical_bytes(core) + b"\n")
        for key in sorted(self._records):
            record = self._records[key]
            for start in record.window_starts:
                digest.update(_canonical_bytes((key.task_class, key.task_slug, key.episode_index, start)) + b"\n")
        return digest.hexdigest()

    def record_for(self, key: ExactWindowEpisodeKey) -> ExactWindowEpisodeRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"episode 不在 cache manifest：{key}") from exc


class RoboCasaExactWindowEpisodeReader:
    """Load only requested declared episodes; keep a bounded CPU payload LRU."""

    def __init__(self, catalog: RoboCasaExactWindowCacheCatalog, *, max_cached_episodes: int = 8) -> None:
        self.catalog = catalog
        self.max_cached_episodes = _positive_int(max_cached_episodes, "max_cached_episodes")
        self._loaded: OrderedDict[ExactWindowEpisodeKey, Mapping[str, Any]] = OrderedDict()

    def _episode(self, record: ExactWindowEpisodeRecord) -> Mapping[str, Any]:
        cached = self._loaded.get(record.key)
        if cached is not None:
            self._loaded.move_to_end(record.key)
            return cached
        path = self.catalog.cache_root / record.relative_path
        if not path.resolve().is_relative_to(self.catalog.cache_root.resolve()):
            raise ValueError(f"cache episode 路径越界：{path}")
        payload = _mapping(torch.load(path, map_location="cpu", weights_only=True), f"episode payload {record.key}")
        for name, expected in (("format", "exact_window_v1"), ("source_format", "lerobot_v3")):
            if payload.get(name) != expected:
                raise ValueError(f"episode {record.key} 的 {name} 不匹配")
        for name, expected in (
            ("suite", self.catalog.suite),
            ("task_class", record.key.task_class),
            ("episode_index", record.key.episode_index),
        ):
            if name in payload and payload[name] != expected:
                raise ValueError(f"episode {record.key} 的 {name} 不匹配")
        metadata = _mapping(payload.get("metadata"), "episode metadata")
        if metadata.get("camera_set") != "left_wrist":
            raise ValueError(f"episode {record.key} 的 camera_set 不匹配")
        for name, expected in (
            ("task_class", record.key.task_class),
            ("task_slug", record.key.task_slug),
            ("episode_index", record.key.episode_index),
        ):
            if name in metadata and metadata[name] != expected:
                raise ValueError(f"episode {record.key} metadata 的 {name} 不匹配")
        windows = _mapping(payload.get("windows"), "episode windows")
        if len(windows) != record.window_count or set(windows) != {str(start) for start in record.window_starts}:
            raise ValueError(f"episode {record.key} 的窗口键或 window_count 与 manifest 不一致")
        self._loaded[record.key] = payload
        if len(self._loaded) > self.max_cached_episodes:
            self._loaded.popitem(last=False)
        return payload

    def read_window(self, key: ExactWindowEpisodeKey, start_frame: int) -> torch.Tensor:
        """Return exact fp32 [5,48,H,W] latent; latent[0] is current z_t."""
        record = self.catalog.record_for(key)
        if type(start_frame) is not int or start_frame not in record.window_starts:
            raise KeyError(f"cache window 不存在：{key}, start_frame={start_frame!r}")
        windows = self._episode(record)["windows"]
        window = _mapping(windows.get(str(start_frame)), f"window {key}/{start_frame}")
        latent = window.get("latent")
        if (
            not isinstance(latent, torch.Tensor)
            or latent.dtype != torch.float32
            or tuple(latent.shape) != self.catalog.latent_shape
            or not bool(torch.isfinite(latent).all())
        ):
            raise ValueError(f"cache latent 无效：{key}, start_frame={start_frame}")
        for name, expected in (
            ("window_frame_indices", torch.arange(start_frame, start_frame + _WINDOW_FRAMES)),
            ("latent_source_frame_indices", torch.arange(start_frame, start_frame + _WINDOW_FRAMES, _ANCHOR_STRIDE)),
        ):
            indices = window.get(name)
            if (
                not isinstance(indices, torch.Tensor)
                or indices.dtype != torch.long
                or not torch.equal(indices, expected)
            ):
                raise ValueError(f"cache {name} 无效：{key}, start_frame={start_frame}")
        rows = window.get("global_row_indices")
        if rows is not None:
            if not isinstance(rows, torch.Tensor) or rows.shape != (_WINDOW_FRAMES,) or rows.dtype == torch.bool:
                raise ValueError(f"cache global_row_indices 无效：{key}, start_frame={start_frame}")
            if rows.is_floating_point():
                if not bool(torch.isfinite(rows).all()) or not bool(torch.equal(rows, rows.trunc())):
                    raise ValueError(f"cache global_row_indices 非整数：{key}, start_frame={start_frame}")
            elif rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"cache global_row_indices dtype 无效：{key}, start_frame={start_frame}")
        return latent.contiguous()

    def read_identity(self, key: ExactWindowEpisodeKey, start_frame: int) -> ExactWindowIdentity:
        """Read the exact cache witness without changing the Phase1A latent API."""
        record = self.catalog.record_for(key)
        if type(start_frame) is not int or start_frame not in record.window_starts:
            raise KeyError(f"cache window 不存在：{key}, start_frame={start_frame!r}")
        window = _mapping(self._episode(record)["windows"].get(str(start_frame)), f"window {key}/{start_frame}")
        expected_frames = torch.arange(start_frame, start_frame + _WINDOW_FRAMES)
        expected_anchors = expected_frames[::_ANCHOR_STRIDE]
        for name, expected in (
            ("window_frame_indices", expected_frames),
            ("latent_source_frame_indices", expected_anchors),
        ):
            value = window.get(name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or not torch.equal(value, expected):
                raise ValueError(f"cache {name} 无效：{key}, start_frame={start_frame}")
        rows = window.get("global_row_indices")
        if rows is not None:
            if not isinstance(rows, torch.Tensor) or rows.shape != (_WINDOW_FRAMES,) or rows.dtype == torch.bool:
                raise ValueError(f"cache global_row_indices 无效：{key}, start_frame={start_frame}")
            if rows.is_floating_point():
                if not bool(torch.isfinite(rows).all()) or not bool(torch.equal(rows, rows.trunc())):
                    raise ValueError(f"cache global_row_indices 非整数：{key}, start_frame={start_frame}")
            elif rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"cache global_row_indices dtype 无效：{key}, start_frame={start_frame}")
        return ExactWindowIdentity(
            key,
            start_frame,
            tuple(int(value) for value in rows.tolist()) if rows is not None else None,
            tuple(int(value) for value in expected_frames.tolist()),
            tuple(int(value) for value in expected_anchors.tolist()),
        )

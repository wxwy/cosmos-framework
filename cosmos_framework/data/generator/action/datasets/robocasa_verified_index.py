# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only, versioned cold-build receipt for the corrected RoboCasa dataset.

The receipt contains *only* static episode/source bindings and the absolute-row
lookup table, never VAE tensors, the LeRobot Dataset, or per-slot TTT state.
Warm load is fail-closed: no silent return to the expensive cold verification.

Payload .pt and source-data parquet files are pinned by stat identity, not by
re-hashing ~451 GiB on every boot. This requires an immutable, controlled data
snapshot. Runtime exact-window identity/tensor checks remain enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "psm_v3_robocasa_verified_index_v1"
RECEIPT = "receipt.json"
ROWS = "absolute_row_mapping.npy"


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _file_witness(root: Path, relative: str, *, hash_content: bool) -> dict[str, Any]:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative:
        raise ValueError(f"verified index 路径不是规范相对路径：{relative}")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"verified index 路径越界或符号链接：{path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"verified index 非普通文件：{path}")
    witness: dict[str, Any] = {
        "path": relative,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }
    if hash_content:
        witness["sha256"] = _sha256(path)
    return witness


def _source_metadata_files(source_root: Path) -> list[str]:
    meta = source_root / "meta"
    if not meta.is_dir():
        raise FileNotFoundError(f"verified index source 缺少 meta：{meta}")
    files = sorted(path for path in meta.rglob("*") if path.is_file() or path.is_symlink())
    return [path.relative_to(source_root).as_posix() for path in files]


class AbsoluteRowLookup(Mapping[int, int]):
    """Memory-mapped, sorted absolute-index -> filtered-LeRobot-row lookup."""

    def __init__(self, rows: np.ndarray) -> None:
        if rows.dtype != np.int64 or rows.ndim != 2 or rows.shape[1] != 2 or not len(rows):
            raise ValueError("verified index absolute-row array shape/dtype 无效")
        self._rows = rows

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def __iter__(self) -> Iterator[int]:
        return (int(key) for key in self._rows[:, 0])

    def __getitem__(self, absolute: int) -> int:
        if type(absolute) is not int:
            raise KeyError(absolute)
        location = int(np.searchsorted(self._rows[:, 0], absolute))
        if location >= len(self) or int(self._rows[location, 0]) != absolute:
            raise KeyError(absolute)
        return int(self._rows[location, 1])


class VerifiedExactWindowIndex:
    """Load a prebuilt immutable receipt, or explicitly build one from a cold reader."""

    def __init__(self, root: Path, receipt: dict[str, Any], rows_path: Path) -> None:
        self.root = root
        self.receipt = receipt
        self.rows_path = rows_path

    @property
    def cache_corpus_digest(self) -> str:
        return str(self.receipt["cache_corpus_digest"])

    @property
    def cache_manifest_sha256(self) -> str:
        return str(self.receipt["cache_manifest_sha256"])

    @property
    def source_binding_digest(self) -> str:
        return str(self.receipt["source_binding_digest"])

    @classmethod
    def open(cls, root: str | Path, *, cache_root: str | Path, source_root: str | Path) -> VerifiedExactWindowIndex:
        directory = Path(root)
        raw = (directory / RECEIPT).read_bytes()
        receipt = json.loads(raw)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != SCHEMA
            or not isinstance(receipt.get("bindings"), list)
            or not isinstance(receipt.get("files"), dict)
            or set(receipt["files"]) != {"cache_payload", "source_data", "source_meta"}
            or not isinstance(receipt.get("absolute_row_count"), int)
            or receipt["absolute_row_count"] <= 0
            or any(
                not isinstance(receipt.get(name), str) or len(receipt[name]) != 64
                for name in (
                    "cache_manifest_sha256",
                    "cache_corpus_digest",
                    "source_binding_digest",
                    "row_mapping_sha256",
                )
            )
        ):
            raise ValueError("verified index schema/receipt 不合法")
        if _sha256(directory / ROWS) != receipt["row_mapping_sha256"]:
            raise ValueError("verified index absolute-row 映射哈希不匹配")
        store = cls(directory, receipt, directory / ROWS)
        store.verify_files(cache_root=cache_root, source_root=source_root)
        return store

    def verify_files(self, *, cache_root: str | Path, source_root: str | Path) -> None:
        cache, source = Path(cache_root), Path(source_root)
        files = self.receipt["files"]
        # This also detects added/deleted metadata records; selected parquet and
        # .pt membership is independently checked by catalog + source binding.
        expected_meta = [item["path"] for item in files["source_meta"]]
        if expected_meta != _source_metadata_files(source):
            raise ValueError("verified index source metadata 文件集合发生变化")
        for field, directory, hash_content in (
            ("cache_payload", cache, False),
            ("source_data", source, False),
            ("source_meta", source, True),
        ):
            for entry in files[field]:
                if not isinstance(entry, dict) or "path" not in entry:
                    raise ValueError("verified index 文件 witness 结构不合法")
                try:
                    actual = _file_witness(directory, entry["path"], hash_content=hash_content)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"verified index 文件已丢失或不合法：{entry['path']}") from exc
                if actual != entry:
                    raise ValueError(f"verified index 文件发生变化，禁止 warm load：{entry['path']}")
        if _sha256(cache / "dataset_manifest.json") != self.cache_manifest_sha256:
            raise ValueError("verified index cache manifest 发生变化")

    def check_catalog(self, catalog: Any) -> None:
        if (
            catalog.manifest_sha256 != self.cache_manifest_sha256
            or catalog.corpus_digest != self.cache_corpus_digest
            or len(catalog.episodes) != len(self.receipt["bindings"])
            or catalog.stats.exact_window_count != self.receipt["exact_window_count"]
        ):
            raise ValueError("verified index cache catalog 身份或几何不匹配")

    def open_rows(self) -> AbsoluteRowLookup:
        rows = np.load(self.rows_path, mmap_mode="r", allow_pickle=False)
        if rows.shape != (self.receipt["absolute_row_count"], 2) or rows.dtype != np.int64:
            raise ValueError("verified index absolute-row 映射长度不匹配")
        # The cold builder stores sorted unique keys and bijective, dense relative
        # positions. File checksum is verified at open, so warm load is O(1).
        return AbsoluteRowLookup(rows)

    @classmethod
    def build(
        cls,
        root: str | Path,
        *,
        cache_catalog: Any,
        source_reader: Any,
    ) -> VerifiedExactWindowIndex:
        destination = Path(root)
        if destination.exists():
            raise FileExistsError(f"verified index 已存在，禁止覆盖：{destination}")
        if source_reader.catalog is not cache_catalog or len(source_reader._bound) != len(cache_catalog.episodes):
            raise ValueError("verified index 只能从完成 strict cold binding 的 reader 构建")
        cache_root, source_root = cache_catalog.cache_root, source_reader.source_root
        if not isinstance(source_reader.abs_to_relative, dict):
            raise ValueError("verified index cold build 要求完整 absolute-row map")
        bindings = []
        for record in cache_catalog.episodes:
            bound = source_reader._bound[record.key]
            bindings.append(
                {
                    "task_class": record.key.task_class,
                    "task_slug": record.key.task_slug,
                    "episode_index": record.key.episode_index,
                    "window_count": record.window_count,
                    "annotation_index": bound.annotation_index,
                    "length": bound.length,
                    "data_file": bound.data_file,
                    "dataset_from_index": bound.dataset_from_index,
                    "dataset_to_index": bound.dataset_to_index,
                    "first_rows": list(bound.first_rows),
                    "terminal_rows": list(bound.terminal_rows),
                }
            )
        cache_files = sorted(record.relative_path.as_posix() for record in cache_catalog.episodes)
        selected_data_files = sorted({bound.data_file for bound in source_reader._bound.values()})
        source_meta_files = _source_metadata_files(source_root)
        files = {
            "cache_payload": [_file_witness(cache_root, path, hash_content=False) for path in cache_files],
            "source_data": [_file_witness(source_root, path, hash_content=False) for path in selected_data_files],
            "source_meta": [_file_witness(source_root, path, hash_content=True) for path in source_meta_files],
        }
        count = len(source_reader.abs_to_relative)
        values = np.fromiter(
            (v for pair in source_reader.abs_to_relative.items() for v in pair),
            dtype=np.int64,
            count=count * 2,
        ).reshape(count, 2)
        values = values[np.argsort(values[:, 0], kind="stable")]
        if count and (
            (values[1:, 0] <= values[:-1, 0]).any() or (np.sort(values[:, 1]) != np.arange(count, dtype=np.int64)).any()
        ):
            raise ValueError("verified index cold absolute-row map 非一一映射")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staged = parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        staged.mkdir()
        try:
            with (staged / ROWS).open("wb") as file:
                np.save(file, values, allow_pickle=False)
            receipt = {
                "schema": SCHEMA,
                "cache_manifest_sha256": cache_catalog.manifest_sha256,
                "cache_corpus_digest": cache_catalog.corpus_digest,
                "source_binding_digest": source_reader.source_binding_digest,
                "exact_window_count": cache_catalog.stats.exact_window_count,
                "absolute_row_count": count,
                "row_mapping_sha256": _sha256(staged / ROWS),
                "bindings": bindings,
                "files": files,
                "integrity_scope": "metadata_content_sha256_plus_payload_file_stat_and_runtime_window_checks",
            }
            (staged / RECEIPT).write_text(
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            if destination.exists():
                raise FileExistsError(f"verified index 并发构建冲突：{destination}")
            os.rename(staged, destination)
        finally:
            if staged.exists():
                import shutil

                shutil.rmtree(staged)
        return cls.open(destination, cache_root=cache_root, source_root=source_root)

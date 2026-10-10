# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-shared payload cache; inherited per-window identity validation is unchanged."""
from __future__ import annotations

import os
from collections.abc import Mapping
from threading import RLock
from typing import Any

import torch

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import RoboCasaExactWindowEpisodeReader
from cosmos_framework.utils.singleflight_cache import SingleFlightLRU


def tensor_storage_bytes(value: Any) -> int:
    """Count unique CPU tensor storages, not logical expanded tensor elements."""
    seen: set[tuple[int, int]] = set()
    total = 0
    def visit(item: Any) -> None:
        nonlocal total
        if isinstance(item, torch.Tensor):
            if item.device.type != 'cpu':
                raise ValueError('episode payload must stay on CPU')
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return total


def _env_positive(name: str, default: int) -> int:
    text = os.environ.get(name, str(default))
    if not text.isascii() or not text.isdigit() or int(text) <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(text)


class SharedRoboCasaEpisodeReader(RoboCasaExactWindowEpisodeReader):
    """One immutable payload per episode across source/raw reads and CPU workers.

    The original reader owns the load/validation code. A fresh loader for a MISS
    avoids sharing its mutable OrderedDict. Inherited read_window/read_identity
    keep all original per-read validation; cache hits do not waive these checks.
    """

    def __init__(self, catalog, *, max_cached_episodes: int = 8, max_bytes: int = 1024**3) -> None:
        super().__init__(catalog, max_cached_episodes=max_cached_episodes)
        self._shared = SingleFlightLRU(max_entries=max_cached_episodes, max_bytes=max_bytes, sizeof=tensor_storage_bytes)
        self._file_lock = RLock()
        self._identities: dict[Any, tuple[int, ...]] = {}

    def _version(self, record) -> tuple[int, ...]:
        path = self.catalog.cache_root / record.relative_path
        if not path.resolve().is_relative_to(self.catalog.cache_root.resolve()):
            raise ValueError('episode payload escapes cache root')
        st = path.stat()
        if not path.is_file() or st.st_size <= 0:
            raise ValueError('episode payload missing or empty')
        version = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        with self._file_lock:
            old = self._identities.setdefault(record.key, version)
            if old != version:
                raise ValueError('episode file changed during this reader lifetime')
        return version

    def _episode(self, record):
        version = self._version(record)
        def load():
            reader = RoboCasaExactWindowEpisodeReader(self.catalog, max_cached_episodes=1)
            payload = reader._episode(record)
            if self._version(record) != version:
                raise ValueError('episode payload changed while loading')
            return payload
        return self._shared.get((record.key, version), load)

    def cache_stats(self) -> dict[str, int | float]:
        return self._shared.stats()


def install_rank_shared_reader(raw) -> SharedRoboCasaEpisodeReader | None:
    """Construction-time wiring only. No dataset reinitialization or DCP mutation."""
    flag = os.environ.get('PSM_V3_SHARED_EPISODE_CACHE', '1')
    if flag not in {'0', '1'}:
        raise ValueError('PSM_V3_SHARED_EPISODE_CACHE must be 0 or 1')
    if flag == '0':
        return None
    if isinstance(raw.cache_reader, SharedRoboCasaEpisodeReader):
        return raw.cache_reader
    shared = SharedRoboCasaEpisodeReader(
        raw.catalog,
        max_cached_episodes=_env_positive('PSM_V3_EPISODE_CACHE_ENTRIES', 8),
        max_bytes=_env_positive('PSM_V3_EPISODE_CACHE_MIB', 1024) * 1024**2,
    )
    raw.cache_reader = shared
    raw.source_reader.cache_reader = shared
    return shared

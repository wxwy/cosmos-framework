# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded shared read-only payload cache: one concurrent load per identity."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from threading import RLock
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class SingleFlightLRU(Generic[K, V]):
    """Cache immutable values; load different keys outside the lock.

    Bounds cover cached references, not outstanding callers or in-flight loader
    allocations. A payload larger than the byte budget is returned but not kept.
    Failed loads are never cached. No background threads are owned here.
    """

    def __init__(self, *, max_entries: int, max_bytes: int, sizeof: Callable[[V], int]) -> None:
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_entries, self.max_bytes, self.sizeof = max_entries, max_bytes, sizeof
        self._lock = RLock()
        self._cache: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._pending: dict[K, Future[V]] = {}
        self._bytes = 0
        self._stats: dict[str, int | float] = {
            "hits": 0,
            "misses": 0,
            "coalesced": 0,
            "loads": 0,
            "failures": 0,
            "evictions": 0,
            "oversized": 0,
            "load_ms": 0.0,
            "wait_ms": 0.0,
            "peak_cached_bytes": 0,
        }

    def get(self, key: K, load: Callable[[], V]) -> V:
        with self._lock:
            if key in self._cache:
                value, _ = self._cache[key]
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
                return value
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._pending[key] = future
                self._stats["misses"] += 1
            else:
                self._stats["coalesced"] += 1
        assert future is not None
        started = time.perf_counter()
        if not owner:
            try:
                return future.result()
            finally:
                with self._lock:
                    self._stats["wait_ms"] += (time.perf_counter() - started) * 1000
        try:
            value = load()
            size = self.sizeof(value)
            if type(size) is not int or size < 0:
                raise ValueError("sizeof must return a nonnegative integer")
            with self._lock:
                if size <= self.max_bytes:
                    while self._cache and (len(self._cache) >= self.max_entries or self._bytes + size > self.max_bytes):
                        _, (_, removed_size) = self._cache.popitem(last=False)
                        self._bytes -= removed_size
                        self._stats["evictions"] += 1
                    self._cache[key] = (value, size)
                    self._bytes += size
                    self._stats["peak_cached_bytes"] = max(self._stats["peak_cached_bytes"], self._bytes)
                else:
                    self._stats["oversized"] += 1
                self._stats["loads"] += 1
            # Publish while the in-flight entry is still owned: a waiter cannot
            # start a duplicate load during the completion boundary.
            future.set_result(value)
            return value
        except BaseException as error:
            with self._lock:
                self._stats["failures"] += 1
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)
                self._stats["load_ms"] += (time.perf_counter() - started) * 1000

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                **self._stats,
                "cached_entries": len(self._cache),
                "cached_bytes": self._bytes,
                "inflight_keys": len(self._pending),
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
            }

    def clear(self) -> None:
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot clear a cache with in-flight reads")
            self._cache.clear()
            self._bytes = 0

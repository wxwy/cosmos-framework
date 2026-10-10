# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded CPU-only raw-window prefetch, never a speculative Local/optimizer step.

Every worker owns independent VAE episode LRUs and a shallow LeRobot SourceReader
view. The loaded HF/Arrow table, immutable metadata and static verified index
are shared read-only. Official transforms/tokenization run on the trainer thread
in exactly the original slot/window order. No async state enters a DCP.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import local
from typing import Any

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowSegmentProducer,
    ExactWindowSegmentRequest,
    PreparedExactWindowSegment,
)


class AsyncExactWindowRawPrefetcher:
    """Bound memory to ONE GA member (<= B_stream prepared segments) per rank."""

    def __init__(self, producer: ExactWindowSegmentProducer, *, num_workers: int) -> None:
        if not isinstance(producer, ExactWindowSegmentProducer):
            raise TypeError("async prefetch requires exact-window producer")
        if type(num_workers) is not int or not 1 <= num_workers <= 16:
            raise ValueError("async raw num_workers must be 1..16; 0 disables async")
        self.producer = producer
        self.num_workers = num_workers
        self._threads = local()
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="v3_raw")
        self._pending: tuple[
            tuple[ExactWindowSegmentRequest, ...], tuple[Future[PreparedExactWindowSegment], ...]
        ] | None = None
        self.iteration_wait_ms = 0.0
        self._closed = False

    def _worker_raw(self) -> Any:
        existing = getattr(self._threads, "raw", None)
        if existing is not None:
            return existing
        original = self.producer.catalog.raw
        official = getattr(original.source_reader, "dataset", None)
        if official is not None and (
            getattr(official, "hf_dataset", None) is None or getattr(official, "_lazy_loading", False)
        ):
            raise RuntimeError("async source reader requires a fully loaded, read-only HF dataset")
        source = copy.copy(original.source_reader)
        # Never mutate the main rank reader or share its mutable OrderedDict LRU.
        source.cache_reader = RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        raw = copy.copy(original)
        raw.source_reader = source
        raw.cache_reader = RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        self._threads.raw = raw
        return raw

    def _prepare(self, request: ExactWindowSegmentRequest) -> PreparedExactWindowSegment:
        return self.producer.prepare(request, raw_getter=self._worker_raw().__getitem__)

    def reset_iteration(self) -> None:
        if self._closed or self._pending is not None:
            raise RuntimeError("async prefetch previous GA member not consumed or prefetch closed")
        self.iteration_wait_ms = 0.0

    def schedule(self, requests: tuple[ExactWindowSegmentRequest, ...]) -> None:
        if self._closed or self._pending is not None:
            raise RuntimeError("async prefetch queue must be empty before schedule")
        if not requests:
            raise ValueError("async prefetch cannot schedule an empty member")
        futures = tuple(self._executor.submit(self._prepare, request) for request in requests)
        self._pending = (requests, futures)

    def load_member(
        self, requests: tuple[ExactWindowSegmentRequest, ...]
    ) -> tuple[PreparedExactWindowSegment, ...]:
        if self._pending is None:
            self.schedule(requests)
        assert self._pending is not None
        planned, futures = self._pending
        if requests != planned:
            raise ValueError("async prefetch pending member identity/order changed")
        started = time.perf_counter()
        try:
            prepared = tuple(future.result() for future in futures)
            if any(item.request != request for item, request in zip(prepared, planned, strict=True)):
                raise ValueError("async prefetch produced a different ordered member")
        except BaseException:
            self.abort()
            raise
        finally:
            self.iteration_wait_ms += (time.perf_counter() - started) * 1000.0
        self._pending = None
        return prepared

    def abort(self) -> None:
        """Drain/cancel pending pure reads before trainer aborts the candidate Frontier."""
        current, self._pending = self._pending, None
        if current is None:
            return
        for future in current[1]:
            future.cancel()
        for future in current[1]:
            try:
                future.result()
            except BaseException:
                # The original failure is raised by load_member/trainer; never mask it.
                pass

    def close(self) -> None:
        if self._closed:
            return
        self.abort()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

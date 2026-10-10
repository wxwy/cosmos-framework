# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded raw-window prefetch; transforms, Local state and commit stay on trainer thread."""

from __future__ import annotations

import copy
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
from cosmos_framework.utils.ordered_prefetch import OrderedMemberPrefetch


class AsyncExactWindowRawPrefetcher(OrderedMemberPrefetch[ExactWindowSegmentRequest, PreparedExactWindowSegment]):
    """One GA member per rank, private reader LRUs, shared immutable HF/Arrow table."""

    def __init__(self, producer: ExactWindowSegmentProducer, *, num_workers: int) -> None:
        if not isinstance(producer, ExactWindowSegmentProducer):
            raise TypeError("async prefetch requires exact-window producer")
        self.producer = producer
        self._threads = local()
        # Lambda resolves _prepare at invocation time so failure-injection probes
        # exercise the same worker callback used by the training adapter.
        super().__init__(lambda request: self._prepare(request), lambda result: result.request, num_workers=num_workers)

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
        # A worker gets its own LeRobot wrapper too, not just its own SourceReader.
        # HF tables/metadata are still shared read-only; no dataset re-initialization.
        if official is not None:
            source.dataset = copy.copy(official)
        source.cache_reader = RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        raw = copy.copy(original)
        raw.source_reader = source
        raw.cache_reader = RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        self._threads.raw = raw
        return raw

    def _prepare(self, request: ExactWindowSegmentRequest) -> PreparedExactWindowSegment:
        return self.producer.prepare(request, raw_getter=self._worker_raw().__getitem__)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in bounded, one-member-ahead CPU Segment/Native-collate prefetch.

Only immutable request capabilities cross the thread boundary. The worker does
not access CUDA, Local state, optimizer, trainer scheduler or checkpoint state.
One transform thread prevents concurrent tokenizer processor use. Cross-window
lookahead is speculative and must be discarded when the old window aborts.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import local
from typing import Any

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_shared_episode_reader import SharedRoboCasaEpisodeReader
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowSegmentProducer,
    ExactWindowSegmentRequest,
)


@dataclass(frozen=True)
class FullSegmentMember:
    requests: tuple[ExactWindowSegmentRequest, ...]
    segments: tuple[SegmentBatch, ...]
    payloads: tuple[dict[str, Any], ...]
    collated: dict[str, Any]


class AsyncExactWindowFullSegmentPrefetcher:
    """Exactly one pending member; never owns Local scan/commit or CUDA.

    This intentionally does not create a multi-thread tokenizer pool. There
    are B=8 segments * T=16 transforms per single worker task. Transform RNG
    is stateless per consumer under producer.consumer_keyed_v1.
    """

    def __init__(self, producer: ExactWindowSegmentProducer) -> None:
        if not isinstance(producer, ExactWindowSegmentProducer) or producer.transform_rng_mode != "consumer_keyed_v1":
            raise ValueError("full segment prefetch requires a consumer-keyed exact-window producer")
        self.producer = producer
        self._thread = local()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3_full_seg")
        self._pending: tuple[tuple[ExactWindowSegmentRequest, ...], Future[FullSegmentMember]] | None = None
        self._closed = False
        self.iteration_wait_ms = 0.0

    def _worker_raw(self) -> Any:
        raw = getattr(self._thread, "raw", None)
        if raw is not None:
            return raw
        original = self.producer.catalog.raw
        official = getattr(original.source_reader, "dataset", None)
        if official is not None and (
            getattr(official, "hf_dataset", None) is None or getattr(official, "_lazy_loading", False)
        ):
            raise RuntimeError("full segment prefetch requires loaded, read-only HF source")
        source = copy.copy(original.source_reader)
        if official is not None:
            source.dataset = copy.copy(official)
        shared = getattr(original, "cache_reader", None)
        shared = shared if isinstance(shared, SharedRoboCasaEpisodeReader) else None
        source.cache_reader = shared or RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        raw = copy.copy(original)
        raw.source_reader = source
        raw.cache_reader = shared or RoboCasaExactWindowEpisodeReader(original.catalog, max_cached_episodes=2)
        self._thread.raw = raw
        return raw

    @staticmethod
    def _collate(requests: tuple[ExactWindowSegmentRequest, ...], segments: tuple[SegmentBatch, ...]) -> FullSegmentMember:
        # Lazy import avoids trainer/producer circular dependency.
        from cosmos_framework.trainer.local_memory_grouped import collate_grouped_native_batch

        if not requests or len(requests) != len(segments):
            raise ValueError("full member requests/segments do not match")
        payloads: list[dict[str, Any]] = []
        for request, segment in zip(requests, segments, strict=True):
            identity = request.identity
            if (
                int(segment.slot_id[0]) != identity.slot_id
                or segment.episode_id[0] != identity.episode_id
                or segment.segment_provenance.source_digest != identity.source_digest
                or segment.segment_provenance.segment_id != identity.segment_id
                or int(segment.consumer_valid.sum()) != request.valid_count
            ):
                raise ValueError("prefetched segment request identity/valid_count mismatch")
            for index in segment.consumer_valid[0].nonzero(as_tuple=False).flatten().tolist():
                payload = segment.consumer_payload[0][index]
                if not isinstance(payload, dict):
                    raise ValueError("prefetched native consumer must have dict payload")
                payloads.append(payload)
        return FullSegmentMember(requests, segments, tuple(payloads), collate_grouped_native_batch(tuple(payloads)))

    def _build(self, requests: tuple[ExactWindowSegmentRequest, ...]) -> FullSegmentMember:
        raw = self._worker_raw()
        # The sole worker reads, transforms, validates and collates an entire
        # member sequentially, while the trainer runs the previous member.
        segments = tuple(
            self.producer.materialize(self.producer.prepare(request, raw_getter=raw.__getitem__))
            for request in requests
        )
        return self._collate(requests, segments)

    def direct_member(self, requests: tuple[ExactWindowSegmentRequest, ...]) -> FullSegmentMember:
        """Main-thread GA0 on first window; same keyed transform as worker."""
        if type(requests) is not tuple or not requests:
            raise ValueError("full member requires frozen tuple requests")
        segments = tuple(self.producer.produce(request) for request in requests)
        return self._collate(requests, segments)

    def reset_iteration(self) -> None:
        if self._closed:
            raise RuntimeError("full member prefetch is closed")
        # Unlike the Raw GA1-only prefetcher, keep a speculative next-window
        # member0 future. On failure abort() drains and discards it.
        self.iteration_wait_ms = 0.0

    def schedule(self, requests: tuple[ExactWindowSegmentRequest, ...]) -> None:
        if self._closed or self._pending is not None:
            raise RuntimeError("full member prefetch must be open with empty bounded slot")
        if type(requests) is not tuple or not requests:
            raise ValueError("full member prefetch requires nonempty immutable tuple")
        self._pending = requests, self._executor.submit(self._build, requests)

    def load_member(self, requests: tuple[ExactWindowSegmentRequest, ...]) -> FullSegmentMember:
        if self._pending is None:
            self.schedule(requests)
        assert self._pending is not None
        planned, future = self._pending
        start = time.perf_counter()
        try:
            if planned != requests:
                raise ValueError("full segment prefetch request identity/order changed")
            result = future.result()
            if result.requests != planned:
                raise ValueError("full segment prefetch produced wrong request identity")
        except BaseException:
            self.abort()
            raise
        finally:
            self.iteration_wait_ms += (time.perf_counter() - start) * 1000.0
        self._pending = None
        return result

    def abort(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        future = pending[1]
        future.cancel()
        try:
            future.result()
        except BaseException:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.abort()
        self._executor.shutdown(wait=True, cancel_futures=True)

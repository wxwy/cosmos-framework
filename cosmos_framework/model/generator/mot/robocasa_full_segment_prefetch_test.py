# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU full Segment Prefetch: exact order, identity and one bounded future."""

from __future__ import annotations

import random

import pytest

import cosmos_framework.trainer.local_memory_grouped as grouped_module
from cosmos_framework.model.generator.mot.robocasa_async_segment_prefetch_test import _assert_segment_equal
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowSegmentProducer
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _request, _setup
from cosmos_framework.model.generator.mot.robocasa_full_segment_prefetch import (
    AsyncExactWindowFullSegmentPrefetcher,
)


def _keyed_producer():
    _, transform, catalog, _ = _setup((33, 33), t=16)
    return transform, catalog, ExactWindowSegmentProducer(
        catalog, config_digest="config", transform_rng_mode="consumer_keyed_v1"
    )


def test_keyed_consumer_transform_never_advances_process_random(monkeypatch: pytest.MonkeyPatch) -> None:
    transform, catalog, producer = _keyed_producer()
    source_type = type(transform)
    old = source_type.__call__
    from cosmos_framework.data.generator.action.utils.transform_random import transform_draw

    def stochastic(self, item, resolution):
        item["test_cfg_rng"] = transform_draw(random.random)
        return old(self, item, resolution)

    monkeypatch.setattr(source_type, "__call__", stochastic)
    random.seed(191)
    before = random.getstate()
    one = producer.produce(_request(catalog, slot=0))
    two = producer.produce(_request(catalog, slot=0))
    assert random.getstate() == before
    assert tuple(p["test_cfg_rng"] for p in one.consumer_payload[0]) == tuple(
        p["test_cfg_rng"] for p in two.consumer_payload[0]
    )


def test_one_worker_full_member_matches_keyed_sync_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _, catalog, producer = _keyed_producer()
    reqs = (_request(catalog, slot=0, episode_index=0), _request(catalog, slot=1, episode_index=1))
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda items: {"count": len(items)})
    sync = tuple(producer.produce(req) for req in reqs)
    queue = AsyncExactWindowFullSegmentPrefetcher(producer)
    monkeypatch.setattr(queue, "_worker_raw", lambda: catalog.raw)
    try:
        queue.reset_iteration()
        queue.schedule(reqs)
        with pytest.raises(RuntimeError, match="empty bounded"):
            queue.schedule(reqs)
        output = queue.load_member(reqs)
        assert output.requests == reqs
        assert output.collated["count"] == 32
        for lhs, rhs in zip(sync, output.segments, strict=True):
            _assert_segment_equal(lhs, rhs)
        assert output.payloads[0] is output.segments[0].consumer_payload[0][0]
        assert output.payloads[16] is output.segments[1].consumer_payload[0][0]
        assert queue.iteration_wait_ms >= 0
        assert queue._pending is None
    finally:
        queue.close()


def test_full_member_cross_window_pending_survives_reset_and_is_identity_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, catalog, producer = _keyed_producer()
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda items: {"count": len(items)})
    reqs = (_request(catalog, slot=0, episode_index=0),)
    wrong = (_request(catalog, slot=1, episode_index=1),)
    queue = AsyncExactWindowFullSegmentPrefetcher(producer)
    monkeypatch.setattr(queue, "_worker_raw", lambda: catalog.raw)
    try:
        queue.schedule(reqs)
        queue.reset_iteration()
        assert queue.load_member(reqs).collated["count"] == 16
        queue.schedule(reqs)
        with pytest.raises(ValueError, match="identity/order"):
            queue.load_member(wrong)
        assert queue._pending is None
    finally:
        queue.close()


def test_full_member_worker_error_discards_speculative_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    _, catalog, producer = _keyed_producer()
    queue = AsyncExactWindowFullSegmentPrefetcher(producer)
    monkeypatch.setattr(queue, "_worker_raw", lambda: catalog.raw)

    def broken(_item):
        raise ValueError("bad transform")

    monkeypatch.setattr(producer, "materialize", broken)
    try:
        queue.schedule((_request(catalog, slot=0),))
        with pytest.raises(ValueError, match="bad transform"):
            queue.load_member((_request(catalog, slot=0),))
        assert queue._pending is None
        queue.reset_iteration()
    finally:
        queue.close()


def test_full_member_requires_keyed_rng() -> None:
    _, _, catalog, old_producer = _setup()
    with pytest.raises(ValueError, match="consumer-keyed"):
        AsyncExactWindowFullSegmentPrefetcher(old_producer)

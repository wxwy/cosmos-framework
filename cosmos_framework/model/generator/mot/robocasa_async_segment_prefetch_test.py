# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contracts: bounded raw prefetch may not change Segment ABI or transform order."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.robocasa_async_segment_prefetch import (
    AsyncExactWindowRawPrefetcher,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _request, _setup


def _assert_segment_equal(left, right) -> None:
    assert left.segment_provenance == right.segment_provenance
    assert left.episode_id == right.episode_id
    assert left.category == right.category
    for key in (
        "consumer_visual_summary",
        "consumer_valid",
        "consumer_step",
        "evidence_visual_summary_prev",
        "evidence_executed_action_prev",
        "evidence_valid",
        "evidence_source_step",
        "slot_id",
    ):
        torch.testing.assert_close(getattr(left, key), getattr(right, key), atol=0, rtol=0)
    for a, b in zip(left.consumer_payload[0], right.consumer_payload[0], strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.keys() == b.keys()
            for key in a:
                if isinstance(a[key], torch.Tensor):
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                else:
                    assert a[key] == b[key]


@pytest.mark.parametrize("cursor", [0, 1, 2])
def test_sequential_prepare_materialize_exact_segment_parity(cursor: int) -> None:
    _, transform, catalog, producer = _setup((33,), t=16)
    request = _request(catalog, cursor=cursor)
    baseline = producer.produce(request)
    transform.calls.clear()
    prepared = producer.prepare(request)
    assert not transform.calls
    candidate = producer.materialize(prepared)
    _assert_segment_equal(baseline, candidate)
    expected = min(catalog.ttt_tbptt_steps, catalog.episodes[0].window_count - cursor * 16)
    assert transform.calls == list(range(cursor * 16, cursor * 16 + expected))


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_async_only_reads_raw_and_preserves_ordered_transform(num_workers: int) -> None:
    raw, transform, catalog, producer = _setup((33, 33), t=16)
    req0 = _request(catalog, cursor=1, episode_index=0, slot=0)
    req1 = _request(catalog, cursor=0, episode_index=1, slot=1)
    baseline = tuple(producer.produce(request) for request in (req0, req1))
    transform.calls.clear()
    prefetcher = AsyncExactWindowRawPrefetcher(producer, num_workers=num_workers)
    try:
        prefetcher.reset_iteration()
        prefetcher.schedule((req0, req1))
        assert not transform.calls
        prepared = prefetcher.load_member((req0, req1))
        assert not transform.calls
        actual = tuple(producer.materialize(item) for item in prepared)
        for expected, observed in zip(baseline, actual, strict=True):
            _assert_segment_equal(expected, observed)
        assert transform.calls == list(range(16, 32)) + list(range(16))
        assert prefetcher.iteration_wait_ms >= 0
        assert prefetcher._pending is None
        assert raw.source_reader.source_binding_digest == "binding"
    finally:
        prefetcher.close()


def test_async_does_not_consume_transform_rng_before_main_thread() -> None:
    _, transform, catalog, producer = _setup((17,), t=16)
    request = _request(catalog)
    original = transform.__call__

    def randomized(item, resolution):
        item["test_rng"] = (random.random(), float(torch.rand(())))
        return original(item, resolution)

    # Instance __call__ is bound by type, so patch the class instead.
    transform.__class__.__call__ = randomized
    try:
        random.seed(723)
        torch.manual_seed(723)
        baseline = producer.produce(request)
        expected_rng = [payload["test_rng"] for payload in baseline.consumer_payload[0]]
        random.seed(723)
        torch.manual_seed(723)
        prefetcher = AsyncExactWindowRawPrefetcher(producer, num_workers=2)
        try:
            prefetcher.reset_iteration()
            prefetcher.schedule((request,))
            prepared = prefetcher.load_member((request,))
            observed = producer.materialize(prepared[0])
            assert [payload["test_rng"] for payload in observed.consumer_payload[0]] == expected_rng
        finally:
            prefetcher.close()
    finally:
        transform.__class__.__call__ = original


def test_async_rejects_request_reordering_and_drains_after_abort() -> None:
    _, _, catalog, producer = _setup((33,), t=16)
    first = _request(catalog, cursor=0)
    second = _request(catalog, cursor=1)
    prefetcher = AsyncExactWindowRawPrefetcher(producer, num_workers=2)
    try:
        prefetcher.reset_iteration()
        prefetcher.schedule((second,))
        with pytest.raises(ValueError, match="order changed"):
            prefetcher.load_member((first,))
        prefetcher.abort()
        assert prefetcher._pending is None
        prefetcher.reset_iteration()
        prefetcher.schedule((first,))
        assert len(prefetcher.load_member((first,))) == 1
    finally:
        prefetcher.close()


def test_async_worker_failure_never_leaves_pending_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, catalog, producer = _setup((33,), t=16)
    request = _request(catalog)

    def failed_prepare(_request, **kwargs):
        raise RuntimeError("read failure")

    monkeypatch.setattr(producer, "prepare", failed_prepare)
    prefetcher = AsyncExactWindowRawPrefetcher(producer, num_workers=1)
    try:
        prefetcher.reset_iteration()
        prefetcher.schedule((request,))
        with pytest.raises(RuntimeError, match="read failure"):
            prefetcher.load_member((request,))
        assert prefetcher._pending is None
        prefetcher.reset_iteration()
    finally:
        prefetcher.close()


@pytest.mark.parametrize("workers", [-1, 0, True, 17])
def test_bad_worker_count_fail_closed(workers) -> None:
    _, _, _, producer = _setup()
    with pytest.raises((ValueError, TypeError)):
        AsyncExactWindowRawPrefetcher(producer, num_workers=workers)


def test_async_close_rejects_future_work() -> None:
    _, _, catalog, producer = _setup()
    prefetcher = AsyncExactWindowRawPrefetcher(producer, num_workers=1)
    prefetcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        prefetcher.schedule((_request(catalog),))

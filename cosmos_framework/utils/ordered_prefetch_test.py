# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise the real CPU executor with deterministic synthetic requests, not a model."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event, get_ident

import pytest

from cosmos_framework.utils.ordered_prefetch import OrderedMemberPrefetch


@pytest.mark.parametrize("workers", [1, 2, 4, 16])
def test_worker_threads_are_real_and_results_remain_in_request_order(workers: int) -> None:
    main_id = get_ident()
    worker_ids = []

    def prepare(request: int) -> tuple[int, int]:
        worker_ids.append(get_ident())
        return request, request * request

    queue = OrderedMemberPrefetch(prepare, lambda result: result[0], num_workers=workers)
    try:
        queue.reset_iteration()
        queue.schedule((5, 2, 7))
        assert queue.load_member((5, 2, 7)) == ((5, 25), (2, 4), (7, 49))
        assert all(thread_id != main_id for thread_id in worker_ids)
        assert queue._pending is None
        assert queue.iteration_wait_ms >= 0
    finally:
        queue.close()


def test_out_of_order_completion_does_not_reorder_consumption() -> None:
    second_finished = Event()

    def prepare(request: int) -> int:
        if request == 0:
            assert second_finished.wait(timeout=5)
        else:
            second_finished.set()
        return request

    queue = OrderedMemberPrefetch(prepare, lambda result: result, num_workers=2)
    try:
        queue.schedule((0, 1))
        assert queue.load_member((0, 1)) == (0, 1)
    finally:
        second_finished.set()
        queue.close()


def test_partial_submit_failure_retains_and_cancels_prior_future(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=1)
    pending: Future[int] = Future()
    calls = 0

    def broken_submit(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return pending
        raise RuntimeError("synthetic submit failure")

    monkeypatch.setattr(queue._executor, "submit", broken_submit)
    try:
        with pytest.raises(RuntimeError, match="submit failure"):
            queue.schedule((0, 1))
        assert pending.cancelled()
        assert queue._pending is None
        queue.reset_iteration()
    finally:
        queue.close()


def test_wrong_request_order_is_terminal_for_pending_member() -> None:
    queue = OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=2)
    try:
        queue.schedule((0, 1))
        with pytest.raises(ValueError, match="order changed"):
            queue.load_member((1, 0))
        assert queue._pending is None
        queue.reset_iteration()
    finally:
        queue.close()


def test_worker_result_identity_is_validated() -> None:
    queue = OrderedMemberPrefetch(lambda request: request + 1, lambda result: result, num_workers=1)
    try:
        with pytest.raises(ValueError, match="different ordered member"):
            queue.load_member((1,))
        assert queue._pending is None
    finally:
        queue.close()


def test_worker_failure_is_not_swallowed_and_other_reads_are_drained() -> None:
    completed = Event()

    def prepare(request: int) -> int:
        if request == 0:
            assert completed.wait(timeout=5)
            raise OSError("synthetic read failure")
        completed.set()
        return request

    queue = OrderedMemberPrefetch(prepare, lambda result: result, num_workers=2)
    try:
        with pytest.raises(OSError, match="read failure"):
            queue.load_member((0, 1))
        assert completed.is_set()
        assert queue._pending is None
        queue.reset_iteration()
    finally:
        completed.set()
        queue.close()


def test_no_second_member_can_be_scheduled_before_first_is_consumed() -> None:
    queue = OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=1)
    try:
        queue.schedule((0,))
        with pytest.raises(RuntimeError, match="empty"):
            queue.schedule((1,))
        with pytest.raises(RuntimeError, match="previous"):
            queue.reset_iteration()
        assert queue.load_member((0,)) == (0,)
        assert queue.load_member((1,)) == (1,)
    finally:
        queue.close()


@pytest.mark.parametrize("requests", [(), [], [1]])
def test_mutable_or_empty_request_container_is_rejected(requests) -> None:
    queue = OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=1)
    try:
        with pytest.raises(ValueError, match="immutable"):
            queue.schedule(requests)
    finally:
        queue.close()


@pytest.mark.parametrize("workers", [-1, 0, True, 1.5, 17])
def test_invalid_concurrency_is_rejected(workers) -> None:
    with pytest.raises(ValueError, match="num_workers"):
        OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=workers)


def test_close_is_idempotent_and_rejects_new_work() -> None:
    queue = OrderedMemberPrefetch(lambda request: request, lambda result: result, num_workers=1)
    queue.schedule((0,))
    queue.close()
    queue.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue.schedule((1,))
    with pytest.raises(RuntimeError, match="closed"):
        queue.reset_iteration()

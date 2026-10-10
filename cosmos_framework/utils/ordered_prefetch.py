# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""A one-member CPU future queue; no dataset, model, RNG or checkpoint ownership."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class OrderedMemberPrefetch(Generic[RequestT, ResultT]):
    """Prepare concurrently and consume in frozen request order.

    The caller alone schedules/loads/aborts. It may submit one member only;
    there is no cross-window speculation. Shutdown drains running pure reads
    rather than pretending that a running Python thread can be cancelled.
    """

    def __init__(
        self,
        prepare: Callable[[RequestT], ResultT],
        result_request: Callable[[ResultT], RequestT],
        *,
        num_workers: int,
    ) -> None:
        if type(num_workers) is not int or not 1 <= num_workers <= 16:
            raise ValueError("async raw num_workers must be 1..16; 0 disables async")
        self.num_workers = num_workers
        self._prepare_request = prepare
        self._result_request = result_request
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="v3_raw")
        self._pending: tuple[tuple[RequestT, ...], Sequence[Future[ResultT]]] | None = None
        self.iteration_wait_ms = 0.0
        self._closed = False

    def reset_iteration(self) -> None:
        if self._closed or self._pending is not None:
            raise RuntimeError("async prefetch previous GA member not consumed or prefetch closed")
        self.iteration_wait_ms = 0.0

    def schedule(self, requests: tuple[RequestT, ...]) -> None:
        if self._closed or self._pending is not None:
            raise RuntimeError("async prefetch queue must be empty and not closed before schedule")
        if type(requests) is not tuple or not requests:
            raise ValueError("async prefetch requires a nonempty immutable request tuple")
        # Publish ownership before submit: even a mid-submission failure must drain
        # previously submitted futures. A tuple comprehension loses those handles.
        futures: list[Future[ResultT]] = []
        self._pending = (requests, futures)
        try:
            for request in requests:
                futures.append(self._executor.submit(self._prepare_request, request))
        except BaseException:
            self.abort()
            raise
        self._pending = (requests, tuple(futures))

    def load_member(self, requests: tuple[RequestT, ...]) -> tuple[ResultT, ...]:
        if self._pending is None:
            self.schedule(requests)
        assert self._pending is not None
        planned, futures = self._pending
        started = time.perf_counter()
        try:
            if requests != planned:
                raise ValueError("async prefetch pending member identity/order changed")
            prepared = tuple(future.result() for future in futures)
            if any(self._result_request(item) != request for item, request in zip(prepared, planned, strict=True)):
                raise ValueError("async prefetch produced a different ordered member")
        except BaseException:
            self.abort()
            raise
        finally:
            self.iteration_wait_ms += (time.perf_counter() - started) * 1000.0
        self._pending = None
        return prepared

    def abort(self) -> None:
        current, self._pending = self._pending, None
        if current is None:
            return
        for future in current[1]:
            future.cancel()
        for future in current[1]:
            try:
                future.result()
            except BaseException:
                # Drain all scheduled reads without masking the original failure.
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.abort()
        self._executor.shutdown(wait=True, cancel_futures=True)

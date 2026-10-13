# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scoped per-consumer CPU transform RNG. Legacy callers keep global random exactly."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_TRANSFORM_RNG: ContextVar[random.Random | None] = ContextVar("psm_v3_transform_rng", default=None)


@contextmanager
def scoped_transform_rng(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    if type(seed) is not int or seed < 0:
        raise ValueError("transform seed must be a nonnegative int")
    token = _TRANSFORM_RNG.set(random.Random(seed))
    try:
        yield
    finally:
        _TRANSFORM_RNG.reset(token)


def transform_draw(legacy_draw: Callable[[], float]) -> float:
    """Keep legacy global RNG when no scope; keyed mode never consumes it."""
    scoped = _TRANSFORM_RNG.get()
    return legacy_draw() if scoped is None else scoped.random()

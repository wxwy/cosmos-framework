# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Legacy RNG preserved and opt-in keyed RNG independent of CPU worker order."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from cosmos_framework.data.generator.action.utils.transform_random import (
    scoped_transform_rng,
    transform_draw,
)


def test_unscoped_transform_uses_exact_legacy_global_random(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "random", lambda: 0.375)
    assert transform_draw(random.random) == 0.375


def test_keyed_transform_is_repeatable_without_consuming_global_random() -> None:
    original = random.getstate()
    try:
        random.seed(141)
        frozen = random.getstate()
        with scoped_transform_rng(123):
            first = [transform_draw(random.random) for _ in range(5)]
        with scoped_transform_rng(123):
            second = [transform_draw(random.random) for _ in range(5)]
        assert first == second
        assert random.getstate() == frozen
    finally:
        random.setstate(original)


def test_keyed_transform_isolated_across_threads_and_nested_scopes() -> None:
    def sample(seed):
        with scoped_transform_rng(seed):
            before = transform_draw(random.random)
            with scoped_transform_rng(999):
                assert transform_draw(random.random) == random.Random(999).random()
            after = transform_draw(random.random)
            return before, after

    with ThreadPoolExecutor(max_workers=3) as workers:
        actual = list(workers.map(sample, (19, 3, 19, 6)))
    assert actual[0] == actual[2]
    assert actual[0] != actual[1]


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_scoped_seed_rejected(invalid) -> None:
    with pytest.raises(ValueError, match="seed"):
        with scoped_transform_rng(invalid):
            pass

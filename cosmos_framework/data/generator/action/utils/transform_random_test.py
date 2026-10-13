# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Legacy RNG preserved and opt-in keyed RNG independent of CPU worker order."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

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

def test_real_text_tokenizer_uses_keyed_cfg_rng_without_touching_global_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.data.generator.augmentors import text_tokenizer

    class FakeProcessor:
        def tokenize_text(self, caption: str, **kwargs):
            return [len(caption)]

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _: FakeProcessor())
    transform = text_tokenizer.TextTokenizerTransform(
        input_keys=["ai_caption"],
        output_keys=["text_token_ids"],
        args={"tokenizer_config": object(), "cfg_dropout_rate": 0.5},
    )
    state = random.getstate()
    try:
        with scoped_transform_rng(141):
            a = transform({"ai_caption": "a mug"})
        with scoped_transform_rng(141):
            b = transform({"ai_caption": "a mug"})
        torch.testing.assert_close(a["text_token_ids"], b["text_token_ids"], rtol=0, atol=0)
        assert a["ai_caption"] == b["ai_caption"]
        assert random.getstate() == state
    finally:
        random.setstate(state)


def test_real_idle_metadata_dropout_uses_scoped_rng_without_global_consumption() -> None:
    from cosmos_framework.data.generator.augmentors.idle_frames_text_info import IdleFramesTextInfo

    transform = IdleFramesTextInfo(
        args={"dropout_rate": 0.5, "enabled": True, "caption_key": "ai_caption"}
    )
    state = random.getstate()
    try:
        with scoped_transform_rng(212):
            first = transform({"ai_caption": "grasp cup", "idle_frames": 2, "action": torch.zeros(17, 15)})
        with scoped_transform_rng(212):
            second = transform({"ai_caption": "grasp cup", "idle_frames": 2, "action": torch.zeros(17, 15)})
        assert first["ai_caption"] == second["ai_caption"]
        assert random.getstate() == state
    finally:
        random.setstate(state)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU contract tests; no real tokenizer/UND/data alignment claim."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.utils import static_task_text as shared
from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.static_task_text import (
    StaticTaskText,
    StaticTaskTextRecipe,
    effective_text_batch,
    effective_text_mask,
    prepare_static_task_text,
    resolve_consumer_text_dropout,
    serialize_static_task_prompt,
)
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline


class ProcessorFixture:
    def tokenize_text(self, text, *, is_video, use_system_prompt):
        assert is_video is False and use_system_prompt is False
        return [7, *text.encode(), 9]  # Real empty input has special tokens.


def task_fixture():
    return StaticTaskText(
        "Close the fridge",
        "Close the fridge. Initial spatial context: fridge in kitchen.",
        "fixture-episode",
        "fixture-subtask",
        "fixture-template",
        "fixture-source",
        "fixture-initial-sg",
    )


def recipe_fixture(**kwargs):
    return StaticTaskTextRecipe(17, 20.0, 192, 320, "fixture-tokenizer", **kwargs)


def pipeline_fixture(monkeypatch, **kwargs):
    import cosmos_framework.data.generator.action.utils.transforms as transforms

    monkeypatch.setattr(
        transforms, "TextTokenizerTransform", lambda **_: SimpleNamespace(_processor=ProcessorFixture())
    )
    recipe = recipe_fixture()
    return ActionTransformPipeline(
        tokenizer_config={},
        cfg_dropout_rate=0.1,
        max_action_dim=64,
        format_prompt_as_json=True,
        append_idle_frame_text=False,
        static_task_text_recipe=recipe,
        **kwargs,
    )


def sample_fixture(**kwargs):
    return {
        "ai_caption": "transient caption must not be used",
        "static_task_text": asdict(task_fixture()),
        "video": torch.zeros(3, 17, 32, 64, dtype=torch.uint8),
        "action": torch.zeros(17, 15),
        "mode": "wam",
        "conditioning_fps": torch.tensor(20),
        "viewpoint": "concat_view",
        "idle_frames": 15,
        "start_frame": 12,
        "current_sg": "moving object",
        **kwargs,
    }


def test_legacy_json_golden_and_disabled_idle_never_reads_action_labels():
    sample = {
        "ai_caption": "Close fridge",
        "video": torch.zeros(3, 17, 2, 4),
        "image_size": (2, 4),
        "conditioning_fps": 20,
        "action": torch.zeros(16, 15),
        "idle_frames": 3,
    }
    legacy = ActionPromptJsonFormatter()(dict(sample))["ai_caption"]
    assert legacy == {
        "actions": [{"time": "0:00-0:01", "description": "Close fridge.", "idle_frame": "3 out of 16."}],
        "duration": "0s",
        "fps": 20.0,
        "resolution": {"H": 2, "W": 4},
        "aspect_ratio": "2,1",
    }
    sample.update(idle_frames=object(), action=object(), relabel_prompt={"idle_frame": "bad future label"})
    new = ActionPromptJsonFormatter(append_idle_frame_text=False)(sample)["ai_caption"]
    assert "idle_frame" not in new["actions"][0]


@pytest.mark.parametrize("json_prompt", [False, True])
def test_explicit_idle_switch_reaches_both_pipeline_paths(json_prompt):
    pipeline = ActionTransformPipeline(
        format_prompt_as_json=json_prompt,
        append_idle_frames=True,
        append_idle_frame_text=False,
    )
    result = pipeline(sample_fixture(), None)
    assert "idle_frame" not in str(result["ai_caption"])
    assert result["idle_frames"] == 15  # Legitimate statistics and real targets survive.
    assert torch.equal(result["action_raw"], torch.zeros(17, 15))


@pytest.mark.parametrize("profile", ["R1", "R2"])
def test_static_tokens_ignore_future_action_window_current_state_and_clean_row_count(monkeypatch, profile):
    del profile  # Same shared recipe, not separate formatters.
    pipeline = pipeline_fixture(monkeypatch)
    left = pipeline(sample_fixture(text_dropout_applied=False), None)
    right = pipeline(
        sample_fixture(
            text_dropout_applied=False,
            action=torch.randn(18, 15),
            start_frame=999,
            idle_frames=0,
            current_sg="other location",
            state15=torch.randn(15),
        ),
        None,
    )
    for key in ("text_token_ids", "text_attention_mask", "text_position_ids"):
        assert torch.equal(left[key], right[key])
    assert left["text_condition_identity"] == right["text_condition_identity"]
    assert "idle_frame" not in left["ai_caption"] and "initial" in left["ai_caption"].lower()
    assert json.loads(left["ai_caption"])["actions"][0]["spatial_context_scope"] == "initial_spatial_context"
    assert "0:00-0:01" in left["ai_caption"]  # Fixed 17/20, not action rows.


def test_train_and_server_use_identical_serialization_tokenizer_and_condition(monkeypatch):
    train = pipeline_fixture(monkeypatch)(sample_fixture(text_dropout_applied=False), None)
    service = prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=False)
    assert (
        train["ai_caption"] == service["ai_caption"] == serialize_static_task_prompt(task_fixture(), recipe_fixture())
    )
    for key in ("text_token_ids", "text_attention_mask", "text_position_ids"):
        assert torch.equal(train[key], service[key])
    assert train["text_condition_identity"] == service["text_condition_identity"]


def test_consumer_draws_once_and_recomputation_reuses_genuine_empty_tokens(monkeypatch):
    draws = []
    monkeypatch.setattr(shared, "transform_draw", lambda _: draws.append(1) or 0.05)
    sample = {}
    first = resolve_consumer_text_dropout(sample, 0.1)
    second = resolve_consumer_text_dropout(sample, 0.1)
    assert first is second is True and draws == [1]
    a = prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=first)
    b = prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=second)
    assert torch.equal(a["text_token_ids"], b["text_token_ids"])
    assert a["text_token_ids"].tolist() == [7, 9] and a["text_attention_mask"].all()
    assert a["ai_caption"] == "" and a["text_condition_identity"] == b["text_condition_identity"]
    full = prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=False)
    assert a["text_condition_identity"] != full["text_condition_identity"]


def test_no_token_empty_input_is_not_an_all_masked_fake_hidden():
    class NoTokenProcessor:
        def tokenize_text(self, text, **kwargs):
            assert text == ""
            return []

    result = prepare_static_task_text(task_fixture(), recipe_fixture(), NoTokenProcessor(), dropout_applied=True)
    assert result["text_token_ids"].shape == result["text_attention_mask"].shape == (0,)
    assert result["text_token_ids"].dtype == torch.long


def test_ragged_prepared_text_uses_existing_native_collator():
    from cosmos_framework.trainer.local_memory_grouped import collate_grouped_native_batch

    values = [
        prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=drop)
        for drop in (False, True)
    ]
    for value in values:
        value["sequence_plan"] = "fixture-plan"
        value["video"] = torch.zeros(3, 17, 2, 4)
    batch = collate_grouped_native_batch(tuple(values))
    assert len(batch["text_token_ids"]) == len(batch["text_attention_mask"]) == 2
    for index, value in enumerate(values):
        for key in ("text_token_ids", "text_attention_mask", "text_position_ids"):
            assert torch.equal(batch[key][index][0], value[key])


def test_recipe_and_asset_fail_closed():
    with pytest.raises(ValueError, match="initial"):
        replace(task_fixture(), context_scope="current_state")
    with pytest.raises(ValueError, match="positive"):
        replace(recipe_fixture(), planned_video_frames=0)
    with pytest.raises(TypeError, match="bool"):
        resolve_consumer_text_dropout({"text_dropout_applied": 0}, 0.1)


def test_c0_final_tokens_include_actual_native_specials_and_mrope():
    samples = [
        prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=drop)
        for drop in (False, True)
    ]
    specials = {"bos_token_id": 101, "eos_token_id": 102, "start_of_generation": 103}
    train = effective_text_batch(samples, recipe_fixture(), special_tokens=specials)
    service = effective_text_batch(samples, recipe_fixture(), special_tokens=specials)
    assert train == service
    assert train.token_ids[1] == (101, 7, 9, 102, 103)
    assert train.position_ids[1] == ((0, 1, 2, 3, 4),) * 3
    assert train.dropped == (False, True)
    mask = effective_text_mask(train)
    assert tuple(mask.sum(dim=1).tolist()) == train.lengths


def test_final_text_protocol_stacks_mixed_full_empty_b1_service_conditions():
    from cosmos_framework.model.generator.mot.g2_contracts import UNDTextFeatures, stack_g2_prepared

    samples = [
        prepare_static_task_text(task_fixture(), recipe_fixture(), ProcessorFixture(), dropout_applied=drop)
        for drop in (False, True)
    ]
    specials = {"bos_token_id": 101, "eos_token_id": 102, "start_of_generation": 103}
    texts = [effective_text_batch([sample], recipe_fixture(), special_tokens=specials) for sample in samples]
    features = tuple(
        UNDTextFeatures(
            text=text, hidden=torch.zeros(1, text.lengths[0], 8), valid_mask=effective_text_mask(text), layers=()
        )
        for text in texts
    )
    carrier = stack_g2_prepared(
        text_features=features, text_kv=(None, None), global_tokens=(None, None), global_kv=(None, None)
    )
    assert carrier.text == effective_text_batch(samples, recipe_fixture(), special_tokens=specials)


def test_g2_episode_static_token_cache_preserves_native_outputs_cfg_rng_and_clones():
    import random
    from collections import OrderedDict
    from types import SimpleNamespace

    import torch

    from cosmos_framework.data.generator.action.utils.static_task_text import (
        prepare_static_task_text,
        resolve_consumer_text_dropout,
    )
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    class Processor:
        def __init__(self):
            self.calls = 0

        def tokenize_text(self, text, **kwargs):
            self.calls += 1
            return [7, len(text) % 101, 8]

    recipe = recipe_fixture()
    task = task_fixture()
    live = Processor()
    transform = object.__new__(ActionTransformPipeline)
    transform._g2_static_token_cache = OrderedDict()
    transform.static_task_text_recipe = recipe
    transform.text_tokenizer = SimpleNamespace(_processor=live)
    reference = Processor()

    cached_rows = []
    uncached_rows = []
    random.seed(732)
    for i in range(60):
        draw = resolve_consumer_text_dropout({}, recipe.cfg_dropout_rate)
        cached_rows.append(transform._prepare_cached_g2_text(task, dropout_applied=draw))
    rng_cached = random.getstate()
    random.seed(732)
    for i in range(60):
        draw = resolve_consumer_text_dropout({}, recipe.cfg_dropout_rate)
        uncached_rows.append(prepare_static_task_text(task, recipe, reference, dropout_applied=draw))
    rng_original = random.getstate()
    assert rng_cached == rng_original
    assert 1 <= live.calls <= 2
    assert reference.calls == 60
    for a, b in zip(cached_rows, uncached_rows, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], torch.Tensor):
                torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
            else:
                assert a[k] == b[k]
    assert cached_rows[0]["text_token_ids"] is not cached_rows[1]["text_token_ids"]
    last = cached_rows[-1]["text_token_ids"].clone()
    cached_rows[0]["text_token_ids"][0] = -1234
    assert torch.equal(cached_rows[-1]["text_token_ids"], last)


def test_g2_static_text_cache_has_bounded_lru():
    from collections import OrderedDict
    from dataclasses import replace
    from types import SimpleNamespace

    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    class Processor:
        def tokenize_text(self, text, **kwargs):
            return [1, len(text) % 997, 2]

    cache = object.__new__(ActionTransformPipeline)
    cache._g2_static_token_cache = OrderedDict()
    cache.static_task_text_recipe = recipe_fixture()
    cache.text_tokenizer = SimpleNamespace(_processor=Processor())
    for i in range(300):
        task = replace(task_fixture(), episode_id="task/episode_%06d" % i)
        cache._prepare_cached_g2_text(task, dropout_applied=False)
        assert len(cache._g2_static_token_cache) <= 256
    assert len(cache._g2_static_token_cache) == 256

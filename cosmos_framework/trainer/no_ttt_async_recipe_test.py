"""Synthetic local files and real Source/Cache/Transform; fake CPU text processor."""

import pickle
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    get_action_robocasa_exact_window_cached_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft_test import roots  # noqa: F401
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
from cosmos_framework.data.generator.augmentors import text_tokenizer
from cosmos_framework.data.generator.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.trainer.no_ttt_async_recipe import AsyncRecipe
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile


class FakeProcessor:
    def __init__(self, path):
        self.name = str(path)

    def tokenize_text(self, caption, **kwargs):
        assert kwargs == {"is_video": False, "use_system_prompt": False}
        return [11, *caption.encode(), 12]


@pytest.fixture
def real_recipe(roots, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PSM_V3_COMPACT_CACHED_VIDEO", "1")
    processor_path = tmp_path / "processor"
    processor_path.mkdir()
    processor = object.__new__(Nemotron3DenseVLProcessor)
    processor.name = str(processor_path)
    monkeypatch.setattr(Nemotron3DenseVLProcessor, "tokenize_text", FakeProcessor.tokenize_text)
    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda config: processor)
    dataset = get_action_robocasa_exact_window_cached_sft_dataset(
        cache_root=roots[0],
        source_root=roots[1],
        tokenizer_config={"_target_": "cosmos_framework.data.generator.processors.build_processor_lazy"},
        cfg_dropout_rate=0.5,
    )
    index = VerifiedExactWindowIndex.build(
        tmp_path / "index", cache_catalog=dataset._dataset.catalog, source_reader=dataset._dataset.source_reader
    )
    dataset._dataset.source_reader.verified_index = index
    catalog = ExactWindowLocalCatalog(dataset, ttt_tbptt_steps=2)
    producer = ExactWindowSegmentProducer(catalog, config_digest="test_config")
    recipe = AsyncRecipe.from_producer(producer, LoaderProfile(2, 1, True, 60))
    return producer, recipe


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_real_transform_two_branches_preserve_values_and_original_rng_interleaving(real_recipe, seed):
    producer, recipe = real_recipe
    worker = recipe.build()
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=1, b_stream=1, active_ga=2)
    requests = tuple(member[0] for member in planner.plan_window(planner.initial_frontier()).members)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    expected = []
    for request in requests:
        expected.append(producer.produce(request))
        # Preserve arbitrary interleaved model/callback draws; don't forecast RNG.
        random.random()
        np.random.rand()
        torch.rand(())
    expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    packets = [worker.prepare(request) for request in requests]
    assert random.getstate() == random.Random(seed).getstate()
    for request, reference, (segment, events) in zip(requests, expected, packets, strict=True):
        assert [event["stage"] for event in events] == ["cpu_source", "cpu_transform"]
        assert all(event["start"] <= event["end"] for event in events)
        for payload in segment.consumer_payload[0][: request.valid_count]:
            if worker.rate > 0 and random.random() < worker.rate:
                payload[worker.caption_key] = ""
                payload[worker.token_key] = worker.empty_tokens.clone()
        _same(reference.consumer_payload, segment.consumer_payload)
        random.random()
        np.random.rand()
        torch.rand(())
    assert random.getstate() == expected_rng[0]
    actual_np = np.random.get_state()
    assert actual_np[0] == expected_rng[1][0]
    np.testing.assert_array_equal(actual_np[1], expected_rng[1][1])
    assert actual_np[2:] == expected_rng[1][2:]
    torch.testing.assert_close(torch.get_rng_state(), expected_rng[2], rtol=0, atol=0)


def test_recipe_roundtrip_contains_only_small_metadata_and_reopens_private_readers(real_recipe):
    producer, recipe = real_recipe
    encoded = pickle.dumps(recipe, protocol=5)
    assert len(encoded) < 1024**2
    restored = pickle.loads(encoded)
    worker = restored.build()
    assert worker.producer is not producer
    assert worker.producer.catalog.raw.source_reader is not producer.catalog.raw.source_reader
    assert worker.producer.catalog.raw.cache_reader is worker.cache_reader
    assert worker.producer.catalog.raw.source_reader.cache_reader is worker.cache_reader
    assert worker.rate == 0.5
    assert worker.producer.catalog.wrapped_sft._transform.text_tokenizer.cfg_dropout_rate == 0
    assert producer.catalog.wrapped_sft._transform.text_tokenizer.cfg_dropout_rate == 0.5
    assert worker.empty_tokens.tolist() == [11, 12]
    assert restored.tokenizer_config["tokenizer_type"].endswith("processor")
    assert "repository" not in restored.tokenizer_config


@pytest.mark.parametrize("field", ["source_digest", "cache_digest", "manifest_digest"])
def test_worker_reconstruction_rejects_frozen_identity_drift(real_recipe, field):
    _, recipe = real_recipe
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(recipe, **{field: "foreign"}).build()


@pytest.mark.parametrize(
    "fault",
    ["keyed", "nonjson", "separate", "rate", "unknown", "remote_processor", "temporal", "formatter", "resize"],
)
def test_unknown_or_changed_transform_contract_rejected_before_workers(real_recipe, fault):
    producer, _ = real_recipe
    transform = producer.catalog.wrapped_sft._transform
    if fault == "keyed":
        producer.transform_rng_mode = "consumer_keyed_v1"
    elif fault == "nonjson":
        transform.prompt_json_formatter = None
    elif fault == "separate":
        transform.text_tokenizer.tokenize_separately = True
    elif fault == "rate":
        transform.text_tokenizer.cfg_dropout_rate = float("nan")
    elif fault == "unknown":
        producer.catalog.wrapped_sft._transform = lambda *_: {}
    elif fault == "temporal":
        transform.video_temporal_downsample += 1
    elif fault == "formatter":
        transform.prompt_json_formatter.float_seconds = True
    elif fault == "resize":
        transform.video_resize.canvas = (320, 320, 320, 320)
    else:
        transform.text_tokenizer._processor.name = "remote/not/staged"
    with pytest.raises(ValueError, match="async No-TTT|reconstruction"):
        AsyncRecipe.from_producer(producer, LoaderProfile(2, 1, True, 60))


def test_foreign_episode_metadata_rejected_but_ipc_copy_rebound(real_recipe):
    producer, recipe = real_recipe
    worker = recipe.build()
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=1, b_stream=1, active_ga=1)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    worker.prepare(deepcopy(request))
    with pytest.raises(ValueError, match="episode metadata"):
        worker.prepare(replace(request, episode=replace(request.episode, flat_start=12345)))


def test_worker_source_and_transform_must_remain_rng_neutral(real_recipe):
    producer, recipe = real_recipe
    worker = recipe.build()
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=1, b_stream=1, active_ga=1)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    original = worker.producer.catalog.wrapped_sft._transform

    def wrong(item, resolution):
        random.random()
        return original(item, resolution)

    worker.producer.catalog.wrapped_sft._transform = wrong
    with pytest.raises(ValueError, match="must not consume"):
        worker.prepare(request)

"""Synthetic local files and real Source/Cache/Transform; fake CPU text processor."""

import json
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


@dataclass(frozen=True)
class _CpuTextRecipe:
    """Spawn-safe fixture; only processor vocabulary is synthetic, not Source."""

    inner: AsyncRecipe

    @property
    def rate(self):
        return self.inner.rate

    @property
    def caption_key(self):
        return self.inner.caption_key

    @property
    def token_key(self):
        return self.inner.token_key

    def build(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        def forbidden(*args, **kwargs):
            raise AssertionError("worker called full LeRobot construction")

        LeRobotDataset.__init__ = forbidden
        text_tokenizer.lazy_instantiate = lambda config: FakeProcessor(config["tokenizer_type"])
        return self.inner.build()


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


def test_worker_recipe_freezes_resolved_data_roots_before_alias_can_change(real_recipe, tmp_path):
    producer, original = real_recipe
    raw = producer.catalog.raw
    links = {}
    for name, target in (
        ("cache", original.cache_root),
        ("source", original.source_root),
        ("index", original.index_root),
    ):
        path = tmp_path / f"{name}_alias"
        path.symlink_to(target, target_is_directory=True)
        links[name] = path
    raw.catalog.cache_root = links["cache"]
    raw.source_reader.source_root = links["source"]
    raw.source_reader.verified_index.root = links["index"]
    frozen = AsyncRecipe.from_producer(producer, LoaderProfile(2, 1, True, 60))
    for name in links:
        assert getattr(frozen, f"{name}_root") == str(Path(getattr(original, f"{name}_root")).resolve())
        links[name].unlink()
        links[name].symlink_to(tmp_path / "missing_alternate_tree", target_is_directory=True)
    assert frozen.build().producer.catalog.manifest_digest == producer.catalog.manifest_digest


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


def test_first_worker_recipe_build_never_calls_full_lerobot_constructor(real_recipe, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    producer, recipe = real_recipe

    def forbidden(*args, **kwargs):
        raise AssertionError("full eager LeRobotDataset construction is forbidden in async workers")

    monkeypatch.setattr(LeRobotDataset, "__init__", forbidden)
    monkeypatch.setattr(LeRobotDataset, "load_hf_dataset", forbidden)
    worker = recipe.build()
    rows = worker.producer.catalog.raw.source_reader.dataset.hf_dataset
    assert not rows.cache
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=1, b_stream=1, active_ga=1)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    segment, events = worker.prepare(request)
    assert int(segment.consumer_valid.sum()) == request.valid_count
    assert list(rows.cache) == [0]
    assert [event["stage"] for event in events] == ["cpu_source", "cpu_transform"]


def test_spawned_real_recipe_source_transform_cfg_and_cross_window_cleanup(real_recipe, tmp_path):
    from cosmos_framework.trainer.no_ttt_async_loader import NoTTTAsyncLoader
    from cosmos_framework.trainer.no_ttt_async_loader_test import consume, equal_tree, seed

    initial, original_recipe = real_recipe
    # B8/GA4 requires 16 distinct two-segment Episodes per Window. Enlarge only
    # this temporary fixture; never bypass the production native-profile guard.
    cache, root = Path(original_recipe.cache_root), Path(original_recipe.source_root)
    manifest_file = cache / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    first_record = manifest["tasks"][0]["episodes"][0]
    payload = torch.load(cache / "tasks/Pick_Mug/episodes/episode_000000.pt", weights_only=True)
    parquet_file = root / "data/chunk-000/file-000.parquet"
    first_rows = pq.read_table(parquet_file).slice(0, 19).to_pylist()
    rows, metadata, records = [], [], []
    for ep in range(16):
        record = deepcopy(first_record)
        record["episode_index"] = ep
        records.append(record)
        copied = deepcopy(payload)
        for window in copied["windows"].values():
            window["global_row_indices"] += 19 * ep
        torch.save(copied, cache / f"tasks/Pick_Mug/episodes/episode_{ep:06d}.pt")
        for row in deepcopy(first_rows):
            row["index"] += 19 * ep
            row["episode_index"] = ep
            rows.append(row)
        metadata.append(
            {
                "length": 19,
                "dataset_from_index": 100 + 19 * ep,
                "dataset_to_index": 119 + 19 * ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        )
    manifest["tasks"][0]["episodes"] = records
    manifest["episode_count"], manifest["window_count"] = 16, 48
    manifest_file.write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist(rows), parquet_file, row_group_size=19)
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    info_file = root / "meta/info.json"
    info = json.loads(info_file.read_text())
    info["total_episodes"], info["total_frames"] = 16, 16 * 19
    info_file.write_text(json.dumps(info))
    dataset = get_action_robocasa_exact_window_cached_sft_dataset(
        cache_root=cache,
        source_root=root,
        tokenizer_config=initial.catalog.wrapped_sft._transform.text_tokenizer.args["tokenizer_config"],
        cfg_dropout_rate=0.5,
    )
    index = VerifiedExactWindowIndex.build(
        tmp_path / "native_index", cache_catalog=dataset._dataset.catalog, source_reader=dataset._dataset.source_reader
    )
    dataset._dataset.source_reader.verified_index = index
    producer = ExactWindowSegmentProducer(
        ExactWindowLocalCatalog(dataset, ttt_tbptt_steps=2), config_digest="test_config"
    )
    recipe = AsyncRecipe.from_producer(producer, LoaderProfile(2, 1, True, 120))
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=1, b_stream=8, active_ga=4)
    seed()
    reference = consume(producer, planner)
    seed()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(2, 1, True, 120), planner, recipe=_CpuTextRecipe(recipe))
    loader.file_witnesses = True  # real temporary source/cache files; retain production guards
    try:
        actual = consume(producer, planner, loader=loader)
        equal_tree(reference[:4], actual[:4])
        assert len(actual[4]) == 3 and loader.preview is not None
        assert all(len(row["worker_pids"]) == 2 for row in actual[4])
        assert all(row["source_read_ms"] == row["transform_host_ms"] == 0 for row in actual[4])
        assert all(row["worker_source_ms_sum"] > 0 and row["worker_transform_ms_sum"] > 0 for row in actual[4])
    finally:
        loader.close()
    assert len(loader.last_cleanup) == 2
    assert all(row["exitcode"] is not None and not Path(f"/proc/{row['pid']}").exists() for row in loader.last_cleanup)

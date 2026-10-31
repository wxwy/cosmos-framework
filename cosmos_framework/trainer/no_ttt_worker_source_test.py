"""Synthetic files only: lazy worker storage versus the pinned official reader."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cosmos_framework.data.generator.action.datasets import robocasa_exact_window_source as source
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    RoboCasaExactWindowCacheCatalog,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source_test import _cache, _source
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
from cosmos_framework.trainer.no_ttt_async_loader_test import equal_tree
from cosmos_framework.trainer.no_ttt_cpu_loader import _require_same_rng, _rng_snapshot
from cosmos_framework.trainer.no_ttt_worker_source import VerifiedWorkerLeRobotDataset


def _two_episodes(tmp_path, *, subset=False, split=False, reverse=False):
    cache, root = _cache(tmp_path / "cache"), _source(tmp_path / "source", extra_episode=True)
    if not subset:
        manifest = json.loads((cache / "dataset_manifest.json").read_text())
        record = deepcopy(manifest["tasks"][0]["episodes"][0])
        record["episode_index"] = 1
        manifest["tasks"][0]["episodes"].append(record)
        manifest["episode_count"], manifest["window_count"] = 2, 6
        (cache / "dataset_manifest.json").write_text(json.dumps(manifest))
        episode = cache / "tasks/Pick_Mug/episodes/episode_000000.pt"
        payload = torch.load(episode, weights_only=True)
        for window in payload["windows"].values():
            window["global_row_indices"] += 19
        torch.save(payload, episode.with_name("episode_000001.pt"))
    file = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(file)
    if split:
        # Selected episode order need not equal physical parquet-file order.
        pq.write_table(table.slice(19), file, row_group_size=7)
        pq.write_table(table.slice(0, 19), file.with_name("file-001.parquet"), row_group_size=7)
        metadata = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(metadata).to_pylist()
        rows[0]["data/file_index"] = 1
        pq.write_table(pa.Table.from_pylist(rows), metadata)
    elif reverse:
        pq.write_table(table.take([*range(19, 38), *range(19)]), file, row_group_size=7)
    catalog = RoboCasaExactWindowCacheCatalog(cache)
    eager = source.RoboCasaExactWindowSourceReader(catalog, root)
    index = VerifiedExactWindowIndex.build(tmp_path / "index", cache_catalog=catalog, source_reader=eager)
    return eager, index


@pytest.fixture
def pair(tmp_path):
    return _two_episodes(tmp_path)


def _lazy(eager, index, *, max_episodes=2, max_bytes=1024**2):
    return source.RoboCasaExactWindowSourceReader(
        eager.catalog,
        eager.source_root,
        verified_index=index,
        dataset_factory=partial(
            VerifiedWorkerLeRobotDataset,
            verified_index=index,
            max_episodes=max_episodes,
            max_bytes=max_bytes,
        ),
    )


@pytest.mark.parametrize("layout", ["shared", "split", "reverse", "subset"])
def test_official_delta_caption_padding_and_source_windows_match(tmp_path, layout):
    eager, index = _two_episodes(
        tmp_path, split=layout == "split", reverse=layout == "reverse", subset=layout == "subset"
    )
    before = _rng_snapshot()
    lazy = _lazy(eager, index)
    assert lazy.source_binding_digest == eager.source_binding_digest
    for record in eager.catalog.episodes:
        for start in record.window_starts:
            expected, actual = eager.read_window(record.key, start), lazy.read_window(record.key, start)
            equal_tree(expected, actual)
            assert actual.ai_caption == "Please pick the mug"  # actual task_index, not class name
            assert actual.action12.dtype == actual.state16.dtype == torch.float32
        bound = eager._bound[record.key]
        # Also exercise official padding outside the accepted Exact-Window range.
        relative = eager.abs_to_relative[bound.dataset_to_index - 1]
        equal_tree(eager.dataset[relative], lazy.dataset[relative])
        assert bool(lazy.dataset[relative]["action_is_pad"].any())
    _require_same_rng(before)
    for name in ("__getitem__", "_get_query_indices", "_query_hf_dataset", "_get_query_timestamps"):
        assert getattr(type(lazy.dataset), name) is getattr(LeRobotDataset, name)


def test_build_does_not_call_eager_constructor_or_read_any_source_parquet(pair, monkeypatch, tmp_path):
    eager, index = pair
    monkeypatch.setattr(LeRobotDataset, "__init__", lambda *a, **k: pytest.fail("eager constructor"))
    monkeypatch.setattr(LeRobotDataset, "load_hf_dataset", lambda *a, **k: pytest.fail("global HF load"))
    opened = []
    original = pq.ParquetFile

    def observe(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", observe)
    started = time.monotonic()
    lazy = _lazy(eager, index)
    built = time.monotonic()
    assert not opened and not lazy.dataset.hf_dataset.cache
    lazy.read_at(0)
    ready = time.monotonic()
    assert opened == [str(eager.source_root / "data/chunk-000/file-000.parquet")]
    assert list(lazy.dataset.hf_dataset.cache) == [0]
    lazy.read_at(1)
    assert len(opened) == 1
    evidence = {
        "scope": "synthetic_not_real_data_or_GPU",
        "eager_constructor_calls": 0,
        "full_source_load_calls": 0,
        "source_parquet_opens_at_build": 0,
        "source_parquet_opens_after_two_reads": len(opened),
        "retained_episodes": list(lazy.dataset.hf_dataset.cache),
        "build_ms": (built - started) * 1000,
        "first_read_ms": (ready - built) * 1000,
        "original_data_timeout_s": 120,
    }
    (tmp_path / "lazy_source_witness.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, sort_keys=True))


def test_lru_eviction_bounds_and_reload_preserve_exact_values(pair, monkeypatch):
    eager, index = pair
    lazy = _lazy(eager, index, max_episodes=1)
    storage = lazy.dataset.hf_dataset
    for ep in (0, 1, 0):
        key = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", ep)
        equal_tree(eager.read_window(key, 0), lazy.read_window(key, 0))
        assert list(storage.cache) == [ep]
        assert 0 < storage.cache_bytes <= storage.max_bytes
        assert storage.cache_bytes == sum(size for _, size in storage.cache.values())
    lazy = _lazy(eager, index, max_bytes=1)
    with pytest.raises(MemoryError, match="bounded Arrow cache"):
        lazy.read_at(0)
    assert not lazy.dataset.hf_dataset.cache and lazy.dataset.hf_dataset.cache_bytes == 0


@pytest.mark.parametrize("fault", ["length", "file", "first_mapping", "interior_mapping", "relative_gap"])
def test_bad_verified_metadata_and_mapping_fail_closed(pair, fault):
    eager, index = pair
    if fault in ("length", "file"):
        index.receipt = deepcopy(index.receipt)
        index.receipt["bindings"][0]["length" if fault == "length" else "data_file"] = (
            123 if fault == "length" else "bad"
        )
        with pytest.raises(ValueError, match="metadata|source metadata"):
            _lazy(eager, index)
        return
    rows = np.load(index.rows_path, mmap_mode="r+", allow_pickle=False)
    location = 0 if fault == "first_mapping" else 1 if fault == "interior_mapping" else slice(19, 38)
    rows[location, 1] += 1
    rows.flush()
    if fault == "interior_mapping":
        lazy = _lazy(eager, index)
        # Official absolute->relative query observes the corrupted mapping;
        # the unchanged SourceReader's exact returned-row check must reject it.
        with pytest.raises(ValueError, match="runtime global index"):
            lazy.read_at(0)
    else:
        with pytest.raises(ValueError, match="row mapping|row spans"):
            _lazy(eager, index)


@pytest.mark.parametrize("fault", ["missing", "escape", "length", "frame", "episode", "annotation", "nan", "caption"])
def test_missing_changed_or_corrupt_source_never_becomes_valid_window(pair, tmp_path, fault):
    eager, index = pair
    lazy = _lazy(eager, index)
    path = eager.source_root / "data/chunk-000/file-000.parquet"
    if fault == "missing":
        path.unlink()
    elif fault == "escape":
        foreign = tmp_path / "foreign.parquet"
        path.rename(foreign)
        path.symlink_to(foreign)
    else:
        rows = pq.read_table(path).to_pylist()
        if fault == "length":
            rows.pop(0)
        elif fault == "frame":
            rows[1]["frame_index"] = 88
        elif fault == "episode":
            rows[1]["episode_index"] = 1
        elif fault == "annotation":
            rows[1]["annotation.human.task_name"] = 1
        elif fault == "nan":
            rows[1]["action"][0] = float("nan")
        elif fault == "caption":
            rows[0]["task_index"] = -100
        pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises((ValueError, FileNotFoundError, IndexError)):
        lazy.read_at(0)


@pytest.mark.parametrize("fault", ["no_index", "hub", "revision", "sync", "videos", "episode_subset"])
def test_unsupported_worker_constructor_does_not_fallback(pair, fault):
    eager, index = pair
    kwargs = dict(
        repo_id="local",
        root=eager.source_root,
        episodes=[0, 1],
        delta_timestamps=source._delta_timestamps(20),
        revision="local",
        force_cache_sync=False,
        download_videos=False,
        verified_index=index,
        max_episodes=1,
        max_bytes=1024**2,
    )
    field, value = {
        "no_index": ("verified_index", None),
        "hub": ("repo_id", "remote/dataset"),
        "revision": ("revision", "main"),
        "sync": ("force_cache_sync", True),
        "videos": ("download_videos", True),
        "episode_subset": ("episodes", [0]),
    }[fault]
    kwargs[field] = value
    with pytest.raises(ValueError, match="verified|selected episode"):
        VerifiedWorkerLeRobotDataset(**kwargs)


def _spawn_read(cache, root, index_root, connection):
    try:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        before = _rng_snapshot()
        started = time.monotonic()
        index = VerifiedExactWindowIndex.open(index_root, cache_root=cache, source_root=root, verify_files=False)
        catalog = RoboCasaExactWindowCacheCatalog(cache, verified_index=index)
        reader = source.RoboCasaExactWindowSourceReader(
            catalog,
            root,
            verified_index=index,
            dataset_factory=partial(
                VerifiedWorkerLeRobotDataset, verified_index=index, max_episodes=1, max_bytes=1024**2
            ),
        )
        built = time.monotonic()
        window = reader.read_at(0)
        _require_same_rng(before)
        connection.send(
            {
                "pid": os.getpid(),
                "build_ms": (built - started) * 1000,
                "first_read_ms": (time.monotonic() - built) * 1000,
                "caption": window.ai_caption,
                "action": window.action12.tolist(),
                "rows": window.global_row_indices,
                "rng_unchanged": True,
                "retained_episodes": list(reader.dataset.hf_dataset.cache),
            }
        )
    except BaseException as exc:
        connection.send({"error": repr(exc)})
        raise
    finally:
        connection.close()


def test_real_spawn_reconstructs_only_small_verified_source_and_reaps(pair, tmp_path):
    eager, index = pair
    before = _rng_snapshot()
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_read, args=(eager.catalog.cache_root, eager.source_root, index.root, child))
    try:
        process.start()
        child.close()
        assert parent.poll(60), "bounded synthetic spawn did not return"
        result = parent.recv()
        process.join(5)
        assert process.exitcode == 0 and "error" not in result
        assert result["pid"] != os.getpid() and result["rng_unchanged"]
        assert result["caption"] == eager.read_at(0).ai_caption
        assert result["rows"] == eager.read_at(0).global_row_indices
        assert result["action"] == eager.read_at(0).action12.tolist()
        assert result["retained_episodes"] == [0]
        (tmp_path / "spawn_source_witness.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, sort_keys=True))
        _require_same_rng(before)
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()
        child.close()
    assert not process.is_alive() and not Path(f"/proc/{process.pid}").exists()

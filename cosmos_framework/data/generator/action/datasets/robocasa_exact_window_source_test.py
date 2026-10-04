# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic local flat LeRobot v3 binding tests; no training assets or GPU."""

from __future__ import annotations

import ast
import json
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.datasets.backward_compatibility import BackwardCompatibilityError
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from cosmos_framework.data.generator.action.datasets import robocasa_exact_window_source as source
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)

_KEY = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 0)
_CACHE_FILE = Path("tasks/Pick_Mug/episodes/episode_000000.pt")


def _cache(root: Path, *, windows: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "exact_window_v1",
        "source_format": "lerobot_v3",
        "suite": "synthetic",
        "chunk_length": 16,
        "sample_stride": 1,
        "fps": 20.0,
        "camera_set": "left_wrist",
        "latent_shape": [5, 48, 2, 2],
        "vae_encode_contract": {
            "compute_dtype": "torch.bfloat16",
            "encode_exact_durations": [17],
            "encode_chunk_frames": {"256": 68},
        },
        "task_class_count": 1,
        "episode_count": 1,
        "window_count": windows,
        "tasks": [
            {
                "task_class": "Pick Mug",
                "task_slug": "Pick_Mug",
                "episodes": [
                    {
                        "task_class": "Pick Mug",
                        "task_slug": "Pick_Mug",
                        "episode_index": 0,
                        "window_count": windows,
                        "source_video_frames": windows + 16,
                    }
                ],
            }
        ],
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = {
        "format": "exact_window_v1",
        "source_format": "lerobot_v3",
        "suite": "synthetic",
        "metadata": {"camera_set": "left_wrist"},
        "windows": {
            str(start): {
                "latent": torch.zeros((5, 48, 2, 2)),
                "window_frame_indices": torch.arange(start, start + 17),
                "latent_source_frame_indices": torch.arange(start, start + 17, 4),
                "global_row_indices": torch.arange(100 + start, 117 + start),
            }
            for start in range(windows)
        },
    }
    path = root / _CACHE_FILE
    path.parent.mkdir(parents=True)
    torch.save(payload, path)
    return root


def _change_cache(root: Path, mutate) -> None:
    path = root / _CACHE_FILE
    payload = torch.load(path, weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def _source(root: Path, *, extra_episode: bool = False) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features = {
        "action": {"dtype": "float32", "shape": [12], "names": None},
        "observation.state": {"dtype": "float32", "shape": [16], "names": None},
        "annotation.human.task_name": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "observation.images.robot0_agentview_left": {
            "dtype": "video",
            "shape": [3, 2, 2],
            "names": None,
        },
    }
    count = 19 + (19 if extra_episode else 0)
    info = {
        "codebase_version": "v3.0",
        "fps": 20,
        "features": features,
        "total_episodes": 2 if extra_episode else 1,
        "total_frames": count,
        "total_tasks": 2,
        "robot_type": "synthetic",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame({"task_index": [0, 1]}, index=["Pick Mug", "Please pick the mug"]).to_parquet(
        root / "meta" / "tasks.parquet"
    )
    episode_rows = [
        {"length": 19, "dataset_from_index": 100, "dataset_to_index": 119, "data/chunk_index": 0, "data/file_index": 0},
    ]
    if extra_episode:
        episode_rows.append(
            {
                "length": 19,
                "dataset_from_index": 119,
                "dataset_to_index": 138,
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(episode_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data = {
        "index": list(range(100, 100 + count)),
        "episode_index": [0] * 19 + ([1] * 19 if extra_episode else []),
        "frame_index": list(range(19)) + (list(range(19)) if extra_episode else []),
        "annotation.human.task_name": [0] * count,
        "task_index": [1] * count,
        "timestamp": [frame / 20 for frame in range(19)] * (2 if extra_episode else 1),
        "action": [[float(frame)] * 12 for frame in range(count)],
        "observation.state": [[float(frame)] * 16 for frame in range(count)],
    }
    pq.write_table(
        pa.table({key: data[key] for key, feature in features.items() if feature["dtype"] != "video"}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    return root


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return _cache(tmp_path / "cache"), _source(tmp_path / "source")


def _reader(roots: tuple[Path, Path]) -> source.RoboCasaExactWindowSourceReader:
    return source.RoboCasaExactWindowSourceReader(RoboCasaExactWindowCacheCatalog(roots[0]), roots[1])


@pytest.mark.level(0)
def test_real_official_nonvisual_reader_without_video_and_exact_window(roots: tuple[Path, Path]) -> None:
    with (
        patch.object(LeRobotDatasetMetadata, "pull_from_repo", side_effect=AssertionError("Hub")) as pull,
        patch.object(
            source._LocalNonVisualLeRobotDataset, "download", side_effect=AssertionError("download")
        ) as download,
        patch.object(LeRobotDataset, "_query_videos", side_effect=AssertionError("video")) as query,
    ):
        reader = _reader(roots)
        assert isinstance(reader.dataset, LeRobotDataset)
        assert reader.dataset.meta.video_keys == []
        assert reader.dataset.episodes == [0]
        assert reader.dataset.meta.info["features"].get("observation.images.robot0_agentview_left") is None
        assert reader.abs_to_relative[100] == 0
        assert reader.abs_to_relative[118] == 18
        result = reader.read_window(_KEY, 1)
        assert (result.key, result.start_frame, result.global_row_indices) == (_KEY, 1, tuple(range(101, 118)))
        assert result.action12.shape == (16, 12) and result.action12.dtype == torch.float32
        assert result.state16.shape == (17, 16) and result.state16.dtype == torch.float32
        assert result.action12.is_contiguous() and result.state16.is_contiguous()
        torch.testing.assert_close(result.action12, torch.arange(1, 17, dtype=torch.float32)[:, None].expand(16, 12))
        torch.testing.assert_close(result.state16, torch.arange(1, 18, dtype=torch.float32)[:, None].expand(17, 16))
        assert result.ai_caption == "Please pick the mug" and result.task_index == 1
        assert result.task_class == "Pick Mug"
        assert reader.read_at(2).start_frame == 2
        assert reader.summary()["offline_only"] is True
        assert reader.summary()["mismatch_counters"] == {
            "missing": 0,
            "ambiguous": 0,
            "task_mismatch": 0,
            "frame_mismatch": 0,
            "row_mismatch": 0,
        }
        pull.assert_not_called()
        download.assert_not_called()
        query.assert_not_called()
    assert (
        "observation.images.robot0_agentview_left" in json.loads((roots[1] / "meta/info.json").read_text())["features"]
    )


@pytest.mark.level(0)
def test_reader_subclass_only_two_overrides_and_delta_contract() -> None:
    assert {key for key in source._LocalNonVisualLeRobotDataset.__dict__ if not key.startswith("__")} == {
        "_check_cached_episodes_sufficient",
        "download",
    }
    delta = source._delta_timestamps(20)
    assert {key: len(value) for key, value in delta.items()} == {
        "action": 16,
        "observation.state": 17,
        "index": 17,
        "episode_index": 17,
        "frame_index": 17,
        "annotation.human.task_name": 17,
    }
    assert "task_index" not in delta and all("image" not in key for key in delta)
    assert delta["action"] == [step / 20 for step in range(16)]


@pytest.mark.level(0)
def test_cache_index_and_source_extras(roots: tuple[Path, Path]) -> None:
    _source(roots[1], extra_episode=True)
    reader = _reader(roots)
    assert len(reader.index) == 3
    assert reader.index.get_shuffle_blocks() == (((_KEY, 0), (_KEY, 1), (_KEY, 2)),)
    assert reader.dataset.episodes == [0]
    assert reader.summary()["source_total_episodes"] == 2
    assert reader.read_at(2).task_class == "Pick Mug"


@pytest.mark.level(0)
def test_dataset_constructor_uses_exact_episode_set_and_disables_videos(roots: tuple[Path, Path]) -> None:
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return source._LocalNonVisualLeRobotDataset(**kwargs)

    reader = source.RoboCasaExactWindowSourceReader(
        RoboCasaExactWindowCacheCatalog(roots[0]),
        roots[1],
        dataset_factory=factory,
    )
    assert seen["repo_id"] == seen["revision"] == "local"
    assert seen["episodes"] == [0]
    assert seen["download_videos"] is False
    assert seen["force_cache_sync"] is False
    assert "task_index" not in seen["delta_timestamps"]
    assert reader.dataset.meta.video_keys == []


@pytest.mark.level(0)
def test_offline_guard_runs_before_factories(roots: tuple[Path, Path]) -> None:
    seen = []

    def metadata_factory(**kwargs):
        seen.append("meta")
        return LeRobotDatasetMetadata(**kwargs)

    def dataset_factory(**kwargs):
        seen.append("dataset")
        return source._LocalNonVisualLeRobotDataset(**kwargs)

    with patch.object(source, "_ensure_hf_hub_offline", side_effect=lambda: seen.append("offline")):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            metadata_factory=metadata_factory,
            dataset_factory=dataset_factory,
        )
    assert seen == ["offline", "meta", "dataset"]


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "missing", ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]
)
def test_missing_metadata_fails_before_constructor(roots: tuple[Path, Path], missing: str) -> None:
    (roots[1] / missing).unlink()
    metadata_factory = Mock(side_effect=AssertionError("metadata factory called"))
    dataset_factory = Mock(side_effect=AssertionError("dataset factory called"))
    with pytest.raises(FileNotFoundError):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            metadata_factory=metadata_factory,
            dataset_factory=dataset_factory,
        )
    metadata_factory.assert_not_called()
    dataset_factory.assert_not_called()


@pytest.mark.level(0)
def test_missing_data_fails_preflight(roots: tuple[Path, Path]) -> None:
    (roots[1] / "data/chunk-000/file-000.parquet").unlink()
    dataset_factory = Mock(side_effect=AssertionError("dataset factory called"))
    with pytest.raises(FileNotFoundError, match="selected data"):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            dataset_factory=dataset_factory,
        )
    dataset_factory.assert_not_called()


@pytest.mark.level(0)
def test_missing_data_directory_fails_before_metadata_factory(roots: tuple[Path, Path]) -> None:
    shutil.rmtree(roots[1] / "data")
    metadata_factory = Mock(side_effect=AssertionError("metadata factory called"))
    with pytest.raises(FileNotFoundError, match="data/"):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            metadata_factory=metadata_factory,
        )
    metadata_factory.assert_not_called()


@pytest.mark.level(0)
def test_official_constructor_fallback_hits_local_download_guard(roots: tuple[Path, Path]) -> None:
    source._ensure_hf_hub_offline()
    original = source._LocalNonVisualLeRobotDataset.download
    with (
        patch.object(LeRobotDatasetMetadata, "pull_from_repo", side_effect=AssertionError("Hub")) as pull,
        patch.object(source._LocalNonVisualLeRobotDataset, "download", autospec=True, side_effect=original) as guard,
    ):
        with pytest.raises(FileNotFoundError, match="禁止下载"):
            source._LocalNonVisualLeRobotDataset(
                repo_id="local",
                root=roots[1],
                episodes=[99],
                revision="local",
                force_cache_sync=False,
                download_videos=False,
            )
        guard.assert_called_once()
        assert guard.call_args.args[1] is False
        pull.assert_not_called()


@pytest.mark.level(0)
def test_selected_data_incomplete_fails_startup_scan(roots: tuple[Path, Path]) -> None:
    path = roots[1] / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 18), path)
    with pytest.raises(ValueError, match="source row bounds"):
        _reader(roots)
    # 空 selected episode 在 startup scan 即拒绝；constructor fallback 另由 direct guard test 覆盖。
    pq.write_table(table.filter(pa.compute.equal(table["episode_index"], 99)), path)
    with pytest.raises(ValueError, match="source episode=0"):
        _reader(roots)


@pytest.mark.level(0)
def test_cache_episode_index_global_uniqueness_across_tasks(roots: tuple[Path, Path]) -> None:
    manifest_path = roots[0] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    second = deepcopy(manifest["tasks"][0])
    second["task_class"] = "Open Drawer"
    second["task_slug"] = "Open_Drawer"
    second["episodes"][0].update(task_class="Open Drawer", task_slug="Open_Drawer")
    manifest["tasks"].append(second)
    manifest["task_class_count"] = 2
    manifest["episode_count"] = 2
    manifest["window_count"] = 6
    manifest_path.write_text(json.dumps(manifest))
    second_path = roots[0] / "tasks/Open_Drawer/episodes/episode_000000.pt"
    second_path.parent.mkdir(parents=True)
    payload = torch.load(roots[0] / _CACHE_FILE, weights_only=True)
    torch.save(payload, second_path)
    with pytest.raises(ValueError, match="跨 task 重复"):
        _reader(roots)


@pytest.mark.level(0)
def test_cache_selected_episode_missing_from_source(roots: tuple[Path, Path]) -> None:
    manifest_path = roots[0] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0]["episodes"][0]["episode_index"] = 1
    manifest_path.write_text(json.dumps(manifest))
    new_path = roots[0] / "tasks/Pick_Mug/episodes/episode_000001.pt"
    (roots[0] / _CACHE_FILE).rename(new_path)
    with pytest.raises(ValueError, match="缺少 cache episode_index"):
        _reader(roots)


@pytest.mark.level(0)
@pytest.mark.parametrize("field,bad", [("length", 18), ("dataset_to_index", 118)])
def test_episode_metadata_length_or_bounds_mismatch(roots: tuple[Path, Path], field: str, bad: int) -> None:
    path = roots[1] / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = bad
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="length/window_count"):
        _reader(roots)


@pytest.mark.level(0)
def test_episode_row_bounds_reject_noncontiguous_index(roots: tuple[Path, Path]) -> None:
    path = roots[1] / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    values = table["index"].to_pylist()
    values[7] = 777
    table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(values))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="row bounds/index"):
        _reader(roots)


@pytest.mark.level(0)
@pytest.mark.parametrize("case", ["zero_match", "duplicate_match", "wrong_class", "multi_class", "non_scalar"])
def test_annotation_task_table_binding_rejects_ambiguous_or_invalid(
    roots: tuple[Path, Path],
    case: str,
) -> None:
    if case == "duplicate_match":
        pd.DataFrame({"task_index": [0, 0, 1]}, index=["Pick Mug", "Again", "Please pick the mug"]).to_parquet(
            roots[1] / "meta/tasks.parquet"
        )
    else:

        def scanner(path, episode_index):
            rows = source._scan_identity(path, episode_index)
            annotation = list(rows[source._ANNOTATION])
            if case == "zero_match":
                annotation = [99] * len(annotation)
            elif case == "wrong_class":
                annotation = [1] * len(annotation)
            elif case == "multi_class":
                annotation[5] = 1
            elif case == "non_scalar":
                annotation = [(0,)] * len(annotation)
            return {**rows, source._ANNOTATION: tuple(annotation)}

    with pytest.raises((ValueError, TypeError), match="annotation|task class|task_class"):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            identity_scanner=source._scan_identity if case == "duplicate_match" else scanner,
        )


@pytest.mark.level(0)
def test_task_table_requires_task_index(roots: tuple[Path, Path]) -> None:
    pd.DataFrame({"wrong": [0, 1]}, index=["Pick Mug", "Please pick the mug"]).to_parquet(
        roots[1] / "meta/tasks.parquet"
    )
    with pytest.raises(ValueError, match="task_index"):
        _reader(roots)


@pytest.mark.level(0)
@pytest.mark.parametrize("field,bad", [("fps", 19), ("codebase_version", "v2.1")])
def test_version_or_fps_mismatch(roots: tuple[Path, Path], field: str, bad: object) -> None:
    path = roots[1] / "meta/info.json"
    info = json.loads(path.read_text())
    info[field] = bad
    path.write_text(json.dumps(info))
    with pytest.raises((ValueError, BackwardCompatibilityError)):
        _reader(roots)


@pytest.mark.level(0)
def test_required_feature_missing(roots: tuple[Path, Path]) -> None:
    path = roots[1] / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"].pop("observation.state")
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="features 缺失"):
        _reader(roots)


@pytest.mark.level(0)
def test_nonflat_multishard_root_rejected(tmp_path: Path) -> None:
    cache = RoboCasaExactWindowCacheCatalog(_cache(tmp_path / "cache"))
    _source(tmp_path / "PickMug" / "2026-01-01" / "lerobot")
    with pytest.raises(FileNotFoundError, match="multi-shard"):
        source.RoboCasaExactWindowSourceReader(cache, tmp_path)


@pytest.mark.level(0)
def test_binding_digest_portable_across_machine_roots(roots: tuple[Path, Path], tmp_path: Path) -> None:
    first = _reader(roots)
    moved = tmp_path / "machine-b" / "source"
    shutil.copytree(roots[1], moved)
    second = _reader((roots[0], moved))
    assert first.source_binding_digest == second.source_binding_digest
    assert first.summary()["source_root"] != second.summary()["source_root"]


@pytest.mark.level(0)
@pytest.mark.parametrize("column,bad", [("episode_index", 2), ("frame_index", 88)])
def test_startup_source_identity_drift(roots: tuple[Path, Path], column: str, bad: int) -> None:
    def scanner(path, episode_index):
        rows = source._scan_identity(path, episode_index)
        values = list(rows[column])
        values[5] = bad
        return {**rows, column: tuple(values)}

    with pytest.raises(ValueError, match="episode_index 漂移|frame_index 漂移"):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            identity_scanner=scanner,
        )


@pytest.mark.level(0)
@pytest.mark.parametrize("case", ["duplicate", "missing_witness"])
def test_absolute_relative_mapping_fails_closed(roots: tuple[Path, Path], case: str) -> None:
    def factory(**kwargs):
        dataset = source._LocalNonVisualLeRobotDataset(**kwargs)
        actual = dataset.hf_dataset
        indices = list(actual["index"])
        if case == "duplicate":
            indices[1] = indices[0]
        else:
            indices[0] = 999

        class IndexProxy:
            def __getitem__(self, key):
                return indices if key == "index" else actual[key]

        dataset.hf_dataset = IndexProxy()
        return dataset

    with pytest.raises(ValueError, match="重复|未在 filtered"):
        source.RoboCasaExactWindowSourceReader(
            RoboCasaExactWindowCacheCatalog(roots[0]),
            roots[1],
            dataset_factory=factory,
        )


@pytest.mark.level(0)
def test_cache_witness_outside_filtered_episode_fails(roots: tuple[Path, Path]) -> None:
    _change_cache(roots[0], lambda p: p["windows"]["1"]["global_row_indices"][5].fill_(999))
    with pytest.raises(ValueError, match="witness 未在 filtered"):
        _reader(roots)


@pytest.mark.level(0)
def test_phase1a_identity_api_preserves_latent_and_optional_rows(roots: tuple[Path, Path]) -> None:
    catalog = RoboCasaExactWindowCacheCatalog(roots[0])
    reader = RoboCasaExactWindowEpisodeReader(catalog)
    identity = reader.read_identity(_KEY, 1)
    assert identity.global_row_indices == tuple(range(101, 118))
    assert identity.window_frame_indices == tuple(range(1, 18))
    assert identity.latent_source_frame_indices == (1, 5, 9, 13, 17)
    assert reader.read_window(_KEY, 1).shape == (5, 48, 2, 2)
    _change_cache(roots[0], lambda payload: payload["windows"]["1"].pop("global_row_indices"))
    reader = RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(roots[0]))
    assert reader.read_identity(_KEY, 1).global_row_indices is None
    assert reader.read_window(_KEY, 1).shape == (5, 48, 2, 2)
    with pytest.raises(ValueError, match="global_row_indices 缺失"):
        _reader(roots)


@pytest.mark.level(0)
@pytest.mark.parametrize("window", [0, 2])
def test_startup_endpoint_row_witness_mismatch(roots: tuple[Path, Path], window: int) -> None:
    _change_cache(roots[0], lambda p: p["windows"][str(window)]["global_row_indices"].add_(1))
    with pytest.raises(ValueError, match="startup first/terminal"):
        _reader(roots)


@pytest.mark.level(0)
def test_runtime_middle_row_witness_mismatch(roots: tuple[Path, Path]) -> None:
    _change_cache(roots[0], lambda p: p["windows"]["1"]["global_row_indices"][5].add_(1))
    reader = _reader(roots)
    with pytest.raises(ValueError, match="runtime global index"):
        reader.read_window(_KEY, 1)


@pytest.mark.level(0)
@pytest.mark.parametrize("column", ["episode_index", "frame_index", "annotation.human.task_name"])
def test_runtime_identity_drift(roots: tuple[Path, Path], column: str) -> None:
    reader = _reader(roots)
    original = type(reader.dataset).__getitem__

    def changed(self, index):
        item = original(self, index)
        item[column] = item[column].clone()
        item[column][5] += 1
        return item

    with patch.object(type(reader.dataset), "__getitem__", changed):
        with pytest.raises(ValueError, match="runtime"):
            reader.read_window(_KEY, 1)


@pytest.mark.level(0)
@pytest.mark.parametrize("column", list(source._DELTA_LENGTHS))
def test_any_pad_mask_rejected(roots: tuple[Path, Path], column: str) -> None:
    reader = _reader(roots)
    original = type(reader.dataset).__getitem__

    def changed(self, index):
        item = original(self, index)
        item[f"{column}_is_pad"] = item[f"{column}_is_pad"].clone()
        item[f"{column}_is_pad"][0] = True
        return item

    with patch.object(type(reader.dataset), "__getitem__", changed):
        with pytest.raises(ValueError, match="padding"):
            reader.read_window(_KEY, 1)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "column,bad",
    [
        ("action", torch.zeros(15, 12)),
        ("action", torch.zeros(16, 12, dtype=torch.int64)),
        ("action", torch.full((16, 12), float("nan"))),
        ("observation.state", torch.zeros(16, 16)),
        ("observation.state", torch.zeros(17, 16, dtype=torch.int64)),
        ("observation.state", torch.full((17, 16), float("inf"))),
    ],
)
def test_action_state_shape_dtype_finite_guard(roots: tuple[Path, Path], column: str, bad: torch.Tensor) -> None:
    reader = _reader(roots)
    original = type(reader.dataset).__getitem__

    def changed(self, index):
        item = original(self, index)
        item[column] = bad
        return item

    with patch.object(type(reader.dataset), "__getitem__", changed):
        with pytest.raises(ValueError, match="action12|state16"):
            reader.read_window(_KEY, 1)


@pytest.mark.level(0)
@pytest.mark.parametrize("caption", ["", None])
def test_caption_empty_or_missing(roots: tuple[Path, Path], caption: object) -> None:
    reader = _reader(roots)
    original = type(reader.dataset).__getitem__

    def changed(self, index):
        item = original(self, index)
        item["task"] = caption
        return item

    with patch.object(type(reader.dataset), "__getitem__", changed):
        with pytest.raises(ValueError, match="ai_caption"):
            reader.read_window(_KEY, 1)


@pytest.mark.level(0)
def test_source_module_has_no_later_phase_imports() -> None:
    tree = ast.parse(Path(source.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(
        any(name in (module or "") for name in ("vae", "local_memory", "action_processing")) for module in imports
    )

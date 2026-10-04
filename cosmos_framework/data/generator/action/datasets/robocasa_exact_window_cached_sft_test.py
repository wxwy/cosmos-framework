# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU checks for exact cache-to-ActionSFT and packed transport."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowCacheCatalog,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    RoboCasaExactWindowCachedDataset,
    get_action_robocasa_exact_window_cached_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import RoboCasaExactWindowSourceReader
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source_test import _cache, _source
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline, VideoResize
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    cache = _cache(tmp_path / "cache")
    source = _source(tmp_path / "source", extra_episode=True)
    geometry = VideoResize()({"video": torch.zeros((3, 1, 256, 512), dtype=torch.uint8)}, None)["image_size"]
    target_h, target_w = (int(value) for value in geometry[:2])
    assert (target_h, target_w) == (192, 320)
    spatial_factor = EDGE_MODEL_CONFIG["tokenizer"]["spatial_compression_factor"]
    assert spatial_factor == 16
    shape = [5, 48, target_h // spatial_factor, target_w // spatial_factor]
    assert shape[-2:] == [12, 20]
    manifest_file = cache / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["latent_shape"] = shape
    manifest_file.write_text(json.dumps(manifest))
    payload_file = cache / "tasks/Pick_Mug/episodes/episode_000000.pt"
    payload = torch.load(payload_file, weights_only=True)
    for start, window in payload["windows"].items():
        window["latent"] = torch.full(shape, float(int(start) + 1))
    torch.save(payload, payload_file)
    data_file = source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_file)
    states = table["observation.state"].to_pylist()
    for state in states:
        state[10:14] = [0.0, 0.0, 0.0, 1.0]
    table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", pa.array(states))
    pq.write_table(table, data_file)
    return cache, source


def _raw(roots: tuple[Path, Path]) -> RoboCasaExactWindowCachedDataset:
    catalog = RoboCasaExactWindowCacheCatalog(roots[0])
    source = RoboCasaExactWindowSourceReader(catalog, roots[1])
    return RoboCasaExactWindowCachedDataset(
        catalog, source, CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog)
    )


@pytest.mark.level(0)
def test_constructor_rejects_wrong_cache_canvas_geometry(roots: tuple[Path, Path]) -> None:
    manifest_file = roots[0] / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["latent_shape"][-2] += 1
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cache latent spatial shape"):
        _raw(roots)


@pytest.mark.level(0)
def test_shuffle_blocks_follow_two_cache_episodes(roots: tuple[Path, Path]) -> None:
    manifest_file = roots[0] / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    second = dict(manifest["tasks"][0]["episodes"][0], episode_index=1)
    manifest["tasks"][0]["episodes"].append(second)
    manifest["episode_count"] = 2
    manifest["window_count"] = 6
    manifest_file.write_text(json.dumps(manifest))
    first_file = roots[0] / "tasks/Pick_Mug/episodes/episode_000000.pt"
    payload = torch.load(first_file, weights_only=True)
    for window in payload["windows"].values():
        window["global_row_indices"] += 19
    torch.save(payload, first_file.with_name("episode_000001.pt"))
    dataset = _raw(roots)
    assert len(dataset) == 6
    assert dataset.get_shuffle_blocks() == [(0, 3), (3, 3)]
    assert dataset[3]["episode_index"] == 1
    assert dataset[3]["start_frame"] == 0


@pytest.mark.level(0)
def test_cache_membership_identity_and_official_idle(roots: tuple[Path, Path]) -> None:
    dataset = _raw(roots)
    assert len(dataset) == 3
    assert dataset.get_shuffle_blocks() == [(0, 3)]
    official_idle = RoboCasaLeRobotDataset._compute_idle_frames
    with patch.object(
        RoboCasaLeRobotDataset,
        "_compute_idle_frames",
        side_effect=lambda proxy, action: official_idle(proxy, action),
    ) as spy:
        first = dataset[0]
    assert spy.call_count == 1
    assert first["idle_frames"] == official_idle(dataset._idle_proxy, first["action"][1:])
    assert tuple(first["video"].shape) == (3, 17, 256, 512)
    assert first["video"].dtype == torch.uint8 and not bool(first["video"].any())
    assert first["action"].shape == (17, 15) and first["action"].dtype == torch.float32
    assert first["video_latent"].shape == dataset.catalog.latent_shape
    assert first["video_latent"].dtype == torch.float32
    assert first["cached_latent_required"] is True
    assert first["domain_id"].item() == get_domain_id("robocasa")
    assert first["viewpoint"] == "concat_view"
    assert first["additional_view_description"] == (
        "The left half is a third-person view of the scene. The right half is from the wrist-mounted camera."
    )
    assert first["task_class"] == "Pick Mug" and first["episode_index"] == 0
    assert first["start_frame"] == 0
    assert first["global_row_indices"].tolist() == list(range(100, 117))
    assert first["window_frame_indices"].tolist() == list(range(17))
    assert first["latent_source_frame_indices"].tolist() == [0, 4, 8, 12, 16]
    assert first["cache_corpus_digest"] == dataset.catalog.corpus_digest
    assert first["source_binding_digest"] == dataset.source_reader.source_binding_digest
    torch.testing.assert_close(dataset[1]["video_latent"], torch.full(dataset.catalog.latent_shape, 2.0))


@pytest.mark.level(0)
@pytest.mark.parametrize("field", ["key", "start_frame", "global_row_indices", "source_binding_digest"])
def test_identity_mismatch_fails_closed(roots: tuple[Path, Path], field: str) -> None:
    dataset = _raw(roots)
    original = dataset.source_reader.read_at

    def wrong(index):
        from dataclasses import replace

        window = original(index)
        value = {
            "key": type(window.key)("Other", "Other", 0),
            "start_frame": 1,
            "global_row_indices": tuple(range(200, 217)),
            "source_binding_digest": "wrong",
        }[field]
        return replace(window, **{field: value})

    with patch.object(dataset.source_reader, "read_at", side_effect=wrong), pytest.raises(ValueError, match="identity"):
        dataset[0]


@pytest.mark.level(0)
def test_transform_geometry_action_latent_and_summary(roots: tuple[Path, Path]) -> None:
    raw = _raw(roots)
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.1,
        max_action_dim=64,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
    )
    sft = ActionSFTDataset(raw, transform, resolution=None)
    original = raw[0]
    sample = sft[0]
    assert sample["video_latent"].shape == original["video_latent"].shape
    assert sample["video_latent"].dtype == original["video_latent"].dtype
    assert sample["cached_latent_required"] is True
    torch.testing.assert_close(sample["video_latent"], original["video_latent"], rtol=0, atol=0)
    assert sample["image_size"].tolist() == raw._canvas_geometry
    assert tuple(sample["video"].shape[-2:]) == tuple(raw._canvas_geometry[:2])
    assert sample["action_raw"].shape == (17, 15)
    torch.testing.assert_close(sample["action_raw"], original["action"], rtol=0, atol=0)
    assert sample["action"].shape == (17, 64)
    torch.testing.assert_close(sample["action"][:, :15], original["action"], rtol=0, atol=0)
    assert not bool(sample["action"][:, 15:].any())
    plan = sample["sequence_plan"]
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    summary = raw.summary()
    assert summary["exact_window_count"] == 3
    assert summary["selected_cache_episodes"] == 1
    assert summary["mismatch_counters"] == dict.fromkeys(
        ("missing", "ambiguous", "task_mismatch", "frame_mismatch", "row_mismatch"), 0
    )
    assert summary["vae_encode_contract"] == raw.catalog.vae_encode_contract
    assert summary["cached_latent_required"] and not summary["online_vae_fallback"]
    assert summary["model_cache_hit_required"]
    assert summary["inner_collate_video_latent_abi"] == "Tensor[B,5,48,H,W]"
    assert summary["model_video_latent_abi"] == "list[B] of Tensor[1,5,48,H,W]"


@pytest.mark.level(0)
def test_actual_collate_and_packing_keep_distinct_abis_and_order(roots: tuple[Path, Path]) -> None:
    class _FakeTextTokenizer:
        def __call__(self, sample):
            sample["ai_caption"] = json.dumps(sample["ai_caption"])
            sample["text_token_ids"] = torch.tensor([1, 2, 3])
            return sample

    with patch(
        "cosmos_framework.data.generator.action.utils.transforms.TextTokenizerTransform",
        return_value=_FakeTextTokenizer(),
    ):
        dataset = get_action_robocasa_exact_window_cached_sft_dataset(
            cache_root=roots[0], source_root=roots[1], tokenizer_config={"synthetic": True}
        )
    assert callable(dataset.summary)
    first, second = dataset[0], dataset[1]
    for batch in ([first], [first, second]):
        inner = custom_collate_fn(batch)
        assert inner["video_latent"].shape == (len(batch), *first["video_latent"].shape)
        assert inner["video_latent"].dtype == torch.float32
        assert inner["cached_latent_required"].tolist() == [True] * len(batch)
    corrupted = dict(second)
    corrupted.pop("video_latent")
    inner_corrupted = custom_collate_fn([first, corrupted])
    assert "video_latent" not in inner_corrupted
    assert inner_corrupted["cached_latent_required"].tolist() == [True, True]
    loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn, shuffle=False, num_workers=0)
    packed = PackingDataLoader(
        dataloader=loader,
        tokenizer_spatial_compression_factor=EDGE_MODEL_CONFIG["tokenizer"]["spatial_compression_factor"],
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=2,
    )
    split = packed._get_next_sample(0)
    assert isinstance(split["video"], list) and len(split["video"]) == 1
    assert split["video"][0].shape == first["video"].shape
    assert split["video_latent"].shape == (1, *first["video_latent"].shape)
    assert split["cached_latent_required"].shape == (1,) and bool(split["cached_latent_required"].item())
    packed.buffers[0].appendleft(split)
    batch = next(iter(packed))
    assert len(batch["video"]) == len(batch["video_latent"]) == 2
    for index, expected in enumerate((first, second)):
        assert isinstance(batch["video"][index], list) and len(batch["video"][index]) == 1
        assert batch["video"][index][0].shape == expected["video"].shape
        assert batch["video_latent"][index].shape == (1, *expected["video_latent"].shape)
        assert batch["cached_latent_required"][index].shape == (1,)
        assert bool(batch["cached_latent_required"][index].item())
        torch.testing.assert_close(batch["video_latent"][index][0], expected["video_latent"], rtol=0, atol=0)
        assert int(batch["start_frame"][index].item()) == index
        assert batch["cache_corpus_digest"][index] == expected["cache_corpus_digest"]
        assert batch["source_binding_digest"][index] == expected["source_binding_digest"]

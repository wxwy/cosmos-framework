# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU contract tests for the cache-first RoboCasa reader."""

from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    ExactWindowEpisodeRecord,
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)


def _episode_path(root: Path, slug: str = "Pick_Mug", index: int = 7) -> Path:
    return root / "tasks" / slug / "episodes" / f"episode_{index:06d}.pt"


def _manifest(root: Path) -> dict:
    return {
        "schema_version": "exact_window_v1",
        "source_format": "lerobot_v3",
        "suite": "robocasa365_target_atomic",
        "source_dataset": "/machine-specific/raw-dataset",
        "chunk_length": 16,
        "sample_stride": 1,
        "fps": 20.0,
        "camera_set": "left_wrist",
        "latent_shape": [5, 48, 2, 3],
        "action_shape_source": [12],
        "state_shape": [16],
        "vae_encode_contract": {
            "compute_dtype": "torch.bfloat16",
            "encode_exact_durations": [17, 61, 73],
            "encode_chunk_frames": {"256": 68, "480": 24},
        },
        "task_class_count": 1,
        "episode_count": 1,
        "window_count": 2,
        "tasks": [
            {
                "task_class": "Pick Mug",
                "task_slug": "Pick_Mug",
                "episodes": [
                    {
                        "task_class": "Pick Mug",
                        "task_slug": "Pick_Mug",
                        "episode_index": 7,
                        "episode_path": str(root / "untrusted-provenance" / "episode_000007.pt"),
                        "window_count": 2,
                        "source_video_frames": 18,
                        "camera_set": "left_wrist",
                        "latent_shape": [5, 48, 2, 3],
                    }
                ],
            }
        ],
    }


def _window(start: int, *, value: float = 1.0) -> dict:
    return {
        "latent": torch.full((5, 48, 2, 3), value + start, dtype=torch.float32),
        "window_frame_indices": torch.arange(start, start + 17),
        "latent_source_frame_indices": torch.arange(start, start + 17, 4),
        "global_row_indices": torch.arange(100 + start, 117 + start),
    }


def _payload(*, task_class: str = "Pick Mug", episode_index: int = 7, window_count: int = 2) -> dict:
    return {
        "format": "exact_window_v1",
        "source_format": "lerobot_v3",
        "suite": "robocasa365_target_atomic",
        "task_class": task_class,
        "episode_index": episode_index,
        "metadata": {
            "camera_set": "left_wrist",
            "task_class": task_class,
            "task_slug": "Pick_Mug",
            "episode_index": episode_index,
        },
        "windows": {str(start): _window(start) for start in range(window_count)},
    }


def _write(root: Path, manifest: dict, payload: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if payload is not None:
        path = _episode_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    return manifest_path


@pytest.fixture
def cache(tmp_path: Path) -> tuple[Path, dict, dict]:
    manifest, payload = _manifest(tmp_path), _payload()
    _write(tmp_path, manifest, payload)
    return tmp_path, manifest, payload


@pytest.mark.level(0)
def test_valid_catalog_stats_digest_and_exact_window(cache: tuple[Path, dict, dict]) -> None:
    root, _, _ = cache
    manifest_bytes = (root / "dataset_manifest.json").read_bytes()
    catalog = RoboCasaExactWindowCacheCatalog(root)
    stats = catalog.stats.as_dict()
    key = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7)

    assert catalog.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert stats["corpus_digest"] == catalog.corpus_digest
    assert len(catalog.corpus_digest) == 64
    assert stats["cache_root"] == str(root)
    assert (stats["schema_version"], stats["source_format"], stats["camera_set"]) == (
        "exact_window_v1",
        "lerobot_v3",
        "left_wrist",
    )
    assert (stats["fps"], stats["chunk_length"]) == (20.0, 16)
    assert (stats["task_class_count"], stats["episode_count"], stats["exact_window_count"]) == (1, 1, 2)
    assert stats["effective_consumer_count"] == sum(record.window_count for record in catalog.episodes)
    assert stats["effective_consumer_count"] == 2  # H_pred=16: one consumer per accepted 17-frame window.
    assert stats["source_unique_frame_count"] == 18
    assert stats["source_unique_frame_count_reason"] is None
    assert (stats["declared_episode_count"], stats["discovered_episode_count"], stats["accepted_episode_count"]) == (
        1,
        1,
        1,
    )
    assert all(stats[f"{name}_episode_count"] == 0 for name in ("rejected", "missing", "extra", "duplicate", "invalid"))
    assert stats["per_task"] == [
        {"task_class": "Pick Mug", "task_slug": "Pick_Mug", "episode_count": 1, "window_count": 2}
    ]
    assert (stats["min_episodes_per_task"], stats["max_episodes_per_task"]) == (1, 1)
    assert (stats["min_windows_per_task"], stats["max_windows_per_task"]) == (2, 2)
    assert stats["completeness_scope"] == "declared_cache_corpus_only"
    assert stats["payload_validation"] == "lazy_on_episode_and_window_read"
    assert catalog.record_for(key).window_starts == range(2)
    assert len(catalog.episodes) == 1
    assert isinstance(catalog.episodes[0], ExactWindowEpisodeRecord)
    assert catalog.episodes[0].key == key
    assert catalog.vae_encode_contract == _manifest(root)["vae_encode_contract"]

    latent = RoboCasaExactWindowEpisodeReader(catalog).read_window(key, 1)
    assert latent.shape == (5, 48, 2, 3) and latent.dtype == torch.float32 and latent.is_contiguous()
    torch.testing.assert_close(latent[0], torch.full((48, 2, 3), 2.0))


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["tasks"].append(deepcopy(m["tasks"][0])),
        lambda m: m["tasks"][0]["episodes"].append(deepcopy(m["tasks"][0]["episodes"][0])),
        lambda m: m["tasks"][0]["episodes"][0].update(task_class="Wrong Task"),
        lambda m: m["tasks"][0].update(task_slug="../escape"),
        lambda m: m["tasks"][0]["episodes"][0].update(episode_index=-1),
    ],
)
def test_duplicate_or_malformed_manifest_identity_fails(cache: tuple[Path, dict, dict], mutate) -> None:
    root, manifest, _ = cache
    mutate(manifest)
    _write(root, manifest)
    with pytest.raises(ValueError):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
@pytest.mark.parametrize("durations", [[33], [1, 61], [], [17, 17], [17, -1]])
def test_exact_durations_must_contain_17(cache: tuple[Path, dict, dict], durations: list[int]) -> None:
    root, manifest, _ = cache
    manifest["vae_encode_contract"]["encode_exact_durations"] = durations
    _write(root, manifest)
    with pytest.raises(ValueError, match="encode_exact_durations"):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("compute_dtype", None),
        ("compute_dtype", ""),
        ("encode_chunk_frames", None),
        ("encode_chunk_frames", []),
        ("encode_chunk_frames", {}),
        ("encode_chunk_frames", {"256": 0}),
        ("encode_chunk_frames", {"256": -1}),
        ("encode_chunk_frames", {"256": "68"}),
    ],
)
def test_invalid_vae_encode_contract_fails(cache: tuple[Path, dict, dict], field: str, bad: object) -> None:
    root, manifest, _ = cache
    manifest["vae_encode_contract"][field] = bad
    _write(root, manifest)
    with pytest.raises(ValueError, match=field):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
def test_missing_compute_dtype_fails(cache: tuple[Path, dict, dict]) -> None:
    root, manifest, _ = cache
    manifest["vae_encode_contract"].pop("compute_dtype")
    _write(root, manifest)
    with pytest.raises(ValueError, match="compute_dtype"):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
@pytest.mark.parametrize("frames, windows", [(18, 1), (17, 2), (16, 1)])
def test_source_video_frames_must_match_window_count(cache: tuple[Path, dict, dict], frames: int, windows: int) -> None:
    root, manifest, _ = cache
    row = manifest["tasks"][0]["episodes"][0]
    row["source_video_frames"] = frames
    row["window_count"] = windows
    manifest["window_count"] = windows
    _write(root, manifest)
    with pytest.raises(ValueError, match="window_count/source_video_frames"):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema_version", "other"),
        ("source_format", "other"),
        ("chunk_length", 32),
        ("sample_stride", 2),
        ("fps", 15.0),
        ("camera_set", "wrist_lr"),
        ("latent_shape", [5, 48, 0, 3]),
        ("action_shape_source", [15]),
        ("state_shape", [10]),
    ],
)
def test_manifest_core_contract_fails_closed(cache: tuple[Path, dict, dict], field: str, bad: object) -> None:
    root, manifest, _ = cache
    manifest[field] = bad
    _write(root, manifest)
    with pytest.raises(ValueError):
        RoboCasaExactWindowCacheCatalog(root)


@pytest.mark.level(0)
def test_declared_missing_file_fails_even_in_audit_mode(cache: tuple[Path, dict, dict]) -> None:
    root, _, _ = cache
    _episode_path(root).unlink()
    for strict in (True, False):
        with pytest.raises(ValueError, match="缺失"):
            RoboCasaExactWindowCacheCatalog(root, strict=strict)


@pytest.mark.level(0)
def test_extra_recognized_payload_is_visible_and_strictly_rejected(cache: tuple[Path, dict, dict]) -> None:
    root, _, payload = cache
    torch.save(payload, _episode_path(root, index=8))
    with pytest.raises(ValueError, match="extra=1"):
        RoboCasaExactWindowCacheCatalog(root)
    stats = RoboCasaExactWindowCacheCatalog(root, strict=False).stats.as_dict()
    assert (stats["declared_episode_count"], stats["discovered_episode_count"]) == (1, 2)
    assert (stats["accepted_episode_count"], stats["rejected_episode_count"], stats["extra_episode_count"]) == (1, 1, 1)
    assert stats["exact_window_count"] == stats["effective_consumer_count"] == 2


@pytest.mark.level(0)
def test_undeclared_task_payload_is_extra_and_fails_strict(cache: tuple[Path, dict, dict]) -> None:
    root, _, payload = cache
    unexpected = _episode_path(root, slug="Unknown_Task", index=3)
    unexpected.parent.mkdir(parents=True)
    torch.save(payload, unexpected)
    with pytest.raises(ValueError, match="extra=1"):
        RoboCasaExactWindowCacheCatalog(root)
    stats = RoboCasaExactWindowCacheCatalog(root, strict=False).stats.as_dict()
    assert stats["discovered_episode_count"] == 2
    assert stats["extra_episode_count"] == stats["rejected_episode_count"] == 1
    assert stats["accepted_episode_count"] == 1


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(format="wrong"),
        lambda p: p.update(source_format="wrong"),
        lambda p: p.update(suite="wrong"),
        lambda p: p.update(task_class="wrong"),
        lambda p: p.update(episode_index=99),
        lambda p: p["metadata"].update(camera_set="wrist_lr"),
        lambda p: p["metadata"].update(task_slug="wrong"),
        lambda p: p["windows"].pop("1"),
    ],
)
def test_wrong_payload_identity_format_camera_or_count_fails(cache: tuple[Path, dict, dict], mutate) -> None:
    root, _, payload = cache
    mutate(payload)
    torch.save(payload, _episode_path(root))
    reader = RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(root))
    with pytest.raises(ValueError):
        reader.read_window(ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7), 0)


@pytest.mark.level(0)
def test_missing_start_has_no_nearest_or_floor_fallback(cache: tuple[Path, dict, dict]) -> None:
    root, _, payload = cache
    key = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7)
    reader = RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(root))
    with pytest.raises(KeyError):
        reader.read_window(key, 2)
    payload["windows"]["2"] = payload["windows"].pop("1")
    torch.save(payload, _episode_path(root))
    with pytest.raises(ValueError, match="窗口键"):
        RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(root)).read_window(key, 1)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "latent",
    [
        torch.ones((5, 48, 2, 3), dtype=torch.float16),
        torch.ones((5, 48, 2, 4)),
        torch.full((5, 48, 2, 3), float("nan")),
    ],
)
def test_bad_latent_dtype_shape_or_finite_fails(cache: tuple[Path, dict, dict], latent: torch.Tensor) -> None:
    root, _, payload = cache
    payload["windows"]["0"]["latent"] = latent
    torch.save(payload, _episode_path(root))
    reader = RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(root))
    with pytest.raises(ValueError, match="latent"):
        reader.read_window(ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7), 0)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    ("field", "indices"),
    [
        ("window_frame_indices", torch.arange(1, 18)),
        ("latent_source_frame_indices", torch.arange(1, 18, 4)),
        ("global_row_indices", torch.tensor([0.5] * 17)),
        ("global_row_indices", torch.tensor([float("nan")] * 17)),
        ("global_row_indices", torch.arange(16)),
    ],
)
def test_wrong_frame_or_row_indices_fail(cache: tuple[Path, dict, dict], field: str, indices: torch.Tensor) -> None:
    root, _, payload = cache
    payload["windows"]["0"][field] = indices
    torch.save(payload, _episode_path(root))
    reader = RoboCasaExactWindowEpisodeReader(RoboCasaExactWindowCacheCatalog(root))
    with pytest.raises(ValueError, match=field):
        reader.read_window(ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7), 0)


@pytest.mark.level(0)
def test_root_relocation_and_manifest_absolute_paths_do_not_change_semantic_digest(tmp_path: Path) -> None:
    roots = (tmp_path / "machine_a", tmp_path / "machine_b")
    digests = []
    manifest_hashes = []
    for root in roots:
        manifest = _manifest(root)
        manifest["source_dataset"] = str(root / "raw")
        _write(root, manifest, _payload())
        catalog = RoboCasaExactWindowCacheCatalog(root)
        digests.append(catalog.corpus_digest)
        manifest_hashes.append(catalog.manifest_sha256)
    assert digests[0] == digests[1]
    assert manifest_hashes[0] != manifest_hashes[1]


@pytest.mark.level(0)
def test_missing_source_frames_reports_unknown_unique_count(cache: tuple[Path, dict, dict]) -> None:
    root, manifest, _ = cache
    manifest["tasks"][0]["episodes"][0].pop("source_video_frames")
    _write(root, manifest)
    stats = RoboCasaExactWindowCacheCatalog(root).stats.as_dict()
    assert stats["source_unique_frame_count"] is None
    assert "source_video_frames" in stats["source_unique_frame_count_reason"]
    assert stats["effective_consumer_count"] == stats["exact_window_count"] == 2


@pytest.mark.level(0)
def test_catalog_does_not_load_payload_and_reader_lru_is_bounded(
    cache: tuple[Path, dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, _ = cache
    second = deepcopy(manifest["tasks"][0]["episodes"][0])
    second.update(episode_index=8, episode_path="/not/the/cache/path", source_video_frames=17, window_count=1)
    manifest["tasks"][0]["episodes"].append(second)
    manifest["episode_count"] = 2
    manifest["window_count"] = 3
    _write(root, manifest)
    torch.save(_payload(episode_index=8, window_count=1), _episode_path(root, index=8))
    original_load = torch.load
    calls: list[Path] = []

    def counting_load(path: Path, *args, **kwargs):
        calls.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    catalog = RoboCasaExactWindowCacheCatalog(root)
    assert not calls  # Membership comes solely from the manifest, never torch.load or raw data.
    assert all(isinstance(record, ExactWindowEpisodeRecord) for record in catalog.episodes)
    assert [record.key.episode_index for record in catalog.episodes] == [7, 8]
    reader = RoboCasaExactWindowEpisodeReader(catalog, max_cached_episodes=1)
    first = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7)
    other = ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 8)
    reader.read_window(first, 0)
    reader.read_window(first, 1)
    reader.read_window(other, 0)
    reader.read_window(first, 0)
    assert calls == [_episode_path(root), _episode_path(root, index=8), _episode_path(root)]
    assert len(reader._loaded) == 1


@pytest.mark.level(0)
def test_imports_have_no_raw_dataset_or_vae_fallback_dependencies() -> None:
    source = Path(__file__).with_name("robocasa_exact_window_cache.py").read_text(encoding="utf-8")
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = " ".join(ast.unparse(node) for node in imports)
    assert "robocasa_lerobot_dataset" not in imported
    assert "vision_vae" not in imported
    assert "tokenizer" not in imported.lower()
    assert "Wan" not in imported


@pytest.mark.level(0)
def test_full_vae_contract_is_preserved_without_hardcoded_duration_list(cache: tuple[Path, dict, dict]) -> None:
    root, manifest, _ = cache
    manifest["vae_encode_contract"]["encode_exact_durations"] = [17, 33]
    manifest["vae_encode_contract"]["extra_provenance"] = {"model": "synthetic"}
    _write(root, manifest)
    catalog = RoboCasaExactWindowCacheCatalog(root)
    assert catalog.vae_encode_contract == manifest["vae_encode_contract"]
    changed = catalog.vae_encode_contract
    changed["encode_exact_durations"].append(99)
    assert catalog.vae_encode_contract["encode_exact_durations"] == [17, 33]

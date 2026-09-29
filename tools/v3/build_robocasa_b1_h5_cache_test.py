# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU/static acceptance tests for the B1 frozen H5 cache builder."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from build_robocasa_b1_h5_cache import (
    CAMERAS,
    LEFT_CAMERA,
    WRIST_CAMERA,
    build_cache,
    endpoint_vector,
    enumerate_train_episodes,
    episode_output_path,
    is_complete_episode,
    static_latents,
    write_episode_h5,
)

V30_SOURCE = Path("/mnt/data1/data_v2_0617/robocasa365_official_v30")
REQUIRES_SOURCE = pytest.mark.skipif(
    not V30_SOURCE.is_dir(),
    reason="canonical V30 source not present on this host",
)


def _valid_fixture(tmp_path: Path, *, episode_id: str = "ep_000000", frame_count: int = 313) -> Path:
    endpoints = endpoint_vector(frame_count)
    latents = static_latents(episode_id, len(endpoints))
    path = episode_output_path(tmp_path, "CloseFridge", "20250816", 0)
    write_episode_h5(
        path,
        episode_id=episode_id,
        frame_count=frame_count,
        latents=latents,
        endpoints={cam: endpoints for cam in CAMERAS},
    )
    return path


def _restore_env(call):
    snapshot = dict(os.environ)
    try:
        return call()
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_endpoint_vector_grid_and_terminal() -> None:
    assert endpoint_vector(313) == list(range(0, 313, 4))
    assert endpoint_vector(313)[-1] == 312
    endpoints = endpoint_vector(299)
    assert endpoints[-2:] == [296, 298]
    assert endpoints[-1] == 298


def test_endpoint_vector_rejects_invalid() -> None:
    for bad in (0, -3, 1.5, True):
        with pytest.raises(ValueError):
            endpoint_vector(bad)


def test_episode_output_path_exact_relative() -> None:
    path = episode_output_path(Path("/out"), "OpenDrawer", "20250816", 42)
    assert path == Path("/out/OpenDrawer/20250816/lerobot/ep_000042.h5")
    with pytest.raises(ValueError):
        episode_output_path(Path("/out"), "", "20250816", 0)
    with pytest.raises(ValueError):
        episode_output_path(Path("/out"), "OpenDrawer", "", 0)


def test_attrs_two_cameras_fp16_valid_all_true(tmp_path: Path) -> None:
    path = _valid_fixture(tmp_path, episode_id="ep_000007", frame_count=299)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["episode_id"] == "ep_000007"
        assert handle.attrs["frame_count"] == 299
        assert handle.attrs["temporal_compression_factor"] == 4
        assert handle.attrs["source_frame_to_latent_policy"] == "causal_endpoint"
        assert set(handle.keys()) == {"latents", "indices", "valid"}
        for cam in CAMERAS:
            latent = handle[f"latents/{cam}"]
            indices = handle[f"indices/latent_source_frame_indices/{cam}"]
            valid = handle[f"valid/{cam}"]
            expected_n = len(endpoint_vector(299))
            assert latent.shape == (expected_n, 48, 16, 16)
            assert latent.dtype == np.dtype("float16")
            assert indices.shape == (expected_n,)
            assert indices.dtype.kind in "iu"
            assert tuple(int(v) for v in indices[:]) == tuple(endpoint_vector(299))
            assert valid.shape == (expected_n,)
            assert valid.dtype == np.dtype("bool")
            assert bool(valid[:].all())


def test_reader_reads_synthetic_fixture(tmp_path: Path) -> None:
    from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader

    path = _valid_fixture(tmp_path, episode_id="ep_000123", frame_count=439)
    reader = RoboCasaLatentReader(
        path,
        expected_episode_id="ep_000123",
        expected_source_frames=439,
    )
    assert tuple(reader.endpoint_indices) == tuple(endpoint_vector(439))
    assert reader._summaries.shape == (len(endpoint_vector(439)), 96)
    assert bool(reader._summaries.isfinite().all())


@pytest.mark.parametrize(
    "corrupt",
    [
        "wrong_episode_id",
        "wrong_frame_count",
        "float32_latents",
        "missing_valid",
        "empty_valid",
        "bad_endpoints",
        "truncated",
    ],
)
def test_malformed_cache_fails_closed(tmp_path: Path, corrupt: str) -> None:
    path = _valid_fixture(tmp_path, episode_id="ep_000005", frame_count=313)
    with h5py.File(path, "a") as handle:
        if corrupt == "wrong_episode_id":
            handle.attrs["episode_id"] = "ep_999999"
        elif corrupt == "wrong_frame_count":
            handle.attrs["frame_count"] = 314
        elif corrupt == "float32_latents":
            data = handle[f"latents/{LEFT_CAMERA}"][:].astype(np.float32)
            del handle[f"latents/{LEFT_CAMERA}"]
            handle.create_dataset(f"latents/{LEFT_CAMERA}", data=data, dtype="float32")
        elif corrupt == "missing_valid":
            del handle[f"valid/{WRIST_CAMERA}"]
        elif corrupt == "empty_valid":
            handle[f"valid/{WRIST_CAMERA}"][0] = False
        elif corrupt == "bad_endpoints":
            handle[f"indices/latent_source_frame_indices/{WRIST_CAMERA}"][-1] = 300
    if corrupt == "truncated":
        size = path.stat().st_size
        with path.open("r+b") as raw:
            raw.truncate(size // 2)
    assert not is_complete_episode(path, episode_id="ep_000005", frame_count=313)


def test_atomic_publication_no_tmp_left(tmp_path: Path) -> None:
    path = _valid_fixture(tmp_path, episode_id="ep_000009", frame_count=351)
    assert path.is_file()
    assert not path.with_suffix(".h5.tmp").exists()
    broken = episode_output_path(tmp_path, "CloseFridge", "20250816", 1)
    bad_latents = {cam: torch.zeros(3, 48, 16, 16, dtype=torch.float16) for cam in CAMERAS}
    with pytest.raises(ValueError):
        write_episode_h5(
            broken,
            episode_id="ep_000001",
            frame_count=313,
            latents=bad_latents,
            endpoints={cam: [0, 4, 8] for cam in CAMERAS},
        )
    assert not broken.exists()
    assert not broken.with_suffix(".h5.tmp").exists()


@REQUIRES_SOURCE
def test_enumerate_train_episodes_exactly_9036() -> None:
    specs = _restore_env(lambda: enumerate_train_episodes(V30_SOURCE))
    assert len(specs) == 9036
    identities = [(spec.task, spec.date, spec.episode_index) for spec in specs]
    assert len(set(identities)) == 9036
    assert len(set(spec.full_id for spec in specs)) == 9036
    for spec in specs:
        assert spec.shard == (V30_SOURCE / spec.task / spec.date / "lerobot").resolve()
        assert spec.episode_id == f"ep_{spec.episode_index:06d}"
        assert spec.valid_count == spec.frame_count - 32
    assert [spec.full_id for spec in specs] == [
        spec.full_id for spec in _restore_env(lambda: enumerate_train_episodes(V30_SOURCE))
    ]


@REQUIRES_SOURCE
def test_resume_skips_only_valid_and_rebuilds_invalid(tmp_path: Path) -> None:
    out = tmp_path / "cache"

    def run():
        return build_cache(
            V30_SOURCE,
            out,
            mode="static",
            vae_path=None,
            device="cpu",
            task_names=None,
            episode_filter=set(),
            limit=3,
            workers=1,
            worker=0,
        )

    first = _restore_env(run)
    assert len(first["built"]) == 3
    assert not first["failed"]
    built = [Path(p) for p in first["built"]]
    victim = built[1]
    with h5py.File(victim, "a") as handle:
        handle.attrs["frame_count"] = 1_000_000
    second = _restore_env(run)
    assert len(second["skipped"]) == 2
    assert len(second["built"]) == 1
    assert str(victim) in second["built"]
    assert not is_complete_episode(
        victim,
        episode_id=victim.stem,
        frame_count=1_000_000,
    )


@REQUIRES_SOURCE
def test_static_build_source_untouched_and_reader_reads(tmp_path: Path) -> None:
    from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader

    out = tmp_path / "cache"
    probe = V30_SOURCE / "CloseFridge" / "20250816" / "lerobot" / "meta" / "info.json"
    before = probe.read_bytes()
    reports = _restore_env(
        lambda: build_cache(
            V30_SOURCE,
            out,
            mode="static",
            vae_path=None,
            device="cpu",
            task_names=None,
            episode_filter=set(),
            limit=1,
            workers=1,
            worker=0,
        )
    )
    assert len(reports["built"]) == 1
    assert not reports["failed"]
    assert reports["records_total"] == 9036
    assert probe.read_bytes() == before
    path = Path(reports["built"][0])
    parts = path.relative_to(out).parts
    assert len(parts) == 4 and parts[2] == "lerobot" and parts[-1].endswith(".h5")
    with h5py.File(path, "r") as handle:
        frame_count = int(handle.attrs["frame_count"])
        episode_id = str(handle.attrs["episode_id"])
    RoboCasaLatentReader(
        path,
        expected_episode_id=episode_id,
        expected_source_frames=frame_count,
    )


@REQUIRES_SOURCE
def test_full_id_filter_selects_exact_shard_local_identity(tmp_path: Path) -> None:
    specs = _restore_env(lambda: enumerate_train_episodes(V30_SOURCE))
    by_episode_id: dict[str, list] = {}
    for spec in specs:
        by_episode_id.setdefault(spec.episode_id, []).append(spec)
    duplicate_group = next(group for group in by_episode_id.values() if len({spec.task for spec in group}) >= 2)
    target = duplicate_group[0]
    out = tmp_path / "cache"
    reports = _restore_env(
        lambda: build_cache(
            V30_SOURCE,
            out,
            mode="static",
            vae_path=None,
            device="cpu",
            task_names=None,
            episode_filter=set(),
            limit=0,
            workers=1,
            worker=0,
            full_id_filter={target.full_id},
        )
    )
    assert reports["selected_full_ids"] == [target.full_id]
    assert len(reports["built"]) == 1
    assert not reports["failed"]
    path = Path(reports["built"][0])
    assert path == episode_output_path(out, target.task, target.date, target.episode_index)

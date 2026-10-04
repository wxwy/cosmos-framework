# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU checks for the read-only exact-window parity probe."""

from __future__ import annotations

import json
from argparse import Namespace
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    ExactWindowEpisodeRecord,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
    CorrectedRoboCasaPolicyContract,
)
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
from cosmos_framework.data.generator.action.utils.transforms import VideoResize
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item
from tools.v3 import verify_robocasa_exact_window_real_parity as probe


def _record(task: str = "A", windows: int = 5, episode: int = 0) -> ExactWindowEpisodeRecord:
    return ExactWindowEpisodeRecord(
        ExactWindowEpisodeKey(task, task, episode), windows, windows + 16, Path("episode.pt")
    )


def _contract() -> CorrectedRoboCasaPolicyContract:
    return CorrectedRoboCasaPolicyContract(
        vae_encode_contract={
            "compute_dtype": "torch.bfloat16",
            "encode_exact_durations": [17, 61, 73],
            "encode_chunk_frames": {"256": 68, "480": 24},
        }
    )


def test_selection_first_middle_terminal_and_invalid() -> None:
    record = _record(windows=6)
    assert probe.select_starts(record) == (0, 2, 5)
    assert probe.select_starts(record, (5, 0, 5)) == (5, 0)
    with pytest.raises(ValueError, match="start"):
        probe.select_starts(record, (6,))
    catalog = SimpleNamespace(episodes=(_record("A"), _record("B"), _record("C")))
    assert len(probe.select_episodes(catalog, (), None, None, 1)) == 1
    assert len(probe.select_episodes(catalog, (), None, None, 3)) == 3
    with pytest.raises(ValueError, match="配合单个 task_class"):
        probe.select_episodes(catalog, (), 0, None, 1)
    with pytest.raises(ValueError, match="不在 exact cache"):
        probe.select_episodes(catalog, ("A",), 9, None, 1)


def test_final_coverage() -> None:
    selection = tuple((_record(task), (0, 2, 4)) for task in "ABC")
    probe.validate_final_coverage(selection)
    with pytest.raises(ValueError, match="3 task"):
        probe.validate_final_coverage(selection[:2])
    with pytest.raises(ValueError, match="9 exact"):
        probe.validate_final_coverage(tuple((record, (0, 4)) for record, _ in selection))


def test_vae_contract_only_locator_overrides(tmp_path: Path) -> None:
    candidate = deepcopy(EDGE_MODEL_CONFIG["tokenizer"])
    expected = _contract().resolve_tokenizer_config(candidate)
    actual = probe.resolve_vae_config(_contract(), tmp_path / "vae.pth")
    assert actual["encode_exact_durations"] == [17, 61, 73]
    assert actual["encode_chunk_frames"] == candidate["encode_chunk_frames"]
    assert {key: value for key, value in actual.items() if key not in ("bucket_name", "vae_path")} == {
        key: value for key, value in expected.items() if key not in ("bucket_name", "vae_path")
    }
    assert candidate == EDGE_MODEL_CONFIG["tokenizer"]
    assert actual["bucket_name"] == "" and actual["vae_path"] == str(tmp_path / "vae.pth")
    assert actual["use_streaming_encode"] is False


def _source_rows(tmp_path: Path, *, bad_index: bool = False, bad_timestamp: bool = False):
    data = tmp_path / "episode.parquet"
    indices = list(range(100, 121))
    if bad_index:
        indices[10] = 999
    timestamps = [i / 20 for i in range(21)]
    if bad_timestamp:
        timestamps[10] = float("nan")
    pq.write_table(pa.table({"index": indices, "timestamp": timestamps, "episode_index": [0] * 21}), data)
    record = _record(windows=5)
    bound = SimpleNamespace(length=21, dataset_from_index=100, dataset_to_index=121, data_file="episode.parquet")
    source = SimpleNamespace(source_root=tmp_path, _bound={record.key: bound})
    return source, record


@pytest.mark.parametrize("bad_index,bad_timestamp", [(False, False), (True, False), (False, True)])
def test_full_episode_rows_and_timestamps(tmp_path: Path, bad_index: bool, bad_timestamp: bool) -> None:
    source, record = _source_rows(tmp_path, bad_index=bad_index, bad_timestamp=bad_timestamp)
    if bad_index or bad_timestamp:
        with pytest.raises(ValueError, match="row/timestamp"):
            probe.episode_rows(source, record)
    else:
        rows, timestamps = probe.episode_rows(source, record)
        assert rows == tuple(range(100, 121))
        assert timestamps[4] == 0.2


def test_global_and_frame_witness() -> None:
    record = _record()
    rows = tuple(range(100, 121))

    class Reader:
        def read_identity(self, key, start):
            return SimpleNamespace(
                key=key,
                start_frame=start,
                global_row_indices=rows[start : start + 17],
                window_frame_indices=tuple(range(start, start + 17)),
                latent_source_frame_indices=tuple(range(start, start + 17, 4)),
            )

    witnesses = probe.verify_witnesses(Reader(), record.key, (0, 2, 4), rows)
    assert len(witnesses) == 3 and witnesses[1]["global_row_indices"][0] == 102
    wrong_rows = list(rows)
    wrong_rows[5] = 999
    with pytest.raises(ValueError, match="witness"):
        probe.verify_witnesses(Reader(), record.key, (2,), tuple(wrong_rows))


def test_video_paths_missing_and_metadata(tmp_path: Path) -> None:
    key = _record().key
    left = "observation.images.robot0_agentview_left"
    wrist = "observation.images.robot0_eye_in_hand"
    for name in (left, wrist):
        (tmp_path / f"{name}.mp4").touch()
    episodes = pa.table({f"videos/{name}/from_timestamp": [1.25] for name in (left, wrist)})
    meta = SimpleNamespace(episodes=episodes, get_video_file_path=lambda _, name: f"{name}.mp4")
    source = SimpleNamespace(meta=meta, source_root=tmp_path)
    assert len(probe.video_paths(source, key)) == 2
    (tmp_path / f"{left}.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="video"):
        probe.video_paths(source, key)


def test_episode_decode_and_official_helpers_once() -> None:
    count = Counter()
    left = "observation.images.robot0_agentview_left"
    wrist = "observation.images.robot0_eye_in_hand"

    def decode(path, timestamps, tolerance, **kwargs):
        count[path.name] += 1
        assert len(timestamps) == 21 and tolerance == 1e-4 and kwargs == {"backend": "pyav"}
        return torch.ones((21, 3, 256, 256), dtype=torch.float32) * 0.25

    def compose(proxy, sample):
        count["compose"] += 1
        return RoboCasaLeRobotDataset._compose_left_wrist(proxy, sample)

    def convert(proxy, value):
        count["convert"] += 1
        return RoboCasaLeRobotDataset._convert_video(proxy, value)

    class Resize:
        def __init__(self, **kwargs):
            assert kwargs == {"pad_keys": ["video"], "keep_aspect_ratio": True}

        def __call__(self, sample, resolution):
            count["resize"] += 1
            return VideoResize(pad_keys=["video"], keep_aspect_ratio=True)(sample, resolution)

    deps = probe.ProbeDependencies(decode=decode, compose=compose, convert=convert, resize_factory=Resize)
    paths = {name: (Path(f"{name}.mp4"), 0.0) for name in (left, wrist)}
    video, geometry = probe.reconstruct_episode(
        None, _record().key, tuple(i / 20 for i in range(21)), paths, 1e-4, "pyav", deps
    )
    assert count == {f"{left}.mp4": 1, f"{wrist}.mp4": 1, "compose": 1, "convert": 1, "resize": 1}
    assert video.shape == (3, 21, 192, 320) and geometry["image_size"].tolist() == [192, 320, 160, 320]
    assert probe._geometry(tuple(video.shape), geometry["image_size"], (5, 48, 12, 20), 16, 21) == [192, 320, 160, 320]
    with pytest.raises(ValueError, match="geometry"):
        probe._geometry(tuple(video.shape), geometry["image_size"], (5, 48, 13, 20), 16, 21)
    with pytest.raises(ValueError, match="C/T"):
        probe._geometry((3, 20, 192, 320), geometry["image_size"], (5, 48, 12, 20), 16, 21)


def test_bad_decoded_frames_fail_before_helpers() -> None:
    called = Counter()

    def decode(*args, **kwargs):
        return torch.full((21, 3, 256, 256), float("nan"))

    deps = probe.ProbeDependencies(decode=decode, compose=lambda *_: called.update(compose=1))
    with pytest.raises(ValueError, match="decoded camera"):
        probe.reconstruct_episode(None, _record().key, (0.0,) * 21, {"x": (Path("x"), 0.0)}, 1e-4, None, deps)
    assert not called


def test_encode_layout_and_normalizer_spy() -> None:
    calls = Counter()

    class FakeTokenizer:
        def encode(self, value):
            calls["encode"] += 1
            assert value.shape == (1, 3, 17, 192, 320) and value.dtype == torch.float32
            return torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1).expand(1, 48, 5, 12, 20)

    def normalize(value, kwargs):
        calls["normalize"] += 1
        return normalize_uint8_item(value, kwargs)

    deps = probe.ProbeDependencies(normalize=normalize)
    for _ in range(3):
        result = probe.encode_window(
            torch.zeros((3, 17, 192, 320), dtype=torch.uint8),
            FakeTokenizer(),
            torch.device("cpu"),
            (5, 48, 12, 20),
            deps,
        )
        assert result.shape == (5, 48, 12, 20) and result[:, 0, 0, 0].tolist() == list(range(5))
    assert calls == {"normalize": 3, "encode": 3}


def test_prepare_tokenizer_moves_model_and_both_scales_without_cuda() -> None:
    calls = []
    device = torch.device("cuda:1")

    class Scale:
        def __init__(self, name):
            self.name = name

        def to(self, target):
            calls.append((self.name, target))
            return (self.name, target)

    class Model:
        def to(self, target):
            calls.append(("model", target))
            return self

        def eval(self):
            calls.append(("eval", None))
            return self

    tokenizer = SimpleNamespace(
        use_streaming_encode=False,
        _keep_encoder_cache=False,
        model=SimpleNamespace(model=Model(), scale=(Scale("mean"), Scale("inv_std"))),
    )
    probe.prepare_tokenizer_device(tokenizer, device)
    assert calls == [("model", device), ("mean", device), ("inv_std", device), ("eval", None)]
    assert tokenizer.model.scale == (("mean", device), ("inv_std", device))
    for flag in ("use_streaming_encode", "_keep_encoder_cache"):
        calls.clear()
        setattr(tokenizer, flag, True)
        with pytest.raises(ValueError, match="normal full encode"):
            probe.prepare_tokenizer_device(tokenizer, device)
        assert calls == []
        setattr(tokenizer, flag, False)


def test_metrics_crop_and_z0_spy() -> None:
    calls = Counter()

    def crop(proxy, latents, sizes):
        calls["crop"] += 1
        return OmniMoTModel._remove_padding_from_latent(proxy, latents, sizes)

    left = torch.zeros((5, 48, 12, 20), dtype=torch.float32)
    right = left.clone()
    right[0, :, :10] = 2
    result = probe.compare_window(left, right, [192, 320, 160, 320], 16, probe.ProbeDependencies(crop=crop))
    assert calls["crop"] == 2
    assert result["pre_crop"]["max_abs"] == 2
    assert result["post_crop"]["max_abs"] == 2
    assert result["z0"]["max_abs"] == 2
    assert len(result["temporal"]) == 5 and result["temporal"][1]["max_abs"] == 0
    assert probe.metrics(left, left)["exact_equal"] is True
    assert probe.metrics(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]))["mean_abs"] == 1.5


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((5, 48, 12, 20), float("nan")),
        torch.zeros((4, 48, 12, 20)),
        torch.zeros((5, 48, 12, 20), dtype=torch.float16),
    ],
)
def test_bad_cache_shape_dtype_finite(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        probe.compare_window(bad, torch.zeros((5, 48, 12, 20)), [192, 320, 160, 320], 16, probe.ProbeDependencies())


def test_status_threshold_and_json() -> None:
    window = {"metrics": {name: {"max_abs": 0.25} for name in ("pre_crop", "post_crop", "z0")}}
    assert probe.gate_result([window], None, False) == ("OBSERVATIONAL_NO_THRESHOLD", None)
    assert probe.gate_result([window], 0.25, False) == ("PASS", True)
    assert probe.gate_result([window], 0.2, False) == ("FAIL", False)
    assert probe.gate_result([], None, True) == ("DRY_RUN_PASS", None)
    json.dumps({"schema": "robocasa_exact_window_real_vae_parity_v1", "windows": [window]}, allow_nan=False)


def test_dry_run_missing_vae_fails_before_tokenizer(tmp_path: Path) -> None:
    calls = Counter()

    def tokenizer_factory(**kwargs):
        calls["tokenizer"] += 1
        raise AssertionError("missing VAE must fail before tokenizer construction")

    args = Namespace(tolerance_s=1e-4, max_abs_threshold=None, vae_path=tmp_path / "missing.pth", dry_run=True)
    with pytest.raises(FileNotFoundError, match="Wan VAE"):
        probe.run(args, deps=probe.ProbeDependencies(tokenizer_factory=tokenizer_factory))
    assert calls["tokenizer"] == 0


def test_dry_run_and_observational_fake_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, record = _source_rows(tmp_path)
    left = "observation.images.robot0_agentview_left"
    wrist = "observation.images.robot0_eye_in_hand"
    for feature in (left, wrist):
        (tmp_path / f"{feature}.mp4").touch()
    source.meta = SimpleNamespace(
        episodes=pa.table({f"videos/{feature}/from_timestamp": [0.0] for feature in (left, wrist)}),
        get_video_file_path=lambda _, feature: f"{feature}.mp4",
    )
    source.source_binding_digest = "binding"
    catalog = SimpleNamespace(
        episodes=(record,), manifest_sha256="manifest", corpus_digest="corpus", latent_shape=(5, 48, 12, 20)
    )
    monkeypatch.setattr(
        probe.CorrectedRoboCasaPolicyContract,
        "from_cache_catalog",
        classmethod(lambda cls, value: _contract()),
    )
    calls = Counter()

    class Reader:
        def __init__(self, _catalog):
            pass

        def read_identity(self, key, start):
            return SimpleNamespace(
                key=key,
                start_frame=start,
                global_row_indices=tuple(range(100 + start, 117 + start)),
                window_frame_indices=tuple(range(start, start + 17)),
                latent_source_frame_indices=tuple(range(start, start + 17, 4)),
            )

        def read_window(self, key, start):
            return torch.zeros((5, 48, 12, 20), dtype=torch.float32)

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    class Tokenizer:
        use_streaming_encode = False
        _keep_encoder_cache = False

        def __init__(self, **kwargs):
            calls["factory"] += 1
            assert kwargs["encode_exact_durations"] == [17, 61, 73]
            self.model = SimpleNamespace(model=Model(), scale=(torch.zeros(1), torch.ones(1)))

        def encode(self, video):
            calls["encode"] += 1
            return torch.zeros((1, 48, 5, 12, 20), dtype=torch.float32)

    def decode(path, timestamps, tolerance, **kwargs):
        calls[path.name] += 1
        return torch.zeros((21, 3, 256, 256), dtype=torch.float32)

    deps = probe.ProbeDependencies(decode=decode, tokenizer_factory=Tokenizer, allow_cpu_for_test=True)
    vae_path = tmp_path / "fake_vae.pth"
    vae_path.touch()
    args = Namespace(
        tolerance_s=1e-4,
        max_abs_threshold=None,
        vae_path=vae_path,
        dry_run=True,
        cache_root=tmp_path,
        source_root=tmp_path,
        task_class=[],
        episode_index=None,
        starts=None,
        num_task_classes=1,
        hash_vae=False,
        device="cpu",
        video_backend=None,
    )
    factories = {
        "catalog_factory": lambda _: catalog,
        "source_factory": lambda *_: source,
        "reader_factory": Reader,
    }
    dry = probe.run(args, deps=deps, **factories)
    assert dry["gate"] == {"dry_run": True, "threshold": None, "status": "DRY_RUN_PASS", "parity_gate_pass": None}
    assert dry["windows"] == []
    assert len(dry["selection"]) == 1
    assert dry["selection"][0]["starts"] == [0, 2, 4]
    assert len(dry["selection"][0]["witnesses"]) == 3
    assert dry["expected_geometry"]["cached_padded_latent_shape"] == [5, 48, 12, 20]
    json.dumps(dry, allow_nan=False)
    assert calls == Counter()
    args.dry_run = False
    observed = probe.run(args, deps=deps, **factories)
    assert observed["gate"]["status"] == "OBSERVATIONAL_NO_THRESHOLD"
    assert observed["gate"]["parity_gate_pass"] is None
    assert len(observed["windows"]) == 3
    assert observed["windows"][0]["geometry"]["camera_keys"] == [left, wrist]
    for window in observed["windows"]:
        assert window["metrics"]["pre_crop"]["max_abs"] == 0
        assert window["metrics"]["post_crop"]["max_abs"] == 0
        assert window["metrics"]["z0"]["max_abs"] == 0
        assert len(window["metrics"]["temporal"]) == 5
        assert window["geometry"]["cropped_shape"] == [1, 48, 5, 10, 20]
    assert observed["aggregate"]["global_max_abs"] == 0
    assert calls == {"factory": 1, "encode": 3, f"{left}.mp4": 1, f"{wrist}.mp4": 1}
    json.dumps(observed, allow_nan=False)


def test_source_static_no_legacy_imports_or_writes() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "from cosmos_framework.model.generator.vision_vae" not in source
    assert "robocasa_latent_evidence" not in source
    assert "build_robocasa_b1_h5_cache" not in source
    assert "torch.save(" not in source
    assert ".write_text(" in source and source.count(".write_text(") == 1

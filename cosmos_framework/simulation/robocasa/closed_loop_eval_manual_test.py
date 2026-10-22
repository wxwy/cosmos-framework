"""CPU media and in-process runtime binding; no Kitchen or policy server imports."""

from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest

from cosmos_framework.simulation.robocasa.eval_utils import b64_png
from cosmos_framework.utils.rollout_evidence import run_recorded_episode


@pytest.fixture
def rollout(monkeypatch):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    monkeypatch.setattr(module, "compose", lambda obs: np.full((16, 32, 3), obs["step"], dtype=np.uint8))
    monkeypatch.setattr(module, "USE_BASE_ACTION", False)
    monkeypatch.setattr(module, "decode_10d_to_env12", lambda *_: np.zeros(12))
    monkeypatch.setattr(module, "check_success", lambda _: False)
    count = 0

    def prediction(*_args, **_kwargs):
        nonlocal count
        count += 1
        return {
            "action": [[0.0] * 10],
            "video": [b64_png(np.full((16, 32, 3), count * 40 + index, dtype=np.uint8)) for index in range(2)],
        }

    predict = Mock(side_effect=prediction)
    monkeypatch.setattr(module, "predict", predict)
    env = Mock()
    env.reset.return_value = {"step": 0}
    env.get_ep_meta.return_value = {"lang": "synthetic CPU fixture"}
    env.step.side_effect = lambda _: ({"step": env.step.call_count}, 0.0, False, {})
    return module, env, predict


def run(module, env, path, *, mode, **kwargs):
    return module.run_policy(
        env,
        server_url="unused",
        image_size=256,
        action_horizon=1,
        max_steps=3,
        latch=1,
        timeout=1,
        gen_video_path=path / "rollout00_generated.mp4" if mode != "none" else None,
        generated_video_mode=mode,
        generated_run_digest="same-identity",
        **kwargs,
    )


def read_video(path):
    with imageio.get_reader(path) as reader:
        return [np.asarray(frame) for frame in reader]


@pytest.mark.parametrize("mode,expected", [("none", 0), ("first", 1), ("all", 3)])
def test_actual_generated_media_modes_and_per_replan_manifest(rollout, tmp_path, mode, expected):
    module, env, predict = rollout
    run(module, env, tmp_path, mode=mode)
    assert predict.call_count == 3
    paths = sorted(tmp_path.glob("*.mp4"))
    assert len(paths) == expected
    if mode == "none":
        assert not list(tmp_path.iterdir())
        return
    manifest = json.loads((tmp_path / "rollout00_generated_manifest.json").read_text())
    assert manifest["mode"] == mode and manifest["evaluation_run_digest"] == "same-identity"
    assert [record["file"] for record in manifest["videos"]] == [path.name for path in paths]
    assert [record["source_step"] for record in manifest["videos"]] == list(range(expected))
    assert [record["replan"] for record in manifest["videos"]] == list(range(expected))
    for index, path in enumerate(paths):
        frames = read_video(path)
        assert len(frames) == manifest["videos"][index]["frames"] == 2
        assert abs(float(np.mean(frames[0])) - (index + 1) * 40) < 5


def test_first_restarts_at_first_prediction_of_each_episode(rollout, tmp_path):
    module, env, predict = rollout
    for episode in range(2):
        destination = tmp_path / str(episode)
        destination.mkdir()
        run(module, env, destination, mode="first")
        assert len(read_video(destination / "rollout00_generated.mp4")) == 2
        manifest = json.loads((destination / "rollout00_generated_manifest.json").read_text())
        assert manifest["videos"][0]["replan"] == 0
        assert manifest["videos"][0]["source_step"] == 0
    assert predict.call_count == 6
    assert abs(float(np.mean(read_video(tmp_path / "1/rollout00_generated.mp4")[0])) - 160) < 5


def test_none_does_not_decode_or_save_generated_media(rollout, tmp_path, monkeypatch):
    module, env, _ = rollout
    monkeypatch.setattr(module, "decode_pred_video", Mock(side_effect=AssertionError("unexpected decode")))
    run(module, env, tmp_path, mode="none")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("video", [None, [], ["invalid base64 image"]])
def test_enabled_missing_or_corrupt_generated_video_fails_and_retains_rollout(rollout, tmp_path, video):
    module, env, predict = rollout
    predict.side_effect = None
    predict.return_value = {"action": [[0.0] * 10], "video": video}
    with pytest.raises(Exception):
        run(module, env, tmp_path, mode="first", save_video=tmp_path / "rollout00.mp4")
    assert len(read_video(tmp_path / "rollout00.mp4")) == 1
    assert not (tmp_path / "rollout00_generated_manifest.json").exists()


def test_later_replan_failure_retains_generated_attempt_and_partial_environment(rollout, tmp_path):
    module, env, predict = rollout
    first = predict.side_effect()
    predict.side_effect = [first, RuntimeError("second prediction failed")]
    with pytest.raises(RuntimeError, match="second prediction"):
        run(module, env, tmp_path, mode="all", save_video=tmp_path / "rollout00.mp4")
    assert len(read_video(tmp_path / "rollout00.mp4")) == 2
    assert len(read_video(tmp_path / "rollout00_generated_replan0000.mp4")) == 2
    manifest = json.loads((tmp_path / "rollout00_generated_manifest.json").read_text())
    assert len(manifest["videos"]) == 1


def test_generated_video_and_initial_png_never_overwrite(rollout, tmp_path):
    module, env, _ = rollout
    destination = tmp_path / "rollout00_generated.mp4"
    destination.write_bytes(b"historical evidence")
    with pytest.raises(FileExistsError):
        run(module, env, tmp_path, mode="first")
    assert destination.read_bytes() == b"historical evidence"
    png = tmp_path / "init.png"
    png.write_bytes(b"historical PNG")
    with pytest.raises(FileExistsError):
        run(module, env, tmp_path, mode="none", save_png=png)
    assert png.read_bytes() == b"historical PNG"


def runtime_args(tmp_path, **overrides):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def probe_runtime(*, lock_path, asset_manifest, tasks):\n"
        "    assert list(tasks) == ['CloseFridge', 'NavigateKitchen']\n"
        "    return {'digest': 'expected', 'protocol': {'official_horizons': "
        "{'CloseFridge': 900, 'NavigateKitchen': 450}}}\n"
    )
    values = {
        "runtime_probe_script": probe,
        "runtime_lock_file": tmp_path / "lock.json",
        "runtime_asset_manifest": None,
        "expected_runtime_digest": "expected",
        "runtime_tasks": "CloseFridge,NavigateKitchen",
    }
    values.update(overrides)
    return Namespace(**values)


def test_actual_in_process_probe_preserves_complete_task_identity(tmp_path):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    runtime = module.verify_runtime(runtime_args(tmp_path), "CloseFridge")
    assert runtime["protocol"]["official_horizons"] == {"CloseFridge": 900, "NavigateKitchen": 450}


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_runtime_digest": "foreign"},
        {"runtime_lock_file": None},
        {"runtime_tasks": "CloseFridge,CloseFridge"},
        {"runtime_tasks": "NavigateKitchen"},
        {"runtime_tasks": "CloseFridge,"},
    ],
)
def test_runtime_mismatch_rejected_before_environment_creation(tmp_path, overrides):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    with pytest.raises(ValueError):
        module.verify_runtime(runtime_args(tmp_path, **overrides), "CloseFridge")


def test_partial_runtime_identity_never_falls_back_to_legacy(tmp_path):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    args = runtime_args(tmp_path, runtime_probe_script=None, runtime_lock_file=None, runtime_tasks=None)
    with pytest.raises(ValueError, match="runtime identity requires"):
        module.verify_runtime(args, "CloseFridge")


@pytest.mark.parametrize("digest", ["expected", "foreign"])
def test_main_attests_before_make_env_and_never_uses_fallback_horizon(tmp_path, monkeypatch, digest):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    args = runtime_args(tmp_path, expected_runtime_digest=digest)
    dataset = tmp_path / "dataset/extras"
    dataset.mkdir(parents=True)
    (dataset / "dataset_meta.json").write_text(json.dumps({"env_args": {"env_name": "CloseFridge"}}))
    make = Mock(side_effect=RuntimeError("STOP before any simulator creation"))
    monkeypatch.setattr(module, "make_env", make)
    monkeypatch.setattr(module, "official_horizon", Mock(side_effect=AssertionError("fallback forbidden")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "closed_loop_eval.py",
            "--dataset-dir",
            str(dataset.parent),
            "--output-dir",
            str(tmp_path / "out"),
            "--runtime-probe-script",
            str(args.runtime_probe_script),
            "--runtime-lock-file",
            str(args.runtime_lock_file),
            "--expected-runtime-digest",
            digest,
            "--runtime-tasks",
            args.runtime_tasks,
        ],
    )
    with pytest.raises((ValueError, RuntimeError)):
        module.main()
    assert make.call_count == int(digest == "expected")
    assert (tmp_path / "out/runtime_attestation.json").exists() == (digest == "expected")


def test_module_does_not_import_vendor_packages_before_runtime_audit():
    import ast

    path = Path(__file__).with_name("closed_loop_eval.py")
    top_level_imports = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.Import)]
    assert not {alias.name for node in top_level_imports for alias in node.names} & {"robocasa", "robosuite"}


def test_recorded_episode_binds_result_and_every_audit_record_to_attempt(tmp_path):
    def runner(env, *, on_prediction, on_action):
        on_prediction({"action": [[0] * 15]}, 1.0)
        on_action({"source_step": 0, "evidence_completed": True})
        return False, 1, "synthetic"

    result = run_recorded_episode(
        runner, None, output_dir=tmp_path, episode=0, run_digest="immutable-run", attempt_id="attempt0002"
    )
    assert result["evaluation_attempt_id"] == "attempt0002"
    for name in ("rollout00_actions.jsonl", "rollout00_predictions.jsonl"):
        row = json.loads((tmp_path / name).read_text())
        assert row["evaluation_attempt_id"] == "attempt0002"
        assert row["evaluation_run_digest"] == "immutable-run"
        assert row["episode"] == 0


@pytest.mark.parametrize(
    "output_task,attempt_id",
    [
        ("CloseFridge", "CloseFridge/attempt0001"),
        ("CloseFridge", "NavigateKitchen/attempt0002"),
        ("NavigateKitchen", "CloseFridge/attempt0002"),
    ],
)
def test_foreign_attempt_namespace_rejected_before_any_write(tmp_path, monkeypatch, output_task, attempt_id):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    dataset = tmp_path / "dataset/extras"
    dataset.mkdir(parents=True)
    (dataset / "dataset_meta.json").write_text(json.dumps({"env_args": {"env_name": "CloseFridge"}}))
    out = tmp_path / "new-run" / output_task / "attempt0002"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "closed_loop_eval.py",
            "--dataset-dir",
            str(dataset.parent),
            "--output-dir",
            str(out),
            "--evaluation-attempt-id",
            attempt_id,
        ],
    )
    with pytest.raises(ValueError, match="attempt identity"):
        module.main()
    assert not out.parent.exists()


@pytest.mark.parametrize("video_args,expected", [(["--save-gen-video"], 1), (["--save-generated-video", "all"], 3)])
def test_actual_main_records_attempt_and_complete_generated_media(rollout, tmp_path, monkeypatch, video_args, expected):
    import requests

    module, env, _ = rollout
    args = runtime_args(tmp_path)
    args.runtime_probe_script.write_text(args.runtime_probe_script.read_text().replace("900", "3"))
    dataset = tmp_path / "dataset/extras"
    dataset.mkdir(parents=True)
    (dataset / "dataset_meta.json").write_text(json.dumps({"env_args": {"env_name": "CloseFridge"}}))
    monkeypatch.setattr(module, "make_env", Mock(return_value=env))
    monkeypatch.setattr(module, "official_horizon", Mock(side_effect=AssertionError("fallback forbidden")))
    response = Namespace(ok=True, json=lambda: {"local_memory": {"mode": "off"}})
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    out = tmp_path / "CloseFridge/attempt0002"
    attempt_id = f"CloseFridge/{out.name}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "closed_loop_eval.py",
            "--dataset-dir",
            str(dataset.parent),
            "--output-dir",
            str(out),
            "--evaluation-run-digest",
            "immutable-run",
            "--evaluation-attempt-id",
            attempt_id,
            "--runtime-probe-script",
            str(args.runtime_probe_script),
            "--runtime-lock-file",
            str(args.runtime_lock_file),
            "--expected-runtime-digest",
            "expected",
            "--runtime-tasks",
            args.runtime_tasks,
            "--num-test-episodes",
            "2",
            "--action-horizon",
            "1",
            "--no-save-rollout-mp4",
            "--no-save-init-png",
            *video_args,
        ],
    )
    module.main()
    results = json.loads((out / "results.json").read_text())
    assert len(results) == 2
    env.close.assert_called_once()
    for episode, row in enumerate(results):
        assert row["evaluation_attempt_id"] == attempt_id
        assert row["evaluation_run_digest"] == "immutable-run"
        assert row["error"] is None and row["replans"] == 3
        assert len(row["generated_videos"]) == expected
        manifest = json.loads((out / f"rollout{episode:02d}_generated_manifest.json").read_text())
        assert manifest["status"] == "COMPLETE"
        assert manifest["evaluation_attempt_id"] == attempt_id
        assert [item["file"] for item in manifest["videos"]] == row["generated_videos"]
        for name in row["generated_videos"]:
            assert len(read_video(out / name)) == 2
        assert not (out / f"rollout{episode:02d}.mp4").exists()
        assert not (out / f"rollout{episode:02d}_init.png").exists()
        for audit in ("actions", "predictions"):
            rows = [json.loads(line) for line in (out / f"rollout{episode:02d}_{audit}.jsonl").read_text().splitlines()]
            assert len(rows) == 3
            assert all(item["evaluation_attempt_id"] == attempt_id for item in rows)

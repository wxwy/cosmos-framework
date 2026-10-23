"""CPU observational progress: original rollout actions and failures stay authoritative."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_framework.utils.rollout_progress import LiveProgressWriter


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def writer(path):
    return LiveProgressWriter(
        path, run_digest="d" * 64, attempt_id="CloseFridge/attempt_0001", task="CloseFridge", episodes=2
    )


def test_jsonl_observations_bind_identity_and_unknowns_without_touching_other_files(tmp_path):
    path = tmp_path / "live_progress.jsonl"
    live = writer(path)
    live.emit(stage="environment_initialization")
    live.emit(event="episode_start", stage="executing", episode=0, max_steps=900, steps=0, predictions=0)
    live.emit(stage="predicting", steps=16, predictions=1, last_prediction_ms=123.5)
    live.emit(event="episode_end", outcome="behavioral_failure", steps=900)
    events = read_events(path)
    assert len(events) == 4
    for event in events:
        assert event["schema"] == "psm_eval_live_progress_v1"
        assert event["evaluation_run_digest"] == "d" * 64
        assert event["evaluation_attempt_id"] == "CloseFridge/attempt_0001"
        assert event["task"] == "CloseFridge" and event["episodes"] == 2
    assert events[0]["episode"] is None and events[0]["last_prediction_ms"] is None
    assert events[2]["steps"] == 16 and events[2]["last_prediction_ms"] == 123.5
    assert events[3]["outcome"] == "behavioral_failure"
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


def test_disabled_writer_never_opens_a_file(monkeypatch):
    monkeypatch.setattr(Path, "open", Mock(side_effect=AssertionError("disabled observer must not open")))
    live = writer(None)
    with live.observe_errors():
        live.emit(stage="executing")


@pytest.mark.parametrize("failure", ["open", "write", "flush", "close"])
def test_writer_io_failures_never_replace_original_exception(tmp_path, monkeypatch, failure):
    class BrokenHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            if failure == "close":
                raise OSError("close failed")

        def write(self, _):
            if failure == "write":
                raise OSError("write failed")

        def flush(self):
            if failure == "flush":
                raise OSError("flush failed")

    monkeypatch.setattr(
        Path,
        "open",
        Mock(side_effect=OSError("open failed")) if failure == "open" else Mock(return_value=BrokenHandle()),
    )
    live = writer(tmp_path / "live.jsonl")
    live.emit(stage="executing")
    original = RuntimeError("original policy failure")
    with pytest.raises(RuntimeError) as captured, live.observe_errors():
        raise original
    assert captured.value is original


def test_serialization_failure_is_observational_only(tmp_path):
    path = tmp_path / "live.jsonl"
    live = writer(path)
    live.emit(stage="predicting", last_prediction_ms=float("nan"))
    live.emit(stage="executing", last_prediction_ms=1.0)
    assert read_events(path)[0]["last_prediction_ms"] == 1.0


def rollout_case(monkeypatch, tmp_path, *, failure=None, callback=None, instruction="synthetic CPU task"):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    calls, actions, witnesses = [], [], []
    original = RuntimeError(f"original {failure}") if failure is not None else None
    clock = iter(float(index) / 10 for index in range(100))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(module, "USE_BASE_ACTION", True)
    monkeypatch.setattr(module, "BASE_ENCODING", "raw")
    monkeypatch.setattr(module, "compose", lambda obs: np.full((16, 32, 3), obs["step"], np.uint8))
    monkeypatch.setattr(module, "build_state_token", lambda _: [0.0] * 15)
    monkeypatch.setattr(module, "check_success", lambda _: False)

    class Env:
        step_count = 0

        def reset(self):
            calls.append(("reset",))
            if failure == "env_reset":
                raise original
            return {"step": 0}

        def get_ep_meta(self):
            return {"lang": instruction}

        def step(self, action):
            calls.append(("step", action.tolist()))
            if failure == "step":
                raise original
            self.step_count += 1
            return {"step": self.step_count}, 0.0, False, {}

    class Client:
        def begin(self, *args, **kwargs):
            calls.append(("begin",))

        def payload(self, slot):
            calls.append(("payload", slot))
            return {"completed": len(actions)}

        def acknowledge(self, *args):
            calls.append(("ack",))

        def record_completed(self, slot, frame, action):
            calls.append(("completed", slot, action.tolist()))

        def end(self, slot):
            calls.append(("end", slot))
            return "session"

    predictions = 0
    raw15 = [0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1]

    def predict(*args, **kwargs):
        nonlocal predictions
        calls.append(("predict", kwargs["local_memory"]))
        predictions += 1
        if failure == "predict" and predictions == 2:
            raise original
        return {"action": [raw15, raw15], "local_memory": {}}

    def save_video(*_):
        calls.append(("save_video",))
        if failure == "media":
            raise original

    def reset_local(*_):
        calls.append(("reset_local",))
        if failure == "local_cleanup":
            raise original

    monkeypatch.setattr(module, "predict", predict)
    monkeypatch.setattr(module, "save_partial_video", save_video)
    monkeypatch.setattr(module, "reset_local_memory", reset_local)
    result, error = None, None
    try:
        result = module.run_policy(
            Env(),
            server_url="synthetic",
            image_size=256,
            action_horizon=2,
            max_steps=5,
            latch=1,
            timeout=1.0,
            use_state=True,
            save_video=tmp_path / "rollout00.mp4",
            local_memory_client=Client(),
            on_action=actions.append,
            on_cleanup=witnesses.append,
            on_progress=callback,
        )
    except Exception as exc:
        error = exc
    if failure is not None:
        assert error is original
    return result, error, calls, actions, witnesses


@pytest.mark.parametrize("failure", [None, "env_reset", "predict", "step", "media", "local_cleanup"])
def test_progress_on_off_and_broken_callback_preserve_actions_order_outcome_and_original_failure(
    tmp_path, monkeypatch, failure
):
    events = []
    baseline = rollout_case(monkeypatch, tmp_path, failure=failure)
    observed = rollout_case(monkeypatch, tmp_path, failure=failure, callback=events.append)
    broken = rollout_case(monkeypatch, tmp_path, failure=failure, callback=Mock(side_effect=OSError("observer failed")))
    assert baseline[0] == observed[0] == broken[0]
    assert baseline[2:] == observed[2:] == broken[2:]
    assert [None if case[1] is None else str(case[1]) for case in (baseline, observed, broken)] == [
        None if failure is None else f"original {failure}"
    ] * 3
    assert events[0]["stage"] == "environment_initialization"
    if failure is None:
        assert [row["stage"] for row in events].count("predicting") == 3
        latency_rows = [row for row in events if row["stage"] == "executing" and row["last_prediction_ms"] is not None]
        assert [row["predictions"] for row in latency_rows] == [1, 2, 3]
        assert [row["steps"] for row in latency_rows] == [0, 2, 4]
        assert all(row["last_prediction_ms"] == pytest.approx(100.0) for row in latency_rows)
        assert events[-1]["stage"] == "local_cleanup" and events[-1]["steps"] == 5
        # Stages/predictions, never one new observation per action.
        assert sum(row["stage"] == "executing" for row in events) == 4
    else:
        assert any(row["event"] == "error" and f"original {failure}" in row["error"] for row in events)


@pytest.mark.parametrize("instruction", ["真实任务指令\nkeep raw  ", ""])
def test_failed_actual_runner_retains_exact_instruction_in_observational_jsonl(tmp_path, monkeypatch, instruction):
    path = tmp_path / "live.jsonl"
    live = writer(path)
    result = rollout_case(
        monkeypatch,
        tmp_path,
        failure="predict",
        instruction=instruction,
        callback=lambda fields: live.emit(episode=0, **fields),
    )
    assert result[0] is None and str(result[1]) == "original predict"
    events = read_events(path)
    assert any(event["instruction"] == instruction for event in events)
    assert events[-1]["instruction"] == instruction
    assert events[-1]["task"] == "CloseFridge"


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_main_live_stream_is_separate_from_exact_2n_commit_progress(tmp_path, monkeypatch, enabled):
    import requests

    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    dataset = tmp_path / "dataset/extras"
    dataset.mkdir(parents=True)
    (dataset / "dataset_meta.json").write_text(json.dumps({"env_args": {"env_name": "CloseFridge"}}))
    out = tmp_path / "CloseFridge/attempt_0001"
    path = out / "live_progress.jsonl"
    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: SimpleNamespace(close=Mock()))
    monkeypatch.setattr(module, "official_horizon", lambda _: 48)
    response = SimpleNamespace(ok=True, json=lambda: {"local_memory": {"mode": "off"}})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    def runner(env, *, on_prediction, on_action, on_progress, **kwargs):
        if on_progress is not None:
            on_progress({"stage": "predicting", "steps": 0, "predictions": 0})
            on_progress({"stage": "executing", "steps": 0, "predictions": 1, "last_prediction_ms": 21.5})
        on_prediction({"action": [[0] * 15]}, 21.5)
        on_action({"source_step": 0})
        return True, 1, "task"

    monkeypatch.setattr(module, "run_policy", runner)
    argv = [
        "closed_loop_eval.py",
        "--dataset-dir",
        str(dataset.parent),
        "--output-dir",
        str(out),
        "--num-test-episodes",
        "2",
        "--evaluation-run-digest",
        "d" * 64,
        "--evaluation-attempt-id",
        "CloseFridge/attempt_0001",
    ]
    if enabled:
        argv += ["--live-progress-file", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    commit = read_events(out / "episode_progress.jsonl")
    assert [(row["event"], row["episode"]) for row in commit] == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]
    assert all(row["schema"] == "psm_episode_progress_v1" for row in commit)
    assert len(json.loads((out / "results.json").read_text())) == 2
    assert path.exists() is enabled
    if enabled:
        events = read_events(path)
        assert [row["episode"] for row in events if row["event"] == "episode_start"] == [0, 1]
        assert [row["outcome"] for row in events if row["event"] == "episode_end"] == ["success", "success"]
        assert any(row["stage"] == "environment_close" for row in events)
        assert all(row["evaluation_attempt_id"] == "CloseFridge/attempt_0001" for row in events)
        assert all(row["evaluation_run_digest"] == "d" * 64 for row in events)
        assert all(row["task"] == "CloseFridge" and row["max_steps"] in (None, 48) for row in events)


def test_main_environment_initialization_failure_is_observed_and_preserved(tmp_path, monkeypatch):
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    dataset = tmp_path / "dataset/extras"
    dataset.mkdir(parents=True)
    (dataset / "dataset_meta.json").write_text(json.dumps({"env_args": {"env_name": "CloseFridge"}}))
    out = tmp_path / "CloseFridge/attempt_0001"
    original = RuntimeError("environment initialization failed")
    monkeypatch.setattr(module, "official_horizon", lambda _: 48)
    monkeypatch.setattr(module, "make_env", Mock(side_effect=original))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "closed_loop_eval.py",
            "--dataset-dir",
            str(dataset.parent),
            "--output-dir",
            str(out),
            "--live-progress-file",
            str(out / "live.jsonl"),
        ],
    )
    with pytest.raises(RuntimeError) as captured:
        module.main()
    assert captured.value is original
    assert read_events(out / "live.jsonl")[-1]["event"] == "error"
    assert not (out / "episode_progress.jsonl").exists()

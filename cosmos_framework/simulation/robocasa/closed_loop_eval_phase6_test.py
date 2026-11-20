from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


class _Client(RoboCasaLocalMemoryClient):
    def __init__(self) -> None:
        super().__init__()
        self.completed = []

    def record_completed(self, slot, composite_image, executed_action15) -> None:
        self.completed.append((composite_image.copy(), np.asarray(executed_action15).copy()))
        super().record_completed(slot, composite_image, executed_action15)


class _Env:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.steps = []

    def reset(self):
        return {"step": 0}

    def get_ep_meta(self):
        return {"lang": "close fridge"}

    def step(self, action):
        if self.fail:
            raise RuntimeError("env step failed")
        self.steps.append(action)
        return {"step": len(self.steps)}, 0.0, True, {}


def _module(monkeypatch):
    monkeypatch.setitem(sys.modules, "robosuite", ModuleType("robosuite"))
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    module = importlib.import_module("cosmos_framework.simulation.robocasa.closed_loop_eval")
    monkeypatch.setattr(module, "compose", lambda obs: np.full((256, 512, 3), obs["step"], dtype=np.uint8))
    monkeypatch.setattr(module, "check_success", lambda env: True)
    monkeypatch.setattr(module, "reset_local_memory", lambda *args: None)
    monkeypatch.setattr(module, "USE_BASE_ACTION", True)
    monkeypatch.setattr(module, "BASE_ENCODING", "raw")
    return module


def _action() -> list[float]:
    return [2.0, -2.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.3]


def _run(module, env, client):
    return module.run_policy(
        env,
        server_url="http://unused",
        image_size=256,
        action_horizon=2,
        max_steps=2,
        latch=1,
        timeout=1.0,
        local_memory_client=client,
    )


def _prediction(actions):
    def respond(*args, **kwargs):
        payload = kwargs["local_memory"]
        return {"action": actions, "video": [], "local_memory": payload}

    return respond


def test_successful_done_publishes_pre_action_canonical_only(monkeypatch) -> None:
    module = _module(monkeypatch)
    monkeypatch.setattr(module, "predict", _prediction([_action(), _action()]))
    env, client = _Env(), _Client()
    success, done_steps, _ = _run(module, env, client)
    assert success and done_steps == 1
    assert len(env.steps) == len(client.completed) == 1
    np.testing.assert_array_equal(client.completed[0][0], np.zeros((256, 512, 3), dtype=np.uint8))
    np.testing.assert_allclose(module.decode_15d_to_env12(client.completed[0][1], False), env.steps[0], atol=2e-6)


@pytest.mark.parametrize("failure", ("decoder", "env"))
def test_failure_never_publishes_completed_evidence(monkeypatch, failure: str) -> None:
    module = _module(monkeypatch)
    actions = [[0.0] * 12] if failure == "decoder" else [_action()]
    monkeypatch.setattr(module, "predict", _prediction(actions))
    env, client = _Env(fail=failure == "env"), _Client()
    with pytest.raises((ValueError, RuntimeError)):
        _run(module, env, client)
    assert client.completed == []


@pytest.mark.parametrize("failure", (None, "env", "prediction"))
def test_g2_replans_use_current_state_and_record_only_completed_steps(monkeypatch, failure):
    module = _module(monkeypatch)
    monkeypatch.setattr(module, "CAMERA_SET", "left_wrist")
    monkeypatch.setattr(module, "check_success", lambda env: len(env.steps) == 18)

    class Env(_Env):
        def get_ep_meta(self):
            raise AssertionError("G2 must use the exact admitted metadata without querying again")

        def step(self, action):
            if failure == "env" and len(self.steps) == 16:
                raise RuntimeError("G2 test simulator error")
            obs, reward, _, info = super().step(action)
            return obs, reward, False, info

    class Runtime:
        def __init__(self):
            self.completed = []
            self.closed = False

        def reset(self, env, observation, **identity):
            assert observation == {"step": 0}
            assert identity == {
                "episode_id": client.payload(0)["episode_id"],
                "task": "CloseFridge",
                "instruction": "close fridge",
                "frame": 0,
                "episode_index": 7,
                "ep_meta": ep_meta,
            }
            assert identity["ep_meta"] is ep_meta

        def decision(self, observation, *, frame, composite, local_memory):
            assert observation["step"] == frame == local_memory["consumer_step"] == len(self.completed)
            assert composite[0, 0, 0] == frame
            assert [row["source_step"] for row in local_memory["evidence"]] == list(range(frame))
            return {
                "state": [2.0, -1.0, 0.3, 1.0, 0.0] + [float(frame)] * 10,
                "prompt": f"close fridge with current facts at {frame}",
                "g2": {"identity": {"episode_id": local_memory["episode_id"], "decision_frame": frame}},
            }

        def record_completed(self, env, observation, *, source_frame, canonical_raw15, submitted_env12):
            assert observation["step"] == source_frame + 1 == len(env.steps)
            np.testing.assert_array_equal(canonical_raw15, client.completed[source_frame][1])
            np.testing.assert_array_equal(submitted_env12, env.steps[source_frame])
            self.completed.append(source_frame)

        def close(self):
            self.closed = True

    frames = []
    prediction_witnesses = []

    def predict(server_url, image, prompt, image_size, timeout, *, state, local_memory, g2):
        frame = int(image[0, 0, 0])
        frames.append(frame)
        assert state[:5] == [2.0, -1.0, 0.3, 1.0, 0.0]
        assert state[-1] == frame
        assert prompt == f"close fridge with current facts at {frame}"
        assert g2["identity"] == {"episode_id": local_memory["episode_id"], "decision_frame": frame}
        actions = [_action()] * 16
        if frame == 16 and failure == "prediction":
            actions[-1] = _action()[:14]
        return {"action": actions, "local_memory": local_memory}

    monkeypatch.setattr(module, "predict", predict)
    env, client, runtime = Env(), _Client(), Runtime()
    ep_meta = {"lang": "close fridge", "fixture_refs": {"fridge": "example"}}

    def run():
        return module.run_policy(
            env,
            server_url="http://unused",
            image_size=256,
            action_horizon=16,
            max_steps=18,
            latch=1,
            timeout=1.0,
            use_state=True,
            local_memory_client=client,
            initial_observation={"step": 0},
            language_instruction=ep_meta["lang"],
            on_prediction=lambda result, latency, *, g2_request: prediction_witnesses.append(g2_request),
            g2_runtime=runtime,
            g2_task="CloseFridge",
            g2_episode_index=7,
            g2_ep_meta=ep_meta,
        )

    if failure is None:
        assert run() == (True, 18, "close fridge")
    else:
        with pytest.raises(RuntimeError if failure == "env" else ValueError):
            run()
    assert frames == [0, 16]
    assert [row["g2"]["identity"]["decision_frame"] for row in prediction_witnesses] == (
        [0] if failure == "prediction" else [0, 16]
    )
    assert runtime.completed == list(range(18 if failure is None else 16))
    assert len(client.completed) == len(runtime.completed)
    assert runtime.closed

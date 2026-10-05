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

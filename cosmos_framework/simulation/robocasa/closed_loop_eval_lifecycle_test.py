"""Run the original rollout body with CPU boundary doubles, without simulator imports."""

from __future__ import annotations

import ast
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


def runner(*, failure=None, reset_failure=None, media_failure=None, env_failure=None):
    path = Path(__file__).with_name("closed_loop_eval.py")
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run_policy"
    )
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    client = Mock()
    client.end.return_value = "abandoned-session"
    cleanup = Mock(side_effect=reset_failure)
    video = Mock(side_effect=media_failure)
    env = Mock()
    env.reset.side_effect = env_failure
    env.reset.return_value = {}
    predict = Mock(side_effect=failure)
    predict.return_value = {"action": [[0.0] * 15], "local_memory": {}}
    scope = {
        "np": np,
        "time": time,
        "compose": lambda _: np.zeros((2, 4, 3), np.uint8),
        "build_state_token": lambda _: [0] * 15,
        "predict": predict,
        "save_partial_video": video,
        "reset_local_memory": cleanup,
        "USE_BASE_ACTION": True,
        "BASE_ENCODING": "raw",
        "_ACTION_DIM_BY_ENCODING": {"raw": 15},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["run_policy"], env, client, cleanup, video


@pytest.mark.parametrize("failure", [RuntimeError("HTTP 400"), TimeoutError("request timeout")])
def test_rollout_failure_remains_primary_when_reset_also_fails(failure):
    run, env, client, reset, _ = runner(failure=failure, reset_failure=RuntimeError("reset failed"))
    witnesses = []
    with pytest.raises(type(failure), match=str(failure)):
        run(
            env,
            server_url="synthetic",
            image_size=256,
            action_horizon=16,
            max_steps=1,
            latch=1,
            timeout=0.1,
            local_memory_client=client,
            on_cleanup=witnesses.append,
        )
    client.end.assert_called_once_with(0)
    reset.assert_called_once()
    assert "reset failed" in witnesses[0]["cleanup_error"]
    assert str(failure) in witnesses[0]["rollout_error"]


@pytest.mark.parametrize("kind", ["normal", "media", "reset", "both", "env_reset"])
def test_cleanup_all_boundaries_and_independent_witness(kind):
    run, env, client, reset, video = runner(
        reset_failure=RuntimeError("reset failed") if kind in ("reset", "both") else None,
        media_failure=RuntimeError("encoder failed") if kind in ("media", "both") else None,
        env_failure=RuntimeError("env reset failed") if kind == "env_reset" else None,
    )
    witnesses = []
    kwargs = dict(
        server_url="synthetic",
        image_size=256,
        action_horizon=16,
        max_steps=0,
        latch=1,
        timeout=0.1,
        local_memory_client=client,
        on_cleanup=witnesses.append,
    )
    if kind == "normal":
        assert run(env, **kwargs)[0] is False
    else:
        primary = {
            "media": "encoder failed",
            "reset": "reset failed",
            "both": "encoder failed",
            "env_reset": "env reset failed",
        }[kind]
        with pytest.raises(RuntimeError, match=primary):
            run(env, **kwargs)
    reset.assert_called_once()
    client.end.assert_called_once_with(0)
    assert len(witnesses) == 1
    assert witnesses[0]["reset_ok"] == (kind not in ("reset", "both"))
    if kind == "both":
        assert witnesses[0]["media_error"] and witnesses[0]["cleanup_error"]


def test_cleanup_witness_callback_cannot_override_original_error():
    run, env, client, _, _ = runner(failure=RuntimeError("original"))
    with pytest.raises(RuntimeError, match="original"):
        run(
            env,
            server_url="synthetic",
            image_size=256,
            action_horizon=16,
            max_steps=1,
            latch=1,
            timeout=0.1,
            local_memory_client=client,
            on_cleanup=Mock(side_effect=OSError("disk full")),
        )

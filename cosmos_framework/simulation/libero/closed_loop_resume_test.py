"""Episode-boundary resume contract for LIBERO closed-loop evaluation."""

from __future__ import annotations

import json

import pytest

from .closed_loop_eval import (
    _INTENTIONAL_SKIP_ERROR,
    _load_resume_task_results,
    _write_task_partial_summary,
)


def _result(episode: int, *, success: bool = False, error=None):
    return {
        "episode": episode,
        "success": success,
        "steps": 42,
        "error": error,
        "elapsed_s": 1.0,
    }


def test_resume_reuses_terminal_results_and_retries_runtime_failures(tmp_path):
    results = [
        _result(0, success=True),
        _result(1, success=False),
        _result(2, error="server error: timeout"),
        {
            "episode": 3,
            "success": False,
            "steps": 0,
            "error": _INTENTIONAL_SKIP_ERROR,
            "elapsed_s": 0.0,
        },
        None,
    ]
    _write_task_partial_summary(
        tmp_path,
        task_id=7,
        task_description="task seven",
        num_trials=5,
        results=results,
    )

    reusable = _load_resume_task_results(
        tmp_path,
        task_id=7,
        task_description="task seven",
        num_trials=5,
    )

    assert sorted(reusable) == [0, 1, 3]
    assert reusable[0]["success"] is True
    assert reusable[1]["success"] is False
    assert reusable[3]["error"] == _INTENTIONAL_SKIP_ERROR


def test_resume_requires_matching_task_and_trial_count(tmp_path):
    _write_task_partial_summary(
        tmp_path,
        task_id=1,
        task_description="original task",
        num_trials=10,
        results=[_result(0)] + [None] * 9,
    )

    with pytest.raises(ValueError, match="task description mismatch"):
        _load_resume_task_results(
            tmp_path,
            task_id=1,
            task_description="different task",
            num_trials=10,
        )
    with pytest.raises(ValueError, match="trial-count mismatch"):
        _load_resume_task_results(
            tmp_path,
            task_id=1,
            task_description="original task",
            num_trials=4,
        )


def test_resume_does_not_trust_actions_or_predictions_without_partial(tmp_path):
    actions = tmp_path / "actions" / "task_009"
    predictions = tmp_path / "predictions" / "task_009"
    actions.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (actions / "episode_000.json").write_text(json.dumps([[0.0] * 7]), encoding="utf-8")
    (predictions / "episode_000.json").write_text(json.dumps([{"step": 10}]), encoding="utf-8")

    assert (
        _load_resume_task_results(
            tmp_path,
            task_id=9,
            task_description="task nine",
            num_trials=10,
        )
        == {}
    )

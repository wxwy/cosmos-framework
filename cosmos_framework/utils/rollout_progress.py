"""Best-effort live observations, separate from authoritative evaluation evidence."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path


class LiveProgressWriter:
    """Append scalar stage observations without changing rollout/error semantics."""

    def __init__(self, path, *, run_digest, attempt_id, task, episodes):
        self.path = Path(path) if path is not None else None
        self.identity = {
            "schema": "psm_eval_live_progress_v1",
            "evaluation_run_digest": run_digest,
            "evaluation_attempt_id": attempt_id,
            "task": task,
            "episodes": episodes,
        }
        self.state = dict.fromkeys(
            (
                "stage",
                "episode",
                "max_steps",
                "steps",
                "predictions",
                "last_prediction_ms",
                "outcome",
                "error",
                "instruction",
            )
        )

    def emit(self, event="stage", **fields):
        if self.path is None:
            return
        try:
            if event not in {"stage", "episode_start", "episode_end", "error"} or set(fields) - set(self.state):
                return
            self.state.update(fields)
            record = {**self.identity, "event": event, **self.state}
            # This is observational only: never create or claim a run namespace.
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
        except Exception:
            pass

    @contextmanager
    def observe_errors(self):
        try:
            yield
        except BaseException as error:
            try:
                self.emit(event="error", error=f"{type(error).__name__}: {error}"[:2048])
            except Exception:
                pass
            raise

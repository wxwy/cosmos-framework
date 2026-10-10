# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Write per-action evidence and partial videos without changing rollout control."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False
    ) as handle:
        pending = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def save_partial_video(frames, path, fps: int) -> None:
    if path is None or not frames:
        return
    import imageio

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite rollout video")
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as handle:
        pending = Path(handle.name)
    try:
        imageio.mimwrite(pending, frames, fps=fps, macro_block_size=None)
        # Atomic publication that never replaces another attempt's artifact.
        os.link(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def run_recorded_episode(runner, env, *, output_dir, episode: int, run_digest: str | None, **kwargs) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / f"rollout{episode:02d}_actions.jsonl"
    predictions_path = output_dir / f"rollout{episode:02d}_predictions.jsonl"
    # Never silently overwrite an interrupted/old episode. A new attempt needs
    # a new output namespace; same-run task resume skips only complete results.
    with actions_path.open("x", encoding="utf-8", buffering=1) as actions:
        try:
            predictions = predictions_path.open("x", encoding="utf-8", buffering=1)
        except BaseException:
            # Preserve the newly created evidence file too: no cleanup of history.
            raise
        count = 0
        rows = []

        def record_action(row):
            nonlocal count
            record = dict(row, evaluation_run_digest=run_digest, episode=episode)
            actions.write(json.dumps(record, allow_nan=False) + "\n")
            actions.flush()
            count += 1

        def record_prediction(result, latency_ms):
            row = {
                "latency_ms": float(latency_ms),
                "action_chunk": result.get("action"),
                "local_memory": result.get("local_memory"),
                "step_before_query": count,
                "evaluation_run_digest": run_digest,
                "episode": episode,
            }
            predictions.write(json.dumps(row, allow_nan=False) + "\n")
            predictions.flush()
            rows.append(row)

        success, steps, prompt, error, interrupted = False, 0, None, None, None
        try:
            success, steps, prompt = runner(env, on_prediction=record_prediction, on_action=record_action, **kwargs)
        except BaseException as exc:
            steps = count
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            predictions.close()
        local_rows = [row["local_memory"] for row in rows if isinstance(row.get("local_memory"), dict)]
        latencies = [row["latency_ms"] for row in rows]
        result = {
            "ep": episode,
            "policy": bool(success) if error is None else False,
            "steps": steps,
            "prompt": prompt,
            "replans": len(rows),
            "error": error,
            "evaluation_run_digest": run_digest,
            "outcome": "runtime_error" if error else ("success" if success else "behavioral_failure"),
            "prediction_latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
            "post_cold_prefix_all": all(bool(row.get("prefix_present")) for row in local_rows[1:])
            if len(local_rows) > 1
            else None,
            "local_memory_predictions": rows,
            "actions_file": actions_path.name,
            "predictions_file": predictions_path.name,
            "completed_action_records": count,
        }
        atomic_json(output_dir / f"rollout{episode:02d}_episode.json", result)
        if interrupted is not None:
            raise interrupted
        return result

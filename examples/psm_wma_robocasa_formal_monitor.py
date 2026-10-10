# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Non-invasive formal30k telemetry journal and read-only checkpoint monitor.

The journal is an opt-in rank0 callback. Plotting and DCP inspection run only
in a separate CPU process, NEVER in the optimizer/checkpointer callback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from cosmos_framework.utils.observed_exposure import observed_task_exposure

TRAIN_PREFIX = "[CorrectedV3][train] "
CHECKPOINT_PREFIX = "[CorrectedV3][checkpoint] "


def _decimal(value: object, places: int = 4) -> str:
    return f"{value:.{places}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def _scientific(value: object) -> str:
    return f"{value:.3e}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def format_progress(record: dict[str, Any], *, max_iter: int) -> str:
    """Readable rank0-only line; retains explicit rank-local gradient semantics."""
    stamp = record.get("ts_local") or datetime.now().astimezone().isoformat(timespec="seconds")
    iteration = record.get("iteration")
    index = f"{iteration}/{max_iter}" if max_iter else str(iteration)
    return (
        f"[{stamp}] iter={index} epoch={record.get('epoch', 'n/a')} "
        f"LOSS outer={_decimal(record.get('outer_loss'))} action={_decimal(record.get('action_loss'))} "
        f"vision={_decimal(record.get('vision_loss'))} "
        f"GRAD[rank0-shard] total={_decimal(record.get('total_grad_norm_rank_local'))} "
        f"missing={record.get('missing_grad_tensors_rank_local', 'n/a')} "
        f"nonfinite={record.get('nonfinite_grad_tensors_rank_local', 'n/a')} "
        f"LOCAL slots={record.get('active_slots', 'n/a')} "
        f"inner={_decimal(record.get('inner_loss_mean'))} "
        f"fast_update={_decimal(record.get('fast_update_norm_mean'))} "
        f"LR={_scientific(record.get('lr_max'))} "
        f"data={_decimal(record.get('data_prepare_ms'), 0)}ms "
        f"wall={_decimal(record.get('step_wall_s'), 1)}s "
        f"peak={_decimal(record.get('peak_reserved_gib'), 2)}GiB "
        f"{record.get('status', 'unknown')}"
    )


class TelemetryJournal:
    """Append-only rank-local text + JSONL, independent of terminal availability."""

    def __init__(self, job_dir: Path, *, rank: int = 0, emit: Callable[[str], None] = print) -> None:
        self.rank = rank
        self.emit = emit
        self.directory = Path(job_dir) / "monitor"
        self.directory.mkdir(parents=True, exist_ok=True)
        metrics = self.directory / f"train_rank{rank}.jsonl"
        # Never glue a resumed record onto a partial write or silently truncate evidence.
        if metrics.exists() and metrics.stat().st_size:
            with metrics.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError(f"unterminated journal {metrics}; preserve and reconcile before resume")
            read_train_records(metrics)
        self._raw = (self.directory / f"rank{rank}.log").open("a", encoding="utf-8", buffering=1)
        try:
            self._metrics = metrics.open("a", encoding="utf-8", buffering=1)
        except BaseException:
            self._raw.close()
            raise
        self._write_failed = False

    def _console(self, line: str) -> None:
        try:
            self.emit(line)
        except Exception:
            # A detached/broken stdout must not prevent writing the durable journal.
            pass

    def _write_error(self, exc: OSError) -> None:
        if not self._write_failed:
            self._write_failed = True
            self._console(f"[CorrectedV3][logging_error] {type(exc).__name__}: {exc}")

    def __call__(self, line: str) -> None:
        # Write the machine-readable record FIRST. The raw and console sinks are independent.
        if line.startswith(TRAIN_PREFIX):
            try:
                self._metrics.write(line[len(TRAIN_PREFIX) :].rstrip("\n") + "\n")
            except OSError as exc:
                self._write_error(exc)
        try:
            self._raw.write(line.rstrip("\n") + "\n")
        except OSError as exc:
            self._write_error(exc)
        self._console(line)

    def close(self) -> None:
        for handle in (self._metrics, self._raw):
            try:
                handle.close()
            except OSError as exc:
                self._write_error(exc)


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite JSON number: {value}")
    return result


def read_train_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Derive a replay-aware view without rewriting append-only historical evidence.

    A rewind from step N to M removes the previous attempt's M..N from this
    derived view, until the new attempt commits those steps again.
    """
    if not path.exists():
        return [], 0
    by_iteration: dict[int, dict[str, Any]] = {}
    superseded = 0
    previous_iteration = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            # Newline is the journal record boundary; do not publish an in-flight final row.
            if not line.endswith("\n"):
                break
            if not line.strip():
                continue
            try:
                entry = json.loads(line, parse_float=_finite_float, parse_constant=_finite_float)
            except ValueError as exc:
                raise ValueError(f"train JSONL invalid line {number}: {exc}") from exc
            iteration = entry.get("iteration") if isinstance(entry, dict) else None
            if type(iteration) is not int or iteration <= 0 or entry.get("status") != "optimizer_committed":
                raise ValueError(f"train JSONL invalid committed iteration line {number}")
            if iteration <= previous_iteration:
                stale = [key for key in by_iteration if key >= iteration]
                superseded += len(stale)
                for key in stale:
                    del by_iteration[key]
            by_iteration[iteration] = entry
            previous_iteration = iteration
    return [by_iteration[key] for key in sorted(by_iteration)], superseded


def _nonempty_file(path: Path, boundary: Path) -> bool:
    try:
        return path.is_file() and path.resolve().is_relative_to(boundary.resolve()) and path.stat().st_size > 0
    except OSError:
        return False


def audit_latest_checkpoint(job_dir: Path, *, world_size: int) -> dict[str, Any]:
    """Check nonempty files/rank coverage only; NEVER deserialize or certify resume parity."""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    folder = Path(job_dir) / "checkpoints"
    latest = folder / "latest_checkpoint.txt"
    if not latest.is_file():
        return {"status": "MISSING", "reason": "latest_checkpoint.txt absent"}
    name = latest.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"iter_[0-9]+", name) is None or int(name[5:]) <= 0:
        return {"status": "INVALID", "reason": "invalid checkpoint pointer"}
    checkpoint = folder / name
    if not checkpoint.is_dir() or not checkpoint.resolve().is_relative_to(folder.resolve()):
        return {"status": "INVALID", "reason": "checkpoint is not a safe directory"}
    parts: dict[str, dict[str, Any]] = {}
    complete = True
    expected_ranks = set(range(world_size))
    for component in ("model", "optim", "scheduler", "trainer"):
        subdir = checkpoint / component
        metadata = _nonempty_file(subdir / ".metadata", checkpoint)
        shards = sorted(subdir.glob("*.distcp")) if subdir.is_dir() else []
        ranks = set()
        files_valid = True
        for shard in shards:
            match = re.fullmatch(r"__([0-9]+)_[0-9]+\.distcp", shard.name)
            if match is None or not _nonempty_file(shard, checkpoint):
                files_valid = False
            else:
                ranks.add(int(match.group(1)))
        valid = metadata and files_valid and ranks == expected_ranks
        complete = complete and valid
        parts[component] = {
            "metadata": metadata,
            "shard_count": len(shards),
            "present_ranks": sorted(ranks),
            "valid": valid,
        }
    states = checkpoint / "dataloader"
    present = [rank for rank in range(world_size) if _nonempty_file(states / f"rank_{rank}.pkl", checkpoint)]
    complete = complete and len(present) == world_size
    try:
        pointer_stable = latest.read_text(encoding="utf-8").strip() == name
    except OSError:
        pointer_stable = False
    return {
        "status": ("COMPLETE" if complete else "INCOMPLETE") if pointer_stable else "IN_PROGRESS",
        "iteration": int(name[5:]),
        "checkpoint": str(checkpoint),
        "components": parts,
        "dataloader_present_ranks": present,
        "world_size": world_size,
        "audit_scope": "filesystem_presence_only_not_strict_resume_parity",
        "layout_scope": "standard_per_rank_distcp_and_rank_pkl",
    }


def _series(rows: list[dict[str, Any]], name: str) -> tuple[list[int], list[float]]:
    pairs = [
        (item["iteration"], float(value))
        for item in rows
        if isinstance((value := item.get(name)), (int, float)) and math.isfinite(value)
    ]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _ema(values: list[float], weight: float = 0.98) -> list[float]:
    smoothed: list[float] = []
    for value in values:
        smoothed.append(value if not smoothed else weight * smoothed[-1] + (1 - weight) * value)
    return smoothed


def render_curves(rows: list[dict[str, Any]], destination: Path) -> list[str]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    figures = {
        "loss": ("outer_loss", "action_loss", "vision_loss"),
        "gradients": (
            "total_grad_norm_rank_local",
            "generation_grad_norm_rank_local",
            "action_grad_norm_rank_local",
            "local_grad_norm_rank_local",
        ),
        "local_memory": ("inner_loss_mean", "fast_state_norm_mean", "fast_update_norm_mean"),
        "learning_rate": ("lr_max", "lr_min"),
        "latency": ("step_wall_s", "data_prepare_ms", "batch_collate_ms", "batch_transfer_ms"),
        "gpu_memory": ("peak_allocated_gib", "peak_reserved_gib"),
    }
    outputs = []
    for category, metrics in figures.items():
        for metric in metrics:
            iterations, values = _series(rows, metric)
            if not iterations:
                continue
            fig = plt.figure(figsize=(11, 4))
            axis = fig.add_subplot(111)
            axis.plot(iterations, values, alpha=0.35, linewidth=0.8, label="raw")
            axis.plot(iterations, _ema(values), linewidth=1.4, label="EMA(0.98)")
            axis.set_ylabel(metric)
            axis.set_xlabel("optimizer iteration")
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
            fig.tight_layout()
            final = destination / f"{category}_{metric}.png"
            pending = destination / f".{category}_{metric}.{os.getpid()}.tmp.png"
            try:
                fig.savefig(pending, dpi=120)
                os.replace(pending, final)
                outputs.append(str(final))
            finally:
                plt.close(fig)
    return outputs


def _write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, destination)


def update_monitor(job_dir: Path, *, world_size: int, max_iter: int, plot: bool) -> dict[str, Any]:
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    job_dir = Path(job_dir)
    records, superseded = read_train_records(job_dir / "monitor/train_rank0.jsonl")
    checkpoint = audit_latest_checkpoint(job_dir, world_size=world_size)
    recent = [
        float(row["step_wall_s"])
        for row in records[-50:]
        if type(row.get("step_wall_s")) in (int, float) and math.isfinite(row["step_wall_s"]) and row["step_wall_s"] > 0
    ]
    latest = records[-1] if records else {}
    avg_step = statistics.median(recent) if recent else None
    estimated_remaining_h = (
        max(0, max_iter - latest["iteration"]) * avg_step / 3600
        if avg_step is not None and isinstance(latest.get("iteration"), int)
        else None
    )
    state = {
        "source": "rank0_only_never_global_reduced",
        "last_iteration": latest.get("iteration"),
        "record_count": len(records),
        **observed_task_exposure(records),
        "latest_memory_inventory": latest.get("memory_inventory"),
        "episode_cache_rank_local": latest.get("episode_cache_rank_local"),
        "replayed_iteration_records": superseded,
        "outer_loss": latest.get("outer_loss"),
        "action_loss": latest.get("action_loss"),
        "vision_loss": latest.get("vision_loss"),
        "missing_grad_tensors_rank_local": latest.get("missing_grad_tensors_rank_local"),
        "nonfinite_grad_tensors_rank_local": latest.get("nonfinite_grad_tensors_rank_local"),
        "median_step_50_s": avg_step,
        "estimated_remaining_hours_excluding_checkpoint": estimated_remaining_h,
        "checkpoint": checkpoint,
    }
    if plot:
        try:
            state["curves"] = render_curves(records, job_dir / "monitor/curves")
        except Exception as exc:
            state["plot_error"] = f"{type(exc).__name__}: {exc}"
    _write_json_atomic(job_dir / "monitor/summary.json", state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=30000)
    parser.add_argument("--follow", action="store_true", help="CPU-only sidecar; no training hooks")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 2:
        parser.error("poll-seconds must be finite and >=2")
    if args.world_size <= 0 or args.max_iter <= 0:
        parser.error("world-size and max-iter must be positive")
    previous = None
    while True:
        state = update_monitor(args.job_dir, world_size=args.world_size, max_iter=args.max_iter, plot=False)
        step = state["last_iteration"]
        signature = (step // 100 if step is not None else None, state["checkpoint"].get("iteration"))
        if not args.follow or signature != previous:
            state = update_monitor(args.job_dir, world_size=args.world_size, max_iter=args.max_iter, plot=True)
            previous = signature
        print(json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
        if not args.follow:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

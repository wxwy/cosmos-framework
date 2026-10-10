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
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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
    """Append-only rank-local text + JSONL. Failures never undo optimizer commits."""

    def __init__(self, job_dir: Path, *, rank: int = 0, emit: Callable[[str], None] = print) -> None:
        self.rank = rank
        self.emit = emit
        self.directory = Path(job_dir) / "monitor"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._raw = (self.directory / f"rank{rank}.log").open("a", encoding="utf-8", buffering=1)
        self._metrics = (self.directory / f"train_rank{rank}.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._write_failed = False

    def __call__(self, line: str) -> None:
        self.emit(line)
        try:
            self._raw.write(line.rstrip("\n") + "\n")
            if line.startswith(TRAIN_PREFIX):
                self._metrics.write(line[len(TRAIN_PREFIX) :].rstrip("\n") + "\n")
        except OSError as exc:
            if not self._write_failed:
                self._write_failed = True
                self.emit(f"[CorrectedV3][logging_error] {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._metrics.close()
        self._raw.close()


def read_train_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return the latest committed trace per iteration; count superseded replay rows."""
    if not path.exists():
        return [], 0
    by_iteration: dict[int, dict[str, Any]] = {}
    superseded = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            # A concurrent writer may have an incomplete FINAL line. Never
            # tolerate malformed fully-terminated or interior records.
            if number == len(lines) and not line.endswith("\n"):
                break
            raise ValueError(f"train JSONL invalid line {number}: {exc}") from exc
        iteration = entry.get("iteration") if isinstance(entry, dict) else None
        if type(iteration) is not int or iteration <= 0 or entry.get("status") != "optimizer_committed":
            raise ValueError(f"train JSONL invalid committed iteration line {number}")
        if iteration in by_iteration:
            superseded += 1
        by_iteration[iteration] = entry
    return [by_iteration[key] for key in sorted(by_iteration)], superseded


def audit_latest_checkpoint(job_dir: Path, *, world_size: int) -> dict[str, Any]:
    """Read metadata ONLY; no torch.load, state mutation, DCP hash or partial-file delete."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    folder = Path(job_dir) / "checkpoints"
    latest = folder / "latest_checkpoint.txt"
    if not latest.is_file():
        return {"status": "MISSING", "reason": "latest_checkpoint.txt absent"}
    name = latest.read_text(encoding="utf-8").strip()
    if not name.startswith("iter_") or not name[5:].isdigit() or int(name[5:]) <= 0:
        return {"status": "INVALID", "reason": "invalid checkpoint pointer"}
    checkpoint = folder / name
    if not checkpoint.is_dir() or not checkpoint.resolve().is_relative_to(folder.resolve()):
        return {"status": "INVALID", "reason": "checkpoint is not a safe directory"}
    parts: dict[str, dict[str, Any]] = {}
    complete = True
    for component in ("model", "optim", "scheduler", "trainer"):
        subdir = checkpoint / component
        metadata = subdir / ".metadata"
        shards = list(subdir.glob("*.distcp")) if subdir.is_dir() else []
        valid = metadata.is_file() and len(shards) >= world_size
        complete = complete and valid
        parts[component] = {"metadata": metadata.is_file(), "shard_count": len(shards), "valid": valid}
    states = checkpoint / "dataloader"
    present = [rank for rank in range(world_size) if (states / f"rank_{rank}.pkl").is_file()]
    complete = complete and len(present) == world_size
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "iteration": int(name[5:]),
        "checkpoint": str(checkpoint),
        "components": parts,
        "dataloader_present_ranks": present,
        "world_size": world_size,
        "audit_scope": "filesystem_presence_only_not_strict_resume_parity",
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
    for filename, metrics in figures.items():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(11, max(3, len(metrics) * 2.2)), squeeze=False)
        for axis, metric in zip(axes[:, 0], metrics, strict=True):
            iterations, values = _series(rows, metric)
            if iterations:
                axis.plot(iterations, values, alpha=0.35, linewidth=0.8, label="raw")
                axis.plot(iterations, _ema(values), linewidth=1.4, label="EMA(0.98)")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.2)
            axis.legend(loc="best") if iterations else None
        axes[-1, 0].set_xlabel("optimizer iteration")
        fig.tight_layout()
        final = destination / f"{filename}.png"
        pending = destination / f".{filename}.tmp.png"
        fig.savefig(pending, dpi=120)
        plt.close(fig)
        os.replace(pending, final)
        outputs.append(str(final))
    return outputs


def _write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, destination)


def update_monitor(job_dir: Path, *, world_size: int, max_iter: int, plot: bool) -> dict[str, Any]:
    job_dir = Path(job_dir)
    records, superseded = read_train_records(job_dir / "monitor/train_rank0.jsonl")
    checkpoint = audit_latest_checkpoint(job_dir, world_size=world_size)
    recent = [float(row["step_wall_s"]) for row in records[-50:] if isinstance(row.get("step_wall_s"), (int, float)) and row["step_wall_s"] > 0]
    latest = records[-1] if records else {}
    avg_step = statistics.median(recent) if recent else None
    estimated_remaining_h = (
        max(0, max_iter - latest["iteration"]) * avg_step / 3600
        if avg_step is not None and isinstance(latest.get("iteration"), int) else None
    )
    state = {
        "source": "rank0_only_never_global_reduced",
        "last_iteration": latest.get("iteration"),
        "record_count": len(records),
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
    _write_json_atomic(job_dir / "monitor/summary.json", state)
    if plot:
        state["curves"] = render_curves(records, job_dir / "monitor/curves")
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=30000)
    parser.add_argument("--follow", action="store_true", help="CPU-only sidecar; no training hooks")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if args.poll_seconds < 2:
        parser.error("poll-seconds must be >=2")
    previous = None
    while True:
        state = update_monitor(
            args.job_dir,
            world_size=args.world_size,
            max_iter=args.max_iter,
            plot=True if not args.follow else previous != (
                (args.job_dir / "checkpoints/latest_checkpoint.txt").read_text().strip()
                if (args.job_dir / "checkpoints/latest_checkpoint.txt").is_file()
                else None
            ),
        )
        checkpoint = state["checkpoint"]
        previous = f"iter_{checkpoint['iteration']:09d}" if checkpoint.get("iteration") else None
        print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.follow:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

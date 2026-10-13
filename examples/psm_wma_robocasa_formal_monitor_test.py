# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fast CPU-only tests for the formal30k read-only sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.psm_wma_robocasa_formal_monitor import (
    TelemetryJournal,
    audit_latest_checkpoint,
    format_progress,
    read_train_records,
    update_monitor,
)


def _record(iteration: int) -> dict:
    return {
        "status": "optimizer_committed",
        "iteration": iteration,
        "epoch": 0,
        "outer_loss": 1.0,
        "action_loss": 0.03,
        "vision_loss": 0.10,
        "lr_max": 0.0002476,
        "step_wall_s": 42.0,
        "peak_reserved_gib": 9.67,
        "total_grad_norm_rank_local": 0.5,
        "missing_grad_tensors_rank_local": 0,
        "nonfinite_grad_tensors_rank_local": 0,
    }


def test_journal_appends_readable_progress_and_raw_json_without_truncating(tmp_path: Path) -> None:
    stdout: list[str] = []
    journal = TelemetryJournal(tmp_path, emit=stdout.append)
    try:
        journal("[CorrectedV3][progress] " + format_progress(_record(801), max_iter=30000))
        journal("[CorrectedV3][train] " + json.dumps(_record(801)))
        journal("[CorrectedV3][checkpoint] " + json.dumps({"iteration": 800, "checkpoint_save_ms": 8000}))
    finally:
        journal.close()
    journal = TelemetryJournal(tmp_path, emit=stdout.append)
    try:
        journal("[CorrectedV3][train] " + json.dumps(_record(802)))
    finally:
        journal.close()
    records, superseded = read_train_records(tmp_path / "monitor/train_rank0.jsonl")
    assert superseded == 0
    assert [row["iteration"] for row in records] == [801, 802]
    text = (tmp_path / "monitor/rank0.log").read_text()
    assert "iter=801/30000" in text
    assert "GRAD[rank0-shard]" in text
    assert "[CorrectedV3][checkpoint]" in text
    assert len(stdout) == 2
    assert "[CorrectedV3][progress]" in stdout[0]
    assert "[CorrectedV3][checkpoint]" in stdout[1]
    assert not any("[CorrectedV3][train]" in line for line in stdout)
    assert text.count("[CorrectedV3][train]") == 2


def test_journal_preserves_non_train_console_messages(tmp_path: Path) -> None:
    stdout: list[str] = []
    journal = TelemetryJournal(tmp_path, emit=stdout.append)
    progress = "[CorrectedV3][progress] iter=803/30000 optimizer_committed"
    checkpoint = "[CorrectedV3][checkpoint] " + json.dumps({"iteration": 800})
    warning = "[CorrectedV3][logging_error] simulated writer warning"
    try:
        journal("[CorrectedV3][train] " + json.dumps(_record(803)))
        journal(progress)
        journal(checkpoint)
        journal(warning)
    finally:
        journal.close()
    assert stdout == [progress, checkpoint, warning]
    assert [item["iteration"] for item in read_train_records(tmp_path / "monitor/train_rank0.jsonl")[0]] == [803]
    assert "[CorrectedV3][train]" in (tmp_path / "monitor/rank0.log").read_text()



@pytest.mark.parametrize(
    ("num_workers", "prepare_ms", "wait_ms", "expected"),
    [
        (0, 11_500.0, None, "data=11.5s "),
        (8, 10_900.0, 0.024, "data=10.9s "),
        (16, 12_300.0, 0.046, "data=12.3s "),
        (8, None, 0.024, "data=n/a "),
        (8, float("nan"), 0.024, "data=n/a "),
    ],
)
def test_progress_data_reflects_main_thread_prepare_time(
    num_workers: int, prepare_ms: float | None, wait_ms: float | None, expected: str
) -> None:
    record = _record(1340)
    record["num_workers"] = num_workers
    record["data_prepare_ms"] = prepare_ms
    record["data_wait_ms"] = wait_ms
    original = dict(record)
    line = format_progress(record, max_iter=30000)
    assert expected in line
    assert "data=0.024ms" not in line
    assert "data=n/ams" not in line
    assert record == original


def test_read_train_records_is_idempotent_on_duplicate_resume_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(i)) for i in (801, 802, 802, 803)) + "\n",
        encoding="utf-8",
    )
    records, superseded = read_train_records(path)
    assert [item["iteration"] for item in records] == [801, 802, 803]
    assert superseded == 1


@pytest.mark.parametrize(
    "text", ["broken_json\n", '{"iteration":4}\n', '{"iteration":0,"status":"optimizer_committed"}\n']
)
def test_invalid_jsonl_fails_closed(tmp_path: Path, text: str) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_train_records(path)


def _write_dcp(job: Path, iteration: int, *, world_size: int) -> None:
    base = job / "checkpoints"
    checkpoint = base / f"iter_{iteration:09d}"
    for component in ("model", "optim", "scheduler", "trainer"):
        folder = checkpoint / component
        folder.mkdir(parents=True)
        (folder / ".metadata").write_bytes(b"metadata")
        for rank in range(world_size):
            (folder / f"__{rank}_0.distcp").write_bytes(b"shard")
    dataloader = checkpoint / "dataloader"
    dataloader.mkdir()
    for rank in range(world_size):
        (dataloader / f"rank_{rank}.pkl").write_bytes(b"state")
    (base / "latest_checkpoint.txt").write_text(f"iter_{iteration:09d}", encoding="utf-8")


def test_checkpoint_audit_metadata_scope_and_missing_rank(tmp_path: Path) -> None:
    _write_dcp(tmp_path, 800, world_size=8)
    report = audit_latest_checkpoint(tmp_path, world_size=8)
    assert report["status"] == "COMPLETE" and report["iteration"] == 800
    assert report["audit_scope"] == "filesystem_presence_only_not_strict_resume_parity"
    (tmp_path / "checkpoints/iter_000000800/dataloader/rank_7.pkl").unlink()
    assert audit_latest_checkpoint(tmp_path, world_size=8)["status"] == "INCOMPLETE"


def test_checkpoint_pointer_must_not_escape_its_directory(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "latest_checkpoint.txt").write_text("../outside", encoding="utf-8")
    assert audit_latest_checkpoint(tmp_path, world_size=8)["status"] == "INVALID"


def test_monitor_summary_is_atomic_and_never_modifies_dcp(tmp_path: Path) -> None:
    _write_dcp(tmp_path, 800, world_size=2)
    output = tmp_path / "monitor/train_rank0.jsonl"
    output.parent.mkdir()
    output.write_text(json.dumps(_record(801)) + "\n" + json.dumps(_record(802)) + "\n")
    report = update_monitor(tmp_path, world_size=2, max_iter=30000, plot=False)
    assert report["record_count"] == 2
    assert report["last_iteration"] == 802
    assert report["checkpoint"]["iteration"] == 800
    assert report["median_step_50_s"] == pytest.approx(42.0)
    assert (tmp_path / "checkpoints/iter_000000800/model/.metadata").exists()
    assert json.loads((tmp_path / "monitor/summary.json").read_text())["last_iteration"] == 802


def test_monitor_tolerates_only_unterminated_concurrent_last_jsonl_row(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_record(801)) + "\n" + '{"iteration":', encoding="utf-8")
    rows, _ = read_train_records(path)
    assert [row["iteration"] for row in rows] == [801]
    path.write_text('{"iteration":\n' + json.dumps(_record(802)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_train_records(path)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local CPU regression tests; synthetic filesystem fixtures, never a real DCP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import psm_wma_robocasa_formal_monitor as monitor


def _record(step: int) -> dict:
    return {"iteration": step, "status": "optimizer_committed", "outer_loss": 1.0, "step_wall_s": 42.0}


def _checkpoint(job: Path, *, world_size: int = 2) -> Path:
    checkpoint = job / "checkpoints/iter_000000800"
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"fixture-not-real-dcp")
        for rank in range(world_size):
            (directory / f"__{rank}_0.distcp").write_bytes(b"fixture-shard")
    (checkpoint / "dataloader").mkdir()
    for rank in range(world_size):
        (checkpoint / "dataloader" / f"rank_{rank}.pkl").write_bytes(b"fixture-state")
    (job / "checkpoints/latest_checkpoint.txt").write_text(checkpoint.name, encoding="utf-8")
    return checkpoint


def test_stdout_failure_does_not_drop_durable_record(tmp_path: Path) -> None:
    def broken_stdout(_line: str) -> None:
        raise BrokenPipeError("synthetic detached terminal")

    journal = monitor.TelemetryJournal(tmp_path, emit=broken_stdout)
    try:
        journal(monitor.TRAIN_PREFIX + json.dumps(_record(801)))
    finally:
        journal.close()
    rows, _ = monitor.read_train_records(tmp_path / "monitor/train_rank0.jsonl")
    assert [row["iteration"] for row in rows] == [801]


def test_resume_rewind_hides_unreplayed_future_without_rewriting_log(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    raw = "".join(json.dumps(_record(step)) + "\n" for step in (801, 802, 803, 802))
    path.write_text(raw, encoding="utf-8")
    rows, superseded = monitor.read_train_records(path)
    assert [row["iteration"] for row in rows] == [801, 802]
    assert superseded == 2
    assert path.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("tail", ['{"iteration":', json.dumps(_record(802))])
def test_unterminated_final_row_is_not_published(tmp_path: Path, tail: str) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_record(801)) + "\n" + tail, encoding="utf-8")
    rows, _ = monitor.read_train_records(path)
    assert [row["iteration"] for row in rows] == [801]


def test_journal_refuses_to_append_after_partial_row_without_mutating_it(tmp_path: Path) -> None:
    path = tmp_path / "monitor/train_rank0.jsonl"
    path.parent.mkdir()
    original = b'{"iteration":'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="unterminated"):
        monitor.TelemetryJournal(tmp_path, emit=lambda _: None)
    assert path.read_bytes() == original


@pytest.mark.parametrize("relative", ["model/__1_0.distcp", "optim/.metadata", "dataloader/rank_1.pkl"])
def test_empty_checkpoint_file_is_not_complete(tmp_path: Path, relative: str) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / relative).write_bytes(b"")
    assert monitor.audit_latest_checkpoint(tmp_path, world_size=2)["status"] != "COMPLETE"


def test_same_rank_extra_shard_does_not_replace_missing_rank(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "model/__1_0.distcp").unlink()
    (checkpoint / "model/__0_1.distcp").write_bytes(b"extra-rank-zero-shard")
    assert monitor.audit_latest_checkpoint(tmp_path, world_size=2)["status"] != "COMPLETE"


def test_plotted_summary_matches_returned_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "monitor/train_rank0.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(_record(801)) + "\n", encoding="utf-8")
    monkeypatch.setattr(monitor, "render_curves", lambda *args: ["synthetic-test.png"])
    result = monitor.update_monitor(tmp_path, world_size=2, max_iter=30000, plot=True)
    saved = json.loads((tmp_path / "monitor/summary.json").read_text())
    assert saved == result


def test_complete_is_only_a_nonempty_filesystem_check(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    result = monitor.audit_latest_checkpoint(tmp_path, world_size=2)
    assert result["status"] == "COMPLETE"
    assert result["audit_scope"] == "filesystem_presence_only_not_strict_resume_parity"


def test_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    record = _record(801)
    record["outer_loss"] = float("nan")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        monitor.read_train_records(path)


def test_real_png_rendering_stays_in_external_sidecar(tmp_path: Path) -> None:
    rows = [{**_record(801), "inner_loss_mean": 0.1, "fast_update_norm_mean": 0.01}]
    outputs = monitor.render_curves(rows, tmp_path / "curves")
    assert len(outputs) == 4
    for output in outputs:
        assert Path(output).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_failure_still_publishes_health_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_plot(*args):
        raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(monitor, "render_curves", fail_plot)
    result = monitor.update_monitor(tmp_path, world_size=2, max_iter=30000, plot=True)
    assert result["plot_error"] == "RuntimeError: synthetic renderer failure"
    assert json.loads((tmp_path / "monitor/summary.json").read_text()) == result

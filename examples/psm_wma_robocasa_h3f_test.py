"""H3-F 30k formal training launcher CPU/static contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples import psm_wma_robocasa_h3f as h3f


def _args(tmp_path: Path, phase: str = "fresh", attempt: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        preflight=True,
        output_root=tmp_path,
        job_name="formal",
        attempt=attempt,
        expected_root="a" * 40,
        expected_child="b" * 40,
    )


def test_overlay_freezes_30k_schedule() -> None:
    config = h3f.load_stage_a_config()
    h3f.overlay_h3f_config(config, phase="fresh", job_name="formal")
    assert config.trainer.max_iter == 30_000
    assert config.trainer.grad_accum_iter == 2
    assert config.scheduler.cycle_lengths == [30_000]
    assert config.scheduler.warm_up_steps == [500]
    assert config.checkpoint.save_iter == 1_000
    assert config.job.group == h3f.H3F_GROUP
    assert all(step % config.checkpoint.save_iter == 0 for step in h3f.H3F_FORMAL_CHECKPOINT_ITERS)


def test_resume_iteration_and_attempt_identity(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "checkpoints").mkdir(parents=True)
    latest = job / "checkpoints/latest_checkpoint.txt"
    latest.write_text("iter_000012000\n")
    assert h3f._resume_iteration(job) == 12_000
    evidence = h3f._evidence_dir(job, phase="resume", attempt=3, start_iteration=12_000)
    assert evidence.name == "attempt_0003_resume_from_000012000"
    latest.write_text("iter_000030000\n")
    with pytest.raises(ValueError, match="1..29999"):
        h3f._resume_iteration(job)


def test_preflight_is_read_only_and_phase_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(h3f, "lock_pair", lambda *_: {"root": "a" * 40, "child": "b" * 40, "gitlink": "b" * 40})
    monkeypatch.setattr(h3f, "validate_h100_asset_authority", lambda: {"config_sha256": "ok"})
    monkeypatch.setattr(h3f, "read_stage_a_contract", lambda *_: {"dcp_keys": 549})
    fake_catalog = SimpleNamespace(episodes=(1,) * 9036, manifest_digest=h3f.MANIFEST_DIGEST)
    monkeypatch.setattr(h3f, "make_catalog", lambda: (object(), fake_catalog))
    monkeypatch.setattr(
        h3f,
        "preflight_native_batch",
        lambda *args, **kwargs: {"native_batch": 8, "preflight_uids": [f"uid-{i}" for i in range(8)]},
    )
    monkeypatch.setattr(h3f, "_disk_free_bytes", lambda _: 123)

    fresh = _args(tmp_path)
    report = h3f.preflight(fresh)
    assert report["max_iter"] == 30_000 and report["save_iter"] == 1_000
    assert report["disk_free_bytes"] == 123
    assert not Path(report["job"]).exists()

    with pytest.raises(ValueError, match="attempt=1"):
        h3f.preflight(_args(tmp_path, attempt=2))

    job = Path(report["job"])
    (job / "checkpoints").mkdir(parents=True)
    (job / "checkpoints/latest_checkpoint.txt").write_text("iter_000004000\n")
    resume = _args(tmp_path, phase="resume", attempt=2)
    resumed = h3f.preflight(resume)
    assert resumed["start_iteration"] == 4_000
    assert not Path(resumed["evidence_dir"]).exists()

    with pytest.raises(ValueError, match="attempt>=2"):
        h3f.preflight(_args(tmp_path, phase="resume", attempt=1))


def test_formal_observer_aggregates_one_record_per_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 11)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 22)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 33)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    model = torch.nn.Module()
    model.register_parameter("local_memory_weight", torch.nn.Parameter(torch.tensor(1.0)))
    model.local_memory_weight.grad = torch.tensor(0.25)
    observer = h3f.FormalObserver(tmp_path / "progress.jsonl", model)
    trainer = SimpleNamespace(
        _grouped_completed_iteration=7,
        _grouped_window=SimpleNamespace(live=SimpleNamespace(frontier=SimpleNamespace(epoch=2))),
    )

    for index in range(32):
        observer(
            phase="native_forward",
            iteration=7,
            member=index // 16,
            index=index % 16,
            loss=torch.tensor(10.0 + index / 100),
            trainer=trainer,
        )
        observer(
            phase="native_backward",
            iteration=7,
            member=index // 16,
            index=index % 16,
            loss=None,
            trainer=trainer,
        )
    observer(phase="pre_optimizer", iteration=7, member=1, index=None, loss=None, trainer=trainer)
    observer(phase="post_commit", iteration=7, member=1, index=None, loss=None, trainer=trainer)

    assert observer.completed == 1
    assert observer.last_record["iteration"] == 8
    assert observer.last_record["native_forward"] == observer.last_record["native_backward"] == 32
    assert observer.last_record["local_grad_nonzero_shards"] == 1
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 1


def test_formal_observer_rejects_incomplete_iteration(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.register_parameter("local_memory_weight", torch.nn.Parameter(torch.tensor(1.0)))
    model.local_memory_weight.grad = torch.tensor(1.0)
    observer = h3f.FormalObserver(tmp_path / "progress.jsonl", model)
    trainer = SimpleNamespace(
        _grouped_completed_iteration=0,
        _grouped_window=SimpleNamespace(live=SimpleNamespace(frontier=SimpleNamespace(epoch=0))),
    )
    observer(phase="native_forward", iteration=0, member=0, index=0, loss=torch.tensor(1.0), trainer=trainer)
    with pytest.raises(RuntimeError, match="event count"):
        observer(phase="post_commit", iteration=0, member=1, index=None, loss=None, trainer=trainer)


def test_config_digest_is_long_run_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h3f, "h3e_config_digest", lambda: "base")
    first = h3f.config_digest()
    monkeypatch.setattr(h3f, "H3F_WARMUP_STEPS", 501)
    assert first != h3f.config_digest()

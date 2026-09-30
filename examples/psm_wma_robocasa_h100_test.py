"""H3-E 8×H100 launcher 的 CPU/static 合同。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer
from examples import psm_wma_robocasa_h100 as h100


def _args(tmp_path: Path, phase: str = "fresh") -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        preflight=True,
        output_root=tmp_path,
        job_name="matched",
        expected_root="a" * 40,
        expected_child="b" * 40,
    )


def _runtime(tmp_path: Path) -> h100.H100RuntimePaths:
    return h100.H100RuntimePaths(
        root_worktree=tmp_path / "root",
        stage_a_checkpoint=tmp_path / "stage/checkpoints/iter_000000001",
        stage_a_config=tmp_path / "stage/config.yaml",
        dataset_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        cache_probe=tmp_path / "cache/CloseFridge/20250816/lerobot/ep_000067.h5",
        edge=tmp_path / "edge",
        vae=tmp_path / "wan.pth",
        base_checkpoint=tmp_path / "base",
    )


def test_parser_requires_formal_pair_and_output() -> None:
    with pytest.raises(SystemExit):
        h100.parser().parse_args(["--phase", "fresh", "--output-root", "/tmp/job"])
    args = h100.parser().parse_args(
        [
            "--phase",
            "resume",
            "--preflight",
            "--output-root",
            "/tmp/job",
            "--expected-root",
            "a" * 40,
            "--expected-child",
            "b" * 40,
        ]
    )
    assert args.phase == "resume" and args.preflight


def test_pair_lock_checks_both_heads_gitlink_and_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, child = "a" * 40, "b" * 40
    values = {"root": root, "child": child, "gitlink": child, "dirty": ""}
    runtime = _runtime(tmp_path)

    def git(repo, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return values["root" if repo == runtime.root_worktree else "child"]
        if arguments == ("ls-tree", "HEAD", "cosmos-framework"):
            return f"160000 commit {values['gitlink']}\tcosmos-framework"
        return values["dirty"]

    monkeypatch.setattr(h100, "_git", git)
    assert h100.lock_pair(root, child, runtime)["gitlink"] == child
    values["gitlink"] = "c" * 40
    with pytest.raises(ValueError, match="formal"):
        h100.lock_pair(root, child, runtime)
    values["gitlink"] = child
    values["dirty"] = " M production.py"
    with pytest.raises(ValueError, match="工作树"):
        h100.lock_pair(root, child, runtime)


def test_load_stage_a_config_scopes_runtime_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    keys = (
        "EDGE_POLICY_CHECKPOINT",
        "WAN_VAE_PATH",
        "ROBOCASA_ROOT",
        "ROBOCASA_LATENT_CACHE_ROOT",
        "BASE_CHECKPOINT_PATH",
    )
    original = {
        "EDGE_POLICY_CHECKPOINT": "/before/edge",
        "WAN_VAE_PATH": "/before/vae",
        "ROBOCASA_ROOT": "/before/robocasa",
        "ROBOCASA_LATENT_CACHE_ROOT": "/before/cache",
    }
    for key, value in original.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("BASE_CHECKPOINT_PATH", raising=False)

    observed: dict[str, str | None] = {}
    sentinel = object()

    def fake_loader(_recipe):
        observed.update({key: os.environ.get(key) for key in keys})
        return sentinel

    monkeypatch.setattr(h100, "load_experiment_from_toml", fake_loader)
    assert h100.load_stage_a_config(runtime) is sentinel
    assert observed == {
        "EDGE_POLICY_CHECKPOINT": str(runtime.edge),
        "WAN_VAE_PATH": str(runtime.vae),
        "ROBOCASA_ROOT": str(runtime.dataset_root),
        "ROBOCASA_LATENT_CACHE_ROOT": str(runtime.cache_root),
        "BASE_CHECKPOINT_PATH": str(runtime.base_checkpoint),
    }
    assert {key: os.environ.get(key) for key in original} == original
    assert "BASE_CHECKPOINT_PATH" not in os.environ


def test_h100_overlay_preserves_edge_and_stage_a_optimizer(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    config = h100.load_stage_a_config(runtime)
    h100.overlay_h100_config(config, phase="fresh", job_name="matched", runtime=runtime)
    assert config.trainer.type is GroupedLocalMemoryTrainer
    assert (config.trainer.max_iter, config.trainer.grad_accum_iter) == (1, 2)
    assert config.model.config.parallelism.data_parallel_shard_degree == 8
    assert config.model.config.parallelism.data_parallel_replicate_degree == 1
    assert config.optimizer.keys_to_select == list(h100.SELECTED_KEYS)
    assert config.optimizer.weight_decay_skip_patterns == [
        r"action2llm\.(fc|bias)\.weight$",
        r"llm2action\.(fc|bias)\.weight$",
    ]
    assert config.checkpoint.keys_to_skip_loading == ["net_ema.", "local_memory"]
    assert config.checkpoint.strict_resume and not config.checkpoint.load_training_state
    assert config.model.config.max_action_dim == 64
    assert config.model.config.num_embodiment_domains == 32
    assert config.model.config.tokenizer.encode_exact_durations == [33]
    assert not config.model.config.ema.enabled
    h100.overlay_h100_config(config, phase="resume", job_name="matched", runtime=runtime)
    assert config.trainer.max_iter == 2 and config.job.name == "matched"


def test_h3f_formal_training_budget_is_30000() -> None:
    assert h100.H3F_FORMAL_MAX_ITER == 30_000
    assert h100.H3F_FORMAL_CHECKPOINT_ITERS[-1] == h100.H3F_FORMAL_MAX_ITER
    assert h100.H3F_FORMAL_CHECKPOINT_ITERS == tuple(sorted(set(h100.H3F_FORMAL_CHECKPOINT_ITERS)))


def test_trigger_resume_starts_at_second_window() -> None:
    fresh = h100.GroupedTriggerLoader(1)
    assert list(fresh) == [{}, {}]
    resumed = h100.GroupedTriggerLoader(2)
    resumed.set_start_iteration(2)
    assert list(resumed) == [{}, {}] and len(resumed) == 2
    with pytest.raises(ValueError):
        resumed.set_start_iteration(5)


def test_preflight_phase_is_fail_closed_without_creating_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path / "assets")
    monkeypatch.setattr(h100, "runtime_paths", lambda: runtime)
    monkeypatch.setattr(h100, "lock_pair", lambda *_: {"root": "a" * 40, "child": "b" * 40})
    monkeypatch.setattr(
        h100,
        "validate_h100_asset_authority",
        lambda *_: {"config_sha256": "configured"},
    )
    monkeypatch.setattr(h100, "read_stage_a_contract", lambda *_: {"dcp_keys": 549})
    monkeypatch.setattr(h100, "config_digest", lambda *_: "digest")
    fake_catalog = SimpleNamespace(episodes=(1,) * 9036, manifest_digest=h100.MANIFEST_DIGEST)
    monkeypatch.setattr(h100, "make_catalog", lambda *_: (object(), fake_catalog))
    monkeypatch.setattr(
        h100,
        "preflight_native_batch",
        lambda *args, **kwargs: {"native_batch": 8, "preflight_uids": [f"uid-{i}" for i in range(8)]},
    )
    args = _args(tmp_path)
    report = h100.preflight(args)
    assert report["native_batch"] == 8 and report["geometry"] == [8, 8, 2, 16, 4]
    assert not (tmp_path / "psm_wma_v3").exists()
    job = tmp_path / "psm_wma_v3/h3e_edge_local_h100/matched"
    job.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        h100.preflight(args)
    args.phase = "resume"
    with pytest.raises(FileNotFoundError):
        h100.preflight(args)
    (job / "checkpoints").mkdir()
    (job / "checkpoints/latest_checkpoint.txt").write_text("iter_000000001")
    assert h100.preflight(args)["job"] == str(job)


def test_config_digest_ignores_phase_and_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    metadata = tmp_path / "checkpoint/model/.metadata"
    metadata.parent.mkdir(parents=True)
    config.write_text("a")
    metadata.write_text("m")
    runtime = _runtime(tmp_path)
    runtime = h100.H100RuntimePaths(
        **{**runtime.__dict__, "stage_a_config": config, "stage_a_checkpoint": tmp_path / "checkpoint"}
    )
    first = h100.config_digest(runtime)
    assert first == h100.config_digest(runtime)
    config.write_text("b")
    assert first != h100.config_digest(runtime)


def test_h100_asset_authority_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    stage = tmp_path / "stage"
    checkpoint = stage / "checkpoints/iter_000000001"
    source = tmp_path / "source"
    cache_root = tmp_path / "cache"
    cache = cache_root / "CloseFridge/20250816/lerobot/ep_000067.h5"
    edge = tmp_path / "edge"
    vae = tmp_path / "wan.pth"
    config = stage / "config.yaml"

    for directory in (root, checkpoint / "model", checkpoint / "trainer", source, cache.parent, edge):
        directory.mkdir(parents=True, exist_ok=True)
    (source / "CloseFridge/20250816/lerobot/meta").mkdir(parents=True)
    config.write_text("stage-a-config")
    (checkpoint / "model/.metadata").write_text("model-metadata")
    (checkpoint / "trainer/.metadata").write_text("trainer-metadata")
    cache.touch()
    (edge / "config.json").write_text("{}")
    vae.touch()
    (source / "CloseFridge/20250816/lerobot/meta/info.json").write_text("{}")

    base = tmp_path / "base"
    (base / "model").mkdir(parents=True)
    (base / "model/.metadata").write_text("base-model-metadata")
    runtime = h100.H100RuntimePaths(
        root_worktree=root,
        stage_a_checkpoint=checkpoint,
        stage_a_config=config,
        dataset_root=source,
        cache_root=cache_root,
        cache_probe=cache,
        edge=edge,
        vae=vae,
        base_checkpoint=base,
    )
    report = h100.validate_h100_asset_authority(runtime)
    assert report["stage_a_checkpoint"] == str(checkpoint)
    first_digest = report["config_sha256"]

    config.write_text("drift")
    assert h100.validate_h100_asset_authority(runtime)["config_sha256"] != first_digest


def test_preflight_native_batch_uses_grouped_catalog_binder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    episodes = tuple(SimpleNamespace(uid=f"task/date/ep_{i:06d}") for i in range(8))
    requests = tuple(SimpleNamespace(episode=episode) for episode in episodes)
    plan = SimpleNamespace(members=(requests, requests))

    class FakePlanner:
        def __init__(self, catalog, *, rank, world_size, seed):
            assert rank == 0 and world_size == 8 and seed == 0

        def initial_frontier(self):
            return "frontier"

        def plan_window(self, frontier):
            assert frontier == "frontier"
            return plan

    class FakeBinder:
        def __init__(self, dataset, catalog, **kwargs):
            assert kwargs["source_root"] == runtime.dataset_root
            assert kwargs["cache_root"] == runtime.cache_root
            assert kwargs["config_digest"] == "digest"
            self.producer_for = object()

    segments = tuple(SimpleNamespace(consumer_payload=(({"slot": index},),)) for index in range(8))
    monkeypatch.setattr(h100, "RankLocalGroupedPlanner", FakePlanner)
    monkeypatch.setattr(h100, "StageARoboCasaEpisodeBinder", FakeBinder)
    monkeypatch.setattr(h100, "build_stage_a_action_transform", lambda paths: ("transform", "resolution"))
    monkeypatch.setattr(
        h100,
        "materialize_member",
        lambda members, producer_for: segments,
    )
    monkeypatch.setattr(
        h100,
        "collate_grouped_native_batch",
        lambda payloads: {
            "sequence_plan": [1] * 8,
            "action": [[SimpleNamespace(shape=(33, 64))]] * 8,
            "action_raw": [[SimpleNamespace(shape=(33, 15))]] * 8,
        },
    )
    evidence = h100.preflight_native_batch(
        object(), object(), SimpleNamespace(), digest="digest", runtime=runtime
    )
    assert evidence["native_batch"] == 8
    assert evidence["preflight_uids"] == [episode.uid for episode in episodes]


def test_optimizer_parameter_ids_unwraps_optimizers_container() -> None:
    from cosmos_framework.utils.generator.optimizer import OptimizersContainer

    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    third = torch.nn.Parameter(torch.tensor(3.0))
    container = object.__new__(OptimizersContainer)
    container.optimizers = [
        torch.optim.SGD([first, second], lr=1e-3),
        torch.optim.AdamW([third], lr=1e-3),
    ]

    assert h100._optimizer_parameter_ids(container) == {id(first), id(second), id(third)}
    assert h100._optimizer_parameter_ids(container.optimizers[0]) == {id(first), id(second)}

    container.optimizers = []
    with pytest.raises(ValueError, match="container"):
        h100._optimizer_parameter_ids(container)


def test_observer_records_consumer_memory_gradient_and_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 12)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 24)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 36)
    model = torch.nn.Module()
    model.register_parameter("local_memory_weight", torch.nn.Parameter(torch.tensor(1.0)))
    model.local_memory_weight.grad = torch.tensor(0.25)
    trace = tmp_path / "rank.jsonl"
    observer = h100.SmokeObserver(trace, model)
    trainer = SimpleNamespace(
        _grouped_completed_iteration=0,
        _grouped_window=SimpleNamespace(live=SimpleNamespace(frontier=SimpleNamespace(epoch=0))),
    )
    for phase in ("native_forward", "native_backward", "pre_optimizer", "post_commit"):
        observer(
            phase=phase,
            iteration=0,
            member=0,
            index=3,
            loss=torch.tensor(1.25) if phase == "native_forward" else None,
            trainer=trainer,
        )
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [line["phase"] for line in lines] == ["native_forward", "native_backward", "pre_optimizer", "post_commit"]
    assert all(line["index"] == 3 and line["allocated_bytes"] == 12 for line in lines)
    assert lines[0]["native_loss"] == 1.25
    assert lines[2]["grad_norms"]["local_memory_weight"] == 0.25
    assert lines[3]["completed_iteration"] == 1


def test_checkpoint_witness_requires_all_components_and_rank_state(tmp_path: Path) -> None:
    folder = tmp_path / "checkpoints"
    checkpoint = folder / "iter_000000001"
    for key in ("model", "optim", "scheduler", "trainer"):
        (checkpoint / key).mkdir(parents=True)
        (checkpoint / key / ".metadata").touch()
    (checkpoint / "dataloader").mkdir()
    (checkpoint / "dataloader/rank_3.pkl").touch()
    (folder / "latest_checkpoint.txt").write_text("iter_000000001\n")
    assert h100.verify_completed_checkpoint(tmp_path, rank=3, iteration=1, phase="fresh", resumed=False) == checkpoint
    with pytest.raises(RuntimeError, match="resume"):
        h100.verify_completed_checkpoint(tmp_path, rank=3, iteration=1, phase="resume", resumed=False)
    assert h100.verify_completed_checkpoint(tmp_path, rank=3, iteration=1, phase="resume", resumed=True) == checkpoint
    (checkpoint / "dataloader/rank_3.pkl").unlink()
    with pytest.raises(RuntimeError, match="DCP"):
        h100.verify_completed_checkpoint(tmp_path, rank=3, iteration=1, phase="resume", resumed=True)

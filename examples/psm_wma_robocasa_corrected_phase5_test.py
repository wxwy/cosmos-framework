"""Corrected Phase5 scratch launcher 的 CPU 合同。"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy_test import _catalog
from examples import psm_wma_robocasa_corrected_phase5 as phase5


@pytest.mark.parametrize("ga", (1, 2, 3))
def test_trigger_length_and_resume_offset(ga: int) -> None:
    loader = phase5.GroupedTriggerLoader(5, ga)
    assert len(loader) == 5 * ga
    assert len(list(loader)) == 5 * ga
    loader.set_start_iteration(2 * ga)
    assert len(loader) == 3 * ga
    assert len(list(loader)) == 3 * ga
    loader.set_start_iteration(5 * ga)
    assert len(loader) == 0
    with pytest.raises(ValueError, match="偏移"):
        loader.set_start_iteration(5 * ga + 1)


def test_observer_uses_actual_plan_prefix_lengths() -> None:
    plan = SimpleNamespace(
        members=((SimpleNamespace(valid_count=3), SimpleNamespace(valid_count=1)), (SimpleNamespace(valid_count=2),))
    )
    trainer = SimpleNamespace(_grouped_window=SimpleNamespace(plan=plan))
    observer = phase5.GroupedPlanObserver()
    for _ in range(5):
        observer(phase="native_forward", trainer=trainer)
        observer(phase="native_backward", trainer=trainer)
    observer(phase="pre_optimizer", trainer=trainer)
    observer(phase="post_commit", trainer=trainer)
    assert observer.completed == 1
    observer(phase="native_forward", trainer=trainer)
    with pytest.raises(RuntimeError, match="调用次数"):
        observer(phase="pre_optimizer", trainer=trainer)


def _config(path: Path, *, t: int = 16, b: int = 8, ga: int = 2, k: int = 4):
    model = SimpleNamespace(
        local_memory_ttt_tbptt_steps=t,
        local_memory_k_local=k,
        local_memory_ttt_dim=64,
        local_memory_fast_hidden_dim=256,
        local_memory_dim=32,
        local_memory_evidence_dim=256,
        local_memory_inner_lr=0.1,
        local_memory_action_dim=15,
        max_action_dim=64,
        tokenizer=deepcopy(EDGE_MODEL_CONFIG["tokenizer"]),
        parallelism=SimpleNamespace(data_parallel_shard_degree=8, data_parallel_replicate_degree=1),
        cache_root=path / "cache",
        source_root=path / "source",
    )
    config = SimpleNamespace(
        model=SimpleNamespace(config=model),
        optimizer=SimpleNamespace(
            keys_to_select=phase5.OPTIMIZER_KEYS,
            optimizer_type="FusedAdam",
            lr=5e-5,
            lr_multipliers={key: 5.0 for key in phase5.ACTION_KEYS},
            weight_decay=0.05,
        ),
        model_parallel=SimpleNamespace(context_parallel_size=1),
        trainer=SimpleNamespace(max_iter=30000, grad_accum_iter=ga),
        scheduler=SimpleNamespace(cycle_lengths=[30000], warm_up_steps=[500]),
        checkpoint=SimpleNamespace(save_iter=100, load_path=str(path / "dcp")),
        job=SimpleNamespace(),
    )
    catalog = SimpleNamespace(manifest_digest="manifest", cache_corpus_digest="corpus", source_binding_digest="source")
    return catalog, config, b, ga


def test_digest_path_invariance_and_semantic_sensitivity(tmp_path: Path) -> None:
    def witnesses(root: Path) -> dict[str, str]:
        edge, base = root / "edge", root / "base"
        (base / "model").mkdir(parents=True)
        edge.mkdir()
        (edge / "config.json").write_bytes(b"edge config")
        (base / "model/.metadata").write_bytes(b"base model metadata")
        return phase5.model_witnesses(edge, base)

    base_witnesses = witnesses(tmp_path / "host_a")
    catalog, config, b, ga = _config(tmp_path / "host_a")
    base = phase5.config_digest(catalog, config, b_stream=b, active_ga=ga, witnesses=base_witnesses)
    moved_catalog, moved_config, _, _ = _config(tmp_path / "host_b")
    moved_witnesses = witnesses(tmp_path / "host_b")
    assert (
        phase5.config_digest(moved_catalog, moved_config, b_stream=b, active_ga=ga, witnesses=moved_witnesses) == base
    )
    for path in (tmp_path / "host_b/edge/config.json", tmp_path / "host_b/base/model/.metadata"):
        original = path.read_bytes()
        path.write_bytes(original + b" changed")
        assert (
            phase5.config_digest(
                catalog,
                config,
                b_stream=b,
                active_ga=ga,
                witnesses=phase5.model_witnesses(tmp_path / "host_b/edge", tmp_path / "host_b/base"),
            )
            != base
        )
        path.write_bytes(original)
    for field, value in (
        ("manifest_digest", "changed"),
        ("cache_corpus_digest", "changed"),
        ("source_binding_digest", "changed"),
    ):
        changed = SimpleNamespace(**vars(catalog))
        setattr(changed, field, value)
        assert phase5.config_digest(changed, config, b_stream=b, active_ga=ga, witnesses=base_witnesses) != base
    for kwargs in ({"t": 32}, {"b": 3}, {"ga": 3}, {"k": 1}, {"k": 8}):
        changed, changed_config, changed_b, changed_ga = _config(tmp_path, **kwargs)
        assert (
            phase5.config_digest(
                changed, changed_config, b_stream=changed_b, active_ga=changed_ga, witnesses=base_witnesses
            )
            != base
        )
    config.model.config.local_memory_evidence_dim = 128
    assert phase5.config_digest(catalog, config, b_stream=b, active_ga=ga, witnesses=base_witnesses) != base
    config.model.config.local_memory_evidence_dim = 256
    config.optimizer.weight_decay = 0.1
    assert phase5.config_digest(catalog, config, b_stream=b, active_ga=ga, witnesses=base_witnesses) != base


def test_overlay_installs_full_manifest_tokenizer_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _catalog(tmp_path / "cache")
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(cache)
    assert contract.vae_encode_contract["encode_exact_durations"] == [17, 61, 73]
    catalog, config, _, _ = _config(tmp_path)
    del catalog
    from examples import psm_wma_robocasa_local_s1

    monkeypatch.setattr(psm_wma_robocasa_local_s1, "overlay_local_config", lambda _: None)
    args = SimpleNamespace(
        t=16,
        k=4,
        world_size=8,
        ga=2,
        max_iter=30000,
        warmup=500,
        save_iter=100,
        base_checkpoint=tmp_path / "base",
        job_name="test",
    )
    phase5.overlay_config(config, args, contract)
    assert config.model.config.tokenizer["encode_exact_durations"] == [17, 61, 73]
    assert config.model.config.tokenizer["encode_chunk_frames"] == EDGE_MODEL_CONFIG["tokenizer"]["encode_chunk_frames"]
    contract.validate_tokenizer_config(config.model.config.tokenizer)
    with pytest.raises(ValueError, match="encode_exact_durations"):
        contract.validate_tokenizer_config({**config.model.config.tokenizer, "encode_exact_durations": [17]})
    bad_chunk = deepcopy(config.model.config.tokenizer)
    bad_chunk["encode_chunk_frames"]["256"] = 64
    with pytest.raises(ValueError, match="encode_chunk_frames"):
        contract.validate_tokenizer_config(bad_chunk)


def test_overlay_validates_actual_dictconfig_tokenizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    _, config, _, _ = _config(tmp_path)
    model = vars(config.model.config).copy()
    model.pop("cache_root")
    model.pop("source_root")
    model["parallelism"] = vars(model["parallelism"])
    config.model.config = OmegaConf.create(model)
    from examples import psm_wma_robocasa_local_s1

    monkeypatch.setattr(psm_wma_robocasa_local_s1, "overlay_local_config", lambda _: None)
    args = SimpleNamespace(
        t=16,
        k=4,
        world_size=8,
        ga=2,
        max_iter=30000,
        warmup=500,
        save_iter=100,
        base_checkpoint=tmp_path / "base",
        job_name="test",
    )
    phase5.overlay_config(config, args, contract)
    assert isinstance(config.model.config.tokenizer, DictConfig)
    assert list(config.model.config.tokenizer.encode_exact_durations) == [17, 61, 73]
    phase5._validate_runtime_tokenizer(contract, config.model.config.tokenizer)
    config.model.config.tokenizer.encode_exact_durations = [17]
    with pytest.raises(ValueError, match="encode_exact_durations"):
        phase5._validate_runtime_tokenizer(contract, config.model.config.tokenizer)


def _checkpoint_args(tmp_path: Path, phase: str) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        output_root=tmp_path,
        job_name="resume_contract",
        world_size=2,
        max_iter=10,
        t=16,
        k=4,
        ga=2,
        warmup=1,
        save_iter=5,
        base_checkpoint=tmp_path / "official_base_dcp",
    )


def _write_same_job_checkpoint(
    args: SimpleNamespace, *, missing: str | None = None, name: str = "iter_000000007"
) -> Path:
    directory = args.output_root / "psm_wma_v3/corrected_phase5" / args.job_name / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "latest_checkpoint.txt").write_text(name)
    checkpoint = directory / name
    for component in ("model", "optim", "scheduler", "trainer"):
        if component != missing:
            path = checkpoint / component / ".metadata"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for rank in range(args.world_size):
        if f"rank_{rank}.pkl" != missing:
            path = checkpoint / "dataloader" / f"rank_{rank}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return checkpoint


def test_fresh_overlay_uses_official_base_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from examples import psm_wma_robocasa_local_s1

    monkeypatch.setattr(psm_wma_robocasa_local_s1, "overlay_local_config", lambda _: None)
    args = _checkpoint_args(tmp_path, "fresh")
    _, config, _, _ = _config(tmp_path)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    assert phase5._resume_checkpoint(args) is None
    phase5.overlay_config(config, args, contract)
    assert config.checkpoint.load_path == str(args.base_checkpoint)
    assert config.checkpoint.load_training_state is False
    assert config.checkpoint.strict_resume is True
    assert set(config.checkpoint.keys_to_skip_loading) == {"net_ema.", "local_memory"}


def test_resume_requires_complete_same_job_dcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from examples import psm_wma_robocasa_local_s1

    monkeypatch.setattr(psm_wma_robocasa_local_s1, "overlay_local_config", lambda _: None)
    args = _checkpoint_args(tmp_path, "resume")
    checkpoint = _write_same_job_checkpoint(args)
    assert phase5._resume_checkpoint(args) == checkpoint
    _, config, _, _ = _config(tmp_path)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    phase5.overlay_config(config, args, contract, resume_checkpoint=checkpoint)
    assert config.checkpoint.load_path == str(checkpoint)
    assert config.checkpoint.load_training_state is True
    assert config.checkpoint.strict_resume is True
    assert config.checkpoint.keys_to_skip_loading == ["net_ema."]
    assert all("local_memory" not in key for key in config.checkpoint.keys_to_skip_loading)


@pytest.mark.parametrize("missing", ("model", "optim", "scheduler", "trainer", "rank_0.pkl", "rank_1.pkl"))
def test_resume_rejects_each_missing_component(tmp_path: Path, missing: str) -> None:
    args = _checkpoint_args(tmp_path, "resume")
    _write_same_job_checkpoint(args, missing=missing)
    with pytest.raises(FileNotFoundError, match="same-job DCP 缺少"):
        phase5._resume_checkpoint(args)


@pytest.mark.parametrize("name", ("invalid", "iter_0", "iter_000000010", "iter_-1"))
def test_resume_rejects_invalid_latest_iteration(tmp_path: Path, name: str) -> None:
    args = _checkpoint_args(tmp_path, "resume")
    _write_same_job_checkpoint(args, name=name)
    with pytest.raises(ValueError, match="iteration 不合法"):
        phase5._resume_checkpoint(args)


@pytest.mark.skipif(
    not all(
        os.environ.get(key)
        for key in (
            "PSM_PHASE5_CACHE_ROOT",
            "PSM_PHASE5_SOURCE_ROOT",
            "EDGE_POLICY_CHECKPOINT",
            "BASE_CHECKPOINT_PATH",
            "WAN_VAE_PATH",
        )
    ),
    reason="未提供真实 exact-window cache/source/Edge/base/VAE 路径",
)
def test_optional_real_strict_snapshot10_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        phase5, "verify_root_child_lock", lambda _: {"root": "test", "child": "test", "gitlink": "test"}
    )
    args = SimpleNamespace(
        phase="fresh",
        t=16,
        b=2,
        ga=1,
        k=4,
        world_size=1,
        max_iter=10,
        save_iter=10,
        warmup=1,
        output_root=tmp_path,
        root_worktree=tmp_path,
        expected_root="test",
        expected_child="test",
        cache_root=Path(os.environ["PSM_PHASE5_CACHE_ROOT"]),
        source_root=Path(os.environ["PSM_PHASE5_SOURCE_ROOT"]),
        edge=Path(os.environ["EDGE_POLICY_CHECKPOINT"]),
        base_checkpoint=Path(os.environ["BASE_CHECKPOINT_PATH"]),
        vae=Path(os.environ["WAN_VAE_PATH"]),
        job_name="snapshot10",
        snapshot10=True,
    )
    report, config, catalog, dataset = phase5.preflight(args)
    summary = dataset._dataset.summary()
    assert summary["cached_latent_required"] is True
    assert summary["online_vae_fallback"] is False
    assert summary["model_cache_hit_required"] is True
    assert dataset._dataset[0]["cached_latent_required"] is True
    phase5._validate_runtime_tokenizer(catalog.raw.contract, config.model.config.tokenizer)
    assert report["cache_manifest_sha256"] == catalog.manifest_digest


class _Net(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.moe_gen = nn.Linear(2, 2)
        self.time_embedder = nn.Linear(2, 2)
        self.vae2llm = nn.Linear(2, 2)
        self.llm2vae = nn.Linear(2, 2)
        self.action2llm = nn.Linear(2, 2)
        self.llm2action = nn.Linear(2, 2)
        self.action_modality_embed = nn.Parameter(torch.ones(2))
        self.local_memory_runtime = nn.Module()
        self.local_memory_runtime.encoder = nn.Linear(2, 2)
        self.local_memory_runtime.core = nn.Module()
        self.local_memory_runtime.core.slot_queries = nn.Parameter(torch.ones(k, 2))
        self.local_memory_runtime.core.w0 = nn.Parameter(torch.ones(k, 2))
        self.local_memory_runtime.core.kqv = nn.Linear(2, 2)
        self.local_memory2llm = nn.Linear(2, 2)
        self.local_memory_modality_embed = nn.Parameter(torch.ones(2))
        self.reasoner = nn.Linear(2, 2)
        self.tokenizer = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)
        self.language_model = nn.Module()
        self.language_model.reasoner = nn.Linear(2, 2)
        self.language_model.block_moe_gen = nn.Linear(2, 2)


@pytest.mark.parametrize("k", (1, 4, 8))
def test_exact_actual_k_inventory_and_missing_extra_fail(k: int) -> None:
    model = nn.Module()
    model.net = _Net(k)
    named = dict(model.named_parameters())
    allowed = [p for name, p in named.items() if phase5._selected_name(name)]
    for name, parameter in named.items():
        parameter.requires_grad_(phase5._selected_name(name))
    optimizer = torch.optim.AdamW(allowed)
    report = phase5.validate_optimizer_inventory(model, optimizer)
    assert report["local_params"] == 4 * k + 20
    assert report["generation_tensors"] == 15
    assert not model.net.reasoner.weight.requires_grad
    assert not model.net.language_model.reasoner.weight.requires_grad
    assert model.net.language_model.block_moe_gen.weight.requires_grad
    assert not model.net.tokenizer.weight.requires_grad
    assert not model.net.vae.weight.requires_grad
    missing = torch.optim.AdamW(allowed[:-1])
    with pytest.raises(ValueError, match="inventory"):
        phase5.validate_optimizer_inventory(model, missing)
    extra = torch.optim.AdamW([*allowed, model.net.reasoner.weight])
    with pytest.raises(ValueError, match="inventory"):
        phase5.validate_optimizer_inventory(model, extra)


def test_trigger_telemetry_start_keeps_exact_yield_and_resume() -> None:
    starts: list[int] = []
    loader = phase5.GroupedTriggerLoader(4, 2, on_iteration_start=starts.append)
    assert list(loader) == [{}, {}, {}, {}, {}, {}, {}, {}]
    assert starts == [0, 1, 2, 3]

    starts.clear()
    loader.set_start_iteration(4)
    assert list(loader) == [{}, {}, {}, {}]
    assert starts == [2, 3]
    starts.clear()
    loader.set_start_iteration(8)
    assert list(loader) == []
    assert starts == []


def test_telemetry_groups_are_strict_subsets_of_optimizer_allowlist() -> None:
    model = nn.Module()
    model.net = _Net(4)
    for name, _ in model.named_parameters():
        group = phase5._telemetry_parameter_group(name)
        assert (group is not None) == phase5._selected_name(name)
        if group == "action":
            assert name.split(".", 2)[1] in phase5.ACTION_KEYS
        elif group == "local":
            assert "local_memory" in name
        elif group is not None:
            assert group == "generation"
    assert phase5._telemetry_parameter_group("net.reasoner.weight") is None


def test_checkpoint_wrapper_observes_only_successful_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    timer = iter((10.0, 10.5, 11.0, 11.5))
    monkeypatch.setattr(phase5.time, "perf_counter", lambda: next(timer))
    checkpointer = SimpleNamespace(save=lambda *args, **kwargs: "DCP_OK")
    observer = SimpleNamespace(log_checkpoint=lambda iteration, ms: events.append((iteration, ms)))
    phase5.install_checkpoint_save_telemetry(checkpointer, observer)
    assert checkpointer.save("model", iteration=100) == "DCP_OK"
    assert events == [(100, pytest.approx(500.0))]

    def fail_save(*args, **kwargs):
        raise OSError("failure")

    checkpointer = SimpleNamespace(save=fail_save)
    phase5.install_checkpoint_save_telemetry(checkpointer, observer)
    with pytest.raises(OSError, match="failure"):
        checkpointer.save("model", iteration=200)
    assert len(events) == 1  # Failed DCP must not print successful save time.


def test_corrected_phase5_keeps_default_callbacks_disabled() -> None:
    """Prevent accidental restoration of upstream callbacks/DCP dataloader owners."""
    import inspect

    source = inspect.getsource(phase5.overlay_config)
    assert "config.trainer.callbacks = {}" in source


def test_grouped_latency_instrumentation_uses_real_producer_and_model_scopes() -> None:
    """Data is produced inside training_step, not by GroupedTriggerLoader."""
    import inspect

    from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
    from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer

    trainer_source = inspect.getsource(GroupedLocalMemoryTrainer.training_step)
    window_source = inspect.getsource(GroupedLocalMemoryWindow.run_member)
    for stage in ("data_prepare", "batch_collate", "batch_transfer", "forward", "backward", "optimizer"):
        assert f'_telemetry_stage("{stage}")' in trainer_source
    assert 'timing("local_scan")' in window_source
    assert 'self._grouped_producer.produce(request)' in trainer_source
    assert "window.finish(observed_optimizer_step)" in trainer_source
    assert "config.trainer.callbacks = {}" in inspect.getsource(phase5.overlay_config)


def _bounded_args(tmp_path: Path, *, stop: int | None = 3) -> SimpleNamespace:
    return SimpleNamespace(
        phase="fresh",
        job_name="bounded_formal_schedule_smoke",
        output_root=tmp_path,
        max_iter=30000,
        warmup=500,
        save_iter=100,
        stop_after_iter=stop,
    )


def test_bounded_execution_preserves_formal_schedule_without_mutation(tmp_path: Path) -> None:
    args = _bounded_args(tmp_path)
    before = (args.max_iter, args.warmup, args.save_iter)
    assert phase5._execution_stop_iteration(args) == 3
    assert (args.max_iter, args.warmup, args.save_iter) == before == (30000, 500, 100)
    assert phase5._execution_stop_iteration(_bounded_args(tmp_path, stop=None)) == 30000

    loader = phase5.GroupedTriggerLoader(args.max_iter, active_ga=2)
    assert len(loader) == 60000  # Actual execution cap is enforced by the trainer loop, not by changing data.
    assert list(next(iter(loader)) for _ in range(1)) == [{}]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"stop_after_iter": 0}, "fresh-only"),
        ({"stop_after_iter": -1}, "fresh-only"),
        ({"stop_after_iter": 30000}, "fresh-only"),
        ({"stop_after_iter": 30001}, "fresh-only"),
        ({"phase": "resume"}, "fresh-only"),
        ({"max_iter": 10}, "30000/500/100"),
        ({"warmup": 1}, "30000/500/100"),
        ({"save_iter": 1}, "30000/500/100"),
        ({"job_name": "edge_local_exact_window"}, "bounded_"),
    ),
)
def test_bounded_smoke_rejects_unsafe_configuration(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    args = _bounded_args(tmp_path)
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        phase5._execution_stop_iteration(args)


def test_bounded_smoke_rejects_namespace_reuse_and_resume(tmp_path: Path) -> None:
    args = _bounded_args(tmp_path)
    output = tmp_path / "psm_wma_v3" / "corrected_phase5" / args.job_name
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="isolated"):
        phase5._execution_stop_iteration(args)

    args.phase, args.stop_after_iter = "resume", None
    with pytest.raises(ValueError, match="must not be resumed"):
        phase5._execution_stop_iteration(args)


def test_bounded_trainer_checks_completed_optimizer_iteration_not_dataloader_length() -> None:
    import inspect

    from cosmos_framework.trainer import ImaginaireTrainer

    train_source = inspect.getsource(ImaginaireTrainer.train)
    assert train_source.count("if iteration >= execution_max_iter:") == 2
    assert 'getattr(self, "_execution_max_iter", None)' in train_source
    assert "if iteration % self.config.checkpoint.save_iter != 0:" in train_source
    main_source = inspect.getsource(phase5.main)
    assert "trainer._execution_max_iter = report" in main_source
    assert "expected_iteration = report" in main_source

"""Matched profile and unchanged G1 configuration/identity contracts on CPU."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy_test import _catalog
from cosmos_framework.trainer.matched_no_ttt import MatchedNoTTTTrainer
from examples import psm_wma_robocasa_corrected_phase5 as phase5
from examples.psm_wma_robocasa_corrected_phase5_test import _config
from examples.psm_wma_robocasa_matched_no_ttt import NoTTTPlanObserver
from examples.psm_wma_robocasa_phase5_f0_test import Serializer, _args, _report
from examples.psm_wma_robocasa_phase5_f0_test import _config as lifecycle_config
from torch import nn


def test_overlay_only_removes_local_and_keeps_data_loss_optimizer_schedule(tmp_path):
    _, local, _, _ = _config(tmp_path)
    local.model.config.ema = SimpleNamespace(enabled=True)
    local.model.config.compile = SimpleNamespace(enabled=True)
    local.model.config.activation_checkpointing = SimpleNamespace(mode="selective")
    baseline = deepcopy(local)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    args = _args(tmp_path)
    phase5.overlay_config(local, args, contract)
    args.no_ttt, args.job_name = True, "no_ttt_matched"
    phase5.overlay_config(baseline, args, contract)
    assert baseline.model.config.local_memory_enabled is False
    assert local.model.config.local_memory_enabled is True
    assert baseline.trainer.type is MatchedNoTTTTrainer
    assert baseline.optimizer.keys_to_select == list(phase5.GENERATION_KEYS)
    assert local.optimizer.keys_to_select == [*baseline.optimizer.keys_to_select, *phase5.LOCAL_KEYS]
    for name in ("lr", "lr_multipliers", "weight_decay", "optimizer_type"):
        assert getattr(baseline.optimizer, name) == getattr(local.optimizer, name)
    assert baseline.optimizer.lr == 5e-5
    assert vars(baseline.scheduler) == vars(local.scheduler)
    assert vars(baseline.checkpoint) == vars(local.checkpoint)
    assert baseline.dataloader_train == local.dataloader_train
    for key, value in vars(local.model.config).items():
        if key not in ("local_memory_enabled", "ema", "compile", "activation_checkpointing", "parallelism"):
            assert getattr(baseline.model.config, key) == value
    assert baseline.trainer.max_iter == 30000 and baseline.trainer.grad_accum_iter == 2
    assert not baseline.model.config.ema.enabled and not baseline.model.config.compile.enabled
    assert baseline.checkpoint.load_training_state is False
    assert baseline.checkpoint.load_path == str(args.base_checkpoint)


def test_digest_separates_no_ttt_but_preserves_legacy_identity_when_absent(tmp_path):
    catalog, config, b, ga = _config(tmp_path)
    witnesses = {"edge": "same_droid"}

    def digest():
        return phase5.config_digest(catalog, config, b_stream=b, active_ga=ga, witnesses=witnesses)

    local_digest = digest()
    config.model.config.local_memory_enabled = True
    assert digest() == local_digest
    config.model.config.local_memory_enabled = False
    baseline_digest = digest()
    assert baseline_digest != local_digest
    config.optimizer.keys_to_select = list(phase5.GENERATION_KEYS)
    assert digest() != baseline_digest


def test_no_ttt_namespace_resume_rejects_mode_switch_without_touching_assets(tmp_path):
    args = _args(tmp_path, "--no-ttt", "--job-name", "no_ttt_formal")
    report = _report(args, {"training_profile": "matched_no_ttt_v1"})
    lifecycle = phase5.JobLifecycle(args, report)
    lifecycle.write_config(lifecycle_config(args), Serializer)
    lifecycle.close()
    path = phase5.job_namespace(args)
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    args.phase = "resume"
    phase5.check_job_namespace(args, report["training_identity"])
    args.no_ttt = False
    with pytest.raises(ValueError, match="identity"):
        phase5.check_job_namespace(args)
    assert {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("job", [phase5.G1_JOB, "formal", "../no_ttt_bad"])
def test_cannot_point_no_ttt_at_g1_or_arbitrary_namespace(tmp_path, job):
    args = _args(tmp_path, "--no-ttt", "--job-name", job)
    with pytest.raises(ValueError):
        phase5.check_job_namespace(args)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_g1_owner_override_cannot_enable_no_ttt(tmp_path):
    args = _args(
        tmp_path,
        "--no-ttt",
        "--phase",
        "resume",
        "--job-name",
        phase5.G1_JOB,
        "--owner-authorized-g1-cross-pair-resume",
    )
    with pytest.raises(ValueError, match="Owner"):
        phase5.check_job_namespace(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("b", 128),
        ("ga", 1),
        ("world_size", 1),
        ("t", 8),
        ("num_workers", 1),
        ("full_segment_prefetch", True),
        ("transform_rng_mode", "consumer_keyed_v1"),
    ],
)
def test_mismatched_no_ttt_geometry_rejected_before_runtime(tmp_path, monkeypatch, field, value):
    args = _args(tmp_path, "--no-ttt", "--job-name", "no_ttt_formal")
    setattr(args, field, value)

    def forbidden():
        pytest.fail("runtime initialized before rejecting unmatched baseline")

    monkeypatch.setattr(phase5, "_load_runtime", forbidden)
    with pytest.raises(ValueError, match="Matched No-TTT"):
        phase5.preflight(args)


@pytest.mark.parametrize(("world_size", "ga"), [(8, 2), (4, 4)])
def test_both_matched_no_ttt_profiles_pass_geometry_gate_before_real_assets(
    tmp_path, monkeypatch, world_size, ga
):
    args = _args(
        tmp_path,
        "--no-ttt",
        "--job-name",
        f"no_ttt_world{world_size}_ga{ga}",
        "--world-size",
        str(world_size),
        "--ga",
        str(ga),
    )

    def admitted():
        raise RuntimeError("reached CUDA/runtime asset preflight")

    monkeypatch.setattr(phase5, "_load_runtime", admitted)
    with pytest.raises(RuntimeError, match="reached CUDA/runtime"):
        phase5.preflight(args)


@pytest.mark.parametrize(("world_size", "ga"), [(4, 2), (8, 4), (2, 8), (4, 8)])
def test_no_ttt_rejects_unreviewed_world_ga_pairs_before_runtime(
    tmp_path, monkeypatch, world_size, ga
):
    args = _args(
        tmp_path,
        "--no-ttt",
        "--job-name",
        f"no_ttt_world{world_size}_ga{ga}",
        "--world-size",
        str(world_size),
        "--ga",
        str(ga),
    )

    def forbidden():
        pytest.fail("invalid profile reached CUDA/runtime preflight")

    monkeypatch.setattr(phase5, "_load_runtime", forbidden)
    with pytest.raises(ValueError, match="Matched No-TTT"):
        phase5.preflight(args)


def test_four_gpu_ga4_overlay_keeps_generation_optimizer_and_2048_global_budget(tmp_path):
    _, config, _, _ = _config(tmp_path)
    config.model.config.ema = SimpleNamespace(enabled=True)
    config.model.config.compile = SimpleNamespace(enabled=True)
    config.model.config.activation_checkpointing = SimpleNamespace(mode="selective")
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    args = _args(
        tmp_path, "--no-ttt", "--job-name", "no_ttt_packed128_4gpu",
        "--world-size", "4", "--ga", "4",
    )
    phase5.overlay_config(config, args, contract)
    assert config.model.config.local_memory_enabled is False
    assert config.trainer.type is MatchedNoTTTTrainer
    assert config.model.config.parallelism.data_parallel_shard_degree == 4
    assert config.model.config.parallelism.data_parallel_replicate_degree == 1
    assert config.trainer.grad_accum_iter == 4
    assert config.optimizer.keys_to_select == list(phase5.GENERATION_KEYS)
    assert config.optimizer.lr == 5e-5
    assert config.optimizer.lr_multipliers == {key: 5.0 for key in phase5.ACTION_KEYS}
    assert config.trainer.max_iter == 30000
    assert config.scheduler.cycle_lengths == [30000]
    assert config.scheduler.warm_up_steps == [500]
    assert config.checkpoint.save_iter == 100
    assert (4 * 8 * 16 * 4) == (8 * 8 * 16 * 2) == 2048


def test_missing_index_retains_original_value_error(tmp_path):
    args = _args(tmp_path, "--no-ttt", "--job-name", "no_ttt_formal")
    args.dataset_index_root = None
    for phase in ("fresh", "resume"):
        args.phase = phase
        with pytest.raises(ValueError, match="dataset-index-root"):
            phase5.preflight(args)


def test_optimizer_inventory_generation_only_and_frozen_understanding():
    model = nn.Module()
    model.config = SimpleNamespace(local_memory_enabled=False)
    model.net = nn.Module()
    model.net.moe_gen = nn.Parameter(torch.ones(1))
    model.net.action2llm = nn.Linear(1, 1, bias=False)
    model.net.understanding = nn.Parameter(torch.ones(1), requires_grad=False)
    optimizer = torch.optim.SGD([model.net.moe_gen, model.net.action2llm.weight], lr=5e-5)
    report = phase5.validate_optimizer_inventory(model, optimizer, no_ttt=True)
    assert report["local_tensors"] == report["local_params"] == 0
    assert report["generation_tensors"] == 2
    with pytest.raises(ValueError, match="inventory"):
        phase5.validate_optimizer_inventory(model, torch.optim.SGD([model.net.moe_gen], lr=5e-5), no_ttt=True)
    model.net.understanding.requires_grad_(True)
    with pytest.raises(ValueError, match="冻结"):
        phase5.validate_optimizer_inventory(model, optimizer, no_ttt=True)


def test_no_ttt_metadata_describes_real_producer_and_no_fast_state(tmp_path):
    _, config, _, _ = _config(tmp_path)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    phase5.synchronize_legacy_chunk_length(config, contract)
    metadata = phase5.describe_effective_data_source(config, contract, ttt_steps=16, local_memory_enabled=False)
    assert metadata["local_ttt_tbptt_steps"] is None and metadata["consumer_block_steps"] == 16
    assert metadata["producer_chain"][-1].endswith("MatchedNoTTTTrainer.bind_grouped_stream")
    assert NoTTTPlanObserver()._materialize_core(None)["local_memory_enabled"] is False


def test_real_composed_no_ttt_config_yaml_pickle_and_semantic_identity(tmp_path, monkeypatch):
    import json
    import pickle

    from cosmos_framework.utils.lazy_config import LazyConfig
    from examples import psm_wma_robocasa_native as native

    args = _args(tmp_path, "--no-ttt", "--job-name", "no_ttt_real_config")
    overrides = {
        "EDGE_POLICY_CHECKPOINT": str(args.edge),
        "WAN_VAE_PATH": str(args.vae),
        "ROBOCASA_ROOT": str(args.source_root),
        "BASE_CHECKPOINT_PATH": str(args.base_checkpoint),
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    config = phase5.load_experiment_from_toml(native.RECIPE)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    phase5.overlay_config(config, args, contract)
    metadata = phase5.describe_effective_data_source(config, contract, ttt_steps=16, local_memory_enabled=False)
    args.dataset_index_root.mkdir()
    (args.dataset_index_root / "receipt.json").write_text("synthetic read-only receipt")
    report = {
        key: "synthetic"
        for key in (
            "cache_manifest_sha256",
            "cache_corpus_digest",
            "source_binding_digest",
            "model_witnesses",
            "transform_rng_mode",
        )
    }
    identity = phase5.training_identity(args, config, report)
    json.dumps(identity, allow_nan=False)
    assert identity["training_profile"] == "matched_no_ttt_v1"
    assert identity["contract"]["h_pred"] == 16 and identity["contract"]["b"] == 8
    with monkeypatch.context() as env:
        env.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "output"))
        LazyConfig.save_yaml(config, str(tmp_path / "config.yaml"))
        LazyConfig.save_pkl(config, str(tmp_path / "config.pkl"))
        phase5.validate_saved_yaml(tmp_path / "config.yaml", metadata)
        saved = pickle.loads((tmp_path / "config.pkl").read_bytes())
        assert saved.model.config.local_memory_enabled is False
        assert saved.trainer.type is MatchedNoTTTTrainer
        assert list(saved.optimizer.keys_to_select) == list(phase5.GENERATION_KEYS)
    assert config.model.config.vlm_config.tokenizer.repository is None
    assert config.model.config.vlm_config.pretrained_weights.enabled is False
    assert config.model.config.diffusion_expert_config.load_weights_from_pretrained is False
    assert config.checkpoint.load_training_state is False


def test_no_ttt_preflight_keeps_official_droid_gate_before_any_data_or_model(tmp_path, monkeypatch):
    args = _args(tmp_path, "--no-ttt", "--job-name", "no_ttt_wrong_init")
    args.base_checkpoint = tmp_path / "G1/checkpoints/iter_000012000"
    seen = []
    monkeypatch.setattr(phase5, "verify_root_child_lock", lambda _: {})
    monkeypatch.setattr(phase5, "check_edge_checkpoint", lambda _: None)

    def reject(path):
        seen.append(path)
        raise ValueError("official DROID initialization required")

    monkeypatch.setattr(phase5, "check_droid_dcp", reject)
    with pytest.raises(ValueError, match="official DROID"):
        phase5.preflight(args)
    assert seen == [args.base_checkpoint]
    assert not phase5.job_namespace(args).exists()

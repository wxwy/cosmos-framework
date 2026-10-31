"""Matched profile and unchanged G1 configuration/identity contracts on CPU."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy_test import _catalog
from cosmos_framework.trainer.matched_no_ttt import MatchedNoTTTTrainer
from examples import psm_wma_robocasa_corrected_phase5 as phase5
from examples.psm_wma_robocasa_corrected_phase5_test import _config
from examples.psm_wma_robocasa_matched_no_ttt import NoTTTPlanObserver
from examples.psm_wma_robocasa_phase5_f0_test import Serializer, _args, _report
from examples.psm_wma_robocasa_phase5_f0_test import _config as lifecycle_config


def _structured_identity_fixture(value):
    """Use supported structured config records; never relax production identity."""
    if isinstance(value, SimpleNamespace):
        values = {key: _structured_identity_fixture(item) for key, item in vars(value).items()}
        record = make_dataclass("IdentityConfigFixture", [(key, object) for key in values])
        return record(**values)
    if isinstance(value, dict):
        return {key: _structured_identity_fixture(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_structured_identity_fixture(item) for item in value)
    return value


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


def test_all_worker_counts_share_strict_training_and_config_identity(tmp_path):
    """Loader processes only change launch/runtime telemetry, never model/data state identity."""
    catalog, config, _, _ = _config(tmp_path, ga=4)
    config.model.config.local_memory_enabled = False
    config.model.config.parallelism.data_parallel_shard_degree = 4
    config.optimizer.keys_to_select = list(phase5.GENERATION_KEYS)
    config.checkpoint.strict_resume = True
    config.data_setting = SimpleNamespace(seed=123)
    config = _structured_identity_fixture(config)
    args = _args(
        tmp_path,
        "--no-ttt",
        "--world-size",
        "4",
        "--ga",
        "4",
        "--job-name",
        "no_ttt_packed128_droid_fresh_30k_world4_ga4",
    )
    args.dataset_index_root.mkdir(parents=True)
    (args.dataset_index_root / "receipt.json").write_text("{}")
    report = {
        "cache_manifest_sha256": catalog.manifest_digest,
        "cache_corpus_digest": catalog.cache_corpus_digest,
        "source_binding_digest": catalog.source_binding_digest,
        "model_witnesses": {"droid": "synthetic"},
        "transform_rng_mode": "legacy_global",
    }
    digest = phase5.config_digest(catalog, config, b_stream=8, active_ga=4, witnesses=report["model_witnesses"])
    baseline = phase5.training_identity(args, config, report)
    frozen = phase5._json_digest({"phase5_digest": digest, "training_identity": baseline})
    assert "no_ttt_cpu_loader" not in baseline
    for workers, factor, persistent, timeout in (
        (0, None, None, None),
        (1, 1, True, 120),
        (2, 2, False, 30),
        (4, 1, True, 90),
        (8, 2, True, 180),
    ):
        args.no_ttt_loader_workers = workers
        args.no_ttt_prefetch_factor = factor
        args.no_ttt_persistent_workers = persistent
        args.no_ttt_loader_timeout_s = timeout
        current = phase5.training_identity(args, config, report)
        assert current == baseline
        assert phase5._json_digest({"phase5_digest": digest, "training_identity": current}) == frozen
    config.optimizer.lr *= 2
    assert phase5.training_identity(args, config, report) != baseline


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
    with pytest.raises(
        ValueError, match=r"^G1 Resume requires unchanged formal T16/B8/GA2/K4/world8 and 30000/500/100$"
    ):
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
def test_both_matched_no_ttt_profiles_pass_geometry_gate_before_real_assets(tmp_path, monkeypatch, world_size, ga):
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
def test_no_ttt_rejects_unreviewed_world_ga_pairs_before_runtime(tmp_path, monkeypatch, world_size, ga):
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


def _owner_nott4_args(tmp_path):
    return _args(
        tmp_path,
        "--no-ttt",
        "--job-name",
        "no_ttt_packed128_droid_fresh_30k_world4_ga4",
        "--world-size",
        "4",
        "--ga",
        "4",
    )


def test_nottt4_root_and_child_git_checks_are_fully_skipped_on_preflight(tmp_path, monkeypatch):
    args = _owner_nott4_args(tmp_path)
    args.expected_root = None
    args.expected_child = None

    def forbidden_git(*_args, **_kwargs):
        pytest.fail("the Owner-exempt 4GPU No-TTT job must never invoke Git")

    monkeypatch.setattr(phase5.subprocess, "run", forbidden_git)
    lock = phase5.verify_root_child_lock(args)
    assert lock["root"] == lock["child"] == lock["gitlink"] == phase5.NOTTT4_UNCHECKED_SOURCE
    assert lock["allowed_root_noncode_changes"] is None
    assert lock["source_validation"] == "owner_disabled_for_nottt_world4_ga4"

    # Prove the real preflight reaches the official Edge asset gate without
    # running any Git status/HEAD command even with no Git metadata available.
    monkeypatch.setattr(phase5, "_load_runtime", lambda: None)

    def reached_edge(_path):
        raise ValueError("Reached Edge identity validation")

    monkeypatch.setattr(phase5, "check_edge_checkpoint", reached_edge)
    with pytest.raises(ValueError, match="Reached Edge identity validation"):
        phase5.preflight(args)
    assert not phase5.job_namespace(args).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("no_ttt", False),
        ("job_name", "no_ttt_world8"),
        ("job_name", "packed128_droid_fresh_g1_20261019"),
        ("world_size", 8),
        ("ga", 2),
        ("num_workers", 1),
        ("full_segment_prefetch", True),
        ("transform_rng_mode", "consumer_keyed_v1"),
    ],
)
def test_nottt4_source_check_bypass_never_leaks_to_other_jobs_or_profiles(tmp_path, monkeypatch, field, value):
    args = _owner_nott4_args(tmp_path)
    assert phase5._nottt4_source_gate_disabled(args)
    setattr(args, field, value)
    assert not phase5._nottt4_source_gate_disabled(args)
    assert phase5._expected_source_identity(args) == (args.expected_root, args.expected_child)

    def must_reach_strict_git(*_args, **_kwargs):
        raise RuntimeError("Strict Git checks still enabled for this profile")

    monkeypatch.setattr(phase5.subprocess, "run", must_reach_strict_git)
    with pytest.raises(RuntimeError, match="Strict Git checks still enabled"):
        phase5.verify_root_child_lock(args)


def test_other_nottt4_job_retains_strict_identity_validation(tmp_path):
    args = _owner_nott4_args(tmp_path)
    args.job_name = "no_ttt_other_world4_ga4"
    args.expected_root = "stale-root-sha"
    args.expected_child = "stale-child-sha"
    identity = {
        "training_profile": "matched_no_ttt_v1",
        "root": phase5.NOTTT4_UNCHECKED_SOURCE,
        "child": phase5.NOTTT4_UNCHECKED_SOURCE,
        "contract": {"world_size": 4, "ga": 4},
    }
    report = _report(args, identity)
    lifecycle = phase5.JobLifecycle(args, report)
    lifecycle.write_config(lifecycle_config(args), Serializer)
    lifecycle.close()
    path = phase5.job_namespace(args)
    original = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    args.phase = "resume"
    args.expected_root = "another-root-value"
    args.expected_child = None
    assert phase5.check_job_namespace(args, identity)["semantic"] == identity
    assert {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()} == original
    wrong = deepcopy(identity)
    wrong["contract"]["ga"] = 8
    with pytest.raises(ValueError, match="semantic"):
        phase5.check_job_namespace(args, wrong)
    with (path / "config.yaml").open("a") as stream:
        stream.write("modified")
    with pytest.raises(ValueError, match="canonical config"):
        phase5.check_job_namespace(args, identity)


@pytest.mark.parametrize("historical_root_child", [True, False])
def test_existing_formal_nottt4_resume_skips_job_identity_without_rewriting_checkpoint(tmp_path, historical_root_child):
    """Legacy real SHA and newer sentinel jobs reuse original digest, regardless of nw."""
    import json

    args = _owner_nott4_args(tmp_path)
    frozen_root = "215305ecb9ab9631af44cbb3469bc4179dd4aeab"
    frozen_child = "a4bc0c728b57ee8aaf49f951f6aed1a17b986573"
    if not historical_root_child:
        frozen_root = frozen_child = phase5.NOTTT4_UNCHECKED_SOURCE
    frozen = {
        "training_profile": "matched_no_ttt_v1",
        "root": frozen_root,
        "child": frozen_child,
        "contract": {"world_size": 4, "ga": 4},
    }
    report = _report(args, frozen)
    lifecycle = phase5.JobLifecycle(args, report)
    lifecycle.write_config(lifecycle_config(args), Serializer)
    lifecycle.close()
    job = phase5.job_namespace(args)
    identity_path = job / phase5.JOB_IDENTITY
    record = json.loads(identity_path.read_text())
    # Reproduce the actual job format created before the SHA-exemption fix.
    record["root"], record["child"] = frozen_root, frozen_child
    identity_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    before = {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()}

    args.phase = "resume"
    args.no_ttt_loader_workers = 8
    args.no_ttt_prefetch_factor = 1
    args.no_ttt_persistent_workers = True
    args.no_ttt_loader_timeout_s = 120
    current_runtime = {
        **frozen,
        "root": phase5.NOTTT4_UNCHECKED_SOURCE,
        "child": phase5.NOTTT4_UNCHECKED_SOURCE,
    }
    read = phase5.check_job_namespace(args, current_runtime)
    assert read["root"] == frozen_root and read["child"] == frozen_child
    runtime_report = {}
    selected = phase5._resume_semantic_without_nottt4_job_identity(args, read, current_runtime, runtime_report)
    assert selected == frozen
    assert runtime_report["nottt4_legacy_source_resume"]["identity_gate"].startswith("disabled_")
    old_digest = phase5._json_digest({"phase5_digest": "a" * 64, "training_identity": frozen})
    resumed_digest = phase5._json_digest({"phase5_digest": "a" * 64, "training_identity": selected})
    assert resumed_digest == old_digest
    if historical_root_child:
        assert old_digest != phase5._json_digest({"phase5_digest": "a" * 64, "training_identity": current_runtime})
    assert {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()} == before


def test_formal_nottt4_resume_only_requires_readable_frozen_semantic(tmp_path):
    import json

    args = _owner_nott4_args(tmp_path)
    frozen = {"training_profile": "matched_no_ttt_v1", "root": "old", "child": "old"}
    lifecycle = phase5.JobLifecycle(args, _report(args, frozen))
    lifecycle.write_config(lifecycle_config(args), Serializer)
    lifecycle.close()
    path = phase5.job_namespace(args) / phase5.JOB_IDENTITY
    record = json.loads(path.read_text())
    record["schema"] = "legacy_schema"
    record["root"] = "old"
    record["child"] = "old"
    record["semantic_digest"] = "historical-different"
    path.write_text(json.dumps(record))
    args.phase = "resume"
    assert phase5.check_job_namespace(args, {"root": "new"})["semantic"] == frozen
    record["semantic"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="readable frozen training state"):
        phase5.check_job_namespace(args)


def test_four_gpu_ga4_overlay_keeps_generation_optimizer_and_2048_global_budget(tmp_path):
    _, config, _, _ = _config(tmp_path)
    config.model.config.ema = SimpleNamespace(enabled=True)
    config.model.config.compile = SimpleNamespace(enabled=True)
    config.model.config.activation_checkpointing = SimpleNamespace(mode="selective")
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path / "cache"))
    args = _args(
        tmp_path,
        "--no-ttt",
        "--job-name",
        "no_ttt_packed128_4gpu",
        "--world-size",
        "4",
        "--ga",
        "4",
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

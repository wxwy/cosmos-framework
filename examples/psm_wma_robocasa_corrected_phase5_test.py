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

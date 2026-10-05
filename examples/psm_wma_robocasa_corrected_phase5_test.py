"""Corrected Phase5 scratch launcher 的 CPU 合同。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
        max_action_dim=64,
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
    )
    catalog = SimpleNamespace(manifest_digest="manifest", cache_corpus_digest="corpus", source_binding_digest="source")
    return catalog, config, b, ga


def test_digest_path_invariance_and_semantic_sensitivity(tmp_path: Path) -> None:
    catalog, config, b, ga = _config(tmp_path / "host_a")
    base = phase5.config_digest(catalog, config, b_stream=b, active_ga=ga)
    moved_catalog, moved_config, _, _ = _config(tmp_path / "host_b")
    assert phase5.config_digest(moved_catalog, moved_config, b_stream=b, active_ga=ga) == base
    for field, value in (
        ("manifest_digest", "changed"),
        ("cache_corpus_digest", "changed"),
        ("source_binding_digest", "changed"),
    ):
        changed = SimpleNamespace(**vars(catalog))
        setattr(changed, field, value)
        assert phase5.config_digest(changed, config, b_stream=b, active_ga=ga) != base
    for kwargs in ({"t": 32}, {"b": 3}, {"ga": 3}, {"k": 1}, {"k": 8}):
        changed, changed_config, changed_b, changed_ga = _config(tmp_path, **kwargs)
        assert phase5.config_digest(changed, changed_config, b_stream=changed_b, active_ga=changed_ga) != base
    config.optimizer.weight_decay = 0.1
    assert phase5.config_digest(catalog, config, b_stream=b, active_ga=ga) != base


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
    assert report["generation_tensors"] == 13
    assert not model.net.reasoner.weight.requires_grad
    missing = torch.optim.AdamW(allowed[:-1])
    with pytest.raises(ValueError, match="inventory"):
        phase5.validate_optimizer_inventory(model, missing)
    extra = torch.optim.AdamW([*allowed, model.net.reasoner.weight])
    with pytest.raises(ValueError, match="inventory"):
        phase5.validate_optimizer_inventory(model, extra)

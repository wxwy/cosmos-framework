# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU configuration/entry/factory checks with explicit synthetic G1 metadata."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from cosmos_framework.data.generator.action.utils.static_task_text_test import recipe_fixture
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig
from cosmos_framework.trainer.g2_training_test import TinyPolicy, equal_tree, plan_fixture
from examples import psm_wma_robocasa_corrected_phase5 as phase5
from examples.psm_wma_robocasa_g2_recipe import G2RunRecipe, install_g2_training, recipe_from_args


def run_recipe_fixture(tmp_path, profile="R2"):
    local = dict(
        local_memory_enabled=True,
        local_memory_inner_lr=0.075,
        local_memory_ttt_tbptt_steps=16,
        local_memory_k_local=4,
        local_memory_dim=32,
        local_memory_evidence_dim=256,
        local_memory_action_dim=15,
        local_memory_ttt_dim=64,
        local_memory_fast_hidden_dim=256,
        rectified_flow_training_config={"loss_scale": 10.0, "image_loss_scale": None},
    )
    optimizer = dict(optimizer_type="AdamW", betas=[0.91, 0.99], eps=1e-8, fused=True, weight_decay=0.07)
    snapshot = tmp_path / "synthetic_g1_config.yaml"
    snapshot.write_text(yaml.safe_dump(dict(model=dict(config=local), optimizer=optimizer, trainer=dict(callbacks={}))))
    text = recipe_fixture()
    plan = replace(plan_fixture(profile), text_recipe_identity=text.identity)
    return G2RunRecipe(
        plan,
        text,
        GlobalMemoryConfig(enabled=profile == "R2"),
        "fixture-alignment",
        "fixture-view-recipe",
        "fixture_provider:open_provider",
        {},
        str(snapshot),
        hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        5,
    )


def test_required_recipe_roundtrip_and_incomplete_formal_run_refused_before_runtime(tmp_path, monkeypatch):
    recipe = run_recipe_fixture(tmp_path)
    path = tmp_path / "g2.json"
    path.write_text(json.dumps(asdict(recipe)))
    assert G2RunRecipe.open(path) == recipe
    path.write_text('{"training": {"profile":"R2"}}')
    monkeypatch.setattr(phase5, "_load_runtime", lambda: pytest.fail("invalid G2 config reached project runtime"))
    with pytest.raises(ValueError, match="incomplete"):
        phase5.main(
            [
                "--phase",
                "fresh",
                "--g2-recipe",
                str(path),
                "--job-name",
                "r2_g2_fixture",
                "--output-root",
                str(tmp_path),
                "--source-root",
                str(tmp_path),
                "--cache-root",
                str(tmp_path),
                "--edge",
                str(tmp_path),
                "--vae",
                str(tmp_path),
                "--base-checkpoint",
                str(tmp_path),
                "--root-worktree",
                str(tmp_path),
            ]
        )


def test_recipe_derives_run_geometry_only_from_explicit_plan_and_g1_snapshot(tmp_path):
    recipe = run_recipe_fixture(tmp_path)
    path = tmp_path / "g2.json"
    path.write_text(json.dumps(asdict(recipe)))
    args = SimpleNamespace(g2_recipe=path, job_name="r2_g2_cpu_fixture")
    assert recipe_from_args(args) == recipe
    assert (args.max_iter, args.b, args.ga, args.world_size) == (4, 1, 1, 1)
    assert (args.t, args.k) == (16, 4)
    assert recipe_from_args(SimpleNamespace()) is None
    with pytest.raises(ValueError, match="separate"):
        recipe.configure_args(SimpleNamespace(job_name="old_g1_job"))


def test_overlay_preserves_g1_local_loss_and_optimizer_low_level_settings(tmp_path):
    recipe = run_recipe_fixture(tmp_path)
    model = SimpleNamespace(global_memory_enabled=False, global_memory_config={})
    config = SimpleNamespace(
        model=SimpleNamespace(config=model),
        optimizer=SimpleNamespace(),
        trainer=SimpleNamespace(),
        checkpoint=SimpleNamespace(load_training_state=False),
    )
    recipe.overlay(config)
    assert model.local_memory_inner_lr == 0.075  # Never replace it by the historical 0.1 default.
    assert model.rectified_flow_training_config == {"loss_scale": 10.0, "image_loss_scale": None}
    assert config.optimizer.betas == [0.91, 0.99] and config.optimizer.weight_decay == 0.07
    assert "local_memory" not in config.optimizer.keys_to_select
    assert config.checkpoint.keys_to_skip_loading == ["net_ema."]
    assert model.global_memory_enabled is True
    assert config.optimizer.lr == 1e-5


def test_overlay_preserves_structured_rectified_flow_type_and_g1_loss(tmp_path):
    from cosmos_framework.configs.base.defaults.model_config import (
        RectifiedFlowTrainingConfig,
    )

    recipe = run_recipe_fixture(tmp_path)
    model = OmegaConf.create(
        {
            "global_memory_enabled": False,
            "global_memory_config": {},
            "g2_text_enabled": False,
            "global_memory_top_variant": "with_robot",
            "rectified_flow_training_config": OmegaConf.structured(
                RectifiedFlowTrainingConfig
            ),
        }
    )
    assert OmegaConf.get_type(model.rectified_flow_training_config) is RectifiedFlowTrainingConfig
    config = SimpleNamespace(
        model=SimpleNamespace(config=model),
        optimizer=SimpleNamespace(),
        trainer=SimpleNamespace(),
        checkpoint=SimpleNamespace(load_training_state=False),
    )
    recipe.overlay(config)
    assert OmegaConf.get_type(model.rectified_flow_training_config) is RectifiedFlowTrainingConfig
    assert model.rectified_flow_training_config.loss_scale == 10.0
    assert model.rectified_flow_training_config.action_loss_weight == 10.0
    assert model.rectified_flow_training_config.image_loss_scale is None
    assert model.global_memory_enabled is True
    assert config.optimizer.lr == 1e-5


def test_recipe_refuses_missing_b_interface_fixture_provider_or_producer_import(tmp_path, monkeypatch):
    recipe = run_recipe_fixture(tmp_path)
    with pytest.raises(ValueError, match="B Global model config"):
        recipe.overlay(SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace())))
    with pytest.raises(ValueError, match="producer"):
        replace(recipe, provider_factory="g2.offline.stage_a:produce")
    module = SimpleNamespace(open_provider=lambda **kwargs: SimpleNamespace(evidence_scope="fixture"))
    monkeypatch.setattr("examples.psm_wma_robocasa_g2_recipe.importlib.import_module", lambda _: module)
    with pytest.raises(ValueError, match="fixtures are CPU only"):
        recipe.data_adapter()


def test_factory_hook_reuses_optimizer_and_reports_actual_mutually_exclusive_inventory(tmp_path):
    recipe = run_recipe_fixture(tmp_path)
    model, trainer, report = TinyPolicy(), SimpleNamespace(g2_evidence_dir=tmp_path), {}
    built = []

    def original(optimizer_config, scheduler_config):
        selected = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(selected, lr=1e-5, weight_decay=0.07)
        built.append(optimizer)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    model.init_optimizer_scheduler = original
    install_g2_training(model, trainer, recipe, report)
    optimizer, scheduler = model.init_optimizer_scheduler(None, None)
    assert built == [optimizer] and trainer._g2_stage_controller is scheduler
    assert not any(name.startswith("net.local_memory") for name in scheduler.roles)
    inventory = report["g2_actual_parameter_inventory"]
    assert len(inventory) == len(list(model.named_parameters()))
    saved = json.loads((tmp_path / "g2_parameter_inventory_before_load_rank_0.json").read_text())
    equal_tree(saved["parameters"], inventory)
    assert saved["scope"] == "actual_runtime_before_checkpoint_load"
    assert set(model._g2_new_global_state_names) == {row["name"] for row in inventory if row["group"] == "global"}


def test_b_actual_inventory_covers_gen_text_norm_and_native_factory(tmp_path):
    from torch import nn

    from cosmos_framework.model.generator.mot.g2_core_test import tiny_network
    from cosmos_framework.model.generator.mot.g2_parameters import parameter_inventory, set_trainable_names
    from cosmos_framework.trainer.g2_training import G2_OPTIMIZER_SELECTION, build_g2_optimizer

    owner = nn.Module()
    owner.net = tiny_network(g2_text=True)
    owner.net.global_memory_reader = nn.Linear(4, 4)
    owner.net.global_memory_projector = nn.Linear(4, 16)
    owner.net.action2llm = nn.Linear(4, 16)
    owner.g2_parameter_inventory = lambda: parameter_inventory(owner, net_prefix="net.")
    owner.set_trainable_names = lambda names: set_trainable_names(owner, names)

    def native_factory(config, schedule):
        optimizer = build_g2_optimizer(
            owner,
            optimizer_type="adamw",
            lr=1e-5,
            fused=True,
            keys_to_select=[G2_OPTIMIZER_SELECTION],
            weight_decay=0.07,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer.optimizers[0], lambda _: 1.0)

    owner.init_optimizer_scheduler = native_factory
    trainer = SimpleNamespace(g2_evidence_dir=tmp_path)
    report = {}
    install_g2_training(owner, trainer, run_recipe_fixture(tmp_path), report)
    optimizer, scheduler = owner.init_optimizer_scheduler(None, None)
    names = [name for name in scheduler.roles if "k_norm_und_for_gen" in name]
    assert names and all(scheduler.roles[name].value == "gen" for name in names)
    selected = {id(p) for inner in optimizer.optimizers for group in inner.param_groups for p in group["params"]}
    assert all(id(dict(owner.named_parameters())[name]) in selected for name in names)
    assert all(not p.requires_grad for name, p in owner.named_parameters() if name in names)


def test_actual_a_reader_b_mot_train_a_gradient_and_frozen_parameters(tmp_path, monkeypatch):
    """Real A Reader/B MoT components, CPU reference kernels; synthetic loss."""
    from torch import nn

    import cosmos_framework.model.generator.mot.attention as attention_module
    import cosmos_framework.model.generator.mot.global_memory as reader_module
    from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
    from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
    from cosmos_framework.model.generator.mot.g2_core_test import TinyLM, cpu_attention, native_pack, tiny_model
    from cosmos_framework.model.generator.mot.g2_parameters import parameter_inventory, set_trainable_names
    from cosmos_framework.model.generator.mot.global_memory_test import cpu_reference
    from cosmos_framework.model.generator.mot.memory_prefix import build_joint_memory_prefix_context
    from cosmos_framework.trainer.g2_training import G2_OPTIMIZER_SELECTION, build_g2_optimizer

    monkeypatch.setattr(attention_module, "attention", cpu_attention)
    monkeypatch.setattr(reader_module, "_dispatch_attention", cpu_reference)
    language_model = tiny_model()
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language_model.config, vision_gen=False, global_memory_enabled=True, global_memory_latent_channels=4
    )
    owner = nn.Module()
    owner.net = Cosmos3VFMNetwork(TinyLM(language_model), config)
    owner.net.action2llm = nn.Linear(4, 16)  # Not used by this component-only synthetic objective.
    owner.g2_parameter_inventory = lambda: parameter_inventory(owner, net_prefix="net.")
    owner.set_trainable_names = lambda names: set_trainable_names(owner, names)

    def factory(config, schedule):
        optimizer = build_g2_optimizer(
            owner,
            optimizer_type="adamw",
            lr=1e-5,
            fused=True,
            keys_to_select=[G2_OPTIMIZER_SELECTION],
            weight_decay=0.07,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer.optimizers[0], lambda _: 1.0)

    owner.init_optimizer_scheduler = factory
    trainer, report = SimpleNamespace(), {}
    install_g2_training(owner, trainer, run_recipe_fixture(tmp_path), report)
    optimizer, controller = owner.init_optimizer_scheduler(None, None)
    before = {name: value.detach().clone() for name, value in owner.state_dict().items()}
    pack, mask, positions, text = native_pack(language_model, (3,))
    features = owner.net.prepare_und_text(text)
    queries = owner.net.prepare_task_queries(features)
    reader = owner.net.read_global(queries, ((synthetic_view(),),))
    projected = owner.net.project_global(reader)
    prefix = build_joint_memory_prefix_context(tuple(projected.unbind(0)), None, target_dtype=torch.float32)
    result, _ = language_model(pack, mask, positions, prepared_text=features, memory_prefix_context=prefix)
    (get_gen_seq(result) * torch.arange(1, 17, dtype=torch.float32)).sum().backward()
    inventory = controller.inventory()
    named = dict(owner.named_parameters())
    for row in inventory:
        gradient = named[row["name"]].grad
        row["gradient_present"] = gradient is not None
        row["gradient_norm"] = None if gradient is None else float(gradient.norm())
        if gradient is not None:
            assert torch.isfinite(gradient).all(), row["name"]
        if row["group"] != "global":
            assert gradient is None
    assert any(
        row["gradient_norm"] is not None and row["gradient_norm"] > 0 for row in inventory if row["group"] == "global"
    )
    controller.prepare_commit(actual_consumers=1, iteration=0)
    optimizer.step()
    controller.step()
    optimizer.zero_grad(set_to_none=True)
    controller.commit_boundary(iteration=1, ga_complete=True, local_transaction_complete=True)
    for name, value in owner.state_dict().items():
        if not name.startswith(("net.global_memory_reader.", "net.global_memory_projector.")):
            assert torch.equal(value, before[name]), name
    output = Path(os.environ.get("G2_C_CPU_EVIDENCE_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "actual_a_reader_tiny_b_parameter_inventory.json").write_text(
        json.dumps(
            {
                "scope": "actual A Reader + B tiny MoT, synthetic CPU/reference attention; NOT full Edge/FSDP/action loss",
                "parameters": inventory,
                "stage_after_commit": controller.state_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )

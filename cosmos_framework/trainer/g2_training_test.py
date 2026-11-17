# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small CPU optimizers + real Local inner core; NOT actual Cosmos/FSDP."""

from __future__ import annotations

import copy
import io
import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cosmos_framework.checkpoint.g2_warm_start import load_g1_model_only
from cosmos_framework.model.generator.mot.g2_contracts import ParameterGroup, TrainingStage
from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore
from cosmos_framework.trainer.g2_training import (
    G2StageController,
    G2TrainingPlan,
    GroupSchedule,
    parameter_roles,
    validate_actual_comparison_budget,
    validate_comparison_plans,
)


class TinyPolicy(nn.Module):
    def __init__(self, global_enabled=True):
        super().__init__()
        self.net = nn.Module()
        if global_enabled:
            self.net.global_memory_reader = nn.Linear(4, 4)
            self.net.global_memory_projector = nn.Linear(4, 4)
        self.net.moe_gen = nn.Linear(4, 4)
        self.net.action2llm = nn.Linear(4, 4)
        self.net.llm2action = nn.Linear(4, 1)
        self.net.action_modality_embed = nn.Parameter(torch.ones(4))
        self.net.local_memory_runtime = nn.Module()
        self.net.local_memory_runtime.encoder = nn.Linear(4, 4)
        self.net.local_memory_runtime.core = ContinualTTTLocalMemoryCore(
            evidence_dim=4,
            local_dim=4,
            ttt_dim=3,
            fast_hidden_dim=5,
        )
        self.net.local_memory2llm = nn.Linear(4, 4)
        self.net.und = nn.Linear(4, 4)
        self.net.vae = nn.Linear(4, 4)

    def g2_parameter_inventory(self):
        roles = roles_fixture(self, "R2" if hasattr(self.net, "global_memory_reader") else "R1")
        groups = {role: [] for role in ("global", "gen", "action_adapter", "frozen")}
        for name, _ in self.named_parameters():
            groups[roles[name].value if name in roles else "frozen"].append(name)
        return {
            "parameters": {key: tuple(value) for key, value in groups.items()},
            "new_state_keys": tuple(
                name
                for name in self.state_dict()
                if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
            ),
        }

    def set_trainable_names(self, names):
        from cosmos_framework.model.generator.mot.g2_parameters import set_trainable_names

        set_trainable_names(self, names)

    def outer_loss(self, fast):
        value = torch.randn(2, 4)
        encoded = self.net.local_memory_runtime.encoder(value)
        local, fast = self.net.local_memory_runtime.core.step_many(encoded, fast, create_graph=True)
        global_value = (
            self.net.global_memory_projector(self.net.global_memory_reader(value))
            if hasattr(self.net, "global_memory_reader")
            else value
        )
        # Autograd MUST pass through frozen GEN in TRAIN_A.
        hidden = self.net.moe_gen(global_value + self.net.local_memory2llm(local.mean(dim=1)))
        hidden = torch.tanh(hidden + self.net.action2llm(value) + self.net.action_modality_embed)
        return self.net.llm2action(hidden).square().mean(), fast


def plan_fixture(profile="R2", *, a=2, b=2, target=7, world=1):
    return G2TrainingPlan(
        profile,
        "fixture-g1-model-sha",
        "fixture-data-c",
        "fixture-split",
        "fixture-text",
        "fixture-state15",
        a,
        b,
        target,
        1,
        1,
        world,
        GroupSchedule(1, "constant", 1.0, a + b) if profile == "R2" else None,
        GroupSchedule(min(2, b), "linear", 0.2, a + b),
        GroupSchedule(1, "constant", 1.0, a + b),
    )


def roles_fixture(model, profile="R2"):
    named = dict(model.named_parameters())
    return parameter_roles(
        model,
        profile=profile,
        global_names=frozenset(
            name for name in named if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
        ),
        gen_names=frozenset(
            name
            for name in named
            if any(
                name.startswith("net." + root)
                for root in (
                    "moe_gen.",
                    "action2llm.",
                    "llm2action.",
                    "action_modality_embed",
                )
            )
        ),
        action_names=frozenset(
            name
            for name in named
            if any(
                name.startswith("net." + root)
                for root in (
                    "action2llm.",
                    "llm2action.",
                    "action_modality_embed",
                )
            )
        ),
    )


def setup_fixture(plan=None):
    plan = plan or plan_fixture()
    model = TinyPolicy(plan.profile == "R2")
    roles = roles_fixture(model, plan.profile)
    # Production uses its existing optimizer factory; this CPU fixture uses
    # unfused AdamW with nonzero decay to detect accidental frozen updates.
    optimizer = torch.optim.AdamW(
        [parameter for name, parameter in model.named_parameters() if name in roles], lr=1e-5, weight_decay=0.3
    )
    scheduler = G2StageController(model, optimizer, plan, roles)
    live = {"cursor": 0, "frontier": ["same-episode"], "fast": model.net.local_memory_runtime.core.initial_state(2)}
    return model, optimizer, scheduler, live


def run_update(model, optimizer, scheduler, live, count):
    loss, fast = model.outer_loss(live["fast"])
    loss.backward()
    scheduler.prepare_commit(actual_consumers=count, iteration=scheduler.completed_steps)
    optimizer.step()
    scheduler.step()
    live["cursor"] += count
    live["fast"] = ContinualTTTFastState(*(value.detach().clone() for value in fast))
    optimizer.zero_grad(set_to_none=True)
    scheduler.commit_boundary(
        iteration=scheduler.completed_steps + 1, ga_complete=True, local_transaction_complete=True
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def test_train_a_global_gradient_through_frozen_gen_and_real_local_inner_update():
    model, optimizer, scheduler, live = setup_fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    fast_before = tuple(value.clone() for value in live["fast"])
    loss, fast = model.outer_loss(live["fast"])
    loss.backward()
    assert (
        model.net.global_memory_reader.weight.grad is not None
        and model.net.global_memory_reader.weight.grad.abs().sum() > 0
    )
    assert not model.net.moe_gen.weight.requires_grad and model.net.moe_gen.weight.grad is None
    assert any(not torch.equal(a, b) for a, b in zip(fast_before, fast, strict=True))
    assert all(parameter.grad is None for parameter in model.net.local_memory_runtime.parameters())
    scheduler.prepare_commit(actual_consumers=2, iteration=0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.commit_boundary(iteration=1, ga_complete=True, local_transaction_complete=True)
    for name, value in model.state_dict().items():
        if not name.startswith(("net.global_memory_reader.", "net.global_memory_projector.")):
            assert torch.equal(value, before[name]), name
    assert not torch.equal(model.net.global_memory_reader.weight, before["net.global_memory_reader.weight"])
    assert model.net.local_memory_runtime.core.inner_lr == 0.1
    with torch.no_grad(), pytest.raises(RuntimeError, match="grad mode"):
        model.outer_loss(live["fast"])


def test_switch_preserves_global_moments_and_external_rng_data_local_state():
    model, optimizer, scheduler, live = setup_fixture()
    run_update(model, optimizer, scheduler, live, 2)
    global_parameter = model.net.global_memory_reader.weight
    moment = optimizer.state[global_parameter]
    assert moment["step"].item() == 1
    # Execute the last A transaction explicitly so the switch itself is audited.
    loss, fast = model.outer_loss(live["fast"])
    loss.backward()
    scheduler.prepare_commit(actual_consumers=1, iteration=1)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    snapshot = copy.deepcopy(live)
    rng = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    moment_snapshot = copy.deepcopy(moment)
    scheduler.commit_boundary(iteration=2, ga_complete=True, local_transaction_complete=True)
    assert scheduler.stage == TrainingStage.TRAIN_B and optimizer.state[global_parameter] is moment
    equal_tree(moment_snapshot, moment)
    equal_tree(snapshot, live)
    equal_tree(rng, (torch.get_rng_state(), random.getstate(), np.random.get_state()))
    assert model.net.moe_gen.weight not in optimizer.state
    assert scheduler.group_updates == {"global": 2, "gen": 0, "action_adapter": 0}
    rates = {group["g2_role"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"global": 5e-5, "gen": 5e-6, "action_adapter": 5e-5}
    run_update(model, optimizer, scheduler, live, 3)
    assert optimizer.state[global_parameter]["step"].item() == 3
    assert optimizer.state[model.net.moe_gen.weight]["step"].item() == 1


@pytest.mark.parametrize("checkpoint_after", [1, 2, 3])
def test_checkpoint_resume_before_at_and_after_stage_boundary_is_exact(checkpoint_after):
    torch.manual_seed(8)
    model, optimizer, scheduler, live = setup_fixture()
    counts = [2, 1, 3, 1]
    for count in counts[:checkpoint_after]:
        run_update(model, optimizer, scheduler, live, count)
    buffer = io.BytesIO()
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            live=live,
            rng=(torch.get_rng_state(), random.getstate(), np.random.get_state()),
        ),
        buffer,
    )
    for count in counts[checkpoint_after:]:
        run_update(model, optimizer, scheduler, live, count)
    expected = copy.deepcopy(
        (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), live, torch.get_rng_state())
    )
    other_model, other_optimizer, other_scheduler, other_live = setup_fixture()
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    other_model.load_state_dict(saved["model"], strict=True)
    other_optimizer.load_state_dict(saved["optimizer"])
    other_scheduler.load_state_dict(saved["scheduler"])
    other_live = saved["live"]
    torch.set_rng_state(saved["rng"][0])
    random.setstate(saved["rng"][1])
    np.random.set_state(saved["rng"][2])
    for count in counts[checkpoint_after:]:
        run_update(other_model, other_optimizer, other_scheduler, other_live, count)
    equal_tree(
        expected,
        (
            other_model.state_dict(),
            other_optimizer.state_dict(),
            other_scheduler.state_dict(),
            other_live,
            torch.get_rng_state(),
        ),
    )


def test_warm_start_inherits_every_g1_tensor_including_local_and_only_initializes_global():
    g1, g2 = TinyPolicy(False), TinyPolicy(True)
    before = copy.deepcopy(g2.net.global_memory_reader.state_dict())
    names = frozenset(
        name
        for name in g2.state_dict()
        if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
    )
    report = load_g1_model_only(g2, g1.state_dict(), new_global_names=names)
    assert report["local_loaded_names"] and report["model_only"]
    for name, value in g1.state_dict().items():
        assert torch.equal(value, g2.state_dict()[name]), name
    equal_tree(before, g2.net.global_memory_reader.state_dict())


@pytest.mark.parametrize("kind", ["missing_local", "shape", "unexpected", "global_allowlist"])
def test_warm_start_refuses_missing_existing_or_wrong_shape_without_mutation(kind):
    g1, g2 = TinyPolicy(False), TinyPolicy(True)
    source = dict(g1.state_dict())
    names = frozenset(
        name
        for name in g2.state_dict()
        if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
    )
    if kind == "missing_local":
        source.pop("net.local_memory_runtime.core.w0_fast_in_weight")
    elif kind == "shape":
        source["net.moe_gen.weight"] = torch.ones(5, 5)
    elif kind == "unexpected":
        source["net.unrecognized.weight"] = torch.ones(1)
    else:
        names |= frozenset(("net.nonexistent_global.weight",))
    before = copy.deepcopy(g2.state_dict())
    with pytest.raises(ValueError):
        load_g1_model_only(g2, source, new_global_names=names)
    equal_tree(before, g2.state_dict())


def test_parameter_partition_no_omissions_duplicates_or_frozen_weight_decay():
    model, optimizer, scheduler, live = setup_fixture()
    inventory = scheduler.inventory()
    assert len(inventory) == len(list(model.named_parameters()))
    selected = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(selected) == len({id(p) for p in selected}) == len(scheduler.roles)
    assert all(group["weight_decay"] == 0 for group in optimizer.param_groups if group["g2_role"] != "global")
    model.net.und.weight.grad = torch.zeros_like(model.net.und.weight)
    with pytest.raises(ValueError, match="stale gradient"):
        scheduler.prepare_commit(actual_consumers=1, iteration=0)
    model.net.und.weight.grad = None
    bad_optimizer = torch.optim.AdamW(selected[:-1], lr=1e-5)
    with pytest.raises(ValueError, match="omissions"):
        G2StageController(model, bad_optimizer, scheduler.plan, scheduler.roles)


def test_stage_resume_rejects_missing_or_inconsistent_clocks_and_budget():
    model, optimizer, scheduler, live = setup_fixture()
    state = scheduler.state_dict()
    for key in state:
        bad = copy.deepcopy(state)
        bad.pop(key)
        with pytest.raises(ValueError):
            scheduler.load_state_dict(bad)
    bad = copy.deepcopy(state)
    bad["stage"] = "TRAIN_B"
    with pytest.raises(ValueError, match="clocks"):
        scheduler.load_state_dict(bad)
    with pytest.raises(ValueError, match="budget"):
        scheduler.prepare_commit(actual_consumers=8, iteration=0)
    with pytest.raises(RuntimeError, match="committed"):
        scheduler.commit_boundary(iteration=1, ga_complete=False, local_transaction_complete=True)


def test_plan_has_no_guessed_formal_defaults_and_r1_budget_checkpoints():
    with pytest.raises(ValueError, match="incomplete"):
        G2TrainingPlan.from_dict({"profile": "R2"})
    with pytest.raises(ValueError, match="historical"):
        replace(plan_fixture(), profile="R0")
    plan = plan_fixture("R1")
    assert plan.required_checkpoint_iterations == (2, 4)
    assert G2TrainingPlan.from_dict(asdict(plan)) == plan
    model, optimizer, scheduler, live = setup_fixture(plan)
    assert not hasattr(model.net, "global_memory_reader") and scheduler.stage == TrainingStage.R1
    assert all(role != ParameterGroup.GLOBAL for role in scheduler.roles.values())
    missing_horizon = asdict(plan)
    missing_horizon["gen_schedule"].pop("horizon_steps")
    with pytest.raises(ValueError, match="incomplete"):
        G2TrainingPlan.from_dict(missing_horizon)


def test_existing_optimizer_container_preserves_factory_decay_and_dcp_state_roundtrip():
    from cosmos_framework.utils.generator.optimizer import build_optimizer

    def build():
        model = TinyPolicy()
        roles = roles_fixture(model)
        optimizer = build_optimizer(
            model,
            optimizer_type="adamw",
            lr=1e-5,
            fused=True,
            weight_decay=0.07,
            keys_to_select=[name.removeprefix("net.") for name in roles],
            weight_decay_skip_patterns=[r"action2llm", r"llm2action"],
        )
        controller = G2StageController(
            model, optimizer, plan_fixture(), roles, set_trainability=model.set_trainable_names
        )
        return model, optimizer, controller, {"cursor": 0, "fast": model.net.local_memory_runtime.core.initial_state(2)}

    model, optimizer, controller, live = build()
    run_update(model, optimizer, controller, live, 2)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    saved_optimizer = copy.deepcopy(optimizer.state_dict())  # Real Torch DCP state-dict utilities.
    for name, before in frozen.items():
        assert torch.equal(before, dict(model.named_parameters())[name])
    other, other_optimizer, other_controller, _ = build()
    other.load_state_dict(model.state_dict(), strict=True)
    other_optimizer.load_state_dict(saved_optimizer)
    other_controller.load_state_dict(controller.state_dict())
    equal_tree(optimizer.state_dict(), other_optimizer.state_dict())
    for inner in other_optimizer.optimizers:
        for group in inner.param_groups:
            if group["g2_role"] != "global":
                assert group["weight_decay"] == 0.0


def test_export_actual_cpu_fixture_parameter_inventory(tmp_path):
    model, optimizer, controller, live = setup_fixture()
    stage_a = controller.inventory()
    run_update(model, optimizer, controller, live, 2)
    run_update(model, optimizer, controller, live, 1)
    stage_b = controller.inventory()
    assert {row["name"] for row in stage_a} == {row["name"] for row in stage_b} == set(dict(model.named_parameters()))
    output = Path(os.environ.get("G2_C_CPU_EVIDENCE_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "actual_cpu_fixture_parameter_inventory.json").write_text(
        json.dumps(
            {
                "scope": "actual named_parameters of synthetic CPU model; NOT full Edge or FSDP",
                "TRAIN_A": stage_a,
                "TRAIN_B": stage_b,
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_r1_r2_comparison_rejects_input_drift_and_nominal_budget_claims():
    r1, r2 = plan_fixture("R1"), plan_fixture("R2")
    assert validate_comparison_plans(r1, r2)["target_effective_consumers"] == 7
    with pytest.raises(ValueError, match="mismatch"):
        validate_comparison_plans(replace(r1, split_identity="other-split"), r2)
    runs = []
    for plan in (r1, r2):
        model, optimizer, controller, live = setup_fixture(plan)
        for count in (2, 1, 3, 1):
            run_update(model, optimizer, controller, live, count)
        runs.append(controller.state_dict())
    comparison = validate_actual_comparison_budget(*runs)
    assert comparison == {
        "effective_consumers": 7,
        "r1_gen_updates": 4,
        "r2_gen_updates": 2,
        "compute_cost": "report separately; equal consumers does not imply equal compute",
    }
    wrong = {**runs[0], "effective_consumers": 8}
    with pytest.raises(ValueError, match="not equal"):
        validate_actual_comparison_budget(wrong, runs[1])
    with pytest.raises(ValueError, match="unproven"):
        validate_actual_comparison_budget({**runs[0], "effective_consumers": 0}, {**runs[1], "effective_consumers": 0})


def test_r1_r2_gen_and_action_lr_match_by_actual_update_count_with_different_budgets():
    """Equal-GEN-update comparison must not silently rescale the decay curve."""
    traces = []
    for profile in ("R1", "R2"):
        plan = replace(
            plan_fixture(profile, a=2, b=3, target=10),
            gen_schedule=GroupSchedule(1, "linear", 0.2, 3),
            action_schedule=GroupSchedule(1, "cosine", 0.3, 3),
        )
        model, optimizer, controller, live = setup_fixture(plan)
        trace = []
        for _ in range(5):
            if ParameterGroup.GEN in controller.active_roles:
                trace.append({g["g2_role"]: g["lr"] for g in optimizer.param_groups if g["g2_role"] != "global"})
            run_update(model, optimizer, controller, live, 2)
        traces.append(trace)
    assert len(traces[0]) == 5 and len(traces[1]) == 3
    assert traces[0][:3] == traces[1]
    assert traces[0][3:] == [traces[1][-1], traces[1][-1]]


def _rank_boundary_worker(rank, rendezvous, output, disagree):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        plan = plan_fixture(a=1, b=1, target=10, world=2)
        if disagree and rank == 1:
            plan = replace(plan, data_identity="different-fixture-data")
        model, optimizer, scheduler, live = setup_fixture(plan)
        if disagree:
            with pytest.raises(ValueError, match="ranks disagree"):
                scheduler.prepare_commit(actual_consumers=rank + 1, iteration=0)
            Path(output, str(rank)).write_text("REJECTED")
        else:
            for count in (rank + 1, rank + 3):
                run_update(model, optimizer, scheduler, live, count)
            Path(output, str(rank)).write_text(json.dumps(scheduler.state_dict(), sort_keys=True))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("disagree", [False, True])
def test_two_rank_cpu_stage_synchronization_and_actual_count_sum(tmp_path, disagree):
    # Actual Gloo collectives, tiny unsharded models; explicitly NOT FSDP.
    mp.start_processes(
        _rank_boundary_worker,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path), disagree),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    assert (tmp_path / "0").read_text() == (tmp_path / "1").read_text()
    if not disagree:
        state = json.loads((tmp_path / "0").read_text())
        assert state["effective_consumers"] == 10
        assert state["group_consumers"] == {"global": 10, "gen": 7, "action_adapter": 7}

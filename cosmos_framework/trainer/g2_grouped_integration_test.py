# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real grouped stream/Local/GA callbacks, synthetic host loss and source only."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MethodType

import pytest
import torch
from torch import nn

from cosmos_framework.trainer import local_memory_grouped as grouped
from cosmos_framework.trainer.g2_training import G2StageController, parameter_roles
from cosmos_framework.trainer.g2_training_test import equal_tree, plan_fixture
from cosmos_framework.trainer.local_memory_grouped_test import _Model, _trainer


def grouped_setup(monkeypatch):
    trainer = _trainer(active_ga=2, b_stream=3)
    model = _Model(t=2)
    model.net.global_memory_reader = nn.Linear(1, 1)
    model.net.global_memory_projector = nn.Linear(1, 1)
    model.net.action_modality_embed = nn.Parameter(torch.tensor(1.0))

    def loss(self, batch, iteration, *, _local_memory_prefixes):
        present = [prefix for prefix in _local_memory_prefixes if prefix is not None]
        local = torch.stack([prefix.square().mean() for prefix in present]).mean() if present else 0.0
        global_value = self.net.global_memory_projector(self.net.global_memory_reader(torch.ones(1, 1))).mean()
        value = self.net.moe_gen * (global_value + 1.0 + local) + self.net.action_modality_embed
        return {"count": batch["count"]}, value.square()

    model.training_step = MethodType(loss, model)
    # Existing source fixtures have only raw/latent fields; collate is a
    # deliberate model boundary stub, Local scan/transactions remain real.
    monkeypatch.setattr(grouped, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    frontier = trainer._grouped_planner.initial_frontier()
    counts = []
    for _ in range(4):
        window = trainer._grouped_planner.plan_window(frontier)
        counts.append(window.n_window)
        frontier = window.candidate_frontier
    plan = replace(plan_fixture(target=sum(counts)), b_stream=3, grad_accum_steps=2)
    named = dict(model.named_parameters())
    roles = parameter_roles(
        model,
        profile="R2",
        global_names=frozenset(
            name for name in named if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
        ),
        gen_names=frozenset(("net.moe_gen", "net.action_modality_embed")),
        action_names=frozenset(("net.action_modality_embed",)),
    )
    optimizer = torch.optim.AdamW([named[name] for name in roles], lr=1e-5, weight_decay=0.05)
    controller = G2StageController(model, optimizer, plan, roles)
    trainer._g2_stage_controller = controller
    return trainer, model, optimizer, controller, counts


def run_grouped(trainer, model, optimizer, controller, iteration):
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    accum = 0
    for _ in range(2):
        _, _, accum = trainer.training_step(model, optimizer, controller, scaler, {}, iteration, accum)
    assert accum == 0


@pytest.mark.parametrize("checkpoint_after", [1, 2, 3])
def test_real_grouped_stage_local_sidecar_resume_and_actual_terminal_counts(monkeypatch, checkpoint_after):
    torch.manual_seed(17)
    trainer, model, optimizer, controller, counts = grouped_setup(monkeypatch)
    slow = copy.deepcopy(model.net.local_memory_runtime.state_dict())
    for iteration in range(checkpoint_after):
        run_grouped(trainer, model, optimizer, controller, iteration)
    callback = trainer.callbacks._callbacks[0]
    saved = copy.deepcopy(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=controller.state_dict(),
            dataloader=callback.state_dict(),
            rng=torch.get_rng_state(),
        )
    )
    for iteration in range(checkpoint_after, 4):
        run_grouped(trainer, model, optimizer, controller, iteration)
    expected = copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            controller.state_dict(),
            callback.state_dict(),
            torch.get_rng_state(),
        )
    )
    resumed, other, other_optimizer, other_controller, other_counts = grouped_setup(monkeypatch)
    other.load_state_dict(saved["model"], strict=True)
    other_optimizer.load_state_dict(saved["optimizer"])
    other_controller.load_state_dict(saved["scheduler"])
    resumed.callbacks._callbacks[0].load_state_dict(saved["dataloader"])
    resumed._resume_required = True
    torch.set_rng_state(saved["rng"])
    for iteration in range(checkpoint_after, 4):
        run_grouped(resumed, other, other_optimizer, other_controller, iteration)
    equal_tree(
        expected,
        (
            other.state_dict(),
            other_optimizer.state_dict(),
            other_controller.state_dict(),
            resumed.callbacks._callbacks[0].state_dict(),
            torch.get_rng_state(),
        ),
    )
    equal_tree(slow, other.net.local_memory_runtime.state_dict())
    assert counts == other_counts and any(count != 3 * 2 * 2 for count in counts)
    assert other_controller.effective_consumers == sum(counts)
    assert other.net.scan_calls > 0


def test_g2_sidecar_is_required_and_cross_checks_scheduler_stage(monkeypatch):
    trainer, model, optimizer, controller, _ = grouped_setup(monkeypatch)
    run_grouped(trainer, model, optimizer, controller, 0)
    saved = trainer.callbacks._callbacks[0].state_dict()
    resumed, other, other_optimizer, other_controller, _ = grouped_setup(monkeypatch)
    with pytest.raises(ValueError, match="explicit stage"):
        resumed.callbacks._callbacks[0].load_state_dict(saved["local"])
    resumed.callbacks._callbacks[0].load_state_dict(saved)
    # Deliberately omit scheduler restoration: do not silently infer stage.
    with pytest.raises(ValueError, match="stage identities"):
        run_grouped(resumed, other, other_optimizer, other_controller, 1)

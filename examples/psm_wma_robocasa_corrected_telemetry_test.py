# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only acceptance for corrected grouped telemetry (no actual optimizer step)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.psm_wma_robocasa_corrected_telemetry import GroupedPlanObserver


class LocalCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slot_queries = nn.Parameter(torch.ones(2))
        self.calls = 0

    def drain_telemetry(self) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {
            "ttt_inner_loss_sum": torch.tensor(8.0),
            "ttt_inner_loss_count": torch.tensor(4.0),
            "ttt_fast_state_norm_sum": torch.tensor(12.0),
            "ttt_fast_state_norm_count": torch.tensor(4.0),
            "ttt_fast_update_norm_sum": torch.tensor(4.0),
            "ttt_fast_update_norm_count": torch.tensor(4.0),
        }


def make_trainer():
    plan = SimpleNamespace(
        members=(
            (SimpleNamespace(valid_count=3), SimpleNamespace(valid_count=1)),
            (SimpleNamespace(valid_count=2),),
        )
    )
    model = nn.Module()
    model.net = nn.Module()
    model.net.moe_gen = nn.Linear(2, 2)
    model.net.action2llm = nn.Linear(2, 2)
    model.net.local_memory_runtime = nn.Module()
    model.net.local_memory_runtime.encoder = nn.Linear(2, 2)
    model.net.local_memory_runtime.core = LocalCore()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    frontier = SimpleNamespace(epoch=2, slots=(SimpleNamespace(uid="A"), SimpleNamespace(uid=None)))
    window = SimpleNamespace(model=model, plan=plan, live=SimpleNamespace(frontier=frontier))
    return SimpleNamespace(_grouped_window=window)


def group(name: str) -> str | None:
    if ".local_memory_runtime." in name:
        return "local"
    if ".action2llm." in name:
        return "action"
    if ".moe_gen." in name:
        return "generation"
    return None


def test_successful_weighted_losses_gradient_norms_and_local_stats():
    emitted: list[str] = []
    now = iter((10.0, 15.5))
    observer = GroupedPlanObserver(rank=0, parameter_group=group, clock=lambda: next(now), emit=emitted.append)
    trainer = make_trainer()
    before = [param.grad.clone() for param in trainer._grouped_window.model.parameters()]
    observer.start_iteration(0)
    for member, index, action in ((0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 1, 5)):
        observer(
            phase="native_forward",
            trainer=trainer,
            iteration=0,
            member=member,
            index=index,
            metrics={
                "flow_matching_loss_action": torch.tensor(float(action)),
                "flow_matching_loss_vision": torch.tensor(float(action * 2)),
            },
        )
        observer(
            phase="native_backward",
            trainer=trainer,
            iteration=0,
            loss=torch.tensor(float(action) / 5),
        )
    assert not emitted  # no successful-iteration log before optimizer commit
    observer(phase="pre_optimizer", trainer=trainer, iteration=0, metrics={"lr_min": 1e-5, "lr_max": 5e-5})
    observer(phase="post_commit", trainer=trainer, iteration=0)

    assert len(emitted) == 1
    record = json.loads(emitted[0].split("[train] ", 1)[1])
    assert record["iteration"] == 1
    assert record["outer_loss"] == pytest.approx(3)
    assert record["action_loss"] == pytest.approx(16 / 6)
    assert record["vision_loss"] == pytest.approx(32 / 6)
    assert record["action_loss_consumer_weight_coverage"] == pytest.approx(1)
    assert record["native_forward_calls"] == record["native_backward_calls"] == 5
    assert record["inner_loss_mean"] == pytest.approx(2)
    assert record["fast_state_norm_mean"] == pytest.approx(3)
    assert record["fast_update_norm_mean"] == pytest.approx(1)
    assert record["epoch"] == 2 and record["active_slots"] == 1
    assert record["step_wall_s"] == pytest.approx(5.5)
    assert record["generation_grad_norm_rank_local"] == pytest.approx(math.sqrt(12))
    assert record["action_grad_norm_rank_local"] == pytest.approx(math.sqrt(6))
    assert record["local_grad_norm_rank_local"] == pytest.approx(math.sqrt(8))
    assert record["total_grad_norm_rank_local"] == pytest.approx(math.sqrt(20))
    assert record["telemetry_errors"] == []
    assert observer.completed == 1
    assert trainer._grouped_window.model.net.local_memory_runtime.core.calls == 1
    assert all(torch.equal(param.grad, grad) for param, grad in zip(trainer._grouped_window.model.parameters(), before))


def test_missing_metric_is_unavailable_not_fabricated_zero():
    emitted: list[str] = []
    observer = GroupedPlanObserver(rank=0, emit=emitted.append)
    trainer = make_trainer()
    for member, index in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        observer(phase="native_forward", trainer=trainer, member=member, index=index, metrics={})
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(0.01))
    observer(phase="pre_optimizer", trainer=trainer)
    observer(phase="post_commit", trainer=trainer, iteration=0)
    record = json.loads(emitted[0].split("[train] ", 1)[1])
    assert record["action_loss"] is None
    assert record["vision_loss"] is None
    assert record["action_loss_consumer_weight_coverage"] == 0


def test_nonrank0_has_no_output_and_bad_counts_fail_before_optimizer():
    emitted: list[str] = []
    trainer = make_trainer()
    observer = GroupedPlanObserver(rank=1, emit=emitted.append)
    observer.start_iteration(0)
    for _ in range(5):
        observer(phase="native_forward", trainer=trainer)
        observer(phase="native_backward", trainer=trainer)
    observer(phase="pre_optimizer", trainer=trainer)
    observer(phase="post_commit", trainer=trainer, iteration=0)
    assert observer.completed == 1 and emitted == []

    observer = GroupedPlanObserver(rank=0, emit=emitted.append)
    observer(phase="native_forward", trainer=trainer)
    with pytest.raises(RuntimeError, match="调用次数"):
        observer(phase="pre_optimizer", trainer=trainer)
    assert not emitted


def test_completed_optimizer_is_not_rolled_back_by_broken_logger():
    def broken_logger(_line: str) -> None:
        raise RuntimeError("broken pipe")

    observer = GroupedPlanObserver(emit=broken_logger)
    observer(phase="post_commit", trainer=make_trainer(), iteration=0)
    assert observer.completed == 1

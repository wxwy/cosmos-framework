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


def test_stage_timing_is_distinct_from_unavailable_async_data_wait():
    emitted: list[str] = []
    ticks = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    observer = GroupedPlanObserver(rank=0, clock=lambda: next(ticks), emit=emitted.append)
    trainer = make_trainer()
    observer.start_iteration(0)
    with observer.time_stage("data_prepare"):
        pass
    with observer.time_stage("forward"):
        pass
    observer(phase="post_commit", trainer=trainer, iteration=0)
    record = json.loads(emitted[0].split("[train] ", 1)[1])
    assert record["data_prepare_ms"] == pytest.approx(1000.0)
    assert record["forward_ms"] == pytest.approx(1000.0)
    assert record["data_wait_ms"] is None
    assert record["data_wait_reason"] == "synchronous_producer_no_background_dataloader"
    assert record["cpu_timing_scope"] == "host_wall_cuda_dispatch_not_gpu_kernel_time"
    assert record["cuda_sampled"] is False


def test_parameter_norms_sample_only_on_interval_100():
    emitted: list[str] = []
    trainer = make_trainer()
    observer = GroupedPlanObserver(rank=0, parameter_group=group, emit=emitted.append, parameter_norm_interval=100)
    observer.start_iteration(0)
    for _member, _index in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        observer(phase="native_forward", trainer=trainer)
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(0.0))
    observer(phase="pre_optimizer", trainer=trainer, metrics={})
    observer(phase="post_commit", trainer=trainer, iteration=0)
    first = json.loads(emitted[-1].split("[train] ", 1)[1])
    assert first["param_norm_sampled"] is False
    assert "local_param_norm_rank_local" not in first

    observer.start_iteration(99)
    for _member, _index in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        observer(phase="native_forward", trainer=trainer)
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(0.0))
    observer(phase="pre_optimizer", trainer=trainer, metrics={})
    observer(phase="post_commit", trainer=trainer, iteration=99)
    second = json.loads(emitted[-1].split("[train] ", 1)[1])
    assert second["param_norm_sampled"] is True
    assert second["local_param_norm_rank_local"] == pytest.approx(math.sqrt(8))
    assert second["action_param_norm_rank_local"] is not None
    assert second["generation_param_norm_rank_local"] is not None
    assert second["nonfinite_grad_tensors_rank_local"] == 0
    assert second["missing_grad_tensors_rank_local"] == 0


def test_log_uses_model_frozen_loss_coefficients_not_double_weighting():
    emitted: list[str] = []
    trainer = make_trainer()
    trainer._grouped_window.model.config = SimpleNamespace(
        rectified_flow_training_config=SimpleNamespace(
            action_loss_weight=10.0,
            loss_scale=10.0,
            image_loss_scale=None,
            sample_level_loss_averaging=False,
        )
    )
    observer = GroupedPlanObserver(rank=0, emit=emitted.append)
    for member, index in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        observer(
            phase="native_forward",
            trainer=trainer,
            member=member,
            index=index,
            metrics={
                "flow_matching_loss_action": torch.tensor(2.0),
                "flow_matching_loss_vision": torch.tensor(3.0),
                "aux_loss_gen": torch.tensor(0.5),
                "aux_loss_und": torch.tensor(0.25),
            },
        )
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(1.0))
    observer(phase="pre_optimizer", trainer=trainer, metrics={"lr_min": 1e-5, "lr_max": 5e-5})
    observer(phase="post_commit", trainer=trainer, iteration=0)
    report = json.loads(emitted[-1].split("[train] ", 1)[1])
    assert report["outer_loss"] == pytest.approx(5.0)
    assert report["action_contribution"] == pytest.approx(20.0)
    assert report["vision_contribution"] == pytest.approx(30.0)
    assert report["aux_loss_gen"] == pytest.approx(0.5)
    assert report["aux_loss_und"] == pytest.approx(0.25)
    assert report["inner_loss_max"] is None  # The minimal fake core publishes no max metric.
    assert report["grad_norm_scope"] == "rank0_local_fsdp_shard"


def test_observer_resets_after_abort_without_successful_log():
    emitted: list[str] = []
    trainer = make_trainer()
    observer = GroupedPlanObserver(rank=0, emit=emitted.append)
    observer.start_iteration(0)
    observer(phase="native_backward", trainer=trainer, loss=torch.tensor(5.0))
    assert not emitted
    # A retry or clean fresh run resets purely observational state.
    observer.start_iteration(0)
    for _ in range(5):
        observer(phase="native_forward", trainer=trainer)
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(1.0))
    observer(phase="pre_optimizer", trainer=trainer)
    observer(phase="post_commit", trainer=trainer, iteration=0)
    report = json.loads(emitted[0].split("[train] ", 1)[1])
    assert report["outer_loss"] == pytest.approx(5.0)
    assert len(emitted) == 1


def test_hot_path_does_not_call_tensor_item(monkeypatch: pytest.MonkeyPatch):
    trainer = make_trainer()
    observer = GroupedPlanObserver(rank=0)

    def forbidden_item(self, *args, **kwargs):
        raise AssertionError("per-consumer .item() forces a CUDA synchronization")

    with monkeypatch.context() as ctx:
        ctx.setattr(torch.Tensor, "item", forbidden_item)
        observer(
            phase="native_forward",
            trainer=trainer,
            member=0,
            index=0,
            metrics={"flow_matching_loss_action": torch.tensor(1.0)},
        )
        observer(phase="native_backward", trainer=trainer, loss=torch.tensor(0.25))
    assert observer.forward == 1 and observer.backward == 1


def test_cuda_event_timing_samples_only_every_100_steps(monkeypatch: pytest.MonkeyPatch):
    emitted: list[str] = []
    state = {"events": 0, "synchronizes": 0}

    class FakeCudaEvent:
        def __init__(self, **_kwargs):
            state["events"] += 1

        def record(self):
            pass

        def elapsed_time(self, other):
            assert isinstance(other, FakeCudaEvent)
            return 2.5

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: state.__setitem__("synchronizes", state["synchronizes"] + 1))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * 1024**3)

    trainer = make_trainer()
    observer = GroupedPlanObserver(rank=0, emit=emitted.append)
    observer.start_iteration(0)
    with observer.time_stage("forward"):
        pass
    observer(phase="post_commit", trainer=trainer, iteration=0)
    assert state["events"] == 0 and state["synchronizes"] == 0

    observer.start_iteration(99)
    with observer.time_stage("forward"):
        pass
    with observer.time_stage("backward"):
        pass
    observer(phase="post_commit", trainer=trainer, iteration=99)
    record = json.loads(emitted[-1].split("[train] ", 1)[1])
    assert state["events"] == 4
    assert state["synchronizes"] == 1
    assert record["forward_cuda_event_ms"] == pytest.approx(2.5)
    assert record["backward_cuda_event_ms"] == pytest.approx(2.5)
    assert record["cuda_sampled"] is True

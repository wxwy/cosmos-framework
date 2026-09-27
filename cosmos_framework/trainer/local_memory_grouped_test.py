"""H3-C ImaginaireTrainer GA2/optimizer 生命周期与原生 batch ABI CPU 合同。"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cosmos_framework.trainer.local_memory_grouped as grouped_module
from cosmos_framework.model.generator.mot.local_memory_grouped_window_test import _rank_catalog, _segments
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import RankLocalGroupedPlanner
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch


class _Hooks:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append(name)

        return record


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.local_memory_runtime = LocalMemoryRuntime()
        self.net.moe_gen = nn.Parameter(torch.tensor(2.0))

    def training_step(self, batch, iteration, *, _local_memory_prefixes):
        assert batch["count"] == len(_local_memory_prefixes)
        present = [prefix for prefix in _local_memory_prefixes if prefix is not None]
        signal = torch.stack([prefix.square().mean() for prefix in present]).mean() if present else 0.0
        return {"count": batch["count"]}, self.net.moe_gen * (signal + 1.0)

    def on_after_backward(self):
        pass

    def on_before_optimizer_step(self, optimizer, scheduler, iteration):
        pass

    def on_before_zero_grad(self, optimizer, scheduler, iteration):
        pass


def _trainer() -> GroupedLocalMemoryTrainer:
    trainer = object.__new__(GroupedLocalMemoryTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=2, distributed_parallelism="fsdp"))
    trainer.callbacks = _Hooks()
    trainer.training_timer = lambda name: nullcontext()
    trainer.bind_grouped_stream(RankLocalGroupedPlanner(_rank_catalog(16, frames=64), rank=0), lambda _: None)
    return trainer


def test_collate_uses_native_joint_dataloader_multi_sample_abi() -> None:
    payloads = tuple(
        {
            "video": torch.zeros(3, 33, 2, 4),
            "action": torch.zeros(33, 64),
            "text_token_ids": torch.tensor([1, 2, index]),
            "sequence_plan": f"plan-{index}",
        }
        for index in range(3)
    )
    batch = collate_grouped_native_batch(payloads)
    assert len(batch["video"]) == len(batch["action"]) == len(batch["sequence_plan"]) == 3
    assert batch["sequence_plan"] == ["plan-0", "plan-1", "plan-2"]
    assert all(len(items) == 1 for items in batch["video"])


def test_trainer_ga2_runs_one_optimizer_then_publishes_and_zeros_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(19)
    model = _Model()
    trainer = _trainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    monkeypatch.setattr(
        grouped_module,
        "materialize_member",
        lambda *_: _segments(trainer._grouped_window, trainer._grouped_window._member_index),
    )
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)

    output, first, accum = trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, 0)
    before = trainer._grouped_window.live
    assert accum == 1 and output["count"] == 8 and trainer._grouped_window.live is before
    assert model.net.moe_gen.grad is not None and first.isfinite()
    old_weight = model.net.moe_gen.detach().clone()

    output, second, accum = trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, accum)
    assert accum == 0 and output["count"] == 8 and second.isfinite()
    assert model.net.moe_gen.item() != old_weight.item()
    assert trainer._grouped_window.live is not before
    assert len(trainer._grouped_window.live.scheduler._committed) == 8
    assert all(parameter.grad is None for parameter in model.parameters())
    assert trainer.callbacks.events.count("on_before_forward") == 32
    assert trainer.callbacks.events.count("on_before_backward") == 32
    assert trainer.callbacks.events.count("on_before_optimizer_step") == 1
    assert scheduler.last_epoch == 1


def test_scaler_skip_does_not_advance_scheduler() -> None:
    optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    class SkippingScaler:
        def __init__(self):
            self.scale = 8.0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            pass

        def update(self):
            self.scale = 4.0

    assert GroupedLocalMemoryTrainer._optimizer_step_success(optimizer, scheduler, SkippingScaler()) is False
    assert scheduler.last_epoch == 0


def test_nonfinite_selected_gradient_fails_before_optimizer() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="非有限"):
        GroupedLocalMemoryTrainer._require_finite_gradients(optimizer)


def test_trainer_scaler_skip_aborts_pending_window_without_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Model()
    trainer = _trainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    class PendingWindow:
        def __init__(self):
            self.plan = SimpleNamespace(members=((), ()))
            self.aborted = False
            self.published = False

        def run_member(self, segments, native_loss, backward):
            model.net.moe_gen.grad = torch.ones_like(model.net.moe_gen)
            return torch.tensor(1.0)

        def finish(self, optimizer_step):
            if not optimizer_step():
                self.abort()
                raise RuntimeError("optimizer skipped")
            self.published = True

        def abort(self):
            self.aborted = True

    class SkippingScaler:
        def __init__(self):
            self.scale = 8.0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.scale

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            pass

        def update(self):
            self.scale = 4.0

    pending = PendingWindow()
    trainer._grouped_window = pending
    monkeypatch.setattr(grouped_module, "materialize_member", lambda *_: ())
    with pytest.raises(RuntimeError, match="skipped"):
        trainer.training_step(model, optimizer, scheduler, SkippingScaler(), {}, 0, 1)
    assert pending.aborted and not pending.published and scheduler.last_epoch == 0

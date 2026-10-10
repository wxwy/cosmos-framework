"""Phase5A corrected trainer 动态 GA/optimizer 生命周期与原生 batch ABI CPU 合同。"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cosmos_framework.trainer.local_memory_grouped as grouped_module
from cosmos_framework.model.generator.mot.local_memory_grouped_window_test import _Net
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _setup as _exact_setup
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


class _Hooks:
    def __init__(self):
        self.events = []
        self._callbacks = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append(name)

        return record


class _Model(nn.Module):
    def __init__(self, t=2):
        super().__init__()
        self.net = _Net(t)

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


def _trainer(active_ga=2, b_stream=8, *, num_workers=0) -> GroupedLocalMemoryTrainer:
    trainer = object.__new__(GroupedLocalMemoryTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=active_ga, distributed_parallelism="fsdp"))
    trainer.callbacks = _Hooks()
    trainer.training_timer = lambda name: nullcontext()
    _, _, catalog, producer = _exact_setup((5,) * 16, t=2)
    trainer.bind_grouped_stream(
        ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=b_stream, active_ga=active_ga),
        producer,
        config_digest="config",
        num_workers=num_workers,
    )
    return trainer


@pytest.mark.parametrize("size", [1, 3, 8])
def test_collate_uses_native_joint_dataloader_multi_sample_abi(size: int) -> None:
    payloads = tuple(
        {
            "video": torch.zeros(3, 17, 2, 4),
            "video_latent": torch.zeros(5, 48, 2, 2),
            "cached_latent_required": True,
            "action": torch.zeros(17, 64),
            "action_raw": torch.zeros(17, 15),
            "text_token_ids": torch.tensor([1, 2, index]),
            "sequence_plan": f"plan-{index}",
        }
        for index in range(size)
    )
    batch = collate_grouped_native_batch(payloads)
    assert len(batch["video"]) == len(batch["action"]) == len(batch["sequence_plan"]) == size
    assert batch["sequence_plan"] == [f"plan-{index}" for index in range(size)]
    assert all(len(items) == 1 for items in batch["video"])
    assert len(batch["video_latent"]) == size
    assert batch["cached_latent_required"] == [True] * size
    assert all(len(item) == 1 and item[0].shape == (17, 64) for item in batch["action"])
    assert all(len(item) == 1 and item[0].shape == (17, 15) for item in batch["action_raw"])


def test_collate_rejects_empty_or_non_dict_payloads() -> None:
    with pytest.raises(ValueError, match="非空"):
        collate_grouped_native_batch(())
    with pytest.raises(ValueError, match="dict"):
        collate_grouped_native_batch((None,))


def test_exact_binding_rejects_catalog_ga_digest_and_duplicate() -> None:
    trainer = _trainer(active_ga=2, b_stream=3)
    with pytest.raises(RuntimeError, match="已绑定"):
        trainer.bind_grouped_stream(trainer._grouped_planner, trainer._grouped_producer, config_digest="config")
    fresh = object.__new__(GroupedLocalMemoryTrainer)
    fresh.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=2))
    fresh.callbacks = _Hooks()
    _, _, catalog, producer = _exact_setup((5,) * 16, t=2)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=3, active_ga=2)
    _, _, other_catalog, _ = _exact_setup((5,) * 16, t=2)
    with pytest.raises(ValueError, match="catalog"):
        fresh.bind_grouped_stream(
            planner, ExactWindowSegmentProducer(other_catalog, config_digest="config"), config_digest="config"
        )
    with pytest.raises(ValueError, match="GA"):
        fresh.bind_grouped_stream(
            ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=3, active_ga=3),
            producer,
            config_digest="config",
        )
    with pytest.raises(ValueError, match="config_digest"):
        fresh.bind_grouped_stream(planner, producer, config_digest="other")
    assert not hasattr(fresh, "_grouped_planner")


def test_trainer_has_no_historical_grouped_import_or_materializer() -> None:
    source = inspect.getsource(grouped_module)
    assert "robocasa_grouped_segment" not in source
    assert "materialize_member" not in source


@pytest.mark.parametrize("active_ga", [1, 2, 3])
def test_trainer_dynamic_ga_runs_one_optimizer_then_publishes_and_zeros_grad(
    monkeypatch: pytest.MonkeyPatch, active_ga: int
) -> None:
    torch.manual_seed(19)
    model = _Model()
    trainer = _trainer(active_ga)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)
    accum = 0
    before = None
    old_weight = model.net.moe_gen.detach().clone()
    for member in range(active_ga):
        output, loss, accum = trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, accum)
        assert output["count"] == 8 and loss.isfinite()
        if member == 0:
            before = trainer._grouped_window.live if active_ga > 1 else None
        if member + 1 < active_ga:
            assert accum == member + 1 and trainer._grouped_window.live is before
            assert model.net.moe_gen.grad is not None
            torch.testing.assert_close(model.net.moe_gen.detach(), old_weight)
            assert scheduler.last_epoch == 0
    assert accum == 0
    assert model.net.moe_gen.item() != old_weight.item()
    if before is not None:
        assert trainer._grouped_window.live is not before
    assert len(trainer._grouped_window.live.scheduler._committed) == 8
    assert all(parameter.grad is None for parameter in model.parameters())
    expected_calls = sum(min(2, 5 - 2 * member) for member in range(active_ga))
    assert trainer.callbacks.events.count("on_before_forward") == expected_calls
    assert trainer.callbacks.events.count("on_before_backward") == expected_calls
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


def test_bind_rejects_catalog_mismatch_and_existing_dataloader_authority() -> None:
    trainer = _trainer()
    _, _, other_catalog, other_producer = _exact_setup((5,) * 16, t=2)
    assert other_catalog is not trainer._grouped_planner.catalog
    fresh = object.__new__(GroupedLocalMemoryTrainer)
    fresh.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=2))
    fresh.callbacks = _Hooks()
    with pytest.raises(ValueError, match="catalog"):
        fresh.bind_grouped_stream(trainer._grouped_planner, other_producer, config_digest="config")
    fresh.callbacks._callbacks.append(SimpleNamespace(checkpoint_component="dataloader"))
    with pytest.raises(ValueError, match="dataloader"):
        fresh.bind_grouped_stream(
            trainer._grouped_planner,
            ExactWindowSegmentProducer(trainer._grouped_planner.catalog, config_digest="config"),
            config_digest="config",
        )


def test_exact_producer_materializes_planned_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(active_ga=1, b_stream=3)
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    producer = trainer._grouped_producer
    seen = []
    original = producer.produce

    def record(request):
        segment = original(request)
        seen.append((request, segment))
        return segment

    monkeypatch.setattr(producer, "produce", record)
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)
    trainer.training_step(model, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False), {}, 0, 0)
    assert len(seen) == 3
    assert tuple(int(segment.slot_id[0]) for _, segment in seen) == (0, 1, 2)
    assert all(segment.segment_provenance.config_digest == "config" for _, segment in seen)


def test_ga_and_context_parallel_must_match_planner() -> None:
    trainer = _trainer(active_ga=3)
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.config.trainer.grad_accum_iter = 2
    with pytest.raises(ValueError, match="active_ga"):
        trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, 0)
    trainer.config.trainer.grad_accum_iter = 3
    trainer.config.model_parallel = SimpleNamespace(context_parallel_size=2)
    with pytest.raises(ValueError, match="context_parallel_size"):
        trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, 0)


def test_optimizer_parameters_supports_container_and_plain_optimizer() -> None:
    first = nn.Parameter(torch.tensor(1.0))
    second = nn.Parameter(torch.tensor(2.0))
    third = nn.Parameter(torch.tensor(3.0))
    first_optimizer = torch.optim.SGD([first, second], lr=0.1)
    second_optimizer = torch.optim.AdamW([third], lr=0.1)
    container = object.__new__(OptimizersContainer)
    container.optimizers = [first_optimizer, second_optimizer]

    plain = GroupedLocalMemoryTrainer._optimizer_parameters(first_optimizer)
    grouped = GroupedLocalMemoryTrainer._optimizer_parameters(container)
    assert [id(parameter) for parameter in plain] == [id(first), id(second)]
    assert [id(parameter) for parameter in grouped] == [id(first), id(second), id(third)]


def test_optimizer_parameters_container_is_fail_closed() -> None:
    container = object.__new__(OptimizersContainer)
    container.optimizers = []
    with pytest.raises(ValueError, match="container"):
        GroupedLocalMemoryTrainer._optimizer_parameters(container)

    container.optimizers = [SimpleNamespace()]
    with pytest.raises(TypeError, match="param_groups"):
        GroupedLocalMemoryTrainer._optimizer_parameters(container)


def test_finite_gradient_check_unwraps_optimizer_container() -> None:
    first = nn.Parameter(torch.tensor(1.0))
    second = nn.Parameter(torch.tensor(2.0))
    first.grad = torch.tensor(0.5)
    second.grad = torch.tensor(0.25)
    container = object.__new__(OptimizersContainer)
    container.optimizers = [
        torch.optim.SGD([first], lr=0.1),
        torch.optim.AdamW([second], lr=0.1),
    ]

    GroupedLocalMemoryTrainer._require_finite_gradients(container)

    second.grad = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="非有限"):
        GroupedLocalMemoryTrainer._require_finite_gradients(container)


def test_nonfinite_selected_gradient_fails_before_optimizer() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="非有限"):
        GroupedLocalMemoryTrainer._require_finite_gradients(optimizer)


def test_absent_w0_gradient_is_legal_when_another_selected_gradient_is_present() -> None:
    absent = nn.Parameter(torch.tensor(1.0))
    present = nn.Parameter(torch.tensor(2.0))
    present.grad = torch.tensor(0.5)
    optimizer = torch.optim.SGD([absent, present], lr=0.1)
    GroupedLocalMemoryTrainer._require_finite_gradients(optimizer)
    present.grad = None
    with pytest.raises(FloatingPointError, match="缺失"):
        GroupedLocalMemoryTrainer._require_finite_gradients(optimizer)


def test_trainer_scaler_skip_aborts_pending_window_without_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Model()
    trainer = _trainer(b_stream=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    class SkippingScaler:
        def __init__(self):
            self.current_scale = 8.0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.current_scale

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            pass

        def update(self):
            self.current_scale = 4.0

    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)
    scaler = SkippingScaler()
    _, _, next_ga = trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, 0)
    live_before = trainer._grouped_window.live
    with pytest.raises(RuntimeError, match="optimizer 未执行成功"):
        trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, next_ga)
    assert trainer._grouped_window.live is live_before
    assert trainer._grouped_window.plan is None
    assert scheduler.last_epoch == 0
    assert trainer._grouped_completed_iteration == 0


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_async_raw_prefetch_keeps_optimizer_and_committed_frontier_parity(
    monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    # Same model seed, same GA/T/B, same commit and next-frontier after one completed window.
    torch.manual_seed(19)
    model = _Model()
    trainer = _trainer(active_ga=2, b_stream=3, num_workers=num_workers)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)
    try:
        accum = 0
        for member in range(2):
            output, loss, accum = trainer.training_step(model, optimizer, scheduler, scaler, {}, 0, accum)
            assert output["count"] == 3
            assert loss.isfinite()
            assert accum == (0 if member == 1 else 1)
        assert trainer._grouped_completed_iteration == 1
        assert scheduler.last_epoch == 1
        assert trainer._grouped_window.live.frontier == trainer._grouped_planner.plan_window(
            trainer._grouped_planner.initial_frontier()
        ).candidate_frontier
        # Numerical reference for the non-prefetched, frozen CPU path with the same seed.
        torch.manual_seed(19)
        baseline_model = _Model()
        baseline = _trainer(active_ga=2, b_stream=3, num_workers=0)
        opt = torch.optim.SGD(baseline_model.parameters(), lr=0.01)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        for member in range(2):
            baseline.training_step(baseline_model, opt, sch, scaler, {}, 0, member)
        for observed, expected in zip(model.parameters(), baseline_model.parameters(), strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert trainer._grouped_window.live.frontier == baseline._grouped_window.live.frontier
        assert set(trainer._grouped_window.live.scheduler._committed) == set(
            baseline._grouped_window.live.scheduler._committed
        )
    finally:
        if trainer._grouped_prefetcher is not None:
            trainer._grouped_prefetcher.close()



def test_async_raw_failure_aborts_candidate_without_optimizer_or_frontier_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(19)
    model = _Model()
    trainer = _trainer(active_ga=2, b_stream=3, num_workers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial_frontier = trainer._grouped_planner.initial_frontier()
    old_weight = model.net.moe_gen.detach().clone()
    assert trainer._grouped_prefetcher is not None

    def fail_read(_request):
        raise ValueError("injected async read failure")

    monkeypatch.setattr(trainer._grouped_prefetcher, "_prepare", fail_read)
    monkeypatch.setattr(grouped_module, "collate_grouped_native_batch", lambda payloads: {"count": len(payloads)})
    monkeypatch.setattr(grouped_module.misc, "to", lambda value, **kwargs: value)
    try:
        _, _, accum = trainer.training_step(
            model, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False), {}, 0, 0
        )
        assert accum == 1
        with pytest.raises(ValueError, match="injected async read failure"):
            trainer.training_step(
                model, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False), {}, 0, accum
            )
        assert trainer._grouped_window.live.frontier == initial_frontier
        assert trainer._grouped_window.plan is None
        assert trainer._grouped_prefetcher._pending is None
        assert trainer._grouped_completed_iteration == 0
        assert scheduler.last_epoch == 0
        torch.testing.assert_close(model.net.moe_gen.detach(), old_weight, rtol=0, atol=0)
    finally:
        trainer._grouped_prefetcher.close()

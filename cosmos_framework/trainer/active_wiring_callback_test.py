from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.mot.ttt_lifecycle import TTTLifecycleCallback
from cosmos_framework.trainer import ImaginaireTrainer, _dispatch_active_callbacks_excluding_ttt
from cosmos_framework.utils.callback import Callback, CallBackGroup


class _Spy:
    def __init__(self, name: str, calls: list[tuple[str, object]]) -> None:
        self.name, self.calls = name, calls

    def on_before_backward(self, **kwargs) -> None:
        self.calls.append((self.name, kwargs["loss"]))

    def on_after_backward(self, **kwargs) -> None:
        self.calls.append((self.name, kwargs["iteration"]))


class _TTTSubclass(TTTLifecycleCallback):
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def on_before_backward(self, **kwargs) -> None:
        self.calls.append(("subclass", kwargs["loss"]))

    def on_after_backward(self, **kwargs) -> None:
        self.calls.append(("subclass", kwargs["iteration"]))


def test_active_callback_filter_preserves_identity_order_and_exact_class_scope() -> None:
    calls: list[tuple[str, object]] = []
    group = object.__new__(CallBackGroup)
    exact = TTTLifecycleCallback()
    first, subclass, second = _Spy("first", calls), _TTTSubclass(calls), _Spy("second", calls)
    group._callbacks = [first, exact, subclass, second]
    before = (id(group._callbacks), tuple(id(item) for item in group._callbacks))

    _dispatch_active_callbacks_excluding_ttt(group, "on_before_backward", model=object(), loss="loss", iteration=3)
    _dispatch_active_callbacks_excluding_ttt(group, "on_after_backward", model=object(), iteration=3)

    assert calls == [
        ("first", "loss"), ("subclass", "loss"), ("second", "loss"),
        ("first", 3), ("subclass", 3), ("second", 3),
    ]
    assert (id(group._callbacks), tuple(id(item) for item in group._callbacks)) == before


def test_active_callback_filter_never_touches_existing_ttt_lifecycle() -> None:
    class Lifecycle:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def observe_loss(self, loss) -> None:
            self.calls.append("observe")

        def on_after_backward(self) -> None:
            self.calls.append("after")

    lifecycle = Lifecycle()
    model = type("Model", (), {"_ttt_lifecycle": lifecycle})()
    group = object.__new__(CallBackGroup)
    group._callbacks = [TTTLifecycleCallback()]

    _dispatch_active_callbacks_excluding_ttt(group, "on_before_backward", model=model, loss="loss", iteration=0)
    _dispatch_active_callbacks_excluding_ttt(group, "on_after_backward", model=model, iteration=0)

    assert lifecycle.calls == []


def test_no_marker_dispatch_keeps_exact_ttt_callback_in_original_order() -> None:
    calls: list[tuple[str, object]] = []
    group = object.__new__(CallBackGroup)
    exact = TTTLifecycleCallback()
    exact.on_before_backward = lambda **kwargs: calls.append(("exact", kwargs["loss"]))
    first, second = _Spy("first", calls), _Spy("second", calls)
    group._callbacks = [first, exact, second]

    group.on_before_backward(model=object(), loss="legacy", iteration=4)

    assert calls == [("first", "legacy"), ("exact", "legacy"), ("second", "legacy")]


class _TrainerCallback(Callback):
    def __init__(self, name: str, calls: list[tuple[str, object]]) -> None:
        self.name, self.calls = name, calls

    def on_before_backward(self, model, loss, iteration: int = 0) -> None:
        self.calls.append((self.name, ("before", loss, iteration)))

    def on_after_backward(self, model, iteration: int = 0) -> None:
        self.calls.append((self.name, ("after", iteration)))


class _LegacyLifecycle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def observe_loss(self, loss: torch.Tensor) -> None:
        self.calls.append(("observe", loss))

    def on_after_backward(self) -> None:
        self.calls.append(("after_backward", None))

    def abort_open_segments(self) -> None:
        self.calls.append(("abort", None))


class _TrainerModel(torch.nn.Module):
    def __init__(self, lifecycle: _LegacyLifecycle) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self._ttt_lifecycle = lifecycle
        self.after_backward_calls = 0

    def training_step(self, data, iteration: int):
        del data, iteration
        return {}, self.weight.square()

    def on_after_backward(self) -> None:
        self.after_backward_calls += 1


def _trainer_with_callbacks(callbacks: list[Callback]) -> ImaginaireTrainer:
    trainer = object.__new__(ImaginaireTrainer)
    group = object.__new__(CallBackGroup)
    group._callbacks = callbacks
    trainer.callbacks = group
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            grad_accum_iter=2,
            distributed_parallelism="none",
            straggler_detection=SimpleNamespace(
                analyze_forward=False, analyze_backward=False, analyze_optimizer=False
            ),
        )
    )
    trainer.training_timer = lambda _: nullcontext()
    trainer.straggler_detector = SimpleNamespace(profile_section=lambda *_: nullcontext())
    return trainer


def test_trainer_no_marker_control_preserves_legacy_ttt_callback_and_lifecycle_route() -> None:
    calls: list[tuple[str, object]] = []
    lifecycle = _LegacyLifecycle()
    first, exact, second = _TrainerCallback("first", calls), TTTLifecycleCallback(), _TrainerCallback("second", calls)
    trainer = _trainer_with_callbacks([first, exact, second])
    model = _TrainerModel(lifecycle)

    _, returned_loss, next_grad_accum_iter = trainer.training_step(
        model, object(), object(), SimpleNamespace(scale=lambda value: value), {}, iteration=7, grad_accum_iter=0
    )

    assert next_grad_accum_iter == 1 and model.after_backward_calls == 1
    assert [name for name, _ in calls] == ["first", "second", "first", "second"]
    assert calls[0][1] == ("before", returned_loss, 7) and calls[1][1] == ("before", returned_loss, 7)
    assert calls[2][1] == ("after", 7) and calls[3][1] == ("after", 7)
    assert lifecycle.calls == [("observe", returned_loss), ("after_backward", None)]


def test_trainer_active_marker_control_skips_legacy_ttt_lifecycle() -> None:
    class ActiveModel(_TrainerModel):
        def training_step(self, data, iteration: int):
            del data, iteration
            loss = self.weight.square()
            return {"psm_local_memory_active_forward": object(), "loss": loss}, loss

    calls: list[tuple[str, object]] = []
    lifecycle = _LegacyLifecycle()
    first, exact, second = _TrainerCallback("first", calls), TTTLifecycleCallback(), _TrainerCallback("second", calls)
    trainer = _trainer_with_callbacks([first, exact, second])
    trainer._run_active_local_memory_backward = lambda model, output, grad_scaler, grad_accum_iter: output["loss"].backward()
    model = ActiveModel(lifecycle)

    _, _, next_grad_accum_iter = trainer.training_step(
        model, object(), object(), SimpleNamespace(scale=lambda value: value), {}, iteration=8, grad_accum_iter=0
    )

    assert next_grad_accum_iter == 1 and model.after_backward_calls == 1
    assert [name for name, _ in calls] == ["first", "second", "first", "second"]
    assert lifecycle.calls == []

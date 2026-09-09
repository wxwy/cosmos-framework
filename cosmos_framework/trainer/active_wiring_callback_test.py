from __future__ import annotations

from cosmos_framework.model.generator.mot.ttt_lifecycle import TTTLifecycleCallback
from cosmos_framework.trainer import _dispatch_active_callbacks_excluding_ttt
from cosmos_framework.utils.callback import CallBackGroup


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B2-B 单 slot T16 串行梯度接力的 CPU 合同测试。"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.local_memory_native_segment import (
    NativeConsumerResult,
    SingleSegmentNativeGradientRelay,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentIdentity,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)
from cosmos_framework.model.generator.mot.local_memory_segment_test import make_segment
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.mot.robocasa_segment_producer_test import make_producer
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata


def _model() -> nn.Module:
    torch.manual_seed(27)
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_memory_runtime = LocalMemoryRuntime()
    model.net.local_memory2llm = nn.Linear(32, 2048)
    model.net.local_memory_modality_embed = nn.Parameter(torch.randn(2048) * 0.02)
    model.net.host = nn.Linear(2048, 1)
    return model


def _runner(model: nn.Module | None = None, *, lr: float = 1e-3) -> SingleSegmentNativeGradientRelay:
    model = _model() if model is None else model
    selected = _build_params_with_metadata(
        model,
        keys_to_select=["local_memory"],
        lr_multipliers={},
        base_lr=lr,
        base_weight_decay=0.0,
        disable_weight_decay_for_1d_params=False,
    )
    assert sum(parameter.numel() for parameter, _ in selected) == 165_312
    optimizer = torch.optim.SGD([parameter for parameter, _ in selected], lr=lr)
    return SingleSegmentNativeGradientRelay(model, optimizer)


def _identity(cursor: int = 0) -> SegmentIdentity:
    return SegmentIdentity(0, "episode", "robocasa", cursor, cursor, "source")


def _plan(identity: SegmentIdentity, count: int = 16) -> GAWindowPlan:
    return GAWindowPlan((identity.member,), (count,))


def _callback(model: nn.Module, seen: list[tuple[int, int, bool]] | None = None):
    def native_forward(payload: dict[str, int], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        step = payload["step"]
        if seen is not None:
            seen.append((index, step, leaf is None))
        if leaf is None:
            hidden = model.net.local_memory_modality_embed.detach().new_zeros(2048)
        else:
            assert leaf.shape == (4, 32)
            hidden = model.net.local_memory2llm(leaf).mean(dim=0) + model.net.local_memory_modality_embed
        prediction = model.net.host(hidden).squeeze()
        loss = (prediction - step / 10).square()
        return NativeConsumerResult(loss, output=step)

    return native_forward


def _gradients(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.net.named_parameters()
        if "local_memory" in name
    }


def test_model_runtime_identity_and_complete_optimizer_inventory() -> None:
    relay = _runner()
    runtime = relay.model.net.local_memory_runtime
    assert relay.adapter.encoder is runtime.encoder
    assert relay.adapter.core is runtime.core
    assert relay.adapter.sidecar is relay.sidecar
    assert not any(parameter.requires_grad for parameter in relay.model.net.host.parameters())
    state = runtime.core.initial_state(1)
    assert all(not isinstance(value, nn.Parameter) for value in state)
    assert not any(id(value) == id(parameter) for value in state for parameter in relay.model.parameters())


def test_t16_stream_order_s0_mapping_and_commit_after_step() -> None:
    relay = _runner()
    identity = _identity()
    segment = make_segment()
    seen: list[tuple[int, int, bool]] = []
    before = {name: parameter.detach().clone() for name, parameter in relay.model.net.named_parameters()}
    real_step = relay.optimizer.step
    real_zero_grad = relay.optimizer.zero_grad
    at_step = []
    zero_calls = []

    def step_with_observation(*args: Any, **kwargs: Any) -> Any:
        at_step.append((relay.sidecar.snapshot(), dict(relay.scheduler._committed)))
        return real_step(*args, **kwargs)

    relay.optimizer.step = step_with_observation

    def zero_with_observation(*args: Any, **kwargs: Any) -> Any:
        zero_calls.append(1)
        return real_zero_grad(*args, **kwargs)

    relay.optimizer.zero_grad = zero_with_observation
    prepared = relay.prepare(segment, identity, _plan(identity))
    assert len(prepared.scan_result.payloads) == 16
    assert prepared.scan_result.locals[0] is None
    assert all(prefix is not None and prefix.shape == (4, 32) for prefix in prepared.scan_result.locals[1:])
    assert prepared.scan_result.identities == tuple((0, "episode", index) for index in range(16))
    assert relay.sidecar.snapshot() == () and relay.scheduler._committed == {}
    native = _callback(relay.model, seen)

    def serial_forward(payload: dict[str, int], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        assert leaf is None or (leaf.is_leaf and leaf.requires_grad)
        return native(payload, leaf, index)

    outcome = relay.execute_prepared(prepared, serial_forward)
    assert outcome.valid_count == 16 and outcome.outputs == tuple(range(16))
    assert seen == [(index, index, index == 0) for index in range(16)]
    assert len(at_step) == 1 and at_step[0] == ((), {})
    assert len(zero_calls) == 1
    assert relay.scheduler._committed[0] is identity
    saved = relay.sidecar.snapshot()[0][2]
    assert all(value.dtype == torch.float32 and value.grad_fn is None and not value.requires_grad for value in saved)
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in relay.model.net.named_parameters()
        if "local_memory" in name
    )
    gradients = _gradients(relay.model)
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients.values())
    assert torch.count_nonzero(relay.model.net.local_memory_runtime.core.slot_queries.grad) > 0
    assert torch.count_nonzero(relay.model.net.local_memory2llm.weight.grad) > 0
    assert torch.count_nonzero(relay.model.net.local_memory_modality_embed.grad) > 0
    assert all(parameter.grad is None for parameter in relay.model.net.host.parameters())


def test_serial_relay_matches_monolithic_native_mean_gradients() -> None:
    serial_model = _model()
    reference_model = copy.deepcopy(serial_model)
    relay = _runner(serial_model, lr=0.0)
    segment = make_segment()
    identity = _identity()
    serial = relay.run(segment, identity, _plan(identity), _callback(serial_model))

    runtime = reference_model.net.local_memory_runtime
    reference_adapter = CanonicalLocalMemorySegmentAdapter(runtime.encoder, runtime.core, LocalMemorySegmentSidecar())
    transaction = LocalMemoryTransaction(_plan(identity), RankLocalSegmentScheduler())
    reference = reference_adapter.scan(segment, identity=identity, transaction=transaction)
    forward = _callback(reference_model)
    losses = [
        forward(payload, prefix, index).loss
        for index, (payload, prefix) in enumerate(zip(reference.payloads, reference.locals, strict=True))
    ]
    expected_loss = torch.stack(losses).mean()
    expected_loss.backward()
    torch.testing.assert_close(serial.mean_loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
    serial_gradients = _gradients(serial_model)
    reference_gradients = _gradients(reference_model)
    assert serial_gradients.keys() == reference_gradients.keys()
    for name in serial_gradients:
        actual, expected = serial_gradients[name], reference_gradients[name]
        if actual is None or expected is None:
            assert actual is expected is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6, msg=name)
    assert all(parameter.grad is None for parameter in serial_model.net.host.parameters())


def test_native_output_metadata_is_detached() -> None:
    relay = _runner()
    identity = _identity()
    native = _callback(relay.model)

    def with_tensor_output(payload: dict[str, int], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        result = native(payload, leaf, index)
        return NativeConsumerResult(result.loss, output={"loss": result.loss})

    outcome = relay.run(make_segment(), identity, _plan(identity), with_tensor_output)
    assert all(not item["loss"].requires_grad and item["loss"].grad_fn is None for item in outcome.outputs)


def test_b1_producer_segment_reaches_serial_native_callback(tmp_path) -> None:
    producer, _, _ = make_producer(tmp_path)
    identity = producer.identity(slot_id=0, cursor=0, segment_id=0)
    segment = producer.produce(identity)
    relay = _runner()
    native = _callback(relay.model)
    seen = []

    def from_b1(payload: dict[str, Any], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        assert payload["policy_chunk_length"] == 32 and len(payload["rgb_frames"]) == 33
        assert "video_latent" not in payload
        seen.append((index, leaf is None))
        return native({"step": payload["rgb_frames"][0]}, leaf, index)

    outcome = relay.run(segment, identity, _plan(identity), from_b1)
    assert outcome.valid_count == 16
    assert seen == [(index, index == 0) for index in range(16)]
    assert relay.scheduler._committed[0] is identity


@pytest.mark.parametrize(
    "failure",
    [
        "forward",
        "nan",
        "nonscalar",
        "missing_leaf",
        "nonfinite_leaf",
        "auxiliary",
        "s0_grad",
        "relay",
        "slow_grad",
        "optimizer",
    ],
)
def test_failure_discards_pending_and_keeps_frontier(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    relay = _runner()
    identity = _identity()
    prepared = relay.prepare(make_segment(), identity, _plan(identity))
    before = {name: parameter.detach().clone() for name, parameter in relay.model.net.named_parameters()}
    callback = _callback(relay.model)

    def failing_forward(payload: dict[str, int], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        if index == 0 and failure == "s0_grad":
            return NativeConsumerResult(relay.model.net.local_memory2llm.weight.square().sum())
        if index == 1:
            if failure == "forward":
                raise RuntimeError("native forward failed")
            if failure == "nan":
                return NativeConsumerResult(torch.tensor(float("nan")))
            if failure == "nonscalar":
                return NativeConsumerResult(torch.ones(2))
            if failure == "missing_leaf":
                return NativeConsumerResult(torch.tensor(1.0, requires_grad=True))
            if failure == "nonfinite_leaf":

                class BadGradient(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, value):
                        return value.sum().new_tensor(1.0)

                    @staticmethod
                    def backward(ctx, grad_output):
                        return torch.full((4, 32), float("nan"))

                assert leaf is not None
                return NativeConsumerResult(BadGradient.apply(leaf))
            if failure == "auxiliary":
                return NativeConsumerResult(torch.tensor(1.0), sample_coupled_auxiliary_loss=torch.tensor(0.1))
        return callback(payload, leaf, index)

    if failure == "relay":
        monkeypatch.setattr(
            relay, "_relay_gradients", lambda prefixes, gradients: (_ for _ in ()).throw(RuntimeError("relay failed"))
        )
    if failure == "slow_grad":
        real_relay = relay._relay_gradients

        def poison_slow_gradient(prefixes, gradients):
            real_relay(prefixes, gradients)
            relay.model.net.local_memory2llm.weight.grad[0, 0] = float("nan")

        monkeypatch.setattr(relay, "_relay_gradients", poison_slow_gradient)
    if failure == "optimizer":
        monkeypatch.setattr(relay.optimizer, "step", lambda: (_ for _ in ()).throw(RuntimeError("step failed")))
    with pytest.raises((RuntimeError, ValueError)):
        relay.execute_prepared(prepared, failing_forward)
    assert relay.sidecar.snapshot() == () and relay.scheduler._committed == {}
    assert relay.adapter._pending is None and relay._prepared is None
    assert prepared.transaction.failure_code == "discard"
    assert all(parameter.grad is None for parameter in relay._selected_params)
    assert all(torch.equal(parameter, before[name]) for name, parameter in relay.model.net.named_parameters())
    if failure == "optimizer":
        with pytest.raises(RuntimeError, match="restart"):
            relay.prepare(make_segment(), identity, _plan(identity))
    else:
        retry = relay.prepare(make_segment(), identity, _plan(identity))
        relay._discard()
        assert retry.transaction.failure_code == "discard"


@pytest.mark.parametrize("copy_kind", ["capability", "identity", "transaction", "result"])
def test_copied_capability_rejected_without_native_work(copy_kind: str) -> None:
    relay = _runner()
    identity = _identity()
    prepared = relay.prepare(make_segment(), identity, _plan(identity))
    if copy_kind == "capability":
        foreign = replace(prepared)
    elif copy_kind == "identity":
        foreign = replace(prepared, identity=replace(prepared.identity))
    elif copy_kind == "transaction":
        foreign = replace(prepared, transaction=LocalMemoryTransaction(_plan(identity), relay.scheduler))
    else:
        foreign = replace(prepared, scan_result=replace(prepared.scan_result))
    calls = []
    with pytest.raises(RuntimeError, match="capability"):
        relay.execute_prepared(foreign, lambda *args: calls.append(args))
    assert calls == []
    assert relay.adapter._pending is None and relay.sidecar.snapshot() == ()
    assert relay.scheduler._committed == {}


def test_multi_member_and_count_mismatch_rejected_before_native_work() -> None:
    relay = _runner()
    identity = _identity()
    segment = make_segment()
    multi = GAWindowPlan((identity.member, (1, "other", 0)), (16, 16))
    with pytest.raises(ValueError, match="one exact"):
        relay.run(segment, identity, multi, lambda *args: pytest.fail("native work started"))
    with pytest.raises(ValueError, match="counts"):
        relay.run(segment, identity, _plan(identity, 15), lambda *args: pytest.fail("native work started"))
    assert relay.adapter._pending is None and relay.sidecar.snapshot() == ()
    assert relay.scheduler._committed == {}


def test_second_prepare_rejected_while_one_scan_is_pending() -> None:
    relay = _runner()
    identity = _identity()
    prepared = relay.prepare(make_segment(), identity, _plan(identity))
    with pytest.raises(RuntimeError, match="pending"):
        relay.prepare(make_segment(), identity, _plan(identity))
    relay._discard()


def test_continuation_failure_preserves_previous_committed_state() -> None:
    relay = _runner()
    first = _identity()
    relay.run(make_segment(), first, _plan(first), _callback(relay.model))
    saved = relay.sidecar.snapshot()
    frontier = dict(relay.scheduler._committed)
    continuation = _identity(cursor=1)
    with pytest.raises(RuntimeError, match="failed"):
        relay.run(
            make_segment(cursor=1),
            continuation,
            _plan(continuation),
            lambda payload, leaf, index: (_ for _ in ()).throw(RuntimeError("native failed")),
        )
    after = relay.sidecar.snapshot()
    assert len(saved) == len(after) == 1
    assert saved[0][:2] == after[0][:2]
    for old, new in zip(saved[0][2], after[0][2], strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert relay.scheduler._committed == frontier
    assert relay.adapter._pending is None

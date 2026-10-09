"""B2-C R1-A model-owned scan 与 FSDP 注册的 CPU/static 合同。"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot import local_memory_native_segment as relay_module
from cosmos_framework.model.generator.mot import parallelize_vfm_network as parallel_module
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.local_memory_native_segment_test import (
    _callback,
    _identity,
    _model,
    _plan,
    _runner,
)
from cosmos_framework.model.generator.mot.local_memory_segment_test import make_segment


def _with_model_scan(model: nn.Module) -> nn.Module:
    model.net.config = SimpleNamespace(local_memory_enabled=True)
    model.net.scan_local_memory = MethodType(Cosmos3VFMNetwork.scan_local_memory, model.net)
    return model


@pytest.mark.parametrize("local_enabled,dp_enabled", [(True, True), (False, True), (True, False)])
def test_root_fsdp_registers_local_scan_once_only_when_enabled(monkeypatch, local_enabled, dp_enabled):
    model = nn.Module()
    model.language_model = nn.Identity()
    model.config = SimpleNamespace(local_memory_enabled=local_enabled)
    registrations = []
    wraps = []
    monkeypatch.setattr(parallel_module, "parallelize_unified_mot", lambda module, **kwargs: module)
    monkeypatch.setattr(parallel_module, "apply_ac", lambda module, config: None)
    monkeypatch.setattr(parallel_module, "fsdp_mesh", lambda dims: "mesh")

    def shard(*, module, mesh, mp_policy):
        wraps.append((module, mesh))
        return module

    monkeypatch.setattr(parallel_module, "fully_shard", shard)
    monkeypatch.setattr(
        parallel_module,
        "register_fsdp_forward_method",
        lambda module, name: registrations.append((module, name)),
    )
    result = parallel_module.parallelize_vfm_network(
        model,
        SimpleNamespace(cp_enabled=False, dp_enabled=dp_enabled),
        SimpleNamespace(enabled=False),
        SimpleNamespace(mode="none"),
    )
    assert result is model
    assert wraps == ([(model, "mesh")] if dp_enabled else [])
    assert registrations == (
        [(model, "generate_reasoner_text"), (model, "scan_local_memory")]
        if local_enabled and dp_enabled
        else [(model, "generate_reasoner_text")]
        if dp_enabled
        else []
    )
    assert getattr(model, "_local_memory_scan_fsdp_registered", False) is (local_enabled and dp_enabled)


def test_model_owned_scan_reuses_exact_modules_and_matches_direct_b0():
    direct = _model()
    owned = _with_model_scan(copy.deepcopy(direct))
    direct_relay = _runner(direct, lr=0.0)
    owned_relay = _runner(owned, lr=0.0)
    runtime = owned.net.local_memory_runtime
    assert owned_relay.adapter.encoder is runtime.encoder
    assert owned_relay.adapter.core is runtime.core
    assert owned_relay.adapter.scan_local_memory.__self__ is owned.net
    assert (
        sum(parameter.numel() for name, parameter in owned.net.named_parameters() if "local_memory" in name) == 165_312
    )
    assert len(list(owned.net.parameters())) == len(list(direct.net.parameters()))
    segment, identity = make_segment(), _identity()
    direct_result = direct_relay.prepare(segment, identity, _plan(identity)).scan_result
    owned_result = owned_relay.prepare(segment, identity, _plan(identity)).scan_result
    torch.testing.assert_close(owned_result.local_tokens, direct_result.local_tokens)
    torch.testing.assert_close(owned_result.local_present, direct_result.local_present)
    assert owned_result.locals[0] is direct_result.locals[0] is None
    for actual, expected in zip(owned_result.state_out, direct_result.state_out, strict=True):
        torch.testing.assert_close(actual, expected)
    assert owned_relay.sidecar.snapshot() == direct_relay.sidecar.snapshot() == ()


def test_model_owned_scan_matches_s0_and_terminal_padding():
    direct = _model()
    owned = _with_model_scan(copy.deepcopy(direct))
    segment = make_segment()
    valid = torch.zeros_like(segment.consumer_valid)
    valid[:, :2] = True
    evidence_valid = torch.zeros_like(segment.evidence_valid)
    evidence_valid[:, 1] = True
    source = torch.full_like(segment.evidence_source_step, -1)
    source[:, 1] = 0
    segment = replace(
        segment,
        consumer_valid=valid,
        evidence_valid=evidence_valid,
        evidence_source_step=source,
        consumer_payload=((segment.consumer_payload[0][0], segment.consumer_payload[0][1], *(None,) * 14),),
    )
    identity = replace(_identity(), training_stream_end=True)
    plan = _plan(identity, count=2)
    reference = _runner(direct, lr=0.0).prepare(segment, identity, plan).scan_result
    actual = _runner(owned, lr=0.0).prepare(segment, identity, plan).scan_result
    assert actual.locals[0] is reference.locals[0] is None
    assert actual.local_present.tolist() == reference.local_present.tolist() == [[False, True] + [False] * 14]
    torch.testing.assert_close(actual.local_tokens, reference.local_tokens)
    for current, expected in zip(actual.state_out, reference.state_out, strict=True):
        torch.testing.assert_close(current, expected)


def test_model_owned_route_skips_direct_adapter_scan_and_preserves_relay_gradients():
    direct = _model()
    owned = _with_model_scan(copy.deepcopy(direct))
    direct_relay = _runner(direct, lr=0.0)
    owned_relay = _runner(owned, lr=0.0)

    owned_scan = owned_relay.adapter.scan_local_memory
    calls = []

    def observed_scan(visual, action, valid, state):
        calls.append((visual, action, valid, state))
        return owned_scan(visual, action, valid, state)

    owned_relay.adapter.scan_local_memory = observed_scan
    segment, identity = make_segment(), _identity()
    direct_out = direct_relay.run(segment, identity, _plan(identity), _callback(direct))
    owned_out = owned_relay.run(segment, identity, _plan(identity), _callback(owned))
    assert len(calls) == 1
    torch.testing.assert_close(owned_out.mean_loss, direct_out.mean_loss)
    for name, parameter in owned.net.named_parameters():
        if "local_memory" in name:
            reference = dict(direct.net.named_parameters())[name]
            assert parameter.grad is not None and reference.grad is not None
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-4, atol=2e-6, msg=name)
    assert owned_relay.scheduler._committed[0] is identity
    assert direct_relay.scheduler._committed[0] is identity
    for _, _, state in owned_relay.sidecar.snapshot():
        assert all(value.dtype == torch.float32 and not value.requires_grad for value in state)


def test_model_owned_route_failure_does_not_publish():
    relay = _runner(_with_model_scan(_model()))
    identity = _identity()
    prepared = relay.prepare(make_segment(), identity, _plan(identity))

    def fail(*args):
        raise RuntimeError("native failed")

    with pytest.raises(RuntimeError, match="native failed"):
        relay.execute_prepared(prepared, fail)
    assert relay.sidecar.snapshot() == ()
    assert relay.scheduler._committed == {}


def test_dtensor_owned_local_without_model_scan_fails_before_b0(monkeypatch):
    monkeypatch.setattr(relay_module, "_local_params_are_dtensors", lambda runtime: True)
    with pytest.raises(RuntimeError, match="require registered model scan_local_memory"):
        _runner(_model())
    with pytest.raises(RuntimeError, match="require registered model scan_local_memory"):
        _runner(_with_model_scan(_model()))


@pytest.mark.parametrize("mode", ("fresh", "continuation", "mixed"))
def test_w0_fast_grad_participation_under_rank_divergent_slot_mix(mode: str) -> None:
    """All-continuation must yield zero (not None) w0 grads for FSDP2 parity."""
    torch.manual_seed(43)
    model = _with_model_scan(_model())
    runtime = model.net.local_memory_runtime
    batch = 2
    visual = torch.randn(batch, 1, 96)
    actions = torch.randn(batch, 1, 15)
    valid = torch.ones(batch, 1, dtype=torch.bool)
    states = None if mode == "fresh" else runtime.core.detach_state(runtime.core.initial_state(batch))
    mask = torch.tensor([True, False]) if mode == "mixed" else None

    result, candidate, present = model.net.scan_local_memory(visual, actions, valid, states, continuation_mask=mask)
    assert bool(present.all())
    if mode == "continuation":
        # Compare the exact unmodified continuation forward, not a reference
        # with fresh resets. The additional path must contribute zero value.
        reference, expected_state, _ = runtime.core.scan_segment_masked_encoded_many(
            runtime.encoder, visual, actions, valid, states
        )
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        for got, expected in zip(candidate, expected_state, strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    result.square().sum().backward()
    w0 = (
        runtime.core.w0_fast_in_weight,
        runtime.core.w0_fast_in_bias,
        runtime.core.w0_fast_out_weight,
        runtime.core.w0_fast_out_bias,
    )
    assert all(parameter.grad is not None for parameter in w0)
    if mode == "continuation":
        assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in w0)
    else:
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in w0)


def test_continuation_mask_requires_state_and_remains_strict() -> None:
    net = _with_model_scan(_model()).net
    visual = torch.randn(2, 1, 96)
    action = torch.randn(2, 1, 15)
    valid = torch.ones(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="batched state"):
        net.scan_local_memory(visual, action, valid, None, continuation_mask=torch.tensor([True, False]))
    state = net.local_memory_runtime.core.detach_state(net.local_memory_runtime.core.initial_state(2))
    with pytest.raises(ValueError, match="fresh/continuation"):
        net.scan_local_memory(visual, action, valid, state, continuation_mask=torch.tensor([True, True]))

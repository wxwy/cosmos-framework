# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU mathematics proof for splitting the unchanged model-owned Local scan.

The actual tiny Cosmos network owns every encoder/core/prefix/action parameter.
Its language tower is an identity: these are not DCP, denoising, or GPU tests.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest
from cosmos_framework.model.generator.mot.action_io_projector import ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTFastState
from cosmos_framework.model.generator.mot.memory_prefix import build_memory_prefix_context


class _IdentityLanguage(nn.Identity):
    def init_weights(self, buffer_device=None):
        pass


def _network(dtype: torch.dtype = torch.float32) -> Cosmos3VFMNetwork:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2207)
        language = PretrainedConfig(
            hidden_size=8, num_attention_heads=2, num_key_value_heads=2, head_dim=4, num_hidden_layers=1
        )
        config = Cosmos3VFMNetworkConfig(
            vlm_config=language,
            vision_gen=True,
            latent_patch_size=1,
            latent_channel_size=1,
            action_gen=True,
            action_dim=15,
            num_embodiment_domains=1,
            action_io_projector_type=ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS,
            local_memory_enabled=True,
            local_memory_evidence_dim=16,
            local_memory_dim=4,
            local_memory_ttt_dim=8,
            local_memory_fast_hidden_dim=16,
            local_memory_ttt_tbptt_steps=16,
            local_memory_k_local=2,
        )
        net = Cosmos3VFMNetwork(_IdentityLanguage(), config)
        net.init_weights(buffer_device=None)
    net = net.to(device="cpu", dtype=dtype).eval()
    assert type(net) is Cosmos3VFMNetwork
    assert net.scan_local_memory.__self__ is net
    assert net.scan_local_memory.__func__ is Cosmos3VFMNetwork.scan_local_memory
    return net


def _evidence(length: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # Distinct, bounded rows make reordering or a repeated first action observable.
    visual = torch.arange(offset * 96, (offset + length) * 96, dtype=torch.float32).reshape(1, length, 96)
    action = torch.arange(offset * 15, (offset + length) * 15, dtype=torch.float32).reshape(1, length, 15)
    return visual.sin(), (action / 17).cos()


def _assert_detached(state: ContinualTTTFastState) -> None:
    assert isinstance(state, ContinualTTTFastState)
    assert len(state) == 4
    for value in state:
        assert type(value) is torch.Tensor
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None and not value.is_inference()
        assert torch.isfinite(value).all()


def _inventory(net: Cosmos3VFMNetwork):
    return (
        {name: (id(parameter), parameter.dtype, parameter.requires_grad) for name, parameter in net.named_parameters()},
        {name: value.detach().clone() for name, value in net.state_dict().items()},
        (id(net.local_memory_runtime), id(net.local_memory_runtime.encoder), id(net.local_memory_runtime.core)),
    )


def _assert_inventory(net: Cosmos3VFMNetwork, before) -> None:
    parameters, values, runtime_ids = before
    assert {name: (id(p), p.dtype, p.requires_grad) for name, p in net.named_parameters()} == parameters
    assert tuple(net.state_dict()) == tuple(values)
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0, check_dtype=True)
    assert all(parameter.grad is None for parameter in net.parameters())
    assert (id(net.local_memory_runtime), id(net.local_memory_runtime.encoder), id(net.local_memory_runtime.core)) == (
        runtime_ids
    )


@dataclass(frozen=True)
class _Result:
    tokens: torch.Tensor
    state: ContinualTTTFastState
    telemetry: dict[str, torch.Tensor]


def _scan(
    net: Cosmos3VFMNetwork,
    visual: torch.Tensor,
    action: torch.Tensor,
    *,
    chunk: int,
    state: ContinualTTTFastState | None = None,
) -> _Result:
    core, encoder = net.local_memory_runtime.core, net.local_memory_runtime.encoder
    core.reset_telemetry()
    encoded_actions, tokens, totals = [], [], {}
    handle = encoder.action_proj.register_forward_pre_hook(
        lambda module, args: encoded_actions.append(args[0].detach().clone())
    )
    try:
        with torch.inference_mode():
            for start in range(0, visual.shape[1], chunk):
                end = min(start + chunk, visual.shape[1])
                if state is not None:
                    _assert_detached(state)
                with torch.inference_mode(False), torch.enable_grad():
                    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                    prefix, candidate, present = net.scan_local_memory(
                        visual[:, start:end],
                        action[:, start:end],
                        torch.ones(1, end - start, dtype=torch.bool),
                        state,
                        create_graph=False,
                    )
                    assert prefix.dtype == torch.float32
                    assert prefix.shape == (1, end - start, core.k_local, core.local_dim)
                    assert present.dtype == torch.bool and present.tolist() == [[True] * (end - start)]
                    state = OnlineLocalMemory._detach_state(candidate)
                    _assert_detached(state)
                    tokens.append(prefix.detach().clone())
                    for key, value in core.drain_telemetry().items():
                        assert not value.requires_grad and value.grad_fn is None
                        if key.endswith(("_sum", "_count")):
                            totals[key] = value.clone() if key not in totals else totals[key] + value
                assert torch.is_inference_mode_enabled()
    finally:
        handle.remove()
    assert state is not None
    assert len(encoded_actions) == visual.shape[1]
    torch.testing.assert_close(
        torch.cat(encoded_actions, dim=1), action.to(encoder.action_proj.weight.dtype), rtol=0, atol=0
    )
    return _Result(torch.cat(tokens, dim=1), state, totals)


def _same_result(actual: _Result, expected: _Result, length: int) -> None:
    torch.testing.assert_close(actual.tokens, expected.tokens, rtol=2e-5, atol=2e-6, check_dtype=True)
    for value, reference in zip(actual.state, expected.state, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-6, check_dtype=True)
    assert actual.telemetry.keys() == expected.telemetry.keys()
    for key, value in actual.telemetry.items():
        torch.testing.assert_close(value, expected.telemetry[key], rtol=2e-5, atol=2e-6, check_dtype=True)
    for name in ("inner_loss", "fast_update_norm", "fast_state_norm"):
        assert actual.telemetry[f"ttt_{name}_count"].item() == length
        assert actual.telemetry[f"ttt_{name}_sum"].item() > 0


@pytest.fixture
def inner_grad_calls(monkeypatch):
    original = torch.autograd.grad
    calls = []

    def observed(*args, **kwargs):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert kwargs["create_graph"] is False
        gradients = original(*args, **kwargs)
        assert all(not value.requires_grad and value.grad_fn is None for value in gradients)
        calls.append(1)
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", observed)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("continuation", [False, True], ids=["fresh", "continuation"])
@pytest.mark.parametrize("length", [1, 4, 16])
def test_real_model_scan_whole_matches_single_steps(length, continuation, dtype, inner_grad_calls, record_property):
    whole = _network(dtype)
    single = copy.deepcopy(whole)
    before_whole, before_single = _inventory(whole), _inventory(single)
    whole_state = single_state = None
    if continuation:
        warm_visual, warm_action = _evidence(3)
        warm_whole = _scan(whole, warm_visual, warm_action, chunk=3)
        warm_single = _scan(single, warm_visual, warm_action, chunk=1)
        _same_result(warm_single, warm_whole, 3)
        whole_state, single_state = warm_whole.state, warm_single.state
    visual, action = _evidence(length, offset=3)
    reference = _scan(whole, visual, action, chunk=length, state=whole_state)
    result = _scan(single, visual, action, chunk=1, state=single_state)
    _same_result(result, reference, length)
    record_property(
        "issue22_scan_parity",
        json.dumps(
            {
                "length": length,
                "continuation": continuation,
                "dtype": str(dtype),
                "prefix_max_abs": float((result.tokens - reference.tokens).abs().max()),
                "state_max_abs": max(
                    float((a - b).abs().max()) for a, b in zip(result.state, reference.state, strict=True)
                ),
                "telemetry_max_abs": max(
                    float((result.telemetry[key] - reference.telemetry[key]).abs().max()) for key in result.telemetry
                ),
                "rtol": 2e-5,
                "atol": 2e-6,
            },
            sort_keys=True,
        ),
    )
    assert len(inner_grad_calls) == 2 * (length + (3 if continuation else 0))
    _assert_inventory(whole, before_whole)
    _assert_inventory(single, before_single)


def _request(consumer: int, start: int, visual: torch.Tensor, action: torch.Tensor) -> OnlineMemoryRequest:
    return OnlineMemoryRequest("session", "episode", consumer, tuple(range(start, consumer)), visual, action)


def _consume_prefix(net: Cosmos3VFMNetwork, prefix: torch.Tensor) -> torch.Tensor:
    # Actual prefix and action projection modules; no transformer/denoiser is claimed.
    with torch.inference_mode():
        context = build_memory_prefix_context(
            (prefix,),
            net.local_memory2llm,
            net.local_memory_modality_embed,
            target_dtype=net.local_memory2llm.weight.dtype,
            k_local=net.local_memory_runtime.core.k_local,
        )
        assert context is not None
        assert context.hidden.dtype == net.local_memory2llm.weight.dtype
        assert context.present.tolist() == [True] and context.sample_offsets.tolist() == [0, 2]
        result = net.llm2action(context.hidden.mean(dim=0, keepdim=True), torch.zeros(1, dtype=torch.long))
        assert result.shape == (1, 15) and result.dtype == net.local_memory2llm.weight.dtype
        return result.detach().clone()


@pytest.mark.parametrize("length", [1, 4, 16])
def test_real_online_partial_requests_match_whole_scan_and_prefix_consumer(length, inner_grad_calls, record_property):
    whole, single = _network(), _network()
    before_whole, before_single = _inventory(whole), _inventory(single)
    memories, scan_calls = [], []
    for net in (whole, single):
        owned_scan = net.scan_local_memory
        calls = []

        def observed(visual, action, valid, state, *, create_graph, scan=owned_scan, calls=calls):
            assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            assert create_graph is False
            if state is not None:
                _assert_detached(state)
            calls.append((action.detach().clone(), state is None))
            return scan(visual, action, valid, state, create_graph=create_graph)

        runtime = net.local_memory_runtime
        memory = OnlineLocalMemory(runtime.encoder, runtime.core, scan_local_memory=observed)
        assert memory.encoder is runtime.encoder and memory.core is runtime.core
        with torch.inference_mode():
            cold = memory.prepare(_request(0, 0, torch.empty(0, 96), torch.empty(0, 15)))
            assert cold.token is None and cold.replacement.state is None
            assert cold.telemetry["adapted_steps"] == 0 and calls == []
            memory.commit(cold)
        memories.append(memory)
        scan_calls.append(calls)
    visual, action = _evidence(length)
    with torch.inference_mode():
        bulk = memories[0].prepare(_request(length, 0, visual[0], action[0]))
        adapted, loss_sum, update_sum = 0.0, 0.0, 0.0
        for index in range(length):
            update = memories[1].prepare(
                _request(index + 1, index, visual[0, index : index + 1], action[0, index : index + 1])
            )
            assert update.telemetry["adapted_steps"] == 1
            _assert_detached(update.replacement.state)
            assert update.token.dtype == torch.float32 and not update.token.requires_grad
            adapted += update.telemetry["adapted_steps"]
            loss_sum += update.telemetry["inner_loss_mean"]
            update_sum += update.telemetry["fast_update_norm"]
            memories[1].commit(update)
        assert adapted == bulk.telemetry["adapted_steps"] == length
        assert loss_sum == pytest.approx(bulk.telemetry["inner_loss_mean"] * length, rel=2e-5, abs=2e-6)
        assert update_sum == pytest.approx(bulk.telemetry["fast_update_norm"] * length, rel=2e-5, abs=2e-6)
        torch.testing.assert_close(update.token, bulk.token, rtol=2e-5, atol=2e-6, check_dtype=True)
        for value, reference in zip(update.replacement.state, bulk.replacement.state, strict=True):
            torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-6, check_dtype=True)
        # Candidates remain unpublished until the existing explicit transaction commit.
        assert memories[0]._records["session"].consumer_step == 0
        memories[0].commit(bulk)
    assert [entry[0].shape[1] for entry in scan_calls[0]] == [length]
    assert [entry[0].shape[1] for entry in scan_calls[1]] == [1] * length
    assert [entry[1] for entry in scan_calls[1]] == [True] + [False] * (length - 1)
    for calls in scan_calls:
        torch.testing.assert_close(torch.cat([entry[0] for entry in calls], dim=1), action, rtol=0, atol=0)
    assert len(inner_grad_calls) == 2 * length
    for memory in memories:
        assert memory._records["session"].consumer_step == length
        _assert_detached(memory._records["session"].state)
    actual_action, expected_action = _consume_prefix(single, update.token), _consume_prefix(whole, bulk.token)
    torch.testing.assert_close(actual_action, expected_action)
    record_property(
        "issue22_online_parity",
        json.dumps(
            {
                "length": length,
                "prefix_max_abs": float((update.token - bulk.token).abs().max()),
                "state_max_abs": max(
                    float((a - b).abs().max())
                    for a, b in zip(update.replacement.state, bulk.replacement.state, strict=True)
                ),
                "inner_loss_sum_abs": abs(loss_sum - bulk.telemetry["inner_loss_mean"] * length),
                "action_max_abs": float((actual_action - expected_action).abs().max()),
                "adapted_steps": adapted,
                "rtol": 2e-5,
                "atol": 2e-6,
            },
            sort_keys=True,
        ),
    )
    _assert_inventory(whole, before_whole)
    _assert_inventory(single, before_single)

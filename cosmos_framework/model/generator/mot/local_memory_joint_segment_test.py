# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-A 单 slot/T16 joint autograd CPU 合同。"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import cosmos_framework.model.generator.mot.attention as attention_module
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, two_way_attention
from cosmos_framework.model.generator.mot.local_memory_joint_segment import (
    SingleSegmentNativeJointAutograd,
)
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, SegmentIdentity
from cosmos_framework.model.generator.mot.local_memory_segment_test import make_segment
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime, build_memory_prefix_context


def _model() -> nn.Module:
    torch.manual_seed(37)
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_memory_runtime = LocalMemoryRuntime()
    model.net.local_memory2llm = nn.Linear(32, 8)
    model.net.local_memory_modality_embed = nn.Parameter(torch.randn(8) * 0.02)
    model.net.moe_gen = nn.Linear(8, 8)
    model.net.action2llm = nn.Linear(8, 8)
    model.net.llm2action = nn.Linear(8, 1)
    model.net.reasoner = nn.Linear(8, 8)
    for parameter in model.net.reasoner.parameters():
        parameter.requires_grad_(False)
    return model


def _identity() -> SegmentIdentity:
    return SegmentIdentity(0, "episode", "robocasa", 0, 0, "source")


def _native_loss(model: nn.Module, seen: list[torch.Tensor | None]):
    def pack(value: torch.Tensor):
        result, _, _ = build_packed_sequence(
            "two_way",
            packed_sequence=value,
            attn_modes=["causal", "full"],
            split_lens=[2, 2],
            sample_lens=[4],
            packed_und_token_indexes=torch.tensor([0, 1]),
            packed_gen_token_indexes=torch.tensor([2, 3]),
            num_heads=1,
            head_dim=8,
            num_layers=1,
        )
        return result

    def callback(payload: dict[str, int], prefix: torch.Tensor | None, index: int) -> torch.Tensor:
        seen.append(prefix)
        context = None
        if prefix is not None:
            context = build_memory_prefix_context(
                (prefix,),
                model.net.local_memory2llm,
                model.net.local_memory_modality_embed,
                target_dtype=torch.float32,
            )
            assert context is not None
        native = model.net.action2llm(model.net.moe_gen(torch.ones(4, 8))).view(4, 1, 8)
        output = two_way_attention(
            pack(native),
            pack(native),
            pack(native),
            memory_prefix_key_states=None if context is None else context.hidden.view(4, 1, 8),
            memory_prefix_value_states=None if context is None else context.hidden.view(4, 1, 8),
            memory_prefix_sample_offsets=None if context is None else context.sample_offsets,
        )
        action = model.net.llm2action(get_gen_seq(output).mean(dim=0).flatten()).squeeze()
        return (action - payload["step"] / 10).square()

    return callback


def _cpu_varlen_attention(query, key, value, **kwargs):
    q_offsets = kwargs.get("cumulative_seqlen_Q")
    kv_offsets = kwargs.get("cumulative_seqlen_KV")
    if q_offsets is None:
        q_offsets = torch.tensor([0, query.shape[1]])
        kv_offsets = torch.tensor([0, key.shape[1]])
    outputs = []
    for row in range(q_offsets.numel() - 1):
        q_part = query[:, q_offsets[row] : q_offsets[row + 1]].transpose(1, 2)
        k_part = key[:, kv_offsets[row] : kv_offsets[row + 1]].transpose(1, 2)
        v_part = value[:, kv_offsets[row] : kv_offsets[row + 1]].transpose(1, 2)
        outputs.append(
            F.scaled_dot_product_attention(q_part, k_part, v_part, is_causal=kwargs.get("is_causal", False)).transpose(
                1, 2
            )
        )
    return torch.cat(outputs, dim=1)


def test_joint_t16_one_outer_backward_reaches_host_and_complete_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "attention", _cpu_varlen_attention)
    model = _model()
    runner = SingleSegmentNativeJointAutograd(model)
    identity = _identity()
    prepared = runner.prepare(make_segment(), identity, GAWindowPlan((identity.member,), (16,)))
    seen: list[torch.Tensor | None] = []
    loss = runner.forward_prepared(prepared, _native_loss(model, seen))
    assert len(seen) == 16 and seen[0] is None
    assert all(seen[index] is prepared.scan_result.locals[index] for index in range(1, 16))
    assert all(prefix is not None and prefix.grad_fn is not None and not prefix.is_leaf for prefix in seen[1:])
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert runner.sidecar.snapshot() == () and runner.scheduler._committed == {}

    loss.backward()
    local = [(name, parameter) for name, parameter in model.net.named_parameters() if "local_memory" in name]
    assert local and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for _, parameter in local)
    for name in (
        "local_memory_runtime.encoder.visual_proj.weight",
        "local_memory_runtime.encoder.action_proj.weight",
        "local_memory_runtime.core.slot_queries",
        "local_memory_runtime.core.w0_fast_in_weight",
        "local_memory2llm.weight",
        "local_memory_modality_embed",
    ):
        assert torch.count_nonzero(dict(local)[name].grad) > 0
    for name in ("moe_gen.weight", "action2llm.weight", "llm2action.weight"):
        assert torch.count_nonzero(dict(model.net.named_parameters())[name].grad) > 0
    assert all(parameter.grad is None for parameter in model.net.reasoner.parameters())
    state = model.net.local_memory_runtime.core.initial_state(1)
    assert all(not isinstance(value, nn.Parameter) for value in state)
    assert not any(id(value) == id(parameter) for value in state for parameter in model.parameters())
    runner.discard(prepared)
    assert runner.sidecar.snapshot() == () and runner.scheduler._committed == {}


def test_callback_failure_discards_candidate_without_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "attention", _cpu_varlen_attention)
    model = _model()
    runner = SingleSegmentNativeJointAutograd(model)
    identity = _identity()
    prepared = runner.prepare(make_segment(), identity, GAWindowPlan((identity.member,), (16,)))

    def fail(payload: dict[str, int], prefix: torch.Tensor | None, index: int) -> torch.Tensor:
        if index == 3:
            raise RuntimeError("native failure")
        return _native_loss(model, [])(payload, prefix, index)

    with pytest.raises(RuntimeError, match="native failure"):
        runner.forward_prepared(prepared, fail)
    assert runner.sidecar.snapshot() == () and runner.scheduler._committed == {}
    assert runner.adapter._pending is None and runner._prepared is None


def test_joint_gradient_matches_independent_direct_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "attention", _cpu_varlen_attention)
    model = _model()
    reference = copy.deepcopy(model)
    segment = make_segment()
    identity = _identity()
    runner = SingleSegmentNativeJointAutograd(model)
    prepared = runner.prepare(segment, identity, GAWindowPlan((identity.member,), (16,)))
    runner.forward_prepared(prepared, _native_loss(model, [])).backward()

    runtime = reference.net.local_memory_runtime
    tokens, _, present = runtime.core.scan_segment_masked_encoded_many(
        runtime.encoder,
        segment.evidence_visual_summary_prev,
        segment.evidence_executed_action_prev,
        segment.evidence_valid,
    )
    payloads, prefixes, _ = segment.gather_consumers(tokens, present)
    callback = _native_loss(reference, [])
    (
        sum(callback(payload, prefix, index) for index, (payload, prefix) in enumerate(zip(payloads, prefixes))) / 16
    ).backward()
    for (name, parameter), (reference_name, expected) in zip(
        model.net.named_parameters(), reference.net.named_parameters(), strict=True
    ):
        assert name == reference_name
        if parameter.grad is None:
            assert expected.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=1e-6)
    runner.discard(prepared)


def test_joint_source_has_no_detached_leaf_or_optimizer_path() -> None:
    source = inspect.getsource(SingleSegmentNativeJointAutograd)
    assert ".detach(" not in source
    assert "relay_gradients" not in source
    assert "optimizer.step" not in source
    assert "torch.no_grad" not in source

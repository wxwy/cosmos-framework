# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B2-A Local prefix CPU contracts; no trainer, checkpoint or GPU execution."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import cosmos_framework.model.generator.mot.attention as attention_module
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, dispatch_attention, two_way_attention
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.memory_prefix import (
    LocalMemoryRuntime,
    MemoryPrefixContext,
    attach_local_prefixes,
    build_memory_prefix_context,
    prepend_memory_kv,
)
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata


def _local_owner(hidden_size: int = 2048) -> nn.Module:
    owner = nn.Module()
    owner.local_memory_runtime = LocalMemoryRuntime()
    owner.local_memory2llm = nn.Linear(32, hidden_size)
    owner.local_memory_modality_embed = nn.Parameter(torch.randn(hidden_size))
    return owner


def _cpu_varlen_attention(query, key, value, **kwargs):
    """CPU reference kernel for the production varlen attention call contract."""
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


def test_selected_inventory_and_fast_state() -> None:
    owner = _local_owner()
    selected = [(name, value) for name, value in owner.named_parameters() if "local_memory" in name]
    assert sum(value.numel() for _, value in selected) == 165_312
    assert len(selected) == len(list(owner.parameters()))
    assert owner.local_memory_runtime.encoder.action_proj.in_features == 15
    state = owner.local_memory_runtime.core.initial_state(2)
    assert all(value.dtype == torch.float32 and not isinstance(value, nn.Parameter) for value in state)
    assert all("fast_state" not in name for name, _ in selected)


def test_optimizer_allowlist_freezes_host_and_selects_complete_local_inventory() -> None:
    model = nn.Module()
    model.net = _local_owner()
    model.net.moe_gen = nn.Linear(4, 4)
    state = model.net.local_memory_runtime.core.initial_state(1)
    local_params = {name: value for name, value in model.net.named_parameters() if "local_memory" in name}
    selected = _build_params_with_metadata(
        model,
        keys_to_select=["local_memory"],
        lr_multipliers={},
        base_lr=1e-4,
        base_weight_decay=0.0,
        disable_weight_decay_for_1d_params=False,
    )
    selected_ids = {id(parameter) for parameter, _ in selected}
    assert sum(parameter.numel() for parameter, _ in selected) == 165_312
    assert selected_ids == {id(parameter) for parameter in local_params.values()}
    assert all(
        "local_memory" in name for name, parameter in model.net.named_parameters() if id(parameter) in selected_ids
    )
    assert not model.net.moe_gen.weight.requires_grad and not model.net.moe_gen.bias.requires_grad
    assert all(id(value) not in selected_ids for value in state)
    assert not any(id(value) == id(parameter) for value in state for parameter in model.parameters())


def test_materialized_encoder_is_initialized() -> None:
    with torch.device("meta"):
        runtime = LocalMemoryRuntime()
        bridge = nn.Linear(32, 2048)
        embed = nn.Parameter(torch.empty(2048))
    holder = nn.Module()
    holder.local_memory_runtime = runtime
    holder.local_memory2llm = bridge
    holder.local_memory_modality_embed = embed
    holder.to_empty(device="cpu")
    holder.hidden_size = 2048
    holder.config = SimpleNamespace(
        local_memory_enabled=True, local_memory_dim=32, vision_gen=False, action_gen=False, sound_gen=False
    )
    holder.language_model = SimpleNamespace(init_weights=lambda buffer_device: None)
    Cosmos3VFMNetwork.init_weights(holder, buffer_device=torch.device("cpu"))
    for module in (runtime.encoder.visual_proj, runtime.encoder.action_proj, runtime.encoder.norm):
        assert all(torch.isfinite(value).all() for value in module.parameters())
    assert torch.count_nonzero(bridge.weight) > 0


def test_first_outer_backward_reaches_ttt_and_bridge_input() -> None:
    owner = _local_owner(hidden_size=32)
    core = owner.local_memory_runtime.core
    evidence = torch.randn(1, 256)
    tokens, _ = core.step_many(evidence, core.initial_state(1))
    tokens.retain_grad()
    context = build_memory_prefix_context(
        (tokens[0],), owner.local_memory2llm, owner.local_memory_modality_embed, target_dtype=torch.float32
    )
    assert context is not None
    query = torch.randn(1, 32)
    logits = query @ context.hidden.T
    loss = (torch.softmax(logits, dim=-1) @ context.hidden).square().mean()
    loss.backward()
    assert torch.count_nonzero(tokens.grad) > 0
    assert torch.count_nonzero(core.slot_queries.grad) > 0
    assert torch.count_nonzero(owner.local_memory2llm.weight.grad) > 0


def test_fp32_bridge_bf16_native_dtype_keeps_gradients() -> None:
    projector = nn.Linear(32, 64, dtype=torch.float32)
    embed = nn.Parameter(torch.randn(64, dtype=torch.float32))
    upstream = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    context = build_memory_prefix_context((upstream,), projector, embed, target_dtype=torch.bfloat16)
    assert context is not None and context.hidden.dtype == torch.bfloat16
    context.hidden.float().square().mean().backward()
    assert torch.isfinite(projector.weight.grad).all() and torch.count_nonzero(projector.weight.grad) > 0
    assert torch.isfinite(upstream.grad).all() and torch.count_nonzero(upstream.grad) > 0


def test_mixed_sample_interleave_is_isolated_and_tensorized() -> None:
    bridge = nn.Linear(32, 8)
    context = build_memory_prefix_context(
        (None, torch.ones(4, 32), None), bridge, nn.Parameter(torch.zeros(8)), target_dtype=torch.float32
    )
    assert context is not None
    assert context.sample_offsets.tolist() == [0, 0, 4, 4]
    native_k = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    native_v = native_k * 10
    local_k = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    local_v = local_k * 10
    key, value, offsets, max_len = prepend_memory_kv(
        local_k,
        local_v,
        context.sample_offsets,
        native_k,
        native_v,
        torch.tensor([0, 1, 3, 4]),
        max_native_len=2,
        k_local=4,
    )
    assert offsets.tolist() == [0, 1, 7, 8]
    torch.testing.assert_close(key.flatten(), torch.tensor([1.0, 5, 6, 7, 8, 2, 3, 4]))
    torch.testing.assert_close(value, key * 10)
    assert max_len == 6
    source = inspect.getsource(prepend_memory_kv)
    assert ".tolist(" not in source and ".item(" not in source and "int(" not in source


def test_native_pad_segment_has_no_local_rows() -> None:
    key, _, offsets, _ = prepend_memory_kv(
        torch.full((4, 1), 9.0),
        torch.full((4, 1), 9.0),
        torch.tensor([0, 4]),
        torch.tensor([[1.0], [2.0], [0.0]]),
        torch.tensor([[1.0], [2.0], [0.0]]),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        max_native_len=2,
        k_local=4,
    )
    assert offsets.tolist() == [0, 6, 7]
    torch.testing.assert_close(key.flatten(), torch.tensor([9, 9, 9, 9, 1, 2, 0], dtype=torch.float32))


def test_two_way_prefix_changes_only_generation_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "attention", _cpu_varlen_attention)

    def pack(value: torch.Tensor):
        result, _, _ = build_packed_sequence(
            "two_way",
            packed_sequence=value,
            attn_modes=["causal", "full", "causal", "full"],
            split_lens=[2, 2, 2, 2],
            sample_lens=[4, 4],
            packed_und_token_indexes=torch.tensor([0, 1, 4, 5]),
            packed_gen_token_indexes=torch.tensor([2, 3, 6, 7]),
            num_heads=1,
            head_dim=8,
            num_layers=1,
        )
        return result

    torch.manual_seed(4)
    q = pack(torch.randn(8, 1, 8))
    k = pack(torch.randn(8, 1, 8))
    v = pack(torch.randn(8, 1, 8))
    native = two_way_attention(q, k, v)
    prefix_k = torch.randn(4, 1, 8)
    prefix_v = torch.randn(4, 1, 8)
    augmented = two_way_attention(
        q,
        k,
        v,
        memory_prefix_key_states=prefix_k,
        memory_prefix_value_states=prefix_v,
        memory_prefix_sample_offsets=torch.tensor([0, 0, 4]),
    )
    torch.testing.assert_close(get_und_seq(augmented), get_und_seq(native), rtol=0, atol=0)
    torch.testing.assert_close(get_gen_seq(augmented)[:2], get_gen_seq(native)[:2], rtol=0, atol=0)
    assert not torch.allclose(get_gen_seq(augmented)[2:], get_gen_seq(native)[2:])


def test_native_two_way_outer_loss_reaches_ttt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "attention", _cpu_varlen_attention)
    owner = _local_owner(hidden_size=8)
    core = owner.local_memory_runtime.core
    tokens, _ = core.step_many(torch.randn(1, 256), core.initial_state(1))
    tokens.retain_grad()
    context = build_memory_prefix_context(
        (tokens[0],),
        owner.local_memory2llm,
        owner.local_memory_modality_embed,
        target_dtype=torch.float32,
    )
    assert context is not None

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

    output = two_way_attention(
        pack(torch.randn(4, 1, 8)),
        pack(torch.randn(4, 1, 8)),
        pack(torch.randn(4, 1, 8)),
        memory_prefix_key_states=context.hidden.view(4, 1, 8),
        memory_prefix_value_states=context.hidden.view(4, 1, 8),
        memory_prefix_sample_offsets=context.sample_offsets,
    )
    get_gen_seq(output)[:2].square().mean().backward()
    assert torch.count_nonzero(tokens.grad) > 0
    assert torch.count_nonzero(core.slot_queries.grad) > 0


def test_layer_projects_only_generator_kv_without_rope() -> None:
    config = Qwen3VLTextConfig(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        attention_bias=False,
    )
    layer = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=LayerTypes("qwen3_vl_dense"),
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
    )

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

    hidden = pack(torch.randn(4, 8))
    cos = pack(torch.zeros(4, 8))
    sin = pack(torch.ones(4, 8))
    prefix_hidden = torch.randn(4, 8)
    context = MemoryPrefixContext(prefix_hidden, torch.tensor([0, 4]), torch.tensor([True]), 4)
    captured = []

    def dispatch(query, key, value, mask, **kwargs):
        captured.append((query, kwargs))
        return query, None

    layer.dispatch_attention_fn = dispatch
    layer(hidden, SimpleNamespace(), (cos, sin), memory_prefix_context=context)
    query, kwargs = captured[-1]
    assert query["_num_full_tokens"] == 2
    expected_k = layer.k_norm_moe_gen(layer.k_proj_moe_gen(prefix_hidden).view(4, 1, 8))
    expected_v = layer.v_proj_moe_gen(prefix_hidden).view(4, 1, 8)
    torch.testing.assert_close(kwargs["memory_prefix_key_states"], expected_k)
    torch.testing.assert_close(kwargs["memory_prefix_value_states"], expected_v)
    layer(hidden, SimpleNamespace(), (cos, sin))
    assert "memory_prefix_key_states" not in captured[-1][1]


def test_private_attach_preserves_native_carriers_and_rejects_duplicate() -> None:
    plans = [SequencePlan(has_text=True, has_vision=True) for _ in range(2)]
    clean = GenerationDataClean(batch_size=2, is_image_batch=False, x0_tokens_vision=[torch.zeros(1, 16, 1, 2, 2)] * 2)
    copied_plans, copied_clean = attach_local_prefixes(plans, clean, (None, torch.randn(4, 32)))
    assert [plan.has_local_memory for plan in copied_plans] == [False, True]
    assert not any(plan.has_local_memory for plan in plans)
    assert clean.x0_tokens_local_memory is None
    assert len(copied_clean.x0_tokens_local_memory) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        attach_local_prefixes(copied_plans, copied_clean, (None, torch.randn(4, 32)))


def test_packing_keeps_native_indexes_and_loss_unchanged() -> None:
    plans = [SequencePlan(has_text=True, has_vision=True) for _ in range(2)]
    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 16, 1, 4, 4) for _ in range(2)],
    )
    tokens = [[30] * 8 for _ in range(2)]
    kwargs = dict(
        input_text_indexes=tokens,
        input_timesteps=torch.tensor([0.5, 0.5]),
        special_tokens={
            "eos_token_id": 1,
            "start_of_generation": 2,
            "end_of_generation": 3,
            "start_of_video": 4,
            "end_of_video": 5,
        },
        latent_patch_size=1,
    )
    native = pack_input_sequence(sequence_plans=plans, gen_data_clean=clean, **kwargs)
    copied_plans, copied_clean = attach_local_prefixes(plans, clean, (None, torch.randn(4, 32)))
    local = pack_input_sequence(sequence_plans=copied_plans, gen_data_clean=copied_clean, **kwargs)
    assert local.local_memory_tokens[0] is None
    assert local.local_memory_tokens[1].shape == (4, 32)
    for name in ("sequence_length", "sample_lens", "split_lens", "attn_modes"):
        assert getattr(local, name) == getattr(native, name)
    for name in ("position_ids", "text_indexes", "ce_loss_indexes"):
        left, right = getattr(local, name), getattr(native, name)
        if left is not None:
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(local.vision.mse_loss_indexes, native.vision.mse_loss_indexes)


@pytest.mark.parametrize("shape", [(3, 32), (4, 31), (4, 33)])
def test_wrong_local_shape_rejected(shape: tuple[int, int]) -> None:
    bridge = nn.Linear(32, 8)
    with pytest.raises(ValueError, match="requires"):
        build_memory_prefix_context((torch.zeros(shape),), bridge, torch.zeros(8), target_dtype=torch.float32)


def test_cp_rejected_before_native_encoding() -> None:
    holder = SimpleNamespace(
        config=SimpleNamespace(local_memory_enabled=True, local_memory_k_local=4),
        training=True,
        parallel_dims=SimpleNamespace(cp_enabled=True),
        pad_for_cuda_graphs=False,
        flex_backend=None,
        multiview_backend=None,
    )
    with pytest.raises(ValueError, match="context parallelism"):
        Cosmos3VFMNetwork.forward(holder, SimpleNamespace(local_memory_tokens=(torch.zeros(4, 32),)))


@pytest.mark.parametrize("override", ["disabled", "cuda_graphs", "multiview"])
def test_unsupported_forward_rejected_before_encoding(override: str) -> None:
    holder = SimpleNamespace(
        config=SimpleNamespace(local_memory_enabled=override != "disabled", local_memory_k_local=4),
        training=True,
        parallel_dims=SimpleNamespace(cp_enabled=False),
        pad_for_cuda_graphs=override == "cuda_graphs",
        flex_backend=object() if override == "multiview" else None,
        multiview_backend=None,
    )
    with pytest.raises(ValueError, match="Local Memory"):
        Cosmos3VFMNetwork.forward(holder, SimpleNamespace(local_memory_tokens=(torch.zeros(4, 32),)))


def test_detached_inference_prefix_passes_local_guard() -> None:
    def stop_after_guard(_):
        raise RuntimeError("passed-local-guard")

    holder = SimpleNamespace(
        config=SimpleNamespace(local_memory_enabled=True, local_memory_k_local=4, local_memory_dim=32),
        training=False,
        parallel_dims=SimpleNamespace(cp_enabled=False),
        pad_for_cuda_graphs=False,
        flex_backend=None,
        multiview_backend=None,
        _encode_text=stop_after_guard,
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="passed-local-guard"):
        Cosmos3VFMNetwork.forward(holder, SimpleNamespace(local_memory_tokens=(torch.zeros(4, 32),)))


def test_inference_prefix_with_grad_or_text_kv_memory_is_rejected() -> None:
    holder = SimpleNamespace(
        config=SimpleNamespace(local_memory_enabled=True, local_memory_k_local=4, local_memory_dim=32),
        training=False,
        parallel_dims=SimpleNamespace(cp_enabled=False),
        pad_for_cuda_graphs=False,
        flex_backend=None,
        multiview_backend=None,
    )
    token = torch.zeros(4, 32, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        Cosmos3VFMNetwork.forward(holder, SimpleNamespace(local_memory_tokens=(token,)))
    with pytest.raises(ValueError, match="no inference text-KV"):
        Cosmos3VFMNetwork.forward(
            holder,
            SimpleNamespace(local_memory_tokens=(torch.zeros(4, 32),)),
            memory=object(),
        )


def test_three_way_dispatch_rejected_before_attention() -> None:
    mask = SimpleNamespace(is_three_way=True, control_stream_token_ranges=None)
    with pytest.raises(ValueError, match="two_way"):
        dispatch_attention(None, None, None, mask, memory_prefix_key_states=torch.zeros(4, 1, 8))


class _AssemblyOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


def _b1_assembly_inputs(native_len=13, pad=0, *, dtype=torch.float32, offset_dtype=torch.int32, prefix_len=4):
    generator = torch.Generator().manual_seed(35)

    def values(rows):
        # Noncontiguous source views must retain their native values and order.
        return torch.randn(rows, 8, 2, generator=generator, dtype=dtype).transpose(1, 2)

    ends = [0, native_len] + ([native_len + pad] if pad else [])
    return dict(
        memory_k=values(prefix_len),
        memory_v=values(prefix_len),
        memory_offsets=torch.tensor([0, prefix_len]),
        native_k=values(native_len + pad),
        native_v=values(native_len + pad),
        native_offsets=torch.tensor(ends, dtype=offset_dtype),
        max_native_len=max(native_len, pad),
        k_local=prefix_len or 4,
    )


def _legacy_assembly(inputs):
    # Grad-enabled execution deliberately retains the original generic routine.
    # Its baseline body is also bound byte-for-byte in the author evidence.
    with torch.enable_grad():
        return prepend_memory_kv(**inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("native_len", [1, 32, 132, 267, 400])
@pytest.mark.parametrize("pad", [0, 2, 8])
def test_b1_inference_layout_matches_generic_with_native_padding(dtype, offset_dtype, native_len, pad):
    inputs = _b1_assembly_inputs(native_len, pad, dtype=dtype, offset_dtype=offset_dtype)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    reference = _legacy_assembly(inputs)
    with torch.inference_mode():
        actual = prepend_memory_kv(**inputs)
    for left, right in zip(actual[:3], reference[:3], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.dtype == right.dtype and left.stride() == right.stride()
    assert actual[3] == reference[3]
    assert actual[2].tolist() == [0, native_len + 4] + ([native_len + pad + 4] if pad else [])
    for name, value in snapshots.items():
        torch.testing.assert_close(inputs[name], value, rtol=0, atol=0)


def test_b1_inference_assembly_has_no_sample_expansion_scan_or_index_copy():
    for pad in (0, 2):
        inputs = _b1_assembly_inputs(pad=pad)
        with torch.inference_mode(), _AssemblyOps() as ops:
            actual = prepend_memory_kv(**inputs)
        forbidden = {
            name: count
            for name, count in ops.counts.items()
            if name
            in {
                "aten::repeat_interleave",
                "aten::cumsum",
                "aten::index_copy",
            }
        }
        assert not forbidden, f"B1 assembly still runs generic sample metadata/scatter: {forbidden}"
        torch.testing.assert_close(actual[0][:4], inputs["memory_k"], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][4:], inputs["native_v"], rtol=0, atol=0)


@pytest.mark.parametrize("prefix_len", [1, 4, 8])
def test_b1_assembly_current_prefix_changes_and_restores(prefix_len):
    inputs = _b1_assembly_inputs(32, 2, dtype=torch.bfloat16, prefix_len=prefix_len)
    with torch.inference_mode():
        original = prepend_memory_kv(**inputs)
        changed_inputs = dict(inputs, memory_k=inputs["memory_k"] + 1, memory_v=inputs["memory_v"] - 1)
        changed = prepend_memory_kv(**changed_inputs)
        restored = prepend_memory_kv(**inputs)
    torch.testing.assert_close(changed[0][:prefix_len], changed_inputs["memory_k"], rtol=0, atol=0)
    torch.testing.assert_close(changed[1][:prefix_len], changed_inputs["memory_v"], rtol=0, atol=0)
    assert not torch.equal(changed[0], original[0]) and not torch.equal(changed[1], original[1])
    for index in range(3):
        torch.testing.assert_close(original[index], restored[index], rtol=0, atol=0)
    torch.testing.assert_close(changed[0][prefix_len:], original[0][prefix_len:], rtol=0, atol=0)
    torch.testing.assert_close(changed[1][prefix_len:], original[1][prefix_len:], rtol=0, atol=0)


@pytest.mark.parametrize("sample_count", [2, 3])
@pytest.mark.parametrize("pad", [0, 2])
def test_multi_sample_inference_keeps_generic_interleaving_and_scans(sample_count, pad):
    inputs = _b1_assembly_inputs(2 * sample_count, pad)
    # Exactly one sample owns a Prefix; others retain their original boundaries.
    inputs["memory_offsets"] = torch.tensor([0] + [4] * sample_count)
    inputs["native_offsets"] = torch.tensor(
        list(range(0, 2 * sample_count + 1, 2)) + ([2 * sample_count + pad] if pad else []), dtype=torch.int32
    )
    reference = _legacy_assembly(inputs)
    with torch.inference_mode(), _AssemblyOps() as ops:
        actual = prepend_memory_kv(**inputs)
    assert ops.counts["aten::repeat_interleave"] == 2 and ops.counts["aten::cumsum"] == 1
    for index in range(3):
        torch.testing.assert_close(actual[index], reference[index], rtol=0, atol=0)
    assert actual[3] == reference[3]


def test_b1_grad_enabled_assembly_preserves_generic_path_and_gradients():
    inputs = _b1_assembly_inputs(13, 2)
    for name in ("memory_k", "memory_v", "native_k", "native_v"):
        inputs[name].requires_grad_()
    with torch.enable_grad(), _AssemblyOps() as ops:
        key, value, _, _ = prepend_memory_kv(**inputs)
    assert ops.counts["aten::repeat_interleave"] == 2 and ops.counts["aten::cumsum"] == 1
    (2 * key.sum() + 3 * value.sum()).backward()
    for name, scale in (("memory_k", 2), ("native_k", 2), ("memory_v", 3), ("native_v", 3)):
        torch.testing.assert_close(inputs[name].grad, torch.full_like(inputs[name], scale), rtol=0, atol=0)


def test_b1_empty_prefix_retains_generic_path():
    inputs = _b1_assembly_inputs(13, 2, prefix_len=0)
    with torch.inference_mode(), _AssemblyOps() as ops:
        actual = prepend_memory_kv(**inputs)
    assert ops.counts["aten::repeat_interleave"] == 2
    torch.testing.assert_close(actual[0], inputs["native_k"], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], inputs["native_offsets"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["key_dtype", "value_dtype", "sample_count", "device", "kv_shape"])
def test_b1_invalid_assembly_still_rejected(bad):
    inputs = _b1_assembly_inputs()
    if bad in ("key_dtype", "value_dtype"):
        key = "memory_k" if bad == "key_dtype" else "memory_v"
        inputs[key] = inputs[key].to(torch.bfloat16)
    elif bad == "sample_count":
        inputs["native_offsets"] = torch.tensor([0, 1, 2, 13], dtype=torch.int32)
    elif bad == "device":
        inputs["memory_k"] = inputs["memory_k"].to("meta")
    else:
        inputs["memory_v"] = inputs["memory_v"][:3]
    for context in (torch.enable_grad, torch.inference_mode):
        with context(), pytest.raises((ValueError, RuntimeError)):
            prepend_memory_kv(**inputs)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_b1_assembly_fullgraph_keeps_exact_layout_without_scans(dynamic, offset_dtype):
    graphs = []

    def backend(graph, _inputs):
        graphs.append([str(node.target) for node in graph.graph.nodes if node.op == "call_function"])
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(prepend_memory_kv, backend=backend, fullgraph=True, dynamic=dynamic)
    try:
        for native_len, pad in ((13, 0), (132, 2), (32, 8)):
            inputs = _b1_assembly_inputs(native_len, pad, dtype=torch.bfloat16, offset_dtype=offset_dtype)
            reference = _legacy_assembly(inputs)
            with torch.inference_mode():
                actual = compiled(**inputs)
            for index in range(3):
                torch.testing.assert_close(actual[index], reference[index], rtol=0, atol=0)
            assert actual[3] == reference[3]
        assert graphs
        assert not any(
            "repeat_interleave" in name or "cumsum" in name or "index_copy" in name
            for graph in graphs
            for name in graph
        )
    finally:
        torch._dynamo.reset()

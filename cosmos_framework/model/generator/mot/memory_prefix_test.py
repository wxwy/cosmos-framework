# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.memory_prefix import (
    MemoryPrefixContext,
    build_memory_prefix_context,
    concat_prefix_with_native_kv,
)
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.unified_mot import (
    LayerTypes,
    PackedAttentionMoT,
    _dispatch_attention_with_optional_memory_prefix,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.mark.L0
def test_memory_prefix_context_projects_present_samples_without_native_rows() -> None:
    projector = nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.arange(15, dtype=torch.float32).view(5, 3))
    tokens = [torch.ones(2, 3), None, torch.full((2, 3), 2.0)]

    context = build_memory_prefix_context(tokens, projector, torch.ones(5), torch.float32)

    assert context is not None
    context.validate()
    assert context.hidden.shape == (4, 5)
    assert context.sample_offsets.tolist() == [0, 2, 2, 4]
    assert context.present.tolist() == [True, False, True]
    torch.testing.assert_close(context.hidden[:2], projector(tokens[0]) + 1)
    torch.testing.assert_close(context.hidden[2:], projector(tokens[2]) + 1)


@pytest.mark.L0
def test_prefix_packer_preserves_native_geometry_and_keeps_payload_out_of_band() -> None:
    special_tokens = {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    text = [[10, 11], [12]]
    timesteps = torch.tensor([0.5, 0.5])
    native_plans = [SequencePlan(has_text=True), SequencePlan(has_text=True)]
    prefix_plans = [SequencePlan(has_text=True, has_local_memory=True), SequencePlan(has_text=True)]
    native = pack_input_sequence(
        native_plans, text, GenerationDataClean(batch_size=2, is_image_batch=False), timesteps, special_tokens
    )
    prefix = pack_input_sequence(
        prefix_plans,
        text,
        GenerationDataClean(batch_size=2, is_image_batch=False, x0_tokens_local_memory=[torch.ones(2, 3)]),
        timesteps,
        special_tokens,
    )
    for field in ("sample_lens", "split_lens", "attn_modes", "sequence_length", "text_indexes", "position_ids", "ce_loss_indexes"):
        left, right = getattr(prefix, field), getattr(native, field)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
    assert prefix.local_memory is None
    assert prefix.local_memory_prefix is not None
    assert prefix.local_memory_prefix.tokens_by_sample[1] is None
    torch.testing.assert_close(prefix.local_memory_prefix.tokens_by_sample[0], torch.ones(2, 3))


@pytest.mark.L0
def test_memory_prefix_context_rejects_invalid_slots_and_inconsistent_k_local() -> None:
    projector = nn.Identity()
    embed = torch.zeros(3)
    with pytest.raises(ValueError, match="K_local"):
        build_memory_prefix_context([torch.ones(1, 3), torch.ones(2, 3)], projector, embed, torch.float32)
    with pytest.raises(ValueError, match="K_local"):
        build_memory_prefix_context([torch.empty(0, 3)], projector, embed, torch.float32)
    with pytest.raises(ValueError, match="present"):
        MemoryPrefixContext(torch.zeros(1, 3), torch.tensor([0, 1]), torch.tensor([False])).validate()
    with pytest.raises(ValueError, match="K_local"):
        MemoryPrefixContext(torch.zeros(3, 3), torch.tensor([0, 1, 3]), torch.tensor([True, True])).validate()


@pytest.mark.L0
def test_memory_prefix_validates_all_slots_before_projecting_any_sample() -> None:
    calls = 0

    def projector(tokens: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return tokens

    with pytest.raises(ValueError, match="K_local"):
        build_memory_prefix_context(
            [torch.ones(1, 3), torch.ones(2, 3)], projector, torch.zeros(3), torch.float32
        )
    assert calls == 0


@pytest.mark.L0
def test_memory_prefix_concat_keeps_each_sample_isolated_and_orders_mem_before_native() -> None:
    prefix_key = torch.tensor([[[10.0]], [[11.0]], [[20.0]]])
    prefix_value = prefix_key + 100
    native_key = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
    native_value = native_key + 1000

    key, value, offsets, max_len = concat_prefix_with_native_kv(
        prefix_key,
        prefix_value,
        torch.tensor([0, 2, 2, 3]),
        native_key,
        native_value,
        torch.tensor([0, 2, 4, 5], dtype=torch.int32),
    )

    assert offsets.tolist() == [0, 4, 6, 8]
    assert max_len == 4
    torch.testing.assert_close(key[:, 0, 0], torch.tensor([10.0, 11.0, 1.0, 2.0, 3.0, 4.0, 20.0, 5.0]))
    torch.testing.assert_close(value[:, 0, 0], torch.tensor([110.0, 111.0, 1001.0, 1002.0, 1003.0, 1004.0, 120.0, 1005.0]))


@pytest.mark.L0
def test_memory_prefix_concat_rejects_incompatible_offsets() -> None:
    tensor = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="sample count"):
        concat_prefix_with_native_kv(
            tensor,
            tensor,
            torch.tensor([0, 1]),
            tensor,
            tensor,
            torch.tensor([0, 1, 1], dtype=torch.int32),
        )


@pytest.mark.L0
def test_memory_prefix_cpu_joint_softmax_changes_dm_but_not_ar_reference() -> None:
    ar_query = torch.tensor([[1.0, 0.0]])
    dm_query = torch.tensor([[0.0, 1.0]])
    ar_key = torch.tensor([[1.0, 0.0]])
    dm_key = torch.tensor([[0.0, 1.0]])
    ar_value = torch.tensor([[3.0, 0.0]])
    dm_value = torch.tensor([[5.0, 0.0]])
    memory_key = torch.tensor([[0.0, 2.0]])
    memory_value = torch.tensor([[11.0, 0.0]])

    def reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(query @ key.T, dim=-1) @ value

    ar_without_memory = reference(ar_query, ar_key, ar_value)
    ar_with_memory = reference(ar_query, ar_key, ar_value)
    torch.testing.assert_close(ar_with_memory, ar_without_memory, rtol=0, atol=0)
    dm_native = reference(dm_query, torch.cat((ar_key, dm_key)), torch.cat((ar_value, dm_value)))
    key, value, offsets, _ = concat_prefix_with_native_kv(
        memory_key[:, None],
        memory_value[:, None],
        torch.tensor([0, 1]),
        torch.cat((ar_key, dm_key))[:, None],
        torch.cat((ar_value, dm_value))[:, None],
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert offsets.tolist() == [0, 3]
    dm_with_memory = reference(dm_query, key[:, 0], value[:, 0])
    expected = reference(
        dm_query, torch.cat((memory_key, ar_key, dm_key)), torch.cat((memory_value, ar_value, dm_value))
    )
    torch.testing.assert_close(dm_with_memory, expected)
    assert not torch.equal(dm_with_memory, dm_native)


@pytest.mark.L0
def test_memory_prefix_k_uses_real_generator_norm_without_rope_or_query_projection() -> None:
    config = Nemotron3DenseVLTextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
        attention_bias=False,
    )
    attention = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
    )
    hidden = torch.arange(1024, dtype=torch.float32).view(2, 512)
    raw = attention.k_proj_moe_gen(hidden).view(2, attention.num_key_value_heads, attention.head_dim)
    normalized = attention.k_norm_moe_gen(raw)
    assert not isinstance(attention.k_norm_moe_gen, nn.Identity)
    assert not torch.allclose(raw, normalized)
    assert not any("memory" in name for name, _ in attention.named_modules())


@pytest.mark.L0
@pytest.mark.parametrize(
    ("pack", "configure", "message"),
    [
        ({"is_sharded": True}, lambda info: None, "context parallel/Ulysses"),
        ({}, lambda info: setattr(info, "is_three_way", True), "three-way"),
        ({}, lambda info: setattr(info, "control_stream_token_ranges", [(0, 1)]), "multi-control"),
        ({}, lambda info: setattr(info, "flex_block_mask", object()), "FlexAttention"),
    ],
)
def test_memory_prefix_dispatcher_rejects_unsupported_paths_before_attention_kernel(
    pack: dict[str, object], configure: object, message: str
) -> None:
    info = SplitInfo([1, 1], ["causal", "full"], [2], actual_len=2)
    configure(info)  # type: ignore[operator]
    prefix = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match=message):
        dispatch_attention(
            pack,
            {},
            {},
            info,
            memory_prefix_key_states=prefix,
            memory_prefix_value_states=prefix,
            memory_prefix_sample_offsets=torch.tensor([0, 1]),
        )


@pytest.mark.L0
def test_memory_prefix_dispatcher_rejects_native_memory_value_before_attention_kernel() -> None:
    info = SplitInfo([1, 1], ["causal", "full"], [2], actual_len=2)
    prefix = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="MemoryValue"):
        dispatch_attention(
            {},
            {},
            {},
            info,
            memory_value=object(),  # type: ignore[arg-type]
            memory_prefix_key_states=prefix,
            memory_prefix_value_states=prefix,
            memory_prefix_sample_offsets=torch.tensor([0, 1]),
        )


@pytest.mark.L0
def test_memory_prefix_preserves_legacy_alternate_dispatch_signature_and_fails_closed_when_present() -> None:
    calls: list[dict[str, object]] = []

    def alternate_dispatch(query: object, key: object, value: object, mask: object, **kwargs: object) -> tuple[object, None]:
        calls.append(kwargs)
        return query, None

    result, _ = _dispatch_attention_with_optional_memory_prefix(
        alternate_dispatch,
        {},
        {},
        {},
        object(),
        natten_metadata=None,
        memory_value=None,
        packed_key_states_normalized=None,
        memory_prefix_context=None,
        memory_prefix_key_states=None,
        memory_prefix_value_states=None,
        memory_prefix_sample_offsets=None,
    )
    assert result == {}
    assert len(calls) == 1
    assert not any(key.startswith("memory_prefix_") for key in calls[0])

    context = MemoryPrefixContext(torch.zeros(1, 1), torch.tensor([0, 1]), torch.tensor([True]))
    with pytest.raises(ValueError, match="alternate attention dispatch"):
        _dispatch_attention_with_optional_memory_prefix(
            alternate_dispatch,
            {},
            {},
            {},
            object(),
            natten_metadata=None,
            memory_value=None,
            packed_key_states_normalized=None,
            memory_prefix_context=context,
            memory_prefix_key_states=torch.zeros(1, 1, 1),
            memory_prefix_value_states=torch.zeros(1, 1, 1),
            memory_prefix_sample_offsets=torch.tensor([0, 1]),
        )

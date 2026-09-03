# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.memory_prefix import (
    MemoryPrefixContext,
    build_memory_prefix_context,
    concat_prefix_with_native_kv,
)
import cosmos_framework.model.generator.mot.attention as attention_module
import cosmos_framework.model.generator.mot.unified_mot as unified_mot_module
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_full_only_seq,
    get_causal_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
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


def _single_sample_attention_pack(values: torch.Tensor) -> dict[str, object]:
    return sequence_pack_from_packed_sequence(
        values,
        ["causal", "full"],
        [1, 1],
        [2],
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )


def _deterministic_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **_: object
) -> torch.Tensor:
    weights = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", query, key), dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, value)


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
def test_memory_prefix_packer_preserves_native_action_generation_metadata() -> None:
    special_tokens = {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    action = torch.arange(6, dtype=torch.float32).view(2, 3)
    native_plan = [SequencePlan(has_text=True, has_action=True)]
    prefix_plan = [SequencePlan(has_text=True, has_action=True, has_local_memory=True)]
    native = pack_input_sequence(
        native_plan,
        [[10, 11]],
        GenerationDataClean(batch_size=1, is_image_batch=False, x0_tokens_action=[action]),
        torch.tensor([0.5]),
        special_tokens,
        action_dim=3,
        include_end_of_generation_token=True,
    )
    prefix = pack_input_sequence(
        prefix_plan,
        [[10, 11]],
        GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_action=[action],
            x0_tokens_local_memory=[torch.ones(2, 3)],
        ),
        torch.tensor([0.5]),
        special_tokens,
        action_dim=3,
        include_end_of_generation_token=True,
    )
    for field in ("sample_lens", "split_lens", "attn_modes", "sequence_length", "text_indexes", "position_ids"):
        left, right = getattr(prefix, field), getattr(native, field)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
    assert native.action is not None and prefix.action is not None
    for field in ("sequence_indexes", "token_shapes", "mse_loss_indexes", "condition_mask", "timesteps"):
        left, right = getattr(prefix.action, field), getattr(native.action, field)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        elif isinstance(left, list):
            assert isinstance(right, list)
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                if isinstance(left_item, torch.Tensor):
                    torch.testing.assert_close(left_item, right_item)
                else:
                    assert left_item == right_item
        else:
            assert left == right
    assert prefix.get_sequence_pack_metadata() == native.get_sequence_pack_metadata()
    assert prefix.local_memory is None
    assert prefix.local_memory_prefix is not None
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
def test_memory_prefix_dispatch_drives_real_two_way_route_with_one_joint_dm_softmax(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def spy_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append((query.detach().clone(), key.detach().clone(), value.detach().clone()))
        return _deterministic_attention(query, key, value, **kwargs)

    monkeypatch.setattr(attention_module, "attention", spy_attention)
    info = SplitInfo([1, 1], ["causal", "full"], [2], actual_len=2)
    prefix_offsets = torch.tensor([0, 1], dtype=torch.long)

    def run(memory_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        query = _single_sample_attention_pack(torch.tensor([[[1.0]], [[2.0]]]))
        key = _single_sample_attention_pack(torch.tensor([[[1.0]], [[2.0]]]))
        value = _single_sample_attention_pack(torch.tensor([[[3.0]], [[5.0]]]))
        output, _ = dispatch_attention(
            query,
            key,
            value,
            info,
            memory_prefix_key_states=torch.tensor([[[2.0]]]),
            memory_prefix_value_states=torch.tensor([[[memory_value]]]),
            memory_prefix_sample_offsets=prefix_offsets,
        )
        return get_causal_seq(output)[0], get_full_only_seq(output)[0]

    ar_first, dm_first = run(11.0)
    ar_second, dm_second = run(29.0)
    assert len(calls) == 4
    torch.testing.assert_close(calls[0][1], torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(calls[1][1], torch.tensor([[[[2.0]], [[1.0]], [[2.0]]]]))
    torch.testing.assert_close(calls[1][2], torch.tensor([[[[11.0]], [[3.0]], [[5.0]]]]))
    torch.testing.assert_close(ar_first, ar_second, rtol=0, atol=0)
    assert not torch.equal(dm_first, dm_second)


@pytest.mark.L0
def test_memory_prefix_packed_attention_forwards_normalized_positionless_memory_k(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Nemotron3DenseVLTextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
        attention_bias=False,
        head_dim=64,
    )
    layer = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
    )
    pack = _single_sample_attention_pack(torch.arange(1024, dtype=torch.float32).view(2, 512))
    positions = _single_sample_attention_pack(torch.zeros(2, layer.head_dim))
    context = MemoryPrefixContext(torch.ones(1, 512), torch.tensor([0, 1]), torch.tensor([True]))
    rotary_calls: list[torch.Tensor] = []
    dispatch_calls: list[dict[str, object]] = []
    query_inputs: list[torch.Tensor] = []

    def identity_rotary(query: torch.Tensor, key: torch.Tensor, *_: object, **__: object) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_calls.append(key.detach().clone())
        return query, key

    def spy_dispatch(*args: object, **kwargs: object) -> tuple[dict[str, object], None]:
        dispatch_calls.append(dict(kwargs))
        source = args[0]
        assert isinstance(source, dict)
        return from_und_gen_splits(
            torch.zeros(1, layer.hidden_size), torch.zeros(1, layer.hidden_size), source
        ), None

    layer._apply_rotary_pos_emb = identity_rotary
    layer.q_proj_moe_gen.register_forward_pre_hook(lambda _module, inputs: query_inputs.append(inputs[0].detach().clone()))
    monkeypatch.setattr(unified_mot_module, "dispatch_attention", spy_dispatch)
    layer.dispatch_attention_fn = unified_mot_module.dispatch_attention
    layer(pack, SplitInfo([1, 1], ["causal", "full"], [2], actual_len=2), (positions, positions), memory_prefix_context=context)

    assert len(rotary_calls) == 2
    assert len(query_inputs) == 1
    assert query_inputs[0].shape == (1, 512)
    assert len(dispatch_calls) == 1
    expected_k = layer.k_norm_moe_gen(layer.k_proj_moe_gen(context.hidden).view(-1, layer.num_key_value_heads, layer.head_dim))
    torch.testing.assert_close(dispatch_calls[0]["memory_prefix_key_states"], expected_k)
    assert dispatch_calls[0]["memory_prefix_sample_offsets"] is context.sample_offsets


@pytest.mark.L0
@pytest.mark.parametrize(
    ("pad_for_cuda_graphs", "memory", "attention_io_layout", "parallel_dims", "message"),
    [
        (False, object(), "sequence_sharded", None, "native MemoryState"),
        (True, None, "sequence_sharded", None, "CUDA graph"),
        (False, None, "replicated", SimpleNamespace(cp_enabled=True), "replicated attention I/O"),
    ],
)
def test_memory_prefix_owner_guards_fail_before_packed_attention_work(
    monkeypatch: pytest.MonkeyPatch,
    pad_for_cuda_graphs: bool,
    memory: object | None,
    attention_io_layout: str,
    parallel_dims: object | None,
    message: str,
) -> None:
    network = object.__new__(Cosmos3VFMNetwork)
    context = MemoryPrefixContext(torch.zeros(1, 1), torch.tensor([0, 1]), torch.tensor([True]))
    object.__setattr__(network, "_encode_text", lambda _packed: (torch.empty(0, 1), torch.float32))
    object.__setattr__(network, "_encode_local_memory", lambda _packed, _dtype: context)
    object.__setattr__(network, "config", SimpleNamespace(vision_gen=False, action_gen=False, sound_gen=False))
    object.__setattr__(network, "pad_for_cuda_graphs", pad_for_cuda_graphs)
    object.__setattr__(network, "attention_io_layout", attention_io_layout)
    object.__setattr__(network, "parallel_dims", parallel_dims)
    object.__setattr__(network, "video_temporal_causal", False)
    monkeypatch.setattr(
        "cosmos_framework.model.generator.mot.cosmos3_vfm_network.build_packed_sequence",
        lambda **_kwargs: pytest.fail("owner guard must run before packed-attention construction"),
    )
    packed = SimpleNamespace(
        attn_modes=[],
        split_lens=[],
        vision=None,
        action=None,
        sound=None,
        num_action_tokens_per_supertoken=0,
        null_action_supertokens=False,
    )
    with pytest.raises(ValueError, match=message):
        Cosmos3VFMNetwork.forward(network, packed, memory=memory)  # type: ignore[arg-type]


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

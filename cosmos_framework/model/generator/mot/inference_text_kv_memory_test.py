# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.model.generator.mot.inference_text_kv_memory as text_kv_module
from cosmos_framework.data.generator.sequence_packing.runtime import sequence_pack_from_packed_sequence
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    InferenceTextKVMemoryValue,
    UndKVCache,
    _attention_gen_with_cached_text,
    dispatch_attention_with_text_kv_memory,
)
from cosmos_framework.model.generator.mot.memory_prefix import MemoryPrefixContext
from cosmos_framework.model.generator.mot.unified_mot import _dispatch_attention_with_optional_memory_prefix
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def _gen_only_pack(values: torch.Tensor) -> dict[str, object]:
    length = values.shape[0]
    return sequence_pack_from_packed_sequence(
        values,
        ["full"],
        [length],
        [length],
        torch.empty(0, dtype=torch.long),
        torch.arange(length, dtype=torch.long),
    )


@pytest.mark.L0
def test_inference_text_kv_state_explicitly_supports_memory_prefix() -> None:
    state = InferenceTextKVMemoryState([UndKVCache()])
    assert state.supports_memory_prefix()


@pytest.mark.L0
def test_cached_text_attention_orders_prefix_before_cached_text_before_gen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_pack = _gen_only_pack(torch.tensor([[[1.0]], [[2.0]]]))
    key_pack = _gen_only_pack(torch.tensor([[[30.0]], [[31.0]]]))
    value_pack = _gen_only_pack(torch.tensor([[[130.0]], [[131.0]]]))
    memory = InferenceTextKVMemoryValue(
        und_k_cached=torch.tensor([[[[20.0]], [[21.0]]]]),
        und_v_cached=torch.tensor([[[[120.0]], [[121.0]]]]),
        frame_idx=1,
        gen_len=2,
    )
    seen: dict[str, torch.Tensor] = {}

    def fake_attention(*, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **_: object) -> torch.Tensor:
        seen["query"] = query.detach().clone()
        seen["key"] = key.detach().clone()
        seen["value"] = value.detach().clone()
        return torch.zeros_like(query)

    monkeypatch.setattr(text_kv_module, "attention", fake_attention)
    output, kv_to_store = _attention_gen_with_cached_text(
        query_pack,
        key_pack,
        value_pack,
        memory,
        memory_prefix_key_states=torch.tensor([[[10.0]], [[11.0]]]),
        memory_prefix_value_states=torch.tensor([[[110.0]], [[111.0]]]),
        memory_prefix_sample_offsets=torch.tensor([0, 2], dtype=torch.long),
        memory_prefix_max_len=2,
    )

    assert kv_to_store is None
    torch.testing.assert_close(
        seen["key"][0, :, 0, 0],
        torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        seen["value"][0, :, 0, 0],
        torch.tensor([110.0, 111.0, 120.0, 121.0, 130.0, 131.0]),
        rtol=0,
        atol=0,
    )
    assert output["causal_seq"].shape[0] == 0
    assert output["full_only_seq"].shape[0] == 2


@pytest.mark.L0
def test_cache_fill_dispatch_forwards_memory_prefix_to_standard_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = _gen_only_pack(torch.ones(2, 1, 1))
    memory = InferenceTextKVMemoryValue(
        und_k_cached=None,
        und_v_cached=None,
        frame_idx=0,
        gen_len=2,
    )
    seen: dict[str, object] = {}

    def fake_dispatch(query: object, key: object, value: object, mask: object, **kwargs: object):
        seen.update(kwargs)
        return query, None

    monkeypatch.setattr(text_kv_module, "dispatch_attention", fake_dispatch)
    prefix = torch.ones(1, 1, 1)
    output, _ = dispatch_attention_with_text_kv_memory(
        pack,
        pack,
        pack,
        object(),
        memory_value=memory,
        memory_prefix_key_states=prefix,
        memory_prefix_value_states=prefix + 1,
        memory_prefix_sample_offsets=torch.tensor([0, 1], dtype=torch.long),
        memory_prefix_max_len=1,
    )

    assert output is pack
    assert seen["memory_value"] is None
    assert seen["memory_prefix_key_states"] is prefix
    assert seen["memory_prefix_max_len"] == 1


@pytest.mark.L0
def test_optional_prefix_dispatch_allows_only_the_text_kv_alternate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = _gen_only_pack(torch.ones(1, 1, 1))
    memory = InferenceTextKVMemoryValue(None, None, frame_idx=0, gen_len=1)
    prefix = torch.ones(1, 1, 1)
    context = MemoryPrefixContext(
        hidden=torch.ones(1, 2),
        sample_offsets=torch.tensor([0, 1], dtype=torch.long),
        present=torch.tensor([True]),
        k_local=1,
    )

    monkeypatch.setattr(text_kv_module, "dispatch_attention", lambda query, *_args, **_kwargs: (query, None))
    output, _ = _dispatch_attention_with_optional_memory_prefix(
        dispatch_attention_with_text_kv_memory,
        pack,
        pack,
        pack,
        object(),
        natten_metadata=None,
        memory_value=memory,
        packed_key_states_normalized=None,
        memory_prefix_context=context,
        memory_prefix_key_states=prefix,
        memory_prefix_value_states=prefix,
        memory_prefix_sample_offsets=context.sample_offsets,
        memory_prefix_max_len=1,
    )
    assert output is pack


@pytest.mark.L0
def test_text_kv_reuse_eligibility_accepts_single_sample_local_memory() -> None:
    fake_model = SimpleNamespace(
        parallel_dims=None,
        config=SimpleNamespace(
            joint_attn_implementation="two_way",
            video_temporal_causal=False,
            sound_gen=False,
        ),
    )
    plan = SimpleNamespace(has_sound=False, has_local_memory=True)
    clean = SimpleNamespace(batch_size=1, num_vision_items_per_sample=None)

    assert OmniMoTModel._can_reuse_inference_text_kv(
        fake_model,
        [plan],
        clean,
        reuse_pack_templates=True,
        has_velocity_postprocess=False,
    )

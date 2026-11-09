# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU numerical/lifecycle contracts for Prefix-on diffusion text K/V reuse.

Uses real MoT decoder layers and packing; only the CUDA attention kernel is
replaced by a per-sequence CPU SDPA reference. This is not a DCP/GPU Gate.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import cosmos_framework.model.generator.mot.attention as attention_module
import cosmos_framework.model.generator.mot.inference_text_kv_memory as cache_module
from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, dispatch_attention
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
    restore_inference_attention_dispatch,
)
from cosmos_framework.model.generator.mot.memory_prefix import build_memory_prefix_context
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def _cpu_attention(query, key, value, **kwargs):
    """Keep the production varlen fences (including padding and grouped heads)."""
    qo = kwargs.get("cumulative_seqlen_Q")
    ko = kwargs.get("cumulative_seqlen_KV")
    if qo is None:
        qo, ko = [0, query.shape[1]], [0, key.shape[1]]
    output = torch.zeros_like(query)
    for i in range(len(qo) - 1):
        qs, qe, ks, ke = map(int, (qo[i], qo[i + 1], ko[i], ko[i + 1]))
        if qe > qs:
            output[:, qs:qe] = F.scaled_dot_product_attention(
                query[:, qs:qe].transpose(1, 2),
                key[:, ks:ke].transpose(1, 2),
                value[:, ks:ke].transpose(1, 2),
                is_causal=kwargs.get("is_causal", False),
                enable_gqa=True,
            ).transpose(1, 2)
    return output


class _TinyMoT(nn.Module):
    def __init__(self, dtype=torch.float32, *, norm_text=True):
        super().__init__()
        cfg = Qwen3VLTextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_hidden_layers=3,
        )
        self.layers = nn.ModuleList(
            [
                MoTDecoderLayer(
                    cfg,
                    layer_idx=i,
                    layer_types=LayerTypes("qwen3_vl_dense"),
                    qk_norm_for_text=norm_text,
                    qk_norm_for_diffusion=True,
                    use_und_k_norm_for_gen=True,
                )
                for i in range(3)
            ]
        )
        self.bridge = nn.Linear(32, 16)
        self.embed = nn.Parameter(torch.randn(16))
        self.to(dtype=dtype).eval()
        # Match the install/restore entry point of the full VFM network.
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=self.layers))

    def step(self, und, gen, prefix, caches=None, alignment=1):
        nu, ng = und.shape[0], gen.shape[0]
        pack, mask, _ = build_packed_sequence(
            "two_way",
            packed_sequence=torch.cat((und, gen)),
            attn_modes=["causal", "full"],
            split_lens=[nu, ng],
            sample_lens=[nu + ng],
            packed_und_token_indexes=torch.arange(nu),
            packed_gen_token_indexes=torch.arange(nu, nu + ng),
            num_heads=2,
            head_dim=8,
            num_layers=3,
            full_seq_alignment=alignment,
            causal_seq_alignment=alignment,
        )
        positions = (
            from_und_gen_splits(
                torch.ones(pack["causal_seq"].shape[0], 8, dtype=gen.dtype),
                torch.ones(pack["full_only_seq"].shape[0], 8, dtype=gen.dtype),
                pack,
            ),
            from_und_gen_splits(
                torch.zeros(pack["causal_seq"].shape[0], 8, dtype=gen.dtype),
                torch.zeros(pack["full_only_seq"].shape[0], 8, dtype=gen.dtype),
                pack,
            ),
        )
        context = (
            build_memory_prefix_context(
                (prefix,),
                self.bridge,
                self.embed,
                target_dtype=gen.dtype,
            )
            if prefix is not None
            else None
        )
        memory = (
            InferenceTextKVMemoryState(caches, local_prefix_length=4 if prefix is not None else 0) if caches else None
        )
        if memory:
            memory.init(pack, gen.device)
        gen_only = memory.is_gen_only() if memory else False
        for i, layer in enumerate(self.layers):
            pack, _, kv = layer(
                pack,
                mask,
                positions,
                memory_value=memory.read_for_layer(i) if memory else None,
                memory_prefix_context=context,
                gen_only=gen_only,
            )
            if memory and kv is not None:
                memory.write_for_layer(i, kv)
        return get_gen_seq(pack)[:ng]


@pytest.fixture(autouse=True)
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(attention_module, "attention", _cpu_attention)
    monkeypatch.setattr(cache_module, "attention", _cpu_attention)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize("norm_text", [False, True])
def test_thirty_denoise_steps_keep_prefix_and_match_uncached(dtype, alignment, norm_text):
    torch.manual_seed(31)
    net = _TinyMoT(dtype, norm_text=norm_text)
    und, gen, prefix = (torch.randn(*s, dtype=dtype) for s in ((5, 16), (6, 16), (4, 32)))
    und_calls = [0, 0, 0]
    prefix_calls = [0, 0, 0]
    hooks = []
    for i, layer in enumerate(net.layers):

        def count_und(_m, _args, _out, i=i):
            und_calls[i] += 1

        def count_prefix(_m, args, _out, i=i):
            if args[0].shape[0] == 4:
                prefix_calls[i] += 1

        hooks.extend(
            (
                layer.mlp.register_forward_hook(count_und),
                layer.self_attn.k_proj_moe_gen.register_forward_hook(count_prefix),
            )
        )
    with torch.inference_mode():
        reference = [net.step(und, gen + i * 0.01, prefix, alignment=alignment) for i in range(30)]
        assert und_calls == [30] * 3 and prefix_calls == [30] * 3
        und_calls[:] = prefix_calls[:] = [0] * 3
        caches = make_inference_text_kv_cache(3)
        previous = install_inference_memory_attention_dispatch(net)
        try:
            for i, expected in enumerate(reference):
                actual = net.step(und, gen + i * 0.01, prefix, caches, alignment)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Real understanding MLPs only run on fill. Prefix K/V still projected
            # once per layer on all thirty forwards, including every cache hit.
            assert und_calls == [1] * 3 and prefix_calls == [30] * 3
        finally:
            restore_inference_attention_dispatch(previous)
    for hook in hooks:
        hook.remove()


def test_prefix_and_text_changes_are_visible_in_independent_requests():
    torch.manual_seed(41)
    net = _TinyMoT()
    und, gen, prefix = torch.randn(5, 16), torch.randn(6, 16), torch.randn(4, 32)
    previous = install_inference_memory_attention_dispatch(net)
    with torch.inference_mode():

        def request(u, p):
            cache = make_inference_text_kv_cache(3)
            fill = net.step(u, gen, p, cache)
            hit = net.step(u, gen, p, cache)
            torch.testing.assert_close(fill, hit, rtol=0, atol=0)
            return hit

        a = request(und, prefix)
        b = request(und, prefix + 3)
        c = request(und + 2, prefix)
        cold = request(und, None)
        assert not torch.allclose(a, b) and not torch.allclose(a, c) and not torch.allclose(a, cold)
        torch.testing.assert_close(a, request(und, prefix), rtol=0, atol=0)
    restore_inference_attention_dispatch(previous)


def test_prefix_is_required_on_every_cache_hit():
    torch.manual_seed(42)
    net = _TinyMoT()
    und, gen, prefix = torch.randn(5, 16), torch.randn(6, 16), torch.randn(4, 32)
    cache = make_inference_text_kv_cache(3)
    previous = install_inference_memory_attention_dispatch(net)
    with torch.inference_mode():
        net.step(und, gen, prefix, cache)
        # Use the Prefix-aware state but deliberately strip the Prefix at attention.
        state = InferenceTextKVMemoryState(cache, local_prefix_length=4)
        state.init(
            {
                "_num_full_tokens": 6,
                "_num_causal_tokens": 5,
                "causal_seq": torch.empty(5, 16),
                "_causal_seq_offsets": torch.tensor([0, 5]),
                "_full_only_seq_offsets": torch.tensor([0, 6]),
            },
            torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="every step"):
            cache_module.dispatch_attention_with_text_kv_memory(
                None, None, None, None, memory_value=state.read_for_layer(0)
            )
    restore_inference_attention_dispatch(previous)


def test_partial_fill_is_not_gen_only_and_cleanup_allows_retry():
    caches = make_inference_text_kv_cache(3)
    state = InferenceTextKVMemoryState(caches, local_prefix_length=4)
    x = torch.randn(1, 5, 1, 8)
    state.write_for_layer(0, (x, x, x, x))
    assert not state.is_gen_only()
    for cache in caches:
        cache.clear()
        assert not cache.is_initialized and cache.k_und is cache.v_und is None
    assert not state.is_gen_only()


def test_prefix_cache_rejects_multiple_native_samples():
    state = InferenceTextKVMemoryState(make_inference_text_kv_cache(1), local_prefix_length=4)
    with pytest.raises(ValueError, match="single native sample"):
        state.init(
            {
                "_num_full_tokens": 6,
                "_num_causal_tokens": 5,
                "_causal_seq_offsets": torch.tensor([0, 2, 5]),
                "_full_only_seq_offsets": torch.tensor([0, 3, 6]),
            },
            torch.device("cpu"),
        )


def test_vfm_guard_allows_only_explicit_prefix_aware_inference_state():
    def encoding_reached(_):
        raise RuntimeError("encoding reached")

    owner = SimpleNamespace(
        config=SimpleNamespace(
            local_memory_enabled=True, local_memory_k_local=4, local_memory_dim=32, joint_attn_implementation="two_way"
        ),
        training=False,
        parallel_dims=None,
        pad_for_cuda_graphs=False,
        flex_backend=None,
        multiview_backend=None,
        _encode_text=encoding_reached,
    )
    seq = SimpleNamespace(local_memory_tokens=(torch.randn(4, 32),))
    for length in (0, 3, 4):
        memory = InferenceTextKVMemoryState(make_inference_text_kv_cache(1), local_prefix_length=length)
        with torch.inference_mode():
            if length == 4:
                with pytest.raises(RuntimeError, match="encoding reached"):
                    Cosmos3VFMNetwork.forward(owner, seq, memory=memory)
            else:
                with pytest.raises(ValueError, match="Prefix-aware B1"):
                    Cosmos3VFMNetwork.forward(owner, seq, memory=memory)
        with pytest.raises(ValueError, match="Prefix-aware B1"):
            Cosmos3VFMNetwork.forward(owner, seq, memory=memory)


def test_unexpected_attention_dispatch_is_restored_after_partial_install():
    net = _TinyMoT()

    def sentinel(*args):
        return None

    net.layers[1].self_attn.dispatch_attention_fn = sentinel
    with pytest.raises(RuntimeError, match="Cannot install"):
        install_inference_memory_attention_dispatch(net)
    assert net.layers[0].self_attn.dispatch_attention_fn is dispatch_attention
    assert net.layers[1].self_attn.dispatch_attention_fn is sentinel


@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_cached_prefix_dispatch_keeps_symbolic_guards_and_current_prefix(monkeypatch, dynamic):
    """Real H100 hit SymBool != bool in the dispatcher under Dynamo.

    CPU eager backend traces the production dispatcher in fullgraph mode. The
    kernel double uses a tensorized per-segment SDPA mask (no Python varlen loop).
    """

    def kernel(query, key, value, **kwargs):
        qo, ko = kwargs.get("cumulative_seqlen_Q"), kwargs.get("cumulative_seqlen_KV")
        mask = None
        if qo is not None:
            qsample = torch.bucketize(torch.arange(query.shape[1]), qo[1:], right=True)
            ksample = torch.bucketize(torch.arange(key.shape[1]), ko[1:], right=True)
            mask = qsample[:, None] == ksample[None, :]
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            enable_gqa=True,
        ).transpose(1, 2)

    monkeypatch.setattr(attention_module, "attention", kernel)
    torch.manual_seed(93)
    caches = make_inference_text_kv_cache(1)
    und_key, und_value = torch.randn(1, 6, 1, 8), torch.randn(1, 6, 1, 8)
    caches[0].store(und_key, und_value)
    state = InferenceTextKVMemoryState(caches, local_prefix_length=4)
    q, k, v = torch.randn(6, 2, 8), torch.randn(6, 1, 8), torch.randn(6, 1, 8)

    def pack(tensor):
        native, _, _ = build_packed_sequence(
            "two_way",
            packed_sequence=torch.cat((tensor.new_zeros(6, *tensor.shape[1:]), tensor)),
            attn_modes=["causal", "full"],
            split_lens=[6, 6],
            sample_lens=[12],
            packed_und_token_indexes=torch.arange(6),
            packed_gen_token_indexes=torch.arange(6, 12),
            num_heads=tensor.shape[1],
            head_dim=8,
            num_layers=1,
        )
        return native

    native_q, native_k, native_v = pack(q), pack(k), pack(v)
    state.init(native_q, torch.device("cpu"))
    memory = state.read_for_layer(0)

    def gen_only(p):
        return from_und_gen_splits(p["causal_seq"][:0], get_gen_seq(p), p)

    mask = attention_module.SplitInfo([6, 6], ["causal", "full"], [12], actual_len=12)
    prefix_key, prefix_value = torch.randn(4, 1, 8), torch.randn(4, 1, 8)
    compiled = torch.compile(
        cache_module.dispatch_attention_with_text_kv_memory, backend="eager", fullgraph=True, dynamic=dynamic
    )
    with torch.inference_mode():
        results = []
        for shift in (0, 1, 0):
            pk, pv = prefix_key + shift, prefix_value + shift
            result, _ = compiled(
                gen_only(native_q),
                gen_only(native_k),
                gen_only(native_v),
                mask,
                memory_value=memory,
                memory_prefix_key_states=pk,
                memory_prefix_value_states=pv,
                memory_prefix_sample_offsets=torch.tensor([0, 4]),
            )
            ref_k = from_und_gen_splits(
                F.pad(und_key[0], (0, 0, 0, 0, 0, memory.causal_pad_length)),
                get_gen_seq(native_k),
                native_k,
            )
            ref_v = from_und_gen_splits(
                F.pad(und_value[0], (0, 0, 0, 0, 0, memory.causal_pad_length)),
                get_gen_seq(native_v),
                native_v,
            )
            expected, _ = dispatch_attention(
                native_q,
                ref_k,
                ref_v,
                mask,
                memory_prefix_key_states=pk,
                memory_prefix_value_states=pv,
                memory_prefix_sample_offsets=torch.tensor([0, 4]),
            )
            torch.testing.assert_close(get_gen_seq(result), get_gen_seq(expected), rtol=0, atol=0)
            assert torch.count_nonzero(get_gen_seq(result)[6:]) == 0
            results.append(get_gen_seq(result).clone())
        assert not torch.equal(results[0], results[1])
        torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)


def test_prefix_cache_fill_uses_materialized_native_kv_and_hit_keeps_native_padding(monkeypatch):
    """Keep the exact native GEN operands/backend contract, including pad ranges."""
    torch.manual_seed(109)

    def pack(tensor):
        return build_packed_sequence(
            "two_way",
            packed_sequence=tensor,
            attn_modes=["causal", "full"],
            split_lens=[5, 6],
            sample_lens=[11],
            packed_und_token_indexes=torch.arange(5),
            packed_gen_token_indexes=torch.arange(5, 11),
            num_heads=tensor.shape[1],
            head_dim=8,
            num_layers=1,
            full_seq_alignment=8,
            causal_seq_alignment=8,
        )[:2]

    query, mask = pack(torch.randn(11, 2, 8))
    key, _ = pack(torch.randn(11, 1, 8))
    value, _ = pack(torch.randn(11, 1, 8))
    norm_key = from_und_gen_splits(key["causal_seq"] * 2, get_gen_seq(key), key)
    kwargs = dict(
        memory_prefix_key_states=torch.randn(4, 1, 8),
        memory_prefix_value_states=torch.randn(4, 1, 8),
        memory_prefix_sample_offsets=torch.tensor([0, 4]),
    )
    calls = []

    def kernel(query, key, value, **kw):
        calls.append(
            dict(
                query=query.clone(),
                key=key.clone(),
                value=value.clone(),
                **{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()},
            )
        )
        return _cpu_attention(query, key, value, **kw)

    monkeypatch.setattr(attention_module, "attention", kernel)
    monkeypatch.setattr(cache_module, "attention", kernel)
    caches = make_inference_text_kv_cache(1)
    state = InferenceTextKVMemoryState(caches, local_prefix_length=4)
    state.init(query, torch.device("cpu"))
    with torch.inference_mode():
        expected, _ = dispatch_attention(query, key, value, mask, packed_key_states_normalized=norm_key, **kwargs)
        native_gen_call = calls[-1]
        fill, kv = cache_module.dispatch_attention_with_text_kv_memory(
            query,
            key,
            value,
            mask,
            memory_value=state.read_for_layer(0),
            packed_key_states_normalized=norm_key,
            **kwargs,
        )
        assert kv is not None and kv[0].numel() == kv[1].numel() == 0
        positions = query["_causal_indices"] + 4
        torch.testing.assert_close(kv[2][0], native_gen_call["key"][0, positions], rtol=0, atol=0)
        torch.testing.assert_close(kv[3][0], native_gen_call["value"][0, positions], rtol=0, atol=0)
        state.write_for_layer(0, kv)

        def gen_only(p):
            return from_und_gen_splits(p["causal_seq"][:0], get_gen_seq(p), p)

        hit, ignored = cache_module.dispatch_attention_with_text_kv_memory(
            gen_only(query),
            gen_only(key),
            gen_only(value),
            mask,
            memory_value=state.read_for_layer(0),
            **kwargs,
        )
        assert ignored is not None and all(t.numel() == 0 for t in ignored)
        hit_gen_call = calls[-1]
        for name in ("query", "key", "value", "cumulative_seqlen_Q", "cumulative_seqlen_KV"):
            torch.testing.assert_close(hit_gen_call[name], native_gen_call[name], rtol=0, atol=0)
        assert hit_gen_call["max_seqlen_Q"] == native_gen_call["max_seqlen_Q"]
        assert hit_gen_call["max_seqlen_KV"] == native_gen_call["max_seqlen_KV"]
        torch.testing.assert_close(get_gen_seq(fill), get_gen_seq(expected), rtol=0, atol=0)
        torch.testing.assert_close(get_gen_seq(hit), get_gen_seq(expected), rtol=0, atol=0)


def _generation_harness():
    """Production generation orchestration, preencoded CPU carriers/tiny MoT.

    VAE/packer/DCP are outside this unit's boundary; it exercises the actual
    request closure, opt-out, cache allocation and finally block, not a copy.
    """
    from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    class Sampler:
        def __call__(self, velocity_fn, noise, num_steps, step_callback=None, **kwargs):
            for i in range(num_steps):
                if step_callback:
                    step_callback(i, num_steps)
                velocity = velocity_fn(noise, torch.tensor([[1000.0 - 10 * i]]))
                noise = [x - 0.01 * v for x, v in zip(noise, velocity, strict=True)]
            return noise

    torch.manual_seed(71)
    obj = SimpleNamespace(
        config=SimpleNamespace(
            local_memory_enabled=True,
            local_memory_k_local=4,
            local_memory_dim=32,
            joint_attn_implementation="two_way",
            video_temporal_causal=False,
            sound_gen=False,
            action_gen=True,
            rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc"),
        ),
        parallel_dims=None,
        net=_TinyMoT(),
        sampler=Sampler(),
        created=[],
        seen=[],
        templates=0,
        fail_at=None,
        text=torch.randn(5, 16),
    )
    obj._maybe_apply_prompt_upsampling = lambda batch, **kw: batch
    for name in ("generate_samples_from_batch", "_can_reuse_inference_pack_templates", "_can_reuse_inference_text_kv"):
        setattr(obj, name, MethodType(getattr(OmniMoTModel, name), obj))

    def prepare(batch, seed, has_negative_prompt):
        clean = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[torch.zeros(16, 1, 2, 3)],
            x0_tokens_action=[torch.zeros(16, 16)],
        )
        noise = torch.linspace(-1, 1, 352)
        return (
            [SequencePlan(has_text=True, has_vision=True, has_action=True)],
            clean,
            [[1, 2]],
            [[3]],
            [noise],
            [torch.zeros_like(noise)],
            [torch.zeros_like(noise)],
            True,
        )

    def make_cache(net=None):
        result = make_inference_text_kv_cache(3)
        obj.created.append(result)
        return result

    def pack_template(*args, **kwargs):
        obj.templates += 1
        return SimpleNamespace(to_cuda=lambda: None)

    def velocity(**kwargs):
        memory = kwargs["memory"]
        obj.seen.append(
            (
                memory is not None,
                memory.is_gen_only() if memory else False,
                kwargs["packed_sequence_template"] is not None,
            )
        )
        if obj.fail_at is not None and len(obj.seen) == obj.fail_at:
            raise RuntimeError("injected generation failure")
        gen = kwargs["noise_x"][0][:96].reshape(6, 16)
        prefix = kwargs["gen_data_clean"].x0_tokens_local_memory
        out = obj.net.step(obj.text, gen, prefix[0] if prefix else None, memory._text_kv_cache if memory else None)
        flat = out.flatten().repeat(4)[:352]
        return [flat]

    obj._prepare_inference_data = prepare
    obj._make_inference_text_kv_cache = make_cache
    obj._pack_input_sequence = pack_template
    obj._derive_include_end_of_generation_token = lambda: False
    obj._get_velocity = velocity
    batch = {"action_processing_record": [ActionProcessingRecord(raw_action_dim=15, action_normalizer=None)]}
    return obj, batch


@pytest.mark.parametrize("setting", ["temporal_causal", "three_way", "postprocess_builder"])
def test_generation_unsupported_prefix_cache_layout_keeps_original_path(setting):
    obj, batch = _generation_harness()
    kwargs = dict(guidance=1.0, num_steps=3, seed=[31], _local_memory_prefixes=(torch.randn(4, 32),))
    if setting == "temporal_causal":
        obj.config.video_temporal_causal = True
    elif setting == "three_way":
        obj.config.joint_attn_implementation = "three_way"
    else:
        kwargs["velocity_postprocess_builder"] = lambda **_kwargs: None
    obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.seen == [(False, False, False)] * 3
    assert not obj.created


def test_generation_entrypoint_same_prefix_actions_cache_hits_and_opt_out():
    obj, batch = _generation_harness()
    prefix = (torch.randn(4, 32),)
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31], _local_memory_prefixes=prefix)
    reference = obj.generate_samples_from_batch(batch, use_prefix_text_kv_cache=False, **kwargs)
    assert obj.seen == [(False, False, False)] * 6 and not obj.created
    obj.seen.clear()
    actual = obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.seen == [(True, False, False)] + [(True, True, False)] * 5
    assert obj.templates == 0  # Pack optimization was not enabled with Prefix.
    assert actual["action"][0].shape == (16, 15)
    for modality in actual:
        torch.testing.assert_close(actual[modality][0], reference[modality][0], rtol=0, atol=0)
    assert all(not c.is_initialized and c.k_und is c.v_und is None for c in obj.created[-1])
    assert all(layer.self_attn.dispatch_attention_fn is dispatch_attention for layer in obj.net.layers)


@pytest.mark.parametrize("fail_at", [1, 3])
def test_generation_exception_clears_cache_and_retry_starts_with_fill(fail_at):
    obj, batch = _generation_harness()
    prefix = (torch.randn(4, 32),)
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31], _local_memory_prefixes=prefix)
    obj.fail_at = fail_at
    with pytest.raises(RuntimeError, match="injected generation failure"):
        obj.generate_samples_from_batch(batch, **kwargs)
    assert all(not c.is_initialized and c.k_und is c.v_und is None for c in obj.created[-1])
    assert all(layer.self_attn.dispatch_attention_fn is dispatch_attention for layer in obj.net.layers)
    obj.fail_at = None
    obj.seen.clear()
    actual = obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.seen[0] == (True, False, False)
    reference = obj.generate_samples_from_batch(batch, use_prefix_text_kv_cache=False, **kwargs)
    torch.testing.assert_close(actual["action"][0], reference["action"][0], rtol=0, atol=0)


def test_generation_no_prefix_native_path_and_option_are_unchanged():
    obj, batch = _generation_harness()
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31])
    reference = obj.generate_samples_from_batch(batch, use_prefix_text_kv_cache=False, **kwargs)
    assert obj.seen == [(True, False, True)] + [(True, True, True)] * 5
    obj.seen.clear()
    actual = obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.seen == [(True, False, True)] + [(True, True, True)] * 5
    torch.testing.assert_close(actual["action"][0], reference["action"][0], rtol=0, atol=0)


def test_generation_requests_changed_instruction_prefix_and_reset_are_isolated():
    obj, batch = _generation_harness()
    prefix = torch.randn(4, 32)
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31])

    def request(p):
        obj.seen.clear()
        result = obj.generate_samples_from_batch(batch, _local_memory_prefixes=(p,), **kwargs)
        assert obj.seen[0][1] is False
        return result["action"][0]

    a = request(prefix)
    b = request(prefix + 3)
    assert not torch.allclose(a, b)
    obj.text += 2
    c = request(prefix)
    assert not torch.allclose(a, c)
    obj.text -= 2
    torch.testing.assert_close(a, request(prefix), rtol=0, atol=1e-6)
    # End/reset adds no cached state to the owner/network; a cold request then
    # a new prefixed request both start at frame_idx=0 with fresh branch caches.
    request(None)
    torch.testing.assert_close(a, request(prefix), rtol=0, atol=1e-6)

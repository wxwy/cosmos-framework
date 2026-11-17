# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU numerical checks at the production attention dispatcher boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import cosmos_framework.model.generator.mot.attention as attention_module
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.g2_contracts import EffectiveTextBatch, G2ConditionBatch, stack_g2_prepared
from cosmos_framework.model.generator.mot.g2_parameters import parameter_inventory, set_trainable_names
from cosmos_framework.model.generator.mot.memory_prefix import (
    build_joint_memory_prefix_context,
    build_memory_prefix_context,
)
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLTextModel, prepare_und_text_features
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def cpu_attention(query, key, value, **kwargs):
    qo = kwargs.get("cumulative_seqlen_Q", torch.tensor([0, query.shape[1]]))
    ko = kwargs.get("cumulative_seqlen_KV", torch.tensor([0, key.shape[1]]))
    out = []
    for qs, qe, ks, ke in zip(qo[:-1], qo[1:], ko[:-1], ko[1:], strict=True):
        out.append(
            F.scaled_dot_product_attention(
                query[:, qs:qe].transpose(1, 2),
                key[:, ks:ke].transpose(1, 2),
                value[:, ks:ke].transpose(1, 2),
                is_causal=kwargs.get("is_causal", False),
                enable_gqa=True,
            ).transpose(1, 2)
        )
    return torch.cat(out, dim=1)


@pytest.fixture(autouse=True)
def reference_dispatch(monkeypatch):
    monkeypatch.setattr(attention_module, "attention", cpu_attention)


def tiny_model(text_norm=False):
    config = Qwen3VLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=3,
        vocab_size=64,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    model = Qwen3VLTextModel(
        config, qk_norm_for_text=text_norm, qk_norm_for_diffusion=True, use_und_k_norm_for_gen=True
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("moe_gen" in name or "k_norm_und_for_gen" in name)
    return model


def native_pack(model, lengths, gen_rows=None):
    if gen_rows is None:
        gen_rows = [torch.randn(2, 16) for _ in lengths]
    ids = [tuple(range(1, n + 1)) for n in lengths]
    rows, und_indexes, gen_indexes, positions, splits, sample_lens = [], [], [], [], [], []
    offset = 0
    for tokens, gen in zip(ids, gen_rows, strict=True):
        n, m = len(tokens), gen.shape[0]
        rows.extend(
            (model.embed_tokens(torch.tensor(tokens, dtype=torch.long)), gen.to(model.embed_tokens.weight.dtype))
        )
        und_indexes.extend(range(offset, offset + n))
        gen_indexes.extend(range(offset + n, offset + n + m))
        positions.extend(range(n + m))
        splits.extend((n, m))
        sample_lens.append(n + m)
        offset += n + m
    pack, mask, _ = build_packed_sequence(
        "two_way",
        packed_sequence=torch.cat(rows),
        attn_modes=["causal", "full"] * len(lengths),
        split_lens=splits,
        sample_lens=sample_lens,
        packed_und_token_indexes=torch.tensor(und_indexes),
        packed_gen_token_indexes=torch.tensor(gen_indexes),
        num_heads=2,
        head_dim=8,
        num_layers=3,
    )
    text = EffectiveTextBatch(
        token_ids=tuple(ids),
        position_ids=tuple((tuple(range(n)),) * 3 for n in lengths),
        dropped=tuple(n == 0 for n in lengths),
        formatter_id="synthetic",
        tokenizer_id="synthetic",
        dropout_id="once",
    )
    return pack, mask, torch.tensor(positions).expand(3, -1), text


@pytest.mark.parametrize("lengths", [(3,), (0,), (3, 0, 1), (0, 0)])
@pytest.mark.parametrize("text_norm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_prepared_hidden_and_gen_parity(lengths, text_norm, dtype):
    torch.manual_seed(31)
    model = tiny_model(text_norm).to(dtype)
    # Native rotary buffers stay FP32 in production materialization as well.
    model.rotary_emb.float()
    pack, mask, pos, text = native_pack(model, lengths)
    captured_k, captured_v, handles = [], [], []
    for block in model.layers:
        handles.append(block.self_attn.k_norm.register_forward_hook(lambda m, inp, out: captured_k.append(out)))
        handles.append(block.self_attn.v_proj.register_forward_hook(lambda m, inp, out: captured_v.append(out)))
    native, _ = model(pack, mask, pos)
    for handle in handles:
        handle.remove()
    prepared = prepare_und_text_features(model, pack, pos, text)
    actual, _ = model(pack, mask, pos, prepared_text=prepared)
    expected_hidden = get_und_seq(native)[: sum(lengths)]
    flat_hidden = torch.cat([prepared.hidden[i, :n] for i, n in enumerate(lengths)])
    torch.testing.assert_close(flat_hidden, expected_hidden, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        get_gen_seq(actual)[: 2 * len(lengths)], get_gen_seq(native)[: 2 * len(lengths)], atol=2e-6, rtol=2e-5
    )
    assert not prepared.hidden.requires_grad and all(not l.key.requires_grad for l in prepared.layers)
    assert prepared.valid_mask.sum(dim=1).tolist() == list(lengths)
    for i, layer in enumerate(prepared.layers):
        torch.testing.assert_close(layer.key, captured_k[i][: sum(lengths)], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(layer.value, captured_v[i].view(-1, 1, 8)[: sum(lengths)], atol=2e-6, rtol=2e-5)


def test_current_gen_text_norm_remains_trainable():
    model = tiny_model()
    pack, mask, pos, text = native_pack(model, (3,))
    features = prepare_und_text_features(model, pack, pos, text)
    output, _ = model(pack, mask, pos, prepared_text=features)
    get_gen_seq(output)[:2].square().mean().backward()
    for block in model.layers:
        grad = block.self_attn.k_norm_und_for_gen.weight.grad
        assert grad is not None and grad.abs().sum() > 0
        assert block.self_attn.k_proj.weight.grad is None


def test_joint_mixed_presence_isolation_and_frozen_gen_gradient():
    torch.manual_seed(92)
    model = tiny_model()
    model.requires_grad_(False)
    gen = [torch.randn(2, 16) for _ in range(4)]
    pack, mask, pos, text = native_pack(model, (3, 0, 2, 1), gen)
    features = prepare_und_text_features(model, pack, pos, text)
    bridge = nn.Linear(4, 16)
    local = (None, torch.randn(4, 4), None, torch.randn(4, 4))
    global_rows = (None, None, torch.randn(64, 16, requires_grad=True), torch.randn(64, 16, requires_grad=True))
    lc = build_memory_prefix_context(local, bridge, torch.zeros(16), target_dtype=torch.float32)
    joint = build_joint_memory_prefix_context(global_rows, lc, target_dtype=torch.float32)
    assert joint.global_lengths == (0, 0, 64, 64) and joint.local_lengths == (0, 4, 0, 4)
    assert joint.sample_offsets.tolist() == [0, 0, 4, 68, 136] and joint.k_local == 4
    output, _ = model(pack, mask, pos, prepared_text=features, memory_prefix_context=joint)
    for i, n in enumerate((3, 0, 2, 1)):
        one, om, op, ot = native_pack(model, (n,), [gen[i]])
        of = prepare_und_text_features(model, one, op, ot)
        ol = build_memory_prefix_context((local[i],), bridge, torch.zeros(16), target_dtype=torch.float32)
        oj = build_joint_memory_prefix_context((global_rows[i],), ol, target_dtype=torch.float32)
        single, _ = model(one, om, op, prepared_text=of, memory_prefix_context=oj)
        torch.testing.assert_close(
            get_gen_seq(output)[2 * i : 2 * i + 2], get_gen_seq(single)[:2], atol=2e-6, rtol=2e-5
        )
    get_gen_seq(output)[:8].square().mean().backward()
    assert global_rows[2].grad.abs().sum() > 0 and global_rows[3].grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_absent_global_returns_exact_old_context_and_no_rng():
    bridge = nn.Linear(4, 16)
    local = build_memory_prefix_context((torch.randn(4, 4),), bridge, torch.zeros(16), target_dtype=torch.float32)
    rng = torch.random.get_rng_state().clone()
    assert build_joint_memory_prefix_context((None,), local, target_dtype=torch.float32) is local
    assert build_joint_memory_prefix_context(None, None, target_dtype=torch.float32) is None
    assert torch.equal(rng, torch.random.get_rng_state())


def test_zero_value_is_not_an_absent_prefix():
    q = torch.ones(1, 1, 1, 2)
    k, v = torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2)
    off = F.scaled_dot_product_attention(q, k, v)
    on = F.scaled_dot_product_attention(q, torch.cat((k, k), dim=2), torch.cat((v * 0, v), dim=2))
    assert torch.equal(off, torch.ones_like(off)) and torch.equal(on, off * 0.5)


def test_cached_text_and_global_kv_equal_current_projection():
    torch.manual_seed(11)
    model = tiny_model()
    pack, mask, pos, text = native_pack(model, (3, 0, 2, 1))
    features = prepare_und_text_features(model, pack, pos, text)
    bridge = nn.Linear(4, 16)
    local = build_memory_prefix_context(
        (None, torch.randn(4, 4), None, torch.randn(4, 4)), bridge, torch.zeros(16), target_dtype=torch.float32
    )
    globals = (None, None, torch.randn(64, 16), torch.randn(64, 16))
    with torch.no_grad():
        joint = build_joint_memory_prefix_context(globals, local, target_dtype=torch.float32)
        reference, _ = model(pack, mask, pos, prepared_text=features, memory_prefix_context=joint)
        text_kv = tuple(
            block.prepare_gen_text_kv(layer) for block, layer in zip(model.layers, features.layers, strict=True)
        )
        global_kv = tuple(block.prepare_global_kv(torch.cat(globals[2:])) for block in model.layers)
        cached = replace(joint, global_kv=global_kv)
        output, _ = model(
            pack, mask, pos, prepared_text=features, prepared_gen_text_kv=text_kv, memory_prefix_context=cached
        )
    torch.testing.assert_close(get_gen_seq(output)[:8], get_gen_seq(reference)[:8], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="no_grad"):
        model(pack, mask, pos, prepared_text=features, prepared_gen_text_kv=text_kv)


def test_all_layers_prefix_lengths_and_no_new_queries():
    model = tiny_model()
    pack, mask, pos, text = native_pack(model, (3, 0))
    features = prepare_und_text_features(model, pack, pos, text)
    joint = build_joint_memory_prefix_context((torch.randn(64, 16), None), None, target_dtype=torch.float32)
    seen = []
    for block in model.layers:
        dispatch = block.self_attn.dispatch_attention_fn

        def capture(q, k, v, *args, original=dispatch, **kwargs):
            seen.append(
                (
                    q["_num_full_tokens"],
                    kwargs["memory_prefix_key_states"].shape,
                    kwargs["memory_prefix_sample_offsets"].tolist(),
                    kwargs["skip_und_attention"],
                )
            )
            return original(q, k, v, *args, **kwargs)

        block.self_attn.dispatch_attention_fn = capture
    model(pack, mask, pos, prepared_text=features, memory_prefix_context=joint)
    assert seen == [(4, (64, 1, 8), [0, 64, 64], True)] * 3


class TinyLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pack, attention_mask, position_ids, **kwargs):
        return self.model(pack, attention_mask, position_ids, **kwargs)


def tiny_network(*, g2_text=False):
    model = tiny_model()
    config = Cosmos3VFMNetworkConfig(vlm_config=model.config, vision_gen=False, g2_text_enabled=g2_text)
    return Cosmos3VFMNetwork(TinyLM(model), config)


def test_public_und_preparation_and_disabled_construction_rng():
    torch.manual_seed(15)
    old = tiny_network()
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(15)
    enabled = tiny_network(g2_text=True)
    assert not any("global_memory" in n for n, _ in enabled.named_parameters())
    assert not hasattr(enabled, "global_memory_reader")
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert set(old.state_dict()) == set(enabled.state_dict())
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, enabled.state_dict()[name], atol=0, rtol=0)
    pack, _, pos, text = native_pack(enabled.language_model.model, (3, 0, 1))
    reference = prepare_und_text_features(enabled.language_model.model, pack, pos, text)
    public = enabled.prepare_und_text(text)
    torch.testing.assert_close(public.hidden, reference.hidden)
    for p, r in zip(public.layers, reference.layers, strict=True):
        torch.testing.assert_close(p.key, r.key)
        torch.testing.assert_close(p.value, r.value)
    with pytest.raises(ValueError, match="g2_text_enabled"):
        old.prepare_und_text(text)


def test_native_text_validation_with_full_split_eog():
    _, _, _, text = native_pack(tiny_model(), (3,))
    carrier = G2ConditionBatch(text=text, views=(None,))
    packed = PackedSequence(
        sample_lens=[6],
        split_lens=[3, 3],
        attn_modes=["causal", "full"],
        text_ids=torch.tensor([1, 2, 3, 9]),
        text_indexes=torch.tensor([0, 1, 2, 5]),
        position_ids=torch.arange(6).expand(3, -1),
        sequence_length=6,
        g2_condition=carrier,
    )
    packed.validate_g2_text()
    packed.text_ids[1] = 17
    with pytest.raises(ValueError, match="tokens/positions"):
        packed.validate_g2_text()


def test_parameter_inventory_complete_and_switch_preserves_objects():
    net = tiny_network(g2_text=True)
    net.global_memory_reader = nn.Linear(4, 4)
    net.global_memory_projector = nn.Linear(4, 16)
    net.local_memory2llm = nn.Linear(4, 16)
    net.action2llm = nn.Linear(4, 16)
    owner = nn.Module()
    owner.net = net
    owner.frozen_vae = nn.Linear(4, 4)
    inventory = parameter_inventory(owner, net_prefix="net.")
    groups = inventory["parameters"]
    selected = [n for values in groups.values() for n in values]
    assert len(selected) == len(set(selected)) == len(dict(owner.named_parameters()))
    assert all(n.startswith("net.global_memory") for n in groups["global"])
    assert all(n.startswith("net.action2llm") for n in groups["action_adapter"])
    assert "net.local_memory2llm.weight" in groups["frozen"]
    assert "frozen_vae.weight" in groups["frozen"]
    ids = {n: id(p) for n, p in owner.named_parameters()}
    set_trainable_names(owner, groups["global"])
    assert {n for n, p in owner.named_parameters() if p.requires_grad} == set(groups["global"])
    set_trainable_names(owner, groups["global"] + groups["gen"] + groups["action_adapter"])
    assert ids == {n: id(p) for n, p in owner.named_parameters()}
    with pytest.raises(ValueError, match="unknown"):
        set_trainable_names(owner, ("typo",))
    net.global_memory_reader.weight.grad = torch.ones_like(net.global_memory_reader.weight)
    with pytest.raises(RuntimeError, match="zero_grad"):
        set_trainable_names(owner, groups["global"])


def test_fsdp_custom_entries_registered(monkeypatch):
    from cosmos_framework.model.generator.mot import parallelize_unified_mot as layer_module
    from cosmos_framework.model.generator.mot import parallelize_vfm_network as root_module

    net = tiny_network(g2_text=True)
    registered = []
    for module in (root_module, layer_module):
        monkeypatch.setattr(module, "fully_shard", lambda *, module, **kw: module)
        monkeypatch.setattr(module, "fsdp_mesh", lambda dims: "cpu-mock-mesh")
        monkeypatch.setattr(
            module, "register_fsdp_forward_method", lambda module, entry: registered.append((module, entry))
        )
    monkeypatch.setattr(root_module, "parallelize_unified_mot", lambda lm, **kw: lm)
    dims = SimpleNamespace(cp_enabled=False, dp_enabled=True, fsdp_cpu_offload=False)
    root_module.parallelize_vfm_network(net, dims, SimpleNamespace(enabled=False), SimpleNamespace(mode="none"))
    layer_module.apply_fsdp(net.language_model, dims)
    root_entries = {name for module, name in registered if module is net}
    assert {"prepare_und_text", "prepare_gen_text_kv", "prepare_global_kv", "prepare_global_tokens"} <= root_entries
    for block in net.language_model.model.layers:
        assert {name for module, name in registered if module is block} == {
            "reasoner_forward",
            "prepare_und",
            "prepare_gen_text_kv",
            "prepare_global_kv",
        }


def test_request_prepares_und_once_and_preserves_carrier(monkeypatch):
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    net = tiny_network(g2_text=True)
    _, _, _, text = native_pack(net.language_model.model, (3,))
    condition = G2ConditionBatch(text=text, views=(None,))
    owner = SimpleNamespace(net=net, tensor_kwargs={"device": torch.device("cpu")})
    calls = []
    original = net.prepare_und_text
    monkeypatch.setattr(net, "prepare_und_text", lambda text: (calls.append(text) or original(text)))
    prepared = OmniMoTModel.prepare_g2_condition(owner, condition)
    again = OmniMoTModel.prepare_g2_condition(owner, prepared)
    assert len(calls) == 1 and condition.text_features is None
    assert again.text_features.text == condition.text


def test_d_prepared_mixed_batch_uses_tokens_presence_without_views():
    model = tiny_model()
    features, text_kvs, global_kvs, projected = [], [], [], []
    with torch.no_grad():
        for i, n in enumerate((3, 0, 1)):
            pack, _, pos, text = native_pack(model, (n,))
            f = prepare_und_text_features(model, pack, pos, text)
            features.append(f)
            text_kvs.append(tuple(b.prepare_gen_text_kv(l) for b, l in zip(model.layers, f.layers, strict=True)))
            g = torch.randn(1, 64, 16) if i == 1 else None
            projected.append(g)
            global_kvs.append(None if g is None else tuple(b.prepare_global_kv(g[0]) for b in model.layers))
        condition = stack_g2_prepared(
            text_features=features, text_kv=text_kvs, global_tokens=projected, global_kv=global_kvs
        )
    assert condition.views == (None, None, None) and condition.global_present == (False, True, False)
    assert condition.text.lengths == (3, 0, 1) and condition.text_features.hidden.shape == (3, 3, 16)
    assert all(k.key.shape == (4, 1, 8) for k in condition.gen_text_kv)
    assert all(k.key.shape == (64, 1, 8) for k in condition.global_kv)

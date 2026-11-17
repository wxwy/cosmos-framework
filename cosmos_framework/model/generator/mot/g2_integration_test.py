# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Actual A/B modules + native packing/action head; synthetic CPU evidence only."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.utils.static_task_text import (
    StaticTaskText,
    StaticTaskTextRecipe,
    effective_text_batch,
    prepare_static_task_text,
)
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot import attention, global_memory
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import EffectiveTextBatch, G2ConditionBatch
from cosmos_framework.model.generator.mot.g2_core_test import TinyLM, cpu_attention, tiny_model
from cosmos_framework.model.generator.mot.g2_parameters import parameter_inventory, set_trainable_names
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

SPECIAL = {"bos_token_id": 60, "eos_token_id": 61, "start_of_generation": 62, "end_of_generation": 63}


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    monkeypatch.setattr(attention, "attention", cpu_attention)
    monkeypatch.setattr(global_memory, "_dispatch_attention", cpu_attention)


def network():
    lm = TinyLM(tiny_model())
    # The fixture UND has already been initialized by its native constructor.
    lm.init_weights = lambda buffer_device: None
    config = Cosmos3VFMNetworkConfig(
        vlm_config=lm.model.config,
        vision_gen=True,
        action_gen=True,
        latent_patch_size=1,
        latent_channel_size=4,
        action_dim=15,
        action_io_projector_type="shared_weight_bias",
        local_memory_enabled=True,
        global_memory_enabled=True,
        global_memory_latent_channels=4,
        global_memory_config={"enabled": True, "top_view_variant": "with_robot"},
    )
    net = Cosmos3VFMNetwork(lm, config)
    net.init_weights(buffer_device=torch.device("cpu"))
    return net


def packed_batch(net, prompts, views, local, vision=None, actions=None):
    batch = len(prompts)
    vision = [torch.randn(1, 4, 1, 2, 2) for _ in prompts] if vision is None else vision
    actions = [torch.randn(2, 15) for _ in prompts] if actions is None else actions
    clean = GenerationDataClean(
        batch_size=batch,
        is_image_batch=True,
        x0_tokens_vision=vision,
        x0_tokens_action=actions,
        fps_vision=torch.full((batch,), 24.0),
        fps_action=torch.full((batch,), 24.0),
        action_domain_id=[torch.tensor(0) for _ in prompts],
        raw_action_dim=[torch.tensor(15) for _ in prompts],
    )
    plans = [SequencePlan(has_text=True, has_vision=True, has_action=True) for _ in prompts]
    packed = pack_input_sequence(
        plans, prompts, clean, torch.full((batch,), 0.5), SPECIAL, latent_patch_size=1, action_dim=15
    )
    lengths = tuple(len(p) + 3 for p in prompts)
    tokens = tuple(tuple(packed.text_ids.split(lengths)[i].tolist()) for i in range(batch))
    text = EffectiveTextBatch(
        token_ids=tokens,
        position_ids=tuple((tuple(range(n)),) * 3 for n in lengths),
        dropped=tuple(not p for p in prompts),
        formatter_id="native-packing-fixture",
        tokenizer_id="synthetic",
        dropout_id="once",
    )
    packed.g2_condition = G2ConditionBatch(text=text, views=views)
    packed.local_memory_tokens = local
    packed.validate_g2_text()
    return packed, vision, actions


def test_actual_reader_native_action_loss_reaches_registered_global_only():
    torch.manual_seed(611)
    net = network()
    inventory = parameter_inventory(net)
    set_trainable_names(net, inventory["parameters"]["global"])
    # Existing Local inner core executes with frozen slow weights, outside no_grad.
    local_rows, state = net.local_memory_runtime.core.step_many(
        torch.randn(4, 256), net.local_memory_runtime.core.initial_state(4)
    )
    assert all(torch.isfinite(s).all() for s in state)
    views = (None, None, (synthetic_view(view_id="global-only"),), (synthetic_view(view_id="both"),))
    packed, _, _ = packed_batch(net, [[1, 2], [], [3], [4, 5, 6]], views, (None, local_rows[1], None, local_rows[3]))
    output = net(packed)
    assert output["last_hidden_state"].shape == (packed.sequence_length, 16)
    assert [tuple(p.shape) for p in output["preds_action"]] == [(2, 15)] * 4
    loss = sum((p - 0.25).square().mean() for p in output["preds_action"])
    loss.backward()
    params = dict(net.named_parameters())
    for name in (
        "global_memory_reader.queries",
        "global_memory_reader.spatial.image_projection.weight",
        "global_memory_projector.projection.weight",
    ):
        grad = params[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None for n, p in params.items() if n not in inventory["parameters"]["global"])
    assert inventory["new_state_keys"] and all(k in net.state_dict() for k in inventory["new_state_keys"])


def test_actual_reader_mixed_native_batch_equals_separate_and_cached():
    torch.manual_seed(612)
    net = network().eval()
    local = (None, torch.randn(4, 32), None, torch.randn(4, 32))
    views = (None, None, (synthetic_view(view_id="G"),), (synthetic_view(view_id="GL"),))
    prompts = [[1, 2], [], [3], [4, 5, 6]]
    packed, vision, actions = packed_batch(net, prompts, views, local)
    with torch.no_grad():
        reference = net(packed)["preds_action"]
        features = net.prepare_und_text(packed.g2_condition.text)
        tokens = net.prepare_global_tokens(features, views)
        condition = replace(
            packed.g2_condition,
            views=(None,) * 4,
            text_features=features,
            global_tokens=tokens,
            gen_text_kv=net.prepare_gen_text_kv(features),
            global_kv=net.prepare_global_kv(tokens),
        )
        cached = net(replace(packed, g2_condition=condition))["preds_action"]
        for i in range(4):
            single, _, _ = packed_batch(net, [prompts[i]], (views[i],), (local[i],), [vision[i]], [actions[i]])
            output = net(single)["preds_action"][0]
            torch.testing.assert_close(output, reference[i], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(cached[i], reference[i], atol=3e-6, rtol=3e-5)


def test_train_a_all_global_absent_has_zero_grad_without_prefix_or_reader(monkeypatch):
    net = network()
    inventory = parameter_inventory(net)
    set_trainable_names(net, inventory["parameters"]["global"])
    monkeypatch.setattr(net.global_memory_reader, "forward", lambda *a, **kw: pytest.fail("absent Reader was called"))
    prefixes = []
    for block in net.language_model.model.layers:
        original = block.self_attn.dispatch_attention_fn

        def capture(*args, original=original, **kwargs):
            prefixes.append("memory_prefix_key_states" in kwargs)
            return original(*args, **kwargs)

        block.self_attn.dispatch_attention_fn = capture
    packed, _, _ = packed_batch(net, [[], [3]], (None, None), (None, None))
    output = net(packed)
    sum(p.square().mean() for p in output["preds_action"]).backward()
    assert prefixes == [False] * 3
    for name, parameter in net.named_parameters():
        if name in inventory["parameters"]["global"]:
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        else:
            assert parameter.grad is None


def test_c_shared_native_text_normal_and_dropout_use_same_actual_und_input():
    calls = []

    class Tokenizer:
        def tokenize_text(self, text, **kwargs):
            calls.append(text)
            return [7, *(ord(char) % 40 for char in text[:4]), 9]

    task = StaticTaskText(
        "close fridge",
        "Initial spatial context: fridge in kitchen; close fridge.",
        "fixture-episode",
        "fixture-subtask",
        "fixture-template",
        "fixture-source",
        "fixture-sg0",
    )
    recipe = StaticTaskTextRecipe(17, 24.0, 16, 16, "synthetic-tokenizer")
    prepared = [prepare_static_task_text(task, recipe, Tokenizer(), dropout_applied=d) for d in (False, True)]
    effective = effective_text_batch(prepared, recipe, special_tokens=SPECIAL)
    net = network().eval()
    views = ((synthetic_view(view_id="normal"),), (synthetic_view(view_id="dropped"),))
    packed, _, _ = packed_batch(net, [s["text_token_ids"].tolist() for s in prepared], views, (None, None))
    packed.g2_condition = replace(packed.g2_condition, text=effective)
    with torch.no_grad():
        result = net(packed)
    assert len(calls) == 2 and calls[1] == "" and effective.dropped == (False, True)
    assert all(torch.isfinite(p).all() for p in result["preds_action"])
    assert tuple(packed.text_ids.tolist()) == sum(effective.token_ids, ())

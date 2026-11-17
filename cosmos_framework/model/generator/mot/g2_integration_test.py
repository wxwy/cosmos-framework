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
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot import attention, global_memory
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import EffectiveTextBatch, G2ConditionBatch, GlobalMemoryConfig
from cosmos_framework.model.generator.mot.g2_core_test import TinyLM, cpu_attention, native_pack, tiny_model
from cosmos_framework.model.generator.mot.g2_parameters import parameter_inventory, set_trainable_names
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLTextModel, prepare_und_text_features
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

SPECIAL = {"bos_token_id": 60, "eos_token_id": 61, "start_of_generation": 62, "end_of_generation": 63}


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    monkeypatch.setattr(attention, "attention", cpu_attention)
    monkeypatch.setattr(global_memory, "_dispatch_attention", cpu_attention)


def network(*, global_enabled=True):
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
        global_memory_enabled=global_enabled,
        global_memory_latent_channels=4,
        global_memory_config={"enabled": global_enabled, "top_view_variant": "with_robot"},
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(3,), (0,), (3, 0, 1)])
def test_nemotron_family_native_und_and_current_gen_norm(dtype, lengths):
    config = Nemotron3DenseVLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=3,
        vocab_size=64,
        mrope_section=[1, 1, 2],
    )
    model = Nemotron3DenseVLTextModel(
        config, qk_norm_for_text=False, qk_norm_for_diffusion=True, use_und_k_norm_for_gen=True
    ).to(dtype)
    model.rotary_emb.float()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("moe_gen" in name or "k_norm_und_for_gen" in name)
    pack, mask, pos, text = native_pack(model, lengths)
    reference, _ = model(pack, mask, pos)
    features = prepare_und_text_features(model, pack, pos, text)
    output, _ = model(pack, mask, pos, prepared_text=features)
    hidden = torch.cat([features.hidden[i, :n] for i, n in enumerate(lengths)])
    torch.testing.assert_close(hidden, get_und_seq(reference)[: sum(lengths)], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        get_gen_seq(output)[: 2 * len(lengths)], get_gen_seq(reference)[: 2 * len(lengths)], atol=2e-6, rtol=2e-5
    )
    if sum(lengths):
        (get_gen_seq(output)[: 2 * len(lengths)].float() * torch.arange(1, 17)).sum().backward()
        assert all(b.self_attn.k_norm_und_for_gen.weight.grad.abs().sum() > 0 for b in model.layers)


@pytest.mark.parametrize("empty_second", [False, True])
def test_d_actual_cache_sessions_to_native_action_head(empty_second, monkeypatch):
    from cosmos_framework.inference.g2_model_hooks_test import actual_tiny_model, consumer, hooks
    from cosmos_framework.inference.g2_policy_conditions import G2ConditionRuntime
    from cosmos_framework.inference.g2_policy_conditions_test import FixtureProvider, request

    facade, counts = actual_tiny_model(monkeypatch)
    facade.net = network().eval()
    facade.llm_special_tokens = SPECIAL
    provider_on, provider_off = FixtureProvider(), FixtureProvider()
    provider_on.scene_offsets["scene-b"] = provider_off.scene_offsets["scene-b"] = 3.0
    on = G2ConditionRuntime(hooks(facade), profile="g2_r2", provider=provider_on, allow_fixtures=True)
    off = G2ConditionRuntime(
        hooks(facade), profile="g2_r2", provider=provider_off, allow_fixtures=True, cache_enabled=False
    )
    for frame in range(3):
        reqs = [request(session="a", frame=frame), request(session="b", episode="ep-b", scene="scene-b", frame=frame)]
        prepared = [consumer(facade, req, empty=empty_second and i == 1) for i, req in enumerate(reqs)]
        opts = dict(recipe_ids=[p[1] for p in prepared], effective_texts=[p[2] for p in prepared])
        with (
            on.prepare_many(reqs, [p[0] for p in prepared], **opts) as cached,
            off.prepare_many(reqs, [p[0] for p in prepared], **opts) as fresh,
        ):
            # Fixture packer adds precisely the SPECIAL entries C already used.
            prompts = [list(p[2].token_ids[0][1:-2]) for p in prepared]
            local = (None, torch.randn(4, 32))
            packed, _, _ = packed_batch(facade.net, prompts, (None, None), local)
            with torch.no_grad():
                actual = facade.net(replace(packed, g2_condition=cached["_g2_condition"]))["preds_action"]
                expected = facade.net(replace(packed, g2_condition=fresh["_g2_condition"]))["preds_action"]
            for lhs, rhs in zip(actual, expected, strict=True):
                torch.testing.assert_close(lhs, rhs, atol=3e-6, rtol=3e-5)
    assert counts["prepare_g2_text"] == counts["read_g2_global"] == 8  # 2 cached + 6 reference
    assert on.cache.info()["stages"]["global"]["hit"] == 4


def test_c_warm_start_stage_switch_and_resume_with_actual_native_action_heads():
    """Direct C controller/native-head composition; grouped Local/DCP tested separately."""
    import copy
    from types import SimpleNamespace

    from torch import nn

    from cosmos_framework.checkpoint.g2_warm_start import load_g1_model_only
    from cosmos_framework.trainer.g2_training import G2_OPTIMIZER_SELECTION, build_g2_optimizer
    from cosmos_framework.trainer.g2_training_test import equal_tree, plan_fixture
    from examples.psm_wma_robocasa_g2_recipe import install_g2_training

    text_recipe = StaticTaskTextRecipe(17, 24.0, 16, 16, "synthetic-tokenizer")
    plan = replace(plan_fixture(a=1, b=1, target=2), text_recipe_identity=text_recipe.identity)
    recipe = SimpleNamespace(training=plan, text=text_recipe, global_memory=GlobalMemoryConfig(enabled=True))

    def owner():
        model = nn.Module()
        model.net = network()
        model.g2_parameter_inventory = lambda: parameter_inventory(model, net_prefix="net.")
        model.set_trainable_names = lambda names: set_trainable_names(model, names)

        def factory(_optimizer_config, _scheduler_config):
            return build_g2_optimizer(
                model,
                optimizer_type="adamw",
                lr=1e-5,
                fused=True,
                keys_to_select=[G2_OPTIMIZER_SELECTION],
                weight_decay=0.03,
            ), None

        model.init_optimizer_scheduler = factory
        trainer = SimpleNamespace()
        install_g2_training(model, trainer, recipe, {})
        optimizer, controller = model.init_optimizer_scheduler(None, None)
        return model, optimizer, controller

    torch.manual_seed(613)
    g1 = nn.Module()
    g1.net = network(global_enabled=False)
    model, optimizer, controller = owner()
    report = load_g1_model_only(
        model, g1.state_dict(), new_global_names=frozenset(model.g2_parameter_inventory()["new_state_keys"])
    )
    assert report["local_loaded_names"] and controller.stage.value == "TRAIN_A"
    local_weights = copy.deepcopy(model.net.local_memory_runtime.state_dict())

    def step(model, optimizer, controller, iteration):
        packed, _, _ = packed_batch(model.net, [[1, 2]], ((synthetic_view(),),), (None,))
        output = model.net(packed)
        (output["preds_action"][0] - 0.25).square().mean().backward()
        controller.prepare_commit(actual_consumers=1, iteration=iteration)
        optimizer.step()
        controller.step()
        optimizer.zero_grad(set_to_none=True)
        # This fixture has no open grouped Local transaction. The actual grouped
        # transaction + sidecar/GA path is exercised by C's integration tests.
        controller.commit_boundary(iteration=iteration + 1, ga_complete=True, local_transaction_complete=True)

    step(model, optimizer, controller, 0)
    assert controller.stage.value == "TRAIN_B"
    assert all(p.requires_grad == (n in controller.roles) for n, p in model.named_parameters())
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), controller.state_dict(), torch.get_rng_state()))
    step(model, optimizer, controller, 1)
    expected = copy.deepcopy(
        (model.state_dict(), optimizer.state_dict(), controller.state_dict(), torch.get_rng_state())
    )
    resumed, other_optimizer, other_controller = owner()
    resumed.load_state_dict(saved[0], strict=True)
    other_optimizer.load_state_dict(saved[1])
    other_controller.load_state_dict(saved[2])
    torch.set_rng_state(saved[3])
    step(resumed, other_optimizer, other_controller, 1)
    equal_tree(
        expected,
        (resumed.state_dict(), other_optimizer.state_dict(), other_controller.state_dict(), torch.get_rng_state()),
    )
    equal_tree(local_weights, resumed.net.local_memory_runtime.state_dict())

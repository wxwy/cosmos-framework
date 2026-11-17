"""Actual B tiny UND/KV + A Reader CPU integration, never a real asset/GPU run."""

from collections import Counter
from types import MethodType

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.utils.static_task_text import (
    StaticTaskText,
    StaticTaskTextRecipe,
    effective_text_batch,
    prepare_static_task_text,
)
from cosmos_framework.inference.g2_model_hooks import VERSION_FIELDS, G2ModelHooks
from cosmos_framework.inference.g2_policy_conditions import G2ConditionRuntime
from cosmos_framework.inference.g2_policy_conditions_test import FixtureProvider, request
from cosmos_framework.model.generator.mot import attention as attention_module
from cosmos_framework.model.generator.mot import global_memory as reader_module
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig
from cosmos_framework.model.generator.mot.g2_core_test import cpu_attention, tiny_network
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def actual_tiny_model(monkeypatch, *, global_enabled=True):
    monkeypatch.setattr(attention_module, "attention", cpu_attention)
    monkeypatch.setattr(reader_module, "_dispatch_attention", cpu_attention)
    torch.manual_seed(17)
    model = nn.Module()
    model.net = tiny_network(g2_text=True)
    if global_enabled:
        config = GlobalMemoryConfig(enabled=True)
        model.net.global_memory_reader = reader_module.GlobalMemoryReader(16, config, latent_channels=4)
        model.net.global_memory_projector = reader_module.GlobalMemoryProjector(16, config)
        model.net.config.global_memory_enabled = True
    methods = (
        "prepare_g2_text",
        "prepare_g2_gen_text_kv",
        "prepare_g2_task_queries",
        "read_g2_global",
        "project_g2_global",
        "prepare_g2_global_kv",
    )
    counts = Counter()
    for name in methods:
        public = MethodType(getattr(OmniMoTModel, name), model)

        def count(*args, public=public, name=name):
            counts[name] += 1
            return public(*args)

        setattr(model, name, count)
    model.tokenize_text = lambda text, **_: [ord(c) % 50 + 1 for c in text]
    model.llm_special_tokens = {"bos_token_id": 61, "eos_token_id": 62, "start_of_generation": 63}
    model.eval()
    return model, counts


def hooks(model):
    return G2ModelHooks(
        model,
        versions={name: f"TEST-actual-tiny-{name}-v1" for name in VERSION_FIELDS},
        use_float_positions=False,
    )


def consumer(model, req, *, empty=False):
    recipe = StaticTaskTextRecipe(17, 20, 256, 512, "TEST-actual-tiny-tokenizer_identity-v1")
    prepared = prepare_static_task_text(
        StaticTaskText(**req["g2"]["static_text"]),
        recipe,
        model,
        dropout_applied=empty,
    )
    text = effective_text_batch((prepared,), recipe, special_tokens=model.llm_special_tokens)
    # Static subtask asset remains the full instruction even for an explicit
    # future no-text branch; condition identity lives in the effective C0 text.
    full = prepare_static_task_text(StaticTaskText(**req["g2"]["static_text"]), recipe, model, dropout_applied=False)
    return full["ai_caption"], recipe.identity, text


def test_B_public_preparation_A_Reader_and_real_C_native_text_cache_parity(monkeypatch):
    model, counts = actual_tiny_model(monkeypatch)
    cached, reference = hooks(model), hooks(model)
    on = G2ConditionRuntime(cached, profile="g2_r2", provider=FixtureProvider(), allow_fixtures=True)
    off = G2ConditionRuntime(
        reference, profile="g2_r2", provider=FixtureProvider(), allow_fixtures=True, cache_enabled=False
    )
    for frame, empty in enumerate((False, False, True, True, False)):
        req = request(frame=frame)
        serialized, recipe, text = consumer(model, req, empty=empty)
        options = dict(recipe_ids=[recipe], effective_texts=[text])
        with (
            on.prepare_many([req], [serialized], **options) as a,
            off.prepare_many([req], [serialized], **options) as b,
        ):
            actual, expected = a["_g2_condition"], b["_g2_condition"]
            assert actual.views == (None,) and actual.global_present == (True,)
            assert actual.text.dropped == (empty,)
            assert torch.equal(actual.text_features.hidden, expected.text_features.hidden)
            assert torch.equal(actual.global_tokens[0], expected.global_tokens[0])
            for x, y in zip(
                (*actual.gen_text_kv, *actual.global_kv), (*expected.gen_text_kv, *expected.global_kv), strict=True
            ):
                assert torch.equal(x.key, y.key) and torch.equal(x.value, y.value)
    assert counts["prepare_g2_text"] == counts["read_g2_global"] == 7  # cache2 + reference5, not 10
    assert on.cache.info()["stages"]["global"]["hit"] == 3
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("mutation", ["inplace", "replacement", "config"])
def test_actual_parameter_or_config_change_invalidates_old_weight_derived_cache(monkeypatch, mutation):
    model, counts = actual_tiny_model(monkeypatch)
    adapter = hooks(model)
    rt = G2ConditionRuntime(adapter, profile="g2_r2", provider=FixtureProvider(), allow_fixtures=True)
    req = request()
    serialized, recipe, text = consumer(model, req)
    with rt.prepare_many([req], [serialized], recipe_ids=[recipe], effective_texts=[text]) as result:
        before = result["_g2_condition"].global_tokens[0].clone()
    revision = rt.cache.revision
    parameter = model.net.global_memory_projector.modality
    if mutation == "inplace":
        with torch.no_grad():
            parameter.add_(0.5)
    elif mutation == "replacement":
        model.net.global_memory_projector.modality = nn.Parameter(parameter.detach().clone() + 0.5)
    else:
        adapter.versions["inference_config_id"] = "TEST-new-config"
    with rt.prepare_many([req], [serialized], recipe_ids=[recipe], effective_texts=[text]) as result:
        after = result["_g2_condition"].global_tokens[0]
        assert torch.equal(before, after) is (mutation == "config")
    assert rt.cache.revision != revision
    assert counts["prepare_g2_text"] == counts["read_g2_global"] == 2


def test_public_stack_routes_ragged_mixed_Sessions_without_loading_views_on_hit(monkeypatch):
    model, counts = actual_tiny_model(monkeypatch)
    rt = G2ConditionRuntime(hooks(model), profile="g2_r2", provider=FixtureProvider(), allow_fixtures=True)
    reqs = [request(session="a"), request(session="b", episode="ep-b", scene="scene-b")]
    values = [consumer(model, req) for req in reqs]
    for _ in range(2):
        with rt.prepare_many(
            reqs, [v[0] for v in values], recipe_ids=[v[1] for v in values], effective_texts=[v[2] for v in values]
        ) as result:
            condition = result["_g2_condition"]
            assert condition.global_present == (True, True) and condition.views == (None, None)
            assert condition.text.token_ids == tuple(v[2].token_ids[0] for v in values)
            assert all(layer.key.shape[0] == 128 for layer in condition.global_kv)
            assert all(layer.key.shape[0] == sum(condition.text.lengths) for layer in condition.gen_text_kv)
    assert counts["read_g2_global"] == 2


def test_R1_actual_B_public_text_path_has_no_Reader_and_no_Global_data(monkeypatch):
    model, counts = actual_tiny_model(monkeypatch, global_enabled=False)
    rt = G2ConditionRuntime(hooks(model), profile="g2_r1")
    req = request()
    serialized, recipe, text = consumer(model, req)
    with rt.prepare_many([req], [serialized], recipe_ids=[recipe], effective_texts=[text]) as result:
        condition = result["_g2_condition"]
        assert condition.global_present == (False,) and condition.global_kv is None
    assert not hasattr(model.net, "global_memory_reader")
    assert counts["prepare_g2_text"] == 1 and counts["read_g2_global"] == 0


def test_unversioned_inference_parameters_conservatively_disable_cross_request_reuse():
    with torch.inference_mode():
        model = nn.Linear(1, 1).eval()
    adapter = hooks(model)
    previous = adapter.cache_revision
    adapter.refresh_revision()
    assert adapter.cache_revision != previous

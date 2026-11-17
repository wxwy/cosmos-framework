"""Synthetic CPU hook/provider contract tests, not real UND/Reader/GPU evidence."""

from collections import Counter
from dataclasses import asdict, dataclass, replace

import pytest
import torch

from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskText
from cosmos_framework.inference.g2_policy_conditions import (
    G2ConditionRuntime,
    OnlineSpatialSnapshot,
    mask_content_identity,
)
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import (
    EffectiveTextBatch,
    G2SampleIdentity,
    GlobalCacheSignature,
    ReaderOutput,
    TextCacheSignature,
    UNDLayerKV,
    UNDTextFeatures,
)


@dataclass(frozen=True)
class PreparedFixture:
    text: UNDTextFeatures
    text_kv: tuple
    global_tokens: torch.Tensor | None
    global_kv: tuple | None


class FixtureHooks:
    """Deterministic arithmetic double; does not claim the real model ABI is implemented."""

    cache_revision = "test-checkpoint+config-v1"
    tokenizer_identity = "TEST-tokenizer"
    use_float_positions = False
    training = False

    def __init__(self):
        self.calls = Counter()
        self.position_rule = "TEST-mrope"
        self.reader_version = "TEST-reader-v1"
        self.projector_version = "TEST-projector-v1"
        self.geometry_encoding_version = "TEST-geom-v1"
        self.mask_version = "TEST-mask-v1"
        self.empty_text = False
        self.weight = torch.nn.Parameter(torch.ones(1))

    def tokenize_static_prompt(self, serialized):
        self.calls["tokenize"] += 1
        tokens = () if self.empty_text else tuple(ord(c) for c in serialized)
        return EffectiveTextBatch(
            token_ids=(tokens,),
            position_ids=((tuple(range(len(tokens))),) * 3,),
            dropped=(self.empty_text,),
            formatter_id="TEST-C-shared-prompt",
            tokenizer_id=self.tokenizer_identity,
            dropout_id="TEST-explicit-no-random",
        )

    def text_signature(self, session, text):
        return TextCacheSignature(
            session_id=session,
            checkpoint_id=self.cache_revision,
            und_version="TEST-UND-v1",
            gen_version="TEST-GEN-v1",
            position_rule=self.position_rule,
            dtype="torch.float32",
            text=text,
        )

    def global_signature(self, text, variant):
        return GlobalCacheSignature(
            text=text,
            reader_version=self.reader_version,
            projector_version=self.projector_version,
            geometry_encoding_version=self.geometry_encoding_version,
            top_variant=variant,
            views=(),
            projections=(),
            rasters=(),
            view_types=(),
            mask_version=self.mask_version,
        )

    def prepare_text(self, text):
        self.calls["text"] += 1
        n = text.lengths[0]
        hidden = torch.tensor(text.token_ids[0], dtype=torch.float32).reshape(1, n, 1).repeat(1, 1, 8)
        key = hidden.reshape(n, 2, 4)
        return UNDTextFeatures(
            text=text,
            hidden=hidden,
            valid_mask=torch.ones(1, n, dtype=torch.bool),
            layers=(UNDLayerKV(key=key, value=key + 1, cos=torch.ones(n, 4), sin=torch.zeros(n, 4)),),
        )

    def prepare_gen_text_kv(self, text):
        self.calls["text_kv"] += 1
        return tuple((layer.key * 2, layer.value) for layer in text.layers)

    def prepare_task_queries(self, text):
        self.calls["task_queries"] += 1
        return self.weight * torch.ones(1, 64, 512) * text.hidden.sum()

    def read_global(self, queries, views):
        self.calls["global"] += 1
        value = sum((view.latent * view.valid_mask).sum() * view.projection.physical_width_m for view in views)
        return ReaderOutput(tokens=queries + value)

    def project_global(self, result):
        self.calls["global_projection"] += 1
        return result.tokens[:, :, :8] * self.weight

    def prepare_global_kv(self, tokens):
        self.calls["global_kv"] += 1
        return ((tokens.reshape(1, 64, 2, 4), tokens.reshape(1, 64, 2, 4) + 1),)

    def generation_kwargs(self, *, text_features, text_kv, global_tokens, global_kv):
        return {
            "_g2_fixture": tuple(
                PreparedFixture(*items) for items in zip(text_features, text_kv, global_tokens, global_kv, strict=True)
            )
        }


class FixtureProvider:
    def __init__(self):
        self.views = {
            "top-with": synthetic_view(view_id="top-with"),
            "top-without": synthetic_view(view_id="top-without", top_variant="without_robot"),
            "object": replace(
                synthetic_view(view_id="object", top_variant=None), scope="island", direction="plus_X", island_id="i1"
            ),
            "unrelated": replace(synthetic_view(view_id="unrelated", top_variant=None), direction="plus_Y"),
        }
        self.selected = ("object",)
        self.loads = Counter()
        self.origin = "synthetic_fixture"
        self.snapshots = []
        self.scene_offsets = {}

    def snapshot(self, identity, base):
        top = "top-with" if base.top_variant == "with_robot" else "top-without"
        if top not in self.views:
            raise ValueError("requested top variant missing")
        selected = tuple(self.views[name] for name in sorted((*self.selected, top)))
        if self.scene_offsets.get(identity.scene_id, 0):
            selected = tuple(
                replace(
                    v, identity=replace(v.identity, content_version=f"{v.identity.content_version}:{identity.scene_id}")
                )
                for v in selected
            )
        signature = replace(
            base,
            views=tuple(v.identity for v in selected),
            projections=tuple(v.projection for v in selected),
            rasters=tuple(v.raster for v in selected),
            view_types=tuple((v.scope, v.direction, v.room_id, v.island_id, v.top_variant) for v in selected),
        )
        result = OnlineSpatialSnapshot(
            identity=identity,
            signature=signature,
            source=self.origin,
            mask_content_ids=tuple(mask_content_identity(view.valid_mask) for view in selected),
        )
        self.snapshots.append(result)
        return result

    def load_view(self, snapshot, index):
        name = snapshot.signature.views[index].view_id
        self.loads[name] += 1
        view = self.views[name]
        offset = self.scene_offsets.get(snapshot.identity.scene_id, 0)
        return replace(view, identity=snapshot.signature.views[index], latent=view.latent + offset) if offset else view

    def change(self, name, **fields):
        self.views[name] = replace(self.views[name], **fields)

    def pixels(self, name, version="new-pixels"):
        view = self.views[name]
        self.change(name, identity=replace(view.identity, content_version=version), latent=view.latent + 7)


def request(*, session="s", episode="ep", scene="scene", subtask="subtask", frame=0, variant="with_robot"):
    task = StaticTaskText(
        original_instruction="close door",
        enhanced_instruction="close door; initial spatial context: door in kitchen",
        episode_id=episode,
        subtask_id=subtask,
        template_identity="template-v1",
        source_identity="initial-rollout-caption",
        initial_context_identity="initial-SG-v1",
    )
    return {
        "prompt": task.enhanced_instruction,
        "state": [0.0] * 15,
        "image": "current-RGB",
        "local_memory": {"session_id": session, "episode_id": episode, "consumer_step": frame},
        "g2": {
            "identity": asdict(
                G2SampleIdentity(
                    episode_id=episode,
                    scene_id=scene,
                    subtask_id=subtask,
                    decision_frame=frame,
                    logical_state_frame=frame,
                    data_version="online-v1",
                    alignment_version="online-align-v1",
                    state15_recipe_id="state15-v1",
                    task_binding_id="init-GT-binding",
                )
            ),
            "top_variant": variant,
            "static_text": asdict(task),
        },
    }


def runtime(*, profile="g2_r2", cache=True):
    hooks, provider = FixtureHooks(), FixtureProvider()
    return (
        G2ConditionRuntime(hooks, profile=profile, provider=provider, cache_enabled=cache, allow_fixtures=True),
        hooks,
        provider,
    )


def prepare(rt, req=None, *, recipe="TEST-static-recipe"):
    req = req or request()
    return rt.prepare_many([req], [req["prompt"]], recipe_ids=[recipe])


def test_state_local_camera_unselected_sg_and_owner_frame_do_not_rebuild_global():
    rt, hooks, provider = runtime()
    with prepare(rt) as output:
        initial = output["_g2_fixture"][0].global_tokens.clone()
    provider.pixels("unrelated")
    for name in ("top-with", "object"):
        view = provider.views[name]
        provider.change(name, identity=replace(view.identity, image_owner="state1", image_owner_frame=1))
    req = request(frame=1)
    req["state"][0] = 100
    req["image"] = "other-RGB"
    req["local_memory"]["evidence"] = ["Local update stays outside this runtime"]
    with prepare(rt, req) as output:
        assert torch.equal(initial, output["_g2_fixture"][0].global_tokens)
        assert rt.last_request_sources[0].signature.views[0].image_owner_frame == 1
    assert hooks.calls["global"] == hooks.calls["task_queries"] == 1
    assert provider.loads == {"object": 1, "top-with": 1}
    assert rt.cache.info()["stages"]["global"]["hit"] == 1


@pytest.mark.parametrize(
    "dependency",
    [
        "pixels",
        "arrow",
        "geometry-version",
        "geometry-values",
        "grouping-version",
        "grouping-values",
        "selected-views",
        "variant",
        "vae-recipe",
        "raster",
        "reader",
        "projector",
        "mask",
        "mask-content",
        "geom-encoding",
        "tokens",
        "position-rule",
        "text-recipe",
        "model-reload",
        "subtask",
    ],
)
def test_each_actual_dependency_invalidates_global(dependency):
    rt, hooks, provider = runtime()
    with prepare(rt):
        pass
    req, recipe = request(frame=1), "TEST-static-recipe"
    view = provider.views["object"]
    if dependency in ("pixels", "arrow"):
        provider.pixels("top-with" if dependency == "arrow" else "object")
    elif dependency == "geometry-version":
        provider.change("object", identity=replace(view.identity, geometry_version="geom-new"))
    elif dependency == "geometry-values":
        provider.change("object", projection=replace(view.projection, physical_width_m=9.0))
    elif dependency == "grouping-version":
        provider.change("object", identity=replace(view.identity, membership_version="members-new"))
    elif dependency == "grouping-values":
        provider.change("object", island_id="new-island")
    elif dependency == "selected-views":
        provider.selected = ()
    elif dependency == "variant":
        req["g2"]["top_variant"] = "without_robot"
    elif dependency == "vae-recipe":
        provider.change("object", identity=replace(view.identity, recipe_id="vae-new"))
    elif dependency == "raster":
        provider.change("object", raster=replace(view.raster, content_box=(0, 0, 24, 16)))
    elif dependency in ("reader", "projector", "mask", "geom-encoding"):
        field = {
            "reader": "reader_version",
            "projector": "projector_version",
            "mask": "mask_version",
            "geom-encoding": "geometry_encoding_version",
        }[dependency]
        setattr(hooks, field, "version-new")
    elif dependency == "mask-content":
        mask = view.valid_mask.clone()
        mask[0, 0] = False
        provider.change("object", valid_mask=mask)
    elif dependency == "tokens":
        hooks.empty_text = True  # Explicit test identity, not serving-time random dropout.
    elif dependency == "position-rule":
        hooks.position_rule = "new-position-rule"
    elif dependency == "text-recipe":
        # A new subtask may introduce a newly versioned static recipe.
        req = request(frame=1, subtask="next")
        recipe = "new-recipe"
    elif dependency == "model-reload":
        hooks.cache_revision = "new-weights"
    elif dependency == "subtask":
        req = request(frame=1, subtask="next")
    with prepare(rt, req, recipe=recipe):
        pass
    assert hooks.calls["global"] == 2, dependency


def test_without_robot_movement_is_reusable_and_missing_variant_never_falls_back():
    rt, hooks, provider = runtime()
    with prepare(rt, request(variant="without_robot")):
        pass
    provider.pixels("top-with")
    with prepare(rt, request(frame=1, variant="without_robot")):
        pass
    assert hooks.calls["global"] == 1
    del provider.views["top-without"]
    with pytest.raises(ValueError, match="variant missing"):
        with prepare(rt, request(frame=2, variant="without_robot")):
            pass


def test_default_top_is_with_robot_and_matches_explicit_content_identity():
    rt, hooks, _ = runtime()
    req = request()
    del req["g2"]["top_variant"]
    with prepare(rt, req):
        pass
    with prepare(rt, request(frame=1)):
        pass
    assert hooks.calls["global"] == 1
    assert rt.last_request_sources[0].signature.top_variant == "with_robot"


def test_static_text_reset_and_scene_session_isolation():
    rt, hooks, provider = runtime()
    with prepare(rt):
        pass
    req = request(frame=1)
    req["prompt"] = req["g2"]["static_text"]["enhanced_instruction"] = "changed in same subtask"
    with pytest.raises(ValueError, match="static instruction"):
        with prepare(rt, req):
            pass
    provider.pixels("object")
    other = request(session="other", episode="ep-other", scene="another-scene")
    with prepare(rt, other):
        pass
    assert hooks.calls["text"] == hooks.calls["global"] == 2
    rt.reset("s")
    with prepare(rt, request(episode="new-ep", scene="new-scene")):
        pass
    assert hooks.calls["global"] == 3


def test_snapshot_is_owned_and_fixed_for_all_denoising_steps_and_exception_cleanup():
    rt, hooks, provider = runtime()
    with pytest.raises(RuntimeError, match="sampling failure"):
        with prepare(rt) as kwargs:
            item = kwargs["_g2_fixture"][0]
            original = item.global_tokens.clone()
            provider.views["object"].latent.add_(100)
            provider.pixels("top-with")
            for _step in range(30):
                assert torch.equal(item.global_tokens, original)
            assert rt.last_request_sources[0].identity.decision_frame == 0
            raise RuntimeError("sampling failure")
    assert rt.cache.info()["active_requests"] == 0
    assert torch.is_grad_enabled()
    rt.reset("s")
    assert rt.cache.info()["entries"] == 0


def test_cache_on_off_and_full_empty_text_exact_fixture_parity_and_rng_unchanged():
    on, hooks, _ = runtime()
    off, reference, _ = runtime(cache=False)
    rng = torch.random.get_rng_state().clone()
    for empty in (False, False, True, True, False):
        hooks.empty_text = reference.empty_text = empty
        with prepare(on) as actual, prepare(off) as expected:
            a, b = actual["_g2_fixture"][0], expected["_g2_fixture"][0]
            assert a.text.text.dropped == (empty,)
            assert torch.equal(a.text.hidden, b.text.hidden)
            assert torch.equal(a.global_tokens, b.global_tokens)
            for (ak, av), (bk, bv) in zip(a.global_kv, b.global_kv, strict=True):
                assert torch.equal(ak, bk) and torch.equal(av, bv)
    assert hooks.calls["text"] == 2 and reference.calls["text"] == 5
    assert torch.equal(rng, torch.random.get_rng_state())


def test_r1_does_not_call_reader_load_views_or_create_task_queries():
    rt, hooks, provider = runtime(profile="g2_r1")
    provider.snapshot = lambda *_: pytest.fail("R1 opened Global inventory")
    with prepare(rt) as output:
        assert output["_g2_fixture"][0].global_tokens is None
    assert hooks.calls["global"] == hooks.calls["task_queries"] == 0
    assert not provider.loads


def test_future_wrong_source_and_wrong_loaded_inventory_rejected():
    rt, _, provider = runtime()
    provider.origin = "expert_episode"
    with pytest.raises(ValueError, match="playback"):
        with prepare(rt):
            pass
    provider.origin = "synthetic_fixture"
    view = provider.views["object"]
    provider.change("object", identity=replace(view.identity, image_owner_frame=1))
    with pytest.raises(ValueError, match="future"):
        with prepare(rt):
            pass
    provider.change("object", identity=view.identity)
    load = provider.load_view
    provider.load_view = lambda snapshot, i: replace(load(snapshot, i), room_id="wrong")
    with pytest.raises(ValueError, match="inventory"):
        with prepare(rt):
            pass
    assert rt.cache.info()["active_requests"] == 0


def test_training_and_missing_online_provider_explicit_errors():
    with pytest.raises(ValueError, match="provider"):
        G2ConditionRuntime(FixtureHooks(), profile="g2_r2")
    rt, hooks, _ = runtime()
    hooks.training = True
    with pytest.raises(RuntimeError, match="training"):
        with prepare(rt):
            pass


def test_multi_rank_cache_is_explicitly_rejected_before_any_model_hook(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="one model rank"):
        runtime()


def test_loaded_mask_content_must_match_provider_inventory():
    rt, _, provider = runtime()
    load = provider.load_view

    def changed(snapshot, i):
        view = load(snapshot, i)
        mask = view.valid_mask.clone()
        mask[0, 0] = False
        return replace(view, valid_mask=mask)

    provider.load_view = changed
    with pytest.raises(ValueError, match="valid_mask differs"):
        with prepare(rt):
            pass


@pytest.mark.parametrize("variant", ["with_robot", "without_robot"])
def test_fixed_A_Reader_and_projector_cache_parity_with_CPU_reference_attention(monkeypatch, variant):
    # Use A's published implementation unchanged. Only its existing attention
    # dispatcher boundary is a CPU reference, as in A's own CPU tests.
    from cosmos_framework.model.generator.mot import global_memory as reader_module
    from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig

    def cpu_attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(reader_module, "_dispatch_attention", cpu_attention)
    torch.manual_seed(7)
    config = GlobalMemoryConfig(enabled=True, top_view_variant=variant)
    reader = reader_module.GlobalMemoryReader(8, config, latent_channels=4).eval()
    projector = reader_module.GlobalMemoryProjector(8, config).eval()

    class ActualReaderHooks(FixtureHooks):
        def prepare_task_queries(self, features):
            self.calls["task_queries"] += 1
            return reader.prepare_task_queries(features)

        def read_global(self, queries, views):
            self.calls["global"] += 1
            return reader.read_visual(queries, (views,))

        def project_global(self, output):
            self.calls["global_projection"] += 1
            return projector(output.tokens)

    cached = ActualReaderHooks()
    reference = ActualReaderHooks()
    providers = (FixtureProvider(), FixtureProvider())
    on = G2ConditionRuntime(cached, profile="g2_r2", provider=providers[0], allow_fixtures=True)
    off = G2ConditionRuntime(
        reference, profile="g2_r2", provider=providers[1], allow_fixtures=True, cache_enabled=False
    )
    before = {name: value.detach().clone() for name, value in reader.named_parameters()}
    for frame in range(3):
        with (
            prepare(on, request(frame=frame, variant=variant)) as a,
            prepare(off, request(frame=frame, variant=variant)) as b,
        ):
            actual, expected = a["_g2_fixture"][0], b["_g2_fixture"][0]
            assert torch.equal(actual.global_tokens, expected.global_tokens)
            assert all(
                torch.equal(x, y)
                for pair_a, pair_b in zip(actual.global_kv, expected.global_kv, strict=True)
                for x, y in zip(pair_a, pair_b, strict=True)
            )
    assert cached.calls["global"] == 1 and reference.calls["global"] == 3
    for name, value in reader.named_parameters():
        assert torch.equal(before[name], value) and value.grad is None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU Reader wiring/gradients. Dispatcher kernels are explicit test doubles.

These tests do not validate actual GPU kernels, FSDP or RoboCasa success rate.
"""

import io
import subprocess
import sys
from dataclasses import replace
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.mot import global_memory
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_text, synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig, RasterMapping
from cosmos_framework.model.generator.mot.global_memory import GlobalMemoryProjector, GlobalMemoryReader

CONFIG = GlobalMemoryConfig(enabled=True)
PRODUCTION_DISPATCH = global_memory._dispatch_attention


def cpu_reference(q, k, v):
    assert q.shape[0] == k.shape[0] == v.shape[0] == 1
    assert q.shape[-2:] == k.shape[-2:] == v.shape[-2:] == (8, 64)
    assert k.shape[1] > 0
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


@pytest.fixture(autouse=True)
def reference_attention(monkeypatch):
    monkeypatch.setattr(global_memory, "_dispatch_attention", cpu_reference)


def text_fixture(lengths=(5,), width=24):
    features = synthetic_text(lengths=lengths, width=width, layers=0)
    return replace(features, hidden=torch.randn_like(features.hidden))


def view_fixture(*, name="top", hw=(2, 3), scope="floor", direction="floor_top", island=None, variant="with_robot"):
    view = synthetic_view(view_id=name, hw=hw, top_variant=variant if direction == "floor_top" else None)
    return replace(view, latent=torch.randn_like(view.latent), scope=scope, direction=direction, island_id=island)


def one_text(features, row):
    text = replace(
        features.text,
        token_ids=(features.text.token_ids[row],),
        position_ids=(features.text.position_ids[row],),
        dropped=(features.text.dropped[row],),
    )
    length = text.lengths[0]
    return replace(
        features,
        text=text,
        hidden=features.hidden[row : row + 1, :length],
        valid_mask=features.valid_mask[row : row + 1, :length],
    )


@pytest.mark.parametrize("variant", ["with_robot", "without_robot"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_and_ragged_multi_island_views_dtype_device_finite(variant, dtype):
    torch.manual_seed(201)
    reader = GlobalMemoryReader(24, replace(CONFIG, top_view_variant=variant), latent_channels=4).to(dtype=dtype)
    top = view_fixture(variant=variant)
    views = (
        (top,),
        (
            replace(top, identity=replace(top.identity, view_id="other-top")),
            view_fixture(name="room-X", hw=(3, 5), scope="room", direction="plus_X"),
            view_fixture(name="island-1-Y", hw=(8, 8), scope="island", direction="minus_Y", island="one"),
            view_fixture(name="island-2-X", hw=(4, 3), scope="island", direction="minus_X", island="two"),
        ),
    )
    result = reader(text_fixture((5, 2)), views)
    assert result.tokens.shape == (2, 64, 512)
    assert result.tokens.dtype == dtype and result.tokens.device.type == "cpu"
    assert torch.isfinite(result.tokens).all()
    projector = GlobalMemoryProjector(320, CONFIG).to(dtype=dtype)
    projected = projector(result.tokens)
    assert projected.shape == (2, 64, 320) and projected.dtype == dtype


def test_default_48_channel_full_wan_grid_keeps_every_supported_patch(monkeypatch):
    torch.manual_seed(212)
    reader = GlobalMemoryReader(24, CONFIG)
    view = synthetic_view(hw=(16, 16))
    view = replace(
        view,
        latent=torch.randn(48, 16, 16),
        raster=RasterMapping(
            original_hw=(32, 48), resized_hw=(171, 256), canvas_hw=(256, 256), content_box=(0, 42, 256, 213)
        ),
    )
    calls = []

    def record(q, k, v):
        calls.append((q.shape[1], k.shape[1]))
        return cpu_reference(q, k, v)

    monkeypatch.setattr(global_memory, "_dispatch_attention", record)
    assert reader(text_fixture((3,)), ((view,),)).tokens.shape == (1, 64, 512)
    # 12 positive-overlap rows * all 16 columns. No Local Visual96 or image mean.
    assert calls == [(64, 3), (64, 192), (64, 64), (64, 192), (64, 64)]


def test_mixed_batch_equals_separate_samples_and_never_reads_another_sample():
    torch.manual_seed(202)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture((5, 0, 3))
    views = ((view_fixture(),), (view_fixture(name="B"),), (view_fixture(name="C"),))
    together = reader(features, views).tokens
    separate = torch.cat([reader(one_text(features, i), (views[i],)).tokens for i in range(3)])
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
    changed = (
        views[:1]
        + ((replace(views[1][0], latent=views[1][0].latent + torch.randn_like(views[1][0].latent) * 50),),)
        + views[2:]
    )
    again = reader(features, changed).tokens
    torch.testing.assert_close(again[[0, 2]], together[[0, 2]], rtol=0, atol=0)
    assert not torch.allclose(again[1], together[1])


def test_permuting_views_and_metadata_together_preserves_semantics():
    torch.manual_seed(203)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture()
    views = (
        view_fixture(),
        view_fixture(name="room", scope="room", direction="plus_Y", hw=(3, 5)),
        view_fixture(name="island", scope="island", direction="minus_X", island="raw7", hw=(3, 5)),
    )
    before = reader(features, (views,)).tokens
    after = reader(features, ((views[2], views[0], views[1]),)).tokens
    torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-5)
    wrong = (replace(views[1], latent=views[2].latent), replace(views[2], latent=views[1].latent), views[0])
    assert not torch.allclose(before, reader(features, (wrong,)).tokens)


def test_padding_is_removed_before_language_and_visual_attention_and_gradients():
    torch.manual_seed(204)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture((5, 2))
    hidden = features.hidden.clone()
    hidden[~features.valid_mask] = float("nan")
    hidden.requires_grad_(True)
    features = replace(features, hidden=hidden)
    view = view_fixture(hw=(8, 8))
    mask = view.valid_mask.clone()
    mask[3:5, 2:4] = False
    from cosmos_framework.model.generator.mot.global_spatial import patch_layout

    _, support = patch_layout(view.raster, (8, 8), rows_flipped=True)
    latent = view.latent.clone()
    latent[:, ~(support & mask)] = float("nan")
    latent.requires_grad_(True)
    view = replace(view, latent=latent, valid_mask=mask)
    output = reader(features, ((view,), (view,))).tokens
    assert torch.isfinite(output).all()
    (output * torch.randn_like(output)).mean().backward()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(latent.grad).all()
    assert (hidden.grad[~features.valid_mask] == 0).all()
    assert (latent.grad[:, ~(support & mask)] == 0).all()


def test_empty_conditions_and_native_special_tokens_follow_effective_text(monkeypatch):
    torch.manual_seed(205)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    calls = []

    def record(q, k, v):
        calls.append((q.shape[1], k.shape[1]))
        return cpu_reference(q, k, v)

    monkeypatch.setattr(global_memory, "_dispatch_attention", record)
    empty = text_fixture((0, 0))
    prepared = reader.prepare_task_queries(empty)
    assert prepared.shape == (2, 64, 512) and torch.isfinite(prepared).all()
    assert calls == []
    special = text_fixture((1,))
    special = replace(special, text=replace(special.text, dropped=(True,)))
    assert torch.isfinite(reader.prepare_task_queries(special)).all()
    assert calls == [(64, 1)]  # actual native empty special is still a valid token.


def test_external_text_mask_cannot_silently_drop_nonempty_effective_tokens():
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture()
    with pytest.raises(ValueError, match="effective tokens"):
        reader.prepare_task_queries(replace(features, valid_mask=torch.zeros_like(features.valid_mask)))


@pytest.mark.parametrize("condition", ["image", "language", "geometry", "scope", "direction", "patch_position"])
def test_each_condition_has_a_detectable_effect(condition):
    torch.manual_seed(206)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture()
    top = view_fixture()
    side = view_fixture(name="side", scope="room", direction="plus_X", hw=(3, 5))
    expected = reader(features, ((top, side),)).tokens
    if condition == "image":
        side = replace(side, latent=side.latent + torch.randn_like(side.latent) * 4)
    elif condition == "language":
        features = replace(features, hidden=features.hidden + torch.randn_like(features.hidden) * 4)
    elif condition == "geometry":
        side = replace(side, projection=replace(side.projection, origin_world=(7, -5, 3)))
    elif condition == "scope":
        side = replace(side, scope="island", island_id="raw-group")
    elif condition == "direction":
        side = replace(side, direction="minus_X")
    else:
        side = replace(side, raster=replace(side.raster, content_box=(0, 0, 24, 16)))
    actual = reader(features, ((top, side),)).tokens
    assert (actual - expected).abs().max() > 1e-5


def test_all_trainable_parameters_have_a_path_through_frozen_downstream_weights():
    torch.manual_seed(207)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    projector = GlobalMemoryProjector(80, CONFIG)
    frozen_downstream = torch.nn.Linear(80, 7, bias=False).requires_grad_(False)
    features = text_fixture((5, 3))
    views = (
        (view_fixture(), view_fixture(name="island", scope="island", direction="minus_X", island="one")),
        (view_fixture(name="second-top"),),
    )
    result = frozen_downstream(projector(reader(features, views).tokens))
    (result * torch.randn_like(result)).mean().backward()
    for owner, module in (("reader", reader), ("projector", projector)):
        for name, parameter in module.named_parameters():
            assert parameter.requires_grad and parameter.grad is not None, f"{owner}.{name}"
            assert torch.isfinite(parameter.grad).all(), f"{owner}.{name}"
            assert parameter.grad.abs().max() > 0, f"{owner}.{name}"
    assert frozen_downstream.weight.grad is None


def test_reader_rounds_use_two_distinct_cross_attentions_then_memory_self_attention(monkeypatch):
    torch.manual_seed(208)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    calls, first_outputs, second_queries = [], [], []

    def record(q, k, v):
        calls.append((q.shape[1], k.shape[1]))
        return cpu_reference(q, k, v)

    monkeypatch.setattr(global_memory, "_dispatch_attention", record)
    reader.visual[0].register_forward_hook(lambda module, args, output: first_outputs.append(output.detach().clone()))
    reader.visual[1].cross_attention.register_forward_pre_hook(
        lambda module, args: second_queries.append(args[0].detach().clone())
    )
    features = text_fixture()
    views = ((view_fixture(hw=(3, 5)),),)
    prepared = reader.prepare_task_queries(features)
    reader.read_visual(prepared, views)
    assert calls == [(64, 5), (64, 15), (64, 64), (64, 15), (64, 64)]
    torch.testing.assert_close(second_queries[0], reader.visual[1].query_norm(first_outputs[0]))
    assert not torch.allclose(second_queries[0], reader.visual[1].query_norm(prepared))
    assert not set(map(id, reader.visual[0].parameters())) & set(map(id, reader.visual[1].parameters()))


def test_task_preparation_split_repeats_without_rng_or_recurrent_state():
    torch.manual_seed(209)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    features = text_fixture()
    views = ((view_fixture(),),)
    queries = reader.prepare_task_queries(features)
    before, rng = queries.clone(), torch.get_rng_state().clone()
    first = reader.read_visual(queries, views).tokens
    second = reader.read_visual(queries, views).tokens
    full = reader(features, views).tokens
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, full, rtol=0, atol=0)
    torch.testing.assert_close(before, queries, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_state_dict_round_trip_and_reproducible_distinct_queries():
    torch.manual_seed(210)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    torch.manual_seed(210)
    restored = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    assert torch.equal(reader.queries, restored.queries)
    assert reader.queries.unique(dim=0).shape[0] == 64
    projector, other_projector = GlobalMemoryProjector(96, CONFIG), GlobalMemoryProjector(96, CONFIG)
    features, views = text_fixture(), ((view_fixture(),),)
    expected = projector(reader(features, views).tokens)
    buffer = io.BytesIO()
    torch.save({"reader": reader.state_dict(), "projector": projector.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored.load_state_dict(state["reader"], strict=True)
    other_projector.load_state_dict(state["projector"], strict=True)
    torch.testing.assert_close(other_projector(restored(features, views).tokens), expected, rtol=0, atol=0)
    assert {"queries", "spatial.geometry_mlp.0.weight", "visual.1.cross_attention.key.weight"} <= state["reader"].keys()
    assert {"projection.weight", "modality"} <= state["projector"].keys()


def test_reset_after_meta_to_empty_initializes_all_parameters_and_fixed_buffers():
    with torch.device("meta"):
        materialized = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    materialized.to_empty(device="cpu")
    torch.manual_seed(211)
    materialized.reset_parameters()
    reference = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    torch.manual_seed(211)
    reference.reset_parameters()
    for name, tensor in materialized.state_dict().items():
        assert torch.isfinite(tensor).all(), name
        torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault", ["missing_top", "wrong_variant", "both_variants", "duplicate", "world_frame", "nonfinite"]
)
def test_selected_view_errors_never_create_missing_direction_placeholders(fault):
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    top = view_fixture()
    if fault == "missing_top":
        views = (view_fixture(name="side", scope="room", direction="plus_X"),)
    elif fault == "wrong_variant":
        views = (replace(top, top_variant="without_robot"),)
    elif fault == "both_variants":
        views = (top, replace(top, identity=replace(top.identity, view_id="clean"), top_variant="without_robot"))
    elif fault == "duplicate":
        views = (top, top)
    elif fault == "world_frame":
        side = view_fixture(name="side", scope="room", direction="plus_X")
        views = (top, replace(side, projection=replace(side.projection, world_frame="other-scene")))
    else:
        views = (replace(top, latent=torch.full_like(top.latent, float("nan"))),)
    with pytest.raises(ValueError):
        reader(text_fixture(), (views,))


def test_disabled_modules_fail_before_consuming_initialization_rng():
    before = torch.get_rng_state().clone()
    for factory in (
        lambda: GlobalMemoryReader(24, GlobalMemoryConfig()),
        lambda: GlobalMemoryProjector(96, GlobalMemoryConfig()),
    ):
        with pytest.raises(ValueError, match="disabled"):
            factory()
    assert torch.equal(before, torch.get_rng_state())


def test_production_boundary_resolves_the_public_attention_dispatcher(monkeypatch):
    module = ModuleType("cosmos_framework.model.attention")
    calls = []

    def public_spy(q, k, v):
        calls.append(q.shape)
        return cpu_reference(q, k, v)

    module.attention = public_spy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(global_memory, "_dispatch_attention", PRODUCTION_DISPATCH)
    reader = GlobalMemoryReader(24, CONFIG, latent_channels=4)
    reader(text_fixture(), ((view_fixture(),),))
    assert len(calls) == 5


def test_import_and_construction_need_no_vae_policy_ros_or_mujoco():
    code = """
import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(word in fullname for word in ('mujoco', 'robosuite', 'rospy', 'wan2pt2_vae', 'omni_mot_model', 'action_policy_server')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockHeavy())
from cosmos_framework.model.generator.mot.global_memory import GlobalMemoryReader, GlobalMemoryProjector
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig
import torch
config = GlobalMemoryConfig(enabled=True)
GlobalMemoryReader(24, config, latent_channels=4)
GlobalMemoryProjector(96, config)
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""C0 ABI tests only; no renderer, weights, model, server or GPU required."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_text, synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import (
    GlobalCacheSignature,
    GlobalMemoryConfig,
    PrefixLayout,
    ReaderOutput,
    TextCacheSignature,
)


def test_disabled_config_consumes_no_rng():
    before = torch.get_rng_state().clone()
    assert not GlobalMemoryConfig().enabled
    assert torch.equal(before, torch.get_rng_state())


def test_view_full_grid_and_explicit_nonidentity_geometry():
    view = synthetic_view(hw=(3, 5))
    assert view.latent.shape == (4, 3, 5)  # deliberately not production channels/canvas
    assert view.valid_mask.all()
    assert view.projection.origin_world == (2.0, 3.0, 0.0)
    assert view.identity.geometry_version != view.identity.membership_version
    with pytest.raises(ValueError, match="source image"):
        replace(view, projection=replace(view.projection, width_px=49))


def test_missing_projection_is_not_a_default():
    with pytest.raises(TypeError):
        type(synthetic_view().projection)(projection_id="missing")


def test_mask_shape_and_type_rejected():
    view = synthetic_view()
    with pytest.raises(ValueError, match="valid_mask"):
        replace(view, valid_mask=torch.ones(2, 3))
    with pytest.raises(ValueError, match="valid_mask"):
        replace(view, valid_mask=torch.ones(3, 2, dtype=torch.bool))


def test_mapping_does_not_guess_padding_from_pixel_colour():
    view = synthetic_view()
    assert view.raster.content_box == (4, 8, 28, 24)
    with pytest.raises(ValueError, match="resized"):
        replace(view.raster, content_box=(0, 0, 32, 32))


def test_effective_text_mixed_lengths_and_empty():
    features = synthetic_text()
    assert features.text.lengths == (3, 0, 1)
    assert features.valid_mask.sum(1).tolist() == [3, 0, 1]
    assert features.layers[0].key.shape == (4, 2, 4)
    empty = synthetic_text(lengths=(0, 0))
    assert empty.hidden.shape == (2, 0, 8)
    assert empty.valid_mask.shape == (2, 0)
    assert empty.layers[0].key.shape[0] == 0


def test_effective_tokens_positions_dropout_are_one_identity():
    text = synthetic_text().text
    assert hash(text) != hash(replace(text, dropout_id="another-consumer"))
    assert text != replace(text, dropped=(False, False, False))
    with pytest.raises(ValueError, match="mRoPE"):
        replace(text, position_ids=(((), (), ()),) * 3)


def test_prefix_mixed_none_local_global_both_offsets():
    layout = PrefixLayout(global_lengths=(0, 0, 64, 64), local_lengths=(0, 4, 0, 4))
    assert layout.sample_offsets == (0, 0, 4, 68, 136)
    assert layout.global_present == (False, False, True, True)
    assert layout.local_present == (False, True, False, True)
    assert PrefixLayout(global_lengths=(0,), local_lengths=(0,)).sample_offsets == (0, 0)
    with pytest.raises(ValueError):
        PrefixLayout(global_lengths=(64,), local_lengths=(0, 4))


def test_reader_output_keeps_graph():
    x = torch.ones(1, 64, 512, requires_grad=True)
    ReaderOutput(tokens=x).tokens.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_cache_signature_selected_metadata_and_session_binding():
    view = synthetic_view()
    text = TextCacheSignature(
        session_id="session-A",
        checkpoint_id="weights-1",
        und_version="und-1",
        gen_version="gen-1",
        position_rule="native-mrope-v1",
        dtype="float32",
        text=synthetic_text().text,
    )
    signature = GlobalCacheSignature(
        text=text,
        reader_version="reader-1",
        projector_version="proj-1",
        geometry_encoding_version="geom-encoding-1",
        top_variant="with_robot",
        views=(view.identity,),
        projections=(view.projection,),
        rasters=(view.raster,),
        view_types=((view.scope, view.direction, view.room_id, view.island_id, view.top_variant),),
        mask_version="positive-overlap-v1",
    )
    assert signature != replace(signature, text=replace(text, session_id="session-B"))
    assert signature != replace(signature, views=(replace(view.identity, membership_version="members-v3"),))
    assert signature != replace(signature, rasters=(replace(view.raster, content_box=(0, 8, 24, 24)),))
    assert signature != replace(signature, top_variant="without_robot")
    assert hash(signature) == hash(replace(signature))
    with pytest.raises(ValueError, match="sorted"):
        replace(
            signature,
            views=(view.identity, view.identity),
            projections=(view.projection,) * 2,
            rasters=(view.raster,) * 2,
            view_types=signature.view_types * 2,
        )


def test_invalid_config():
    with pytest.raises(ValueError, match="exactly one"):
        GlobalMemoryConfig(top_view_variant="both")
    with pytest.raises(ValueError, match="Reader"):
        GlobalMemoryConfig(k_global=128)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU spatial oracles use shared C0 synthetic evidence, never real assets."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import RasterMapping
from cosmos_framework.model.generator.mot.global_spatial import SpatialFeatureEncoder, patch_layout, projection_vector


def test_native_nonidentity_frame_and_uv_make_the_plane_anchor():
    projection = synthetic_view().projection
    # origin(2,3,0) + (-1)*right(0,1,0) + (-2)*up(-1,0,0).
    assert projection_vector(projection) == (4, 2, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 4.8, 3.2)
    shifted = replace(projection, origin_world=(7, -4, 2))
    assert projection_vector(shifted)[:3] == (9, -5, 2)


def test_partial_cells_have_hand_computed_overlap_centres():
    raster = RasterMapping(original_hw=(32, 64), resized_hw=(32, 64), canvas_hw=(48, 64), content_box=(0, 8, 64, 40))
    uv, valid = patch_layout(raster, (3, 4), rows_flipped=True)
    assert valid.all()
    torch.testing.assert_close(uv[:, 0, 1], torch.tensor([0.875, 0.5, 0.125]))
    torch.testing.assert_close(uv[0, :, 0], torch.tensor([0.125, 0.375, 0.625, 0.875]))
    opposite, _ = patch_layout(raster, (3, 4), rows_flipped=False)
    torch.testing.assert_close(opposite[..., 0], uv[..., 0])
    torch.testing.assert_close(opposite[..., 1], 1 - uv[..., 1])


def test_half_open_boundaries_exclude_full_padding_rows():
    raster = RasterMapping(original_hw=(32, 64), resized_hw=(32, 64), canvas_hw=(64, 64), content_box=(0, 16, 64, 48))
    uv, valid = patch_layout(raster, (4, 4), rows_flipped=True)
    assert valid.tolist() == [[False] * 4, [True] * 4, [True] * 4, [False] * 4]
    assert torch.equal(uv[~valid], torch.zeros(8, 2))


def test_thin_and_odd_padded_content_is_not_lost():
    thin = RasterMapping(original_hw=(1, 1), resized_hw=(1, 1), canvas_hw=(64, 64), content_box=(31, 31, 32, 32))
    uv, valid = patch_layout(thin, (4, 4), rows_flipped=True)
    assert valid.sum() == 1
    torch.testing.assert_close(uv[valid], torch.tensor([[0.5, 0.5]]))
    odd = RasterMapping(original_hw=(1, 3), resized_hw=(1, 3), canvas_hw=(4, 4), content_box=(0, 1, 3, 2))
    uv, valid = patch_layout(odd, (2, 2), rows_flipped=True)
    assert valid.tolist() == [[True, True], [False, False]]
    torch.testing.assert_close(uv[valid], torch.tensor([[1 / 3, 0.5], [5 / 6, 0.5]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_before_projection_ignores_nan_padding_and_keeps_dtype(dtype):
    torch.manual_seed(101)
    encoder = SpatialFeatureEncoder(4).to(dtype=dtype)
    view = synthetic_view(hw=(8, 8))
    _, support = patch_layout(view.raster, (8, 8), rows_flipped=True)
    mask = view.valid_mask.clone()
    mask[3, 2] = False  # A preprocessor may further restrict geometric support.
    clean = replace(view, latent=torch.zeros_like(view.latent), valid_mask=mask)
    dirty = clean.latent.clone()
    dirty[:, ~(support & mask)] = float("nan")
    dirty.requires_grad_(True)
    contaminated = replace(clean, latent=dirty)
    expected, actual = encoder(clean), encoder(contaminated)
    assert actual.shape == (int((support & mask).sum()), 512)
    assert actual.dtype == dtype and actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual.float() * torch.randn_like(actual.float())).mean().backward()
    assert torch.isfinite(dirty.grad).all()
    assert (dirty.grad[:, ~(support & mask)] == 0).all()


def test_projection_is_per_patch_without_whole_image_pooling():
    torch.manual_seed(102)
    encoder = SpatialFeatureEncoder(4)
    view = synthetic_view()
    first = encoder(view)
    changed = view.latent.clone()
    changed[:, 0, 1] += torch.tensor([2, -3, 5, 7])
    second = encoder(replace(view, latent=changed))
    assert first.shape == (6, 512)  # all six full-grid cells intersect content.
    assert not torch.allclose(first[1], second[1])
    torch.testing.assert_close(first[[0, 2, 3, 4, 5]], second[[0, 2, 3, 4, 5]], rtol=0, atol=0)


def test_raw_room_island_and_version_labels_are_not_learned_semantic_ids():
    encoder = SpatialFeatureEncoder(4)
    view = synthetic_view()
    renamed = replace(
        view,
        room_id="unrelated-scene-room-999",
        island_id="raw-id-7",
        identity=replace(
            view.identity, content_version="same-pixels-new-label", membership_version="same-group-new-label"
        ),
    )
    torch.testing.assert_close(encoder(view), encoder(renamed), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["basis", "extent", "offset", "channels", "direction", "no_patches"])
def test_incomplete_or_inconsistent_geometry_and_active_evidence_fail(fault):
    view = synthetic_view()
    if fault == "basis":
        view = replace(view, projection=replace(view.projection, right_world=(0, 2, 0)))
    elif fault == "extent":
        view = replace(view, projection=replace(view.projection, physical_width_m=4.9))
    elif fault == "offset":
        view = replace(view, projection=replace(view.projection, pixel_center_offset=0))
    elif fault == "channels":
        view = replace(view, latent=view.latent[:3])
    elif fault == "direction":
        view = replace(view, direction="missing_unknown_direction")
    else:
        view = replace(view, valid_mask=torch.zeros_like(view.valid_mask))
    with pytest.raises(ValueError):
        SpatialFeatureEncoder(4)(view)


def test_a_nonfinite_valid_patch_fails_instead_of_disappearing():
    view = synthetic_view()
    latent = view.latent.clone()
    latent[:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        SpatialFeatureEncoder(4)(replace(view, latent=latent))

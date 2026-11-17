# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""G2 spatial patch encoding, consuming only the shared C0 native evidence.

The anchor is on an orthographic projection plane, not an object's XYZ. Patch
support is positive-area intersection with the resize/pad content rectangle; it
does not claim the Wan receptive field is an independent rectangular patch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cosmos_framework.model.generator.mot.g2_contracts import RasterMapping, SpatialProjection, SpatialView

SPATIAL_ENCODING_VERSION = "g2_spatial_plane14_fourier8_overlap_centroid_v1"
PATCH_MASK_VERSION = "positive_area_content_intersection_v1"
SCOPES = ("floor", "room", "island")
DIRECTIONS = ("floor_top", "upper_up", "plus_X", "minus_X", "plus_Y", "minus_Y")


def projection_vector(projection: SpatialProjection) -> tuple[float, ...]:
    """[plane origin, right, up, forward, width, height], with lengths in metres."""
    for basis in (
        projection.axes_world,
        (projection.right_world, projection.up_world, projection.forward_world),
    ):
        for i in range(3):
            for j in range(3):
                dot = sum(a * b for a, b in zip(basis[i], basis[j], strict=True))
                if not math.isclose(dot, float(i == j), abs_tol=1e-5):
                    raise ValueError("projection basis must be orthonormal")
    if any(type(n) is not int for n in (projection.width_px, projection.height_px)):
        raise ValueError("projection pixel dimensions must be integers")
    for pixels, extent in (
        (projection.width_px, projection.physical_width_m),
        (projection.height_px, projection.physical_height_m),
    ):
        if not math.isclose(pixels * projection.meters_per_pixel, extent, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("projection metric extent disagrees with pixels/metres-per-pixel")
    if projection.pixel_center_offset != 0.5:
        raise ValueError("spatial encoding v1 requires producer's pixel-centre offset 0.5")
    u, v = projection.uv_min_room
    origin = tuple(
        o + u * right + v * up
        for o, right, up in zip(projection.origin_world, projection.right_world, projection.up_world, strict=True)
    )
    return (
        *origin,
        *projection.right_world,
        *projection.up_world,
        *projection.forward_world,
        projection.physical_width_m,
        projection.physical_height_m,
    )


def patch_layout(
    raster: RasterMapping,
    grid_hw: tuple[int, int],
    *,
    rows_flipped: bool,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 [H_z,W_z,2] UV and bool support, independent of RGB values.

    UV refers to edges of the original effective image, in [0,1]; v follows the
    producer's up axis. Dividing the inverse-resized source anchor by original
    width/height cancels that factor, yielding the expressions below. Inactive
    positions are zeroed only for a finite diagnostic output, never made tokens.
    """
    if len(grid_hw) != 2 or any(type(n) is not int or n <= 0 for n in grid_hw):
        raise ValueError("latent grid must have positive integer height/width")
    height, width = grid_hw
    canvas_h, canvas_w = raster.canvas_hw
    left, top, right, bottom = raster.content_box
    x = torch.arange(width + 1, device=device, dtype=torch.float32) * (canvas_w / width)
    y = torch.arange(height + 1, device=device, dtype=torch.float32) * (canvas_h / height)
    x_lo, x_hi = x[:-1].clamp_min(left), x[1:].clamp_max(right)
    y_lo, y_hi = y[:-1].clamp_min(top), y[1:].clamp_max(bottom)
    valid = (y_hi > y_lo)[:, None] & (x_hi > x_lo)[None, :]
    u = ((x_lo + x_hi) * 0.5 - left) / (right - left)
    v = ((y_lo + y_hi) * 0.5 - top) / (bottom - top)
    if rows_flipped:
        v = 1 - v
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    positions = torch.stack((uu, vv), dim=-1)
    return positions.masked_fill(~valid[..., None], 0), valid


class SpatialFeatureEncoder(nn.Module):
    """Shared per-patch latent projection + scope/direction + geometry + PE.

    Raw Room/Island IDs and source/version strings never index a learned table.
    C0's mask can further restrict support; it can never enable padding cells.
    """

    encoding_version = SPATIAL_ENCODING_VERSION
    mask_version = PATCH_MASK_VERSION

    def __init__(self, latent_channels: int = 48) -> None:
        super().__init__()
        if type(latent_channels) is not int or latent_channels <= 0:
            raise ValueError("latent_channels must match a positive recipe channel count")
        self.image_projection = nn.Linear(latent_channels, 512)
        self.scope_embedding = nn.Embedding(len(SCOPES), 512)
        self.direction_embedding = nn.Embedding(len(DIRECTIONS), 512)
        self.geometry_mlp = nn.Sequential(nn.Linear(14, 512), nn.SiLU(), nn.Linear(512, 512))
        self.position_projection = nn.Linear(34, 512, bias=False)
        # Integer buffers survive dtype conversion, and are reset after to_empty.
        self.register_buffer("position_frequencies", torch.empty(8, dtype=torch.int32))
        self.register_buffer("geometry_divisors", torch.empty(14, dtype=torch.int32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        self.position_frequencies.copy_(
            2 ** torch.arange(8, device=self.position_frequencies.device, dtype=torch.int32)
        )
        self.geometry_divisors.copy_(
            torch.tensor([10] * 3 + [1] * 9 + [10] * 2, device=self.geometry_divisors.device, dtype=torch.int32)
        )

    def forward(self, view: SpatialView) -> torch.Tensor:
        """Return only active [N_patch,512] tokens, without pre-pooling the grid."""
        weight = self.image_projection.weight
        if view.latent.shape[0] != self.image_projection.in_features:
            raise ValueError("view latent channels disagree with configured recipe")
        if view.latent.device != weight.device:
            raise ValueError("view and spatial encoder must share device")
        if view.direction not in DIRECTIONS:
            raise ValueError("unknown native spatial direction")
        if view.scope == "island" and view.direction not in DIRECTIONS[2:]:
            raise ValueError("producer Island views have horizontal directions only")
        geometry = projection_vector(view.projection)
        positions, support = patch_layout(
            view.raster, tuple(view.latent.shape[1:]), rows_flipped=view.projection.rows_flipped, device=weight.device
        )
        active = support & view.valid_mask
        if not bool(active.any()):
            raise ValueError("selected view has no valid patches")
        # Compact BEFORE affine/norm: even NaN padding cannot contaminate forward
        # values or parameter gradients. No missing-direction tokens are added.
        patches = view.latent.permute(1, 2, 0)[active]
        if not bool(torch.isfinite(patches).all()):
            raise ValueError("valid view patches must be finite")
        uv = positions[active]
        phase = (2 * uv - 1)[..., None] * self.position_frequencies.float() * math.pi
        fourier = torch.cat((uv, phase.sin().flatten(1), phase.cos().flatten(1)), dim=-1)
        metric = torch.tensor(geometry, device=weight.device, dtype=torch.float32) / self.geometry_divisors
        kind = self.scope_embedding.weight[SCOPES.index(view.scope)]
        kind = kind + self.direction_embedding.weight[DIRECTIONS.index(view.direction)]
        return (
            self.image_projection(patches.to(weight.dtype))
            + kind
            + self.geometry_mlp(metric.to(weight.dtype))
            + self.position_projection(fourier.to(weight.dtype))
        )

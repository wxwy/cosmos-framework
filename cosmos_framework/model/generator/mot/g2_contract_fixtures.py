# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small synthetic CPU fixtures. Explicit geometry is invented TEST data only."""

from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.g2_contracts import (
    EffectiveTextBatch,
    RasterMapping,
    SpatialProjection,
    SpatialView,
    UNDLayerKV,
    UNDTextFeatures,
    ViewIdentity,
)


def synthetic_view(*, view_id="sixview/floor_top_w_rbs", hw=(2, 3), top_variant="with_robot"):
    projection = SpatialProjection(
        projection_id="synthetic-projection-v1",
        world_frame="synthetic-world",
        origin_world=(2.0, 3.0, 0.0),
        axes_world=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        right_world=(0.0, 1.0, 0.0),
        up_world=(-1.0, 0.0, 0.0),
        forward_world=(0.0, 0.0, 1.0),
        uv_min_room=(-1.0, -2.0),
        width_px=48,
        height_px=32,
        meters_per_pixel=0.1,
        physical_width_m=4.8,
        physical_height_m=3.2,
        rows_flipped=True,
        pixel_center_offset=0.5,
    )
    return SpatialView(
        identity=ViewIdentity(
            view_id=view_id,
            content_version="pixels-v1",
            image_owner="state0",
            image_owner_frame=0,
            geometry_version="geom-v1",
            membership_version="members-v2",
            recipe_id="synthetic-C4-not-real-Wan",
        ),
        latent=torch.arange(4 * hw[0] * hw[1], dtype=torch.float32).reshape(4, *hw),
        valid_mask=torch.ones(hw, dtype=torch.bool),
        projection=projection,
        raster=RasterMapping(original_hw=(32, 48), resized_hw=(16, 24), canvas_hw=(32, 32), content_box=(4, 8, 28, 24)),
        scope="floor",
        direction="floor_top",
        room_id="room-fixture",
        island_id=None,
        top_variant=top_variant,
    )


def synthetic_text(*, lengths=(3, 0, 1), width=8, layers=2):
    tokens = tuple(tuple(range(1, n + 1)) for n in lengths)
    text = EffectiveTextBatch(
        token_ids=tokens,
        position_ids=tuple((tuple(range(n)),) * 3 for n in lengths),
        dropped=tuple(n == 0 for n in lengths),
        formatter_id="synthetic-static-no-idle-v1",
        tokenizer_id="synthetic-not-a-production-tokenizer",
        dropout_id="decision-fixture-v1",
    )
    hidden = torch.arange(len(lengths) * max(lengths) * width, dtype=torch.float32).reshape(
        len(lengths), max(lengths), width
    )
    valid = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    kv = tuple(
        UNDLayerKV(
            key=torch.zeros(sum(lengths), 2, 4),
            value=torch.ones(sum(lengths), 2, 4),
            cos=torch.ones(sum(lengths), 4),
            sin=torch.zeros(sum(lengths), 4),
        )
        for _ in range(layers)
    )
    return UNDTextFeatures(text=text, hidden=hidden, valid_mask=valid, layers=kv)

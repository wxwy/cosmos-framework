# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only checks for the optional packed cached-vision preparation seam."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def _model(*, training: bool = True) -> SimpleNamespace:
    tokenizer = SimpleNamespace(
        spatial_compression_factor=16,
        get_latent_temporal_positions=Mock(side_effect=lambda **kwargs: torch.arange(kwargs["num_latent_frames"])),
        encode=Mock(side_effect=AssertionError("VAE encode")),
    )
    model = SimpleNamespace(
        training=training,
        input_video_key="video",
        input_image_key="images",
        tokenizer_vision_gen=tokenizer,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        config=SimpleNamespace(
            sr_latent_condition_noise=None,
            diffusion_expert_config=SimpleNamespace(
                vision_temporal_position_mode="uniae_source_right_edge", num_view_embeddings=0
            ),
        ),
        _normalize_video_databatch_inplace=Mock(
            side_effect=lambda batch: batch.update(video=[item[0].unsqueeze(0) for item in batch["video"]])
        ),
        _augment_image_dim_inplace=Mock(),
        _encode_vision_x0_tokens=Mock(return_value=[torch.ones((1, 48, 5, 12, 20))]),
        _encode_lidar_stream=Mock(return_value=(None, None, None)),
        _encode_radar_stream=Mock(return_value=(None, None, None)),
        _normalize_action_databatch=Mock(return_value=(torch.zeros((1, 17, 64)), torch.tensor([30]), "action")),
        _normalize_sound_databatch_inplace=Mock(return_value=None),
        tokenizer_sound_gen=None,
    )
    for name in (
        "_has_vision_stream",
        "is_image_batch",
        "_remove_padding_from_latent",
        "_get_temporal_positions_vision",
    ):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))
    model._validate_and_get_num_views = OmniMoTModel._validate_and_get_num_views
    model._unwrap_vision_item = OmniMoTModel._unwrap_vision_item
    return model


def _batch() -> dict:
    latent = torch.arange(5 * 48 * 12 * 20, dtype=torch.float32).reshape(1, 5, 48, 12, 20)
    return {
        "video": [[torch.zeros((3, 17, 192, 320), dtype=torch.uint8)]],
        "video_latent": [latent],
        "cached_latent_required": [torch.tensor([True])],
        "image_size": [torch.tensor([192, 320, 160, 320])],
        "action": [[torch.zeros((17, 64))]],
    }


def _run(model: SimpleNamespace, batch: dict, **kwargs):
    return OmniMoTModel.get_data_and_condition(model, batch, **kwargs)


@pytest.mark.level(0)
def test_valid_cache_hit_preserves_native_crop_temporal_and_action_alignment() -> None:
    model = _model()
    batch = _batch()
    expected = batch["video_latent"][0].squeeze(0).permute(1, 0, 2, 3).unsqueeze(0)[:, :, :, :10, :20]
    result = _run(model, batch, balance_vae_encode=True)
    assert result.batch_size == 1 and not result.is_image_batch
    assert len(result.x0_tokens_vision) == len(result.raw_state_vision) == 1
    assert result.raw_state_vision[0].shape == (1, 3, 17, 192, 320)
    assert result.raw_state_vision[0].dtype == torch.uint8
    assert result.x0_tokens_vision[0].shape == (1, 48, 5, 10, 20)
    torch.testing.assert_close(result.x0_tokens_vision[0], expected, rtol=0, atol=0)
    assert result.temporal_positions_vision[0].shape == (5,)
    assert model.tokenizer_vision_gen.get_latent_temporal_positions.call_args.kwargs["num_pixel_frames"] == 17
    assert result.x0_tokens_action.shape == (1, 17, 64)
    assert result.action_domain_id.item() == 30
    model._encode_vision_x0_tokens.assert_not_called()
    model._normalize_video_databatch_inplace.assert_not_called()
    model.tokenizer_vision_gen.encode.assert_not_called()


@pytest.mark.level(0)
def test_two_cached_samples_keep_order_and_single_vision_item() -> None:
    model = _model()
    batch = _batch()
    batch["video"].append([torch.ones_like(batch["video"][0][0])])
    batch["video_latent"].append(torch.full_like(batch["video_latent"][0], 7.0))
    batch["cached_latent_required"].append(torch.tensor([True]))
    batch["image_size"].append(batch["image_size"][0].clone())
    result = _run(model, batch)
    assert result.batch_size == 2
    assert result.num_vision_items_per_sample is None
    assert result.num_views_per_vision_item is None
    assert len(result.x0_tokens_vision) == len(result.raw_state_vision) == 2
    assert result.raw_state_vision[0].sum() == 0
    assert result.raw_state_vision[1].sum() > 0
    assert result.x0_tokens_vision[0][0, 0, 0, 0, 0] == 0
    assert result.x0_tokens_vision[1][0, 0, 0, 0, 0] == 7
    model._encode_vision_x0_tokens.assert_not_called()


@pytest.mark.level(0)
def test_no_cache_native_off_mode_unchanged_even_in_eval() -> None:
    model = _model(training=False)
    batch = _batch()
    batch.pop("video_latent")
    batch.pop("cached_latent_required")
    result = _run(model, batch)
    assert result.batch_size == 1
    model._encode_vision_x0_tokens.assert_called_once()
    model._normalize_video_databatch_inplace.assert_called_once()


@pytest.mark.level(0)
@pytest.mark.parametrize(
    ("change", "pattern"),
    [
        (lambda batch: batch.pop("video_latent"), "missing"),
        (lambda batch: batch.pop("cached_latent_required"), "marker"),
        (lambda batch: batch.update(cached_latent_required=[torch.tensor([False])]), "marker"),
        (lambda batch: batch.update(cached_latent_required=[None]), "marker"),
        (lambda batch: batch.update(video_latent=[]), "one tensor"),
        (lambda batch: batch.update(video_latent=[torch.zeros((5, 48, 12, 20))]), "finite fp32"),
        (lambda batch: batch.update(video_latent=[torch.zeros((1, 4, 48, 12, 20))]), "finite fp32"),
        (lambda batch: batch.update(video_latent=[torch.zeros((1, 5, 47, 12, 20))]), "finite fp32"),
        (
            lambda batch: batch.update(video_latent=[torch.zeros((1, 5, 48, 12, 20), dtype=torch.float16)]),
            "finite fp32",
        ),
        (lambda batch: batch["video_latent"][0].fill_(float("nan")), "finite fp32"),
        (lambda batch: batch.pop("image_size"), "image_size"),
        (lambda batch: batch.update(images=batch.pop("video")), "marker"),
        (lambda batch: batch.update(num_vision_items_per_sample=[2]), "multi-vision"),
        (lambda batch: batch.update(enable_per_camera_vae_encoding=True), "single-view"),
        (lambda batch: batch["image_size"][0].__setitem__(0, 208), "canvas"),
        (lambda batch: batch.update(video_latent=[torch.zeros((1, 5, 48, 11, 20))]), "canvas"),
    ],
)
def test_bad_cache_never_falls_back(change, pattern: str) -> None:
    model = _model()
    batch = deepcopy(_batch())
    change(batch)
    with pytest.raises((ValueError, TypeError), match=pattern):
        _run(model, batch, balance_vae_encode=True)
    model._encode_vision_x0_tokens.assert_not_called()
    model._normalize_video_databatch_inplace.assert_not_called()
    model.tokenizer_vision_gen.encode.assert_not_called()


@pytest.mark.level(0)
def test_eval_cache_hit_and_inference_indexes_fail_before_encode() -> None:
    eval_model = _model(training=False)
    with pytest.raises(ValueError, match="only supported during training"):
        _run(eval_model, _batch())
    eval_model._encode_vision_x0_tokens.assert_not_called()

    train_model = _model()
    with pytest.raises(ValueError, match="single-view"):
        _run(train_model, _batch(), vision_condition_indexes=[[0]])
    train_model._encode_vision_x0_tokens.assert_not_called()

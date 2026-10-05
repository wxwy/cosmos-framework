from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from cosmos_framework.data.generator.action.utils.transforms import find_closest_target_size, reflection_pad_to_target
from cosmos_framework.inference import robocasa_composite_visual as visual
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import robocasa_current_latent_to_visual96
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


class _Tokenizer:
    use_streaming_encode = False
    _keep_encoder_cache = False

    def __init__(self) -> None:
        self.model = SimpleNamespace(model=nn.Linear(1, 1))
        self.calls: list[torch.Tensor] = []

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.clone())
        return torch.ones((1, 48, 1, 12, 20), dtype=torch.bfloat16)


def _prepared() -> visual.PreparedCompositeFrame:
    frame = torch.arange(3 * 256 * 512, dtype=torch.int64).reshape(3, 256, 512).to(torch.uint8)
    return visual.prepare_robocasa_composite_frame(frame, 256)


@pytest.mark.parametrize("image_size", [128, 192, 256, 320])
def test_shared_spatial_repeat_before_after_pad_identical(image_size: int) -> None:
    composite = torch.arange(3 * 256 * 512, dtype=torch.int64).reshape(3, 256, 512).to(torch.uint8)
    prepared = visual.prepare_robocasa_composite_frame(composite, image_size)
    source = composite
    if image_size != 256:
        resized = Image.fromarray(source.permute(1, 2, 0).numpy()).resize(
            (int(round(512 * image_size / 256)), image_size), resample=Image.Resampling.BILINEAR
        )
        source = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()
    _, height, width = source.shape
    resolution = get_vision_data_resolution((height, width))
    target_w, target_h = find_closest_target_size(height, width, resolution)
    old = {"video": source.unsqueeze(1).repeat(1, 17, 1, 1)}
    reflection_pad_to_target(old, ["video"], True, target_w, target_h)
    assert torch.equal(source, prepared.source_uint8)
    assert torch.equal(old["video"], prepared.padded_single_frame.repeat(1, 17, 1, 1))
    assert torch.equal(old["image_size"], prepared.padded_image_size)
    assert torch.equal(old["video"][:, :1], prepared.padded_single_frame)


def test_encode1_uses_shared_normalizer_and_phase4_visual96(monkeypatch) -> None:
    prepared = _prepared()
    tokenizer = _Tokenizer()
    calls = []
    original = visual.normalize_uint8_item

    def checked(value, kwargs):
        calls.append((tuple(value.shape), kwargs))
        return original(value, kwargs)

    monkeypatch.setattr(visual, "normalize_uint8_item", checked)
    z0, summary = visual.encode_current_composite_visual96(tokenizer, prepared)
    assert calls[0][0][2] == 1
    assert calls[0][1]["device"] == next(tokenizer.model.model.parameters()).device
    assert tokenizer.calls[0].shape == (1, 3, 1, *prepared.padded_single_frame.shape[-2:])
    assert z0.dtype == summary.dtype == torch.float32
    assert torch.isfinite(z0).all() and torch.isfinite(summary).all()
    torch.testing.assert_close(summary, robocasa_current_latent_to_visual96(z0))


def test_encode1_rejects_streaming_and_cache() -> None:
    import pytest

    prepared = _prepared()
    tokenizer = _Tokenizer()
    tokenizer.use_streaming_encode = True
    with pytest.raises(RuntimeError, match="streaming"):
        visual.encode_current_composite_visual96(tokenizer, prepared)
    tokenizer.use_streaming_encode = False
    tokenizer._keep_encoder_cache = True
    with pytest.raises(RuntimeError, match="cache"):
        visual.encode_current_composite_visual96(tokenizer, prepared)
    assert tokenizer.calls == []


def test_encode1_resolves_wan_interface_device_without_parameters() -> None:
    tokenizer = _Tokenizer()
    tokenizer.model = SimpleNamespace(model=None, device=torch.device("cpu"))
    z0, _ = visual.encode_current_composite_visual96(tokenizer, _prepared())
    assert z0.device.type == "cpu"

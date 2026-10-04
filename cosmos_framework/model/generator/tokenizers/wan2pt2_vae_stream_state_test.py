from __future__ import annotations

import torch
from torch import nn

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanEncoderStreamState, WanVAE_


def _bare_vae() -> WanVAE_:
    vae = WanVAE_.__new__(WanVAE_)
    nn.Module.__init__(vae)
    vae._enc_conv_num = 2
    vae._enc_cache = [None, None]
    vae._enc_stream_shape = None
    return vae


def test_encoder_stream_state_snapshot_restore_is_transactional() -> None:
    vae = _bare_vae()
    fresh = vae.new_encoder_stream_state()
    assert fresh == WanEncoderStreamState((None, None), None)

    first = torch.ones(1, 2, 2, 3, 3)
    vae._enc_cache = [first, None]
    vae._enc_stream_shape = (1, 256, 256, torch.device("cpu"), torch.float32)
    committed = vae.snapshot_encoder_stream_state()

    candidate = torch.full_like(first, 2)
    vae._enc_cache = [candidate, None]
    vae.restore_encoder_stream_state(committed)

    assert vae._enc_cache[0] is first
    assert vae._enc_stream_shape == committed.stream_shape
    assert torch.equal(vae._enc_cache[0], torch.ones_like(first))


def test_encoder_stream_state_rejects_inconsistent_fresh_cache() -> None:
    vae = _bare_vae()
    invalid = WanEncoderStreamState((torch.ones(1), None), None)
    try:
        vae.restore_encoder_stream_state(invalid)
    except ValueError as error:
        assert "fresh" in str(error)
    else:
        raise AssertionError("fresh stream state with non-empty cache must fail closed")

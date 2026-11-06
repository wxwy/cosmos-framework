# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Issue22 G0: observe actual Policy/Encode1 paths with a CPU Wan2.2 VAE.

This is a prerequisite proof, not a pipeline implementation or checkpoint/GPU
acceptance. Only VAE channel widths and residual depth are reduced. The real
48-channel latent ABI, spatial/temporal downsampling, encoder, normalization,
prime/cache lifecycle, bf16 cast and loaded-tokenizer entry points are retained.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.inference.robocasa_composite_visual import (
    encode_current_composite_visual96,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import robocasa_current_latent_to_visual96
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def _sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert bool(torch.isfinite(actual).all())


def _composite(variant: int) -> torch.Tensor:
    """Distinct left/wrist RGB, high frequencies and edges survive every resize."""
    y = torch.arange(256, dtype=torch.int64).view(256, 1)
    x = torch.arange(512, dtype=torch.int64).view(1, 512)
    return torch.stack(
        (
            (3 * x + 7 * y + 37 * variant) % 256,
            (11 * x + y + 91 * (x >= 256) + 73 * variant) % 256,
            (x + 13 * y + 127 * (y >= 128) + 19 * variant) % 256,
        )
    ).to(torch.uint8)


def _png(frame: torch.Tensor) -> str:
    output = io.BytesIO()
    Image.fromarray(frame.permute(1, 2, 0).numpy()).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.fixture
def server():
    # Avoid script-wide logging/RNG/grad initialization when importing the actual
    # service. The PNG decoder and _prep_policy_item implementation stay intact.
    with patch("cosmos_framework.inference.common.init._init_script", lambda **_kwargs: None):
        return importlib.import_module("cosmos_framework.scripts.action_policy_server_robocasa")


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def tokenizer(monkeypatch, cpu_threads):
    del cpu_threads
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(220719)
        tiny = wan.WanVAE_(
            dim=8,
            dec_dim=8,
            z_dim=48,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            temporal_window=4,
            dropout=0.0,
        )
    # Only the constructor's weight source is substituted. WanVAE.__init__ still
    # creates the official 48-channel mean/std, freezes/evals the actual module,
    # casts all weights to bf16 and restores its production memory layout.
    with monkeypatch.context() as setup:
        setup.setattr(wan, "_video_vae", lambda **_kwargs: tiny)
        loaded = wan.WanVAE(dtype=torch.bfloat16, device="cpu", vae_pth="")
    result = wan.Wan2pt2VAEInterface.__new__(wan.Wan2pt2VAEInterface)
    result.model = loaded
    result._spatial_compression_factor = 16
    result._temporal_compression_factor = 4
    result._causal = True
    result._keep_encoder_cache = False
    result._keep_decoder_cache = False
    result.use_streaming_encode = False
    result._decoder_override = None
    result.chunk_duration = 17
    assert tiny.z_dim == 48 and tiny.temperal_downsample == [False, True, True]
    assert all(parameter.dtype == torch.bfloat16 for parameter in tiny.parameters())
    assert not tiny.training and not any(parameter.requires_grad for parameter in tiny.parameters())
    return result


def _service(server, tokenizer):
    service = server.ActionModelService.__new__(server.ActionModelService)
    service.cfg = SimpleNamespace(action_chunk_size=16, max_action_dim=15, fps=10)
    service.requires_state = False
    service.raw_action_dim = 15
    service._prompt_json_formatter = None
    service.append_duration_fps = False
    service.append_resolution_info = False
    model = SimpleNamespace(
        training=False,
        input_video_key="video",
        input_image_key="images",
        tokenizer_vision_gen=tokenizer,
        tokenizer_sound_gen=None,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        config=SimpleNamespace(
            sr_latent_condition_noise=None,
            diffusion_expert_config=SimpleNamespace(
                vision_temporal_position_mode="latent_index", num_view_embeddings=0
            ),
        ),
    )
    # Bind production methods, rather than fake any vision, action or inactive
    # modality computation. Avoid constructing the unrelated language model.
    for name in (
        "get_data_and_condition",
        "_has_vision_stream",
        "is_image_batch",
        "_normalize_video_databatch_inplace",
        "_augment_image_dim_inplace",
        "_normalize_uint8_vision_item",
        "_encode_vision_item",
        "_encode_vision_x0_tokens",
        "_encode_vision_x0_tokens_prefix",
        "_remove_padding_from_latent",
        "_get_temporal_positions_vision",
        "_encode_lidar_stream",
        "_encode_radar_stream",
        "_normalize_action_databatch",
        "_normalize_sound_databatch_inplace",
        "encode",
    ):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))
    model._validate_and_get_num_views = OmniMoTModel._validate_and_get_num_views
    model._unwrap_vision_item = OmniMoTModel._unwrap_vision_item
    service.model = model
    return service


def _request(frame: torch.Tensor, image_size: int) -> dict:
    return {
        "image": _png(frame),
        "prompt": "Put the left object in the right container.",
        "domain_name": "robocasa",
        "image_size": image_size,
    }


def _policy_batch(service, preps: list[dict]) -> dict:
    # Same assembly as RoboCasa _transaction_batch. The sole device difference
    # is image_size remaining on CPU; no Tensor.to or encoding is monkeypatched.
    return {
        service._input_video_key(): [[prep["video_padded"]] for prep in preps],
        **make_batched_action_processing_fields(
            ActionProcessingRecord(raw_action_dim=15, action_normalizer=None), batch_size=len(preps)
        ),
        "action": [[service._build_action_input(prep["state_token"])] for prep in preps],
        "mode": ["wam"] * len(preps),
        "ai_caption": [prep["augmented_prompt"] for prep in preps],
        "prompt": [prep["augmented_prompt"] for prep in preps],
        "conditioning_fps": [torch.tensor(service.cfg.fps) for _ in preps],
        "image_size": torch.stack([prep["padded_image_size"] for prep in preps]),
        "domain_id": [torch.tensor(30) for _ in preps],
        "sequence_plan": [prep["sequence_plan"] for prep in preps],
    }


def _observe(monkeypatch, tokenizer):
    records, primes = [], []
    original = tokenizer.encode
    original_chunk = tokenizer.model.model._run_encode_chunk

    def encode(value):
        before = tokenizer.model.model.snapshot_encoder_stream_state()
        assert before.stream_shape is None and all(item is None for item in before.cache)
        assert not tokenizer.use_streaming_encode and not tokenizer._keep_encoder_cache
        seen = value.detach().clone()
        result = original(value)
        after = tokenizer.model.model.snapshot_encoder_stream_state()
        assert after.stream_shape is None and all(item is None for item in after.cache)
        records.append((seen, result.detach().clone()))
        return result

    def chunk(value, cache, scale):
        # Observe the real frozen bf16 prime after WanVAE's fp32 -> bf16 cast.
        assert value.dtype == torch.bfloat16
        assert value.shape[:3] == (1, 12, 1)
        assert all(item is None for item in cache)
        assert scale is tokenizer.model.scale
        primes.append(tuple(value.shape))
        return original_chunk(value, cache, scale)

    monkeypatch.setattr(tokenizer, "encode", encode)
    monkeypatch.setattr(tokenizer.model.model, "_run_encode_chunk", chunk)
    return records, primes


@pytest.mark.parametrize("image_size", [128, 256])
def test_policy_precrop_current_latent_is_bitexact_independent_encode1(
    monkeypatch, server, tokenizer, image_size, record_property
):
    """Two requests, including B=2, observe A -> B -> A without cache reset."""
    service = _service(server, tokenizer)
    frames = [_composite(0), _composite(1), _composite(0)]
    preps = [service._prep_policy_item(_request(frame, image_size)) for frame in frames]
    assert all(prep["sequence_plan"].condition_frame_indexes_vision == [0] for prep in preps)
    state = {key: value.clone() for key, value in tokenizer.model.model.state_dict().items()}
    identities = tuple((name, id(parameter)) for name, parameter in tokenizer.model.model.named_parameters())
    records, primes = _observe(monkeypatch, tokenizer)
    with torch.inference_mode():
        first = service.model.get_data_and_condition(_policy_batch(service, preps[:1]), vision_condition_indexes=[[0]])
        assert len(records) == len(primes) == 1
        following = service.model.get_data_and_condition(
            _policy_batch(service, preps[1:]), vision_condition_indexes=[[0], [0]]
        )
        assert len(records) == len(primes) == 3
        policy_records = tuple(records)
        independent = [
            encode_current_composite_visual96(tokenizer, prepare_robocasa_composite_frame(frame, image_size))
            for frame in frames
        ]
    assert len(records) == len(primes) == 6  # No hidden full-video or repeated Policy encoding.
    cropped = [*first.x0_tokens_vision, *following.x0_tokens_vision]
    witnesses = []
    for index, (frame, prep, (policy_input, policy_latent), (z0, visual), crop) in enumerate(
        zip(frames, preps, policy_records, independent, cropped, strict=True)
    ):
        independent_input, independent_latent = records[index + 3]
        prepared = prepare_robocasa_composite_frame(frame, image_size)
        _exact(prep["video_padded"][:, :1], prepared.padded_single_frame)
        _exact(prep["padded_image_size"], prepared.padded_image_size)
        assert policy_input.shape == (1, 3, 1, *prepared.padded_single_frame.shape[-2:])
        assert policy_input.dtype == torch.float32
        _exact(policy_input, independent_input)
        _exact(policy_latent, independent_latent)
        _exact(policy_latent[0, :, 0].float(), z0)
        _exact(robocasa_current_latent_to_visual96(policy_latent[0, :, 0].float()), visual)
        assert visual.shape == (96,) and visual.dtype == torch.float32
        content_h, content_w = (int(value) // 16 for value in prep["padded_image_size"][2:])
        expected_crop = policy_latent[:, :, :, :content_h, :content_w]
        _exact(crop[:, :, :1], expected_crop)
        assert crop.shape[2] == 5 and torch.count_nonzero(crop[:, :, 1:]) == 0
        assert crop.shape[-2:] != policy_latent.shape[-2:]
        cropped_visual = robocasa_current_latent_to_visual96(crop[0, :, 0].float())
        assert not torch.equal(cropped_visual, visual), "The capture seam must precede the existing latent crop."
        witnesses.append(
            {
                "index": index,
                "image_size": image_size,
                "input_shape": list(policy_input.shape),
                "input_dtype": str(policy_input.dtype),
                "input_sha256": _sha(policy_input),
                "latent_shape": list(policy_latent.shape),
                "latent_sha256": _sha(z0),
                "visual96_sha256": _sha(visual),
                "policy_crop_shape": list(crop.shape),
                "crop_visual96_differs": True,
            }
        )
    _exact(policy_records[0][0], policy_records[2][0])
    _exact(policy_records[0][1], policy_records[2][1])
    assert not torch.equal(policy_records[0][0], policy_records[1][0])
    assert not torch.equal(policy_records[0][1], policy_records[1][1])
    assert tuple(state) == tuple(tokenizer.model.model.state_dict())
    assert identities == tuple((name, id(parameter)) for name, parameter in tokenizer.model.model.named_parameters())
    for key, value in tokenizer.model.model.state_dict().items():
        _exact(value, state[key])
    record_property(
        "issue22_g0_actual_wan_cpu",
        json.dumps(
            {
                "cases": witnesses,
                "encode_calls": len(records),
                "prime_calls": len(primes),
                "wan_weight_dtype": "torch.bfloat16",
                "latent_channels": 48,
                "checkpoint_loaded": False,
                "gpu_validated": False,
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("image_size", [128, 192, 256, 320])
def test_actual_robocasa_policy_spatial_prep_matches_independent_frame(server, image_size):
    # Additional public server preprocessing coverage is cheap and needs no VAE.
    service = _service(server, tokenizer=None)
    frame = _composite(1)
    prep = service._prep_policy_item(_request(frame, image_size))
    prepared = prepare_robocasa_composite_frame(frame, image_size)
    _exact(prep["img_chw_uint8"], prepared.source_uint8)
    _exact(prep["video_padded"], prepared.padded_single_frame.repeat(1, 17, 1, 1))
    _exact(prep["padded_image_size"], prepared.padded_image_size)
    assert prep["sequence_plan"].condition_frame_indexes_vision == [0]

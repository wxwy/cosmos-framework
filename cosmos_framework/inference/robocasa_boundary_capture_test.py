# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contract tests of the actual pre-crop hook; no real Wan/DCP/GPU claim."""

from __future__ import annotations

import ast
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from cosmos_framework.inference.robocasa_boundary_capture import (
    capture_policy_boundary,
    observe_policy_current_encoding,
    policy_current_encode_span,
)
from cosmos_framework.inference.robocasa_vae_encoder import (
    prepare_composite_frame,
    preprocess_profile_digest,
    visual96_contract_digest,
)
from cosmos_framework.inference.robocasa_vae_protocol import source_digest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]
_PACKAGE = Path(__file__).resolve().parents[1]


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    # Resolve actual production method bodies without importing the language
    # model, dataset/trainer graph, optional attention kernels or checkpoints.
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def methods():
    namespace = {
        "torch": torch,
        "Any": Any,
        "F": functional,
        "observe_policy_current_encoding": observe_policy_current_encoding,
        "policy_current_encode_span": policy_current_encode_span,
    }
    _functions(_PACKAGE / "model/generator/vision_encoder.py", {"normalize_uint8_item"}, namespace)
    _functions(
        _PACKAGE / "model/generator/mot/robocasa_exact_window_local.py",
        {"robocasa_current_latent_to_visual96"},
        namespace,
    )
    names = {
        "_encode_vision_item",
        "_encode_vision_x0_tokens_prefix",
        "_remove_padding_from_latent",
        "_normalize_uint8_vision_item",
        "_normalize_video_databatch_inplace",
        "encode",
    }
    _functions(_PACKAGE / "model/generator/omni_mot_model.py", names, namespace)
    # Official single Encode1 wrapper, with its real shared normalizer and pool.
    _functions(_PACKAGE / "inference/robocasa_composite_visual.py", {"encode_current_composite_visual96"}, namespace)
    return SimpleNamespace(**namespace)


def _rgb(variant=0):
    y, x = np.indices((256, 512))
    return (
        np.stack(
            (np.full_like(x, variant * 43), (x + y + variant * 7) % 256, (x * 3 + y * 11 + variant * 17) % 256), axis=-1
        )
        .astype(np.uint8)
        .tobytes()
    )


def _identity(image_size=256):
    return {
        "vae_weights_sha256": "1" * 64,
        "vae_config_sha256": "2" * 64,
        "preprocess_profile_sha256": preprocess_profile_digest(image_size),
        "visual96_contract_sha256": visual96_contract_digest(),
        "encoder_backend": "eager",
        "encoder_dtype": "bfloat16",
    }


def _lease(variant=0):
    return {
        "server_epoch": "server",
        "run_lease": "run",
        "manifest_digest": "a" * 64,
        "slot_id": f"slot{variant}",
        "generation": 1,
        "task": "task",
        "episode_index": 0,
        "attempt_id": f"attempt{variant}",
        "episode_id": f"episode{variant}",
        "session_id": f"session{variant}",
    }


def _prepared(rgb=None, image_size=256):
    raw = _rgb() if rgb is None else rgb
    source = (
        torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(256, 512, 3).copy()).permute(2, 0, 1).contiguous()
    )
    return prepare_composite_frame(source, image_size)


def _model(methods):
    frozen = nn.Linear(1, 1, dtype=torch.bfloat16).eval().requires_grad_(False)
    calls = []

    def encode(state):
        calls.append(state.clone())
        batch, _, time, height, width = state.shape
        latent_time = 1 + (time - 1) // 4
        shape = (batch, 48, latent_time, height // 16, width // 16)
        result = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        return result + state.mean()

    tokenizer = SimpleNamespace(
        model=SimpleNamespace(model=frozen, dtype=torch.bfloat16, device="cpu"),
        is_causal=True,
        use_streaming_encode=False,
        _keep_encoder_cache=False,
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        encode=encode,
        get_latent_num_frames=lambda count: 1 + (count - 1) // 4,
        get_pixel_num_frames=lambda count: (count - 1) * 4 + 1,
    )
    model = SimpleNamespace(
        tokenizer_vision_gen=tokenizer, tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32}
    )
    for name in (
        "_encode_vision_item",
        "_encode_vision_x0_tokens_prefix",
        "_remove_padding_from_latent",
        "_normalize_uint8_vision_item",
        "_normalize_video_databatch_inplace",
        "encode",
    ):
        setattr(model, name, MethodType(getattr(methods, name), model))
    model.input_video_key = "video"
    return model, calls


def _scope(model, *, rgb=None, step=0, variant=0, clock=None, **overrides):
    options = {
        "source_rgb": _rgb(variant) if rgb is None else rgb,
        "identity": _identity(),
        "lease": _lease(variant),
        "request_key": f"request{step}",
        "chunk_id": f"chunk{step}",
        "source_step": step,
        "image_size": 256,
        "deadline_ns": 1000,
        "clock_domain": "synthetic-clock",
        "clock_ns": lambda: 100,
    }
    if clock is not None:
        options["clock_ns"] = clock
    options.update(overrides)
    return capture_policy_boundary(model, **options)


def _policy_prefix(model, rgb=None):
    prepared = _prepared(rgb)
    batch = {"video": [[prepared.padded_single_frame.repeat(1, 17, 1, 1)]]}
    model._normalize_video_databatch_inplace(batch)
    result = model._encode_vision_x0_tokens_prefix(batch["video"], [[0]])
    cropped = model._remove_padding_from_latent(result, [prepared.padded_image_size])
    return result[0], cropped[0], prepared


def test_actual_hook_observes_existing_single_encode_before_crop_and_matches_official_encode1(methods):
    model, calls = _model(methods)
    prepared = _prepared()
    state = {name: value.clone() for name, value in model.tokenizer_vision_gen.model.model.state_dict().items()}
    parameter_ids = tuple(id(parameter) for parameter in model.tokenizer_vision_gen.model.model.parameters())
    model_keys = tuple(vars(model))
    _, expected = methods.encode_current_composite_visual96(model.tokenizer_vision_gen, prepared)
    assert len(calls) == 1
    calls.clear()  # Observational counter only; no model, RNG or cache reset.
    with torch.no_grad(), _scope(model, step=16) as capture:
        with pytest.raises(RuntimeError, match="private/unavailable"):
            _ = capture.boundary
        full, cropped, _ = _policy_prefix(model)
        assert len(calls) == 1 and calls[0].shape[:3] == (1, 3, 1)
    boundary = capture.boundary
    assert boundary.visual96 == struct.pack("<96f", *expected.tolist())
    assert boundary.header["source_rgb_sha256"] == source_digest(_rgb())
    assert boundary.header["source_step"] == 16
    assert boundary.header["lease"] == _lease()
    assert boundary.header["policy_request_key"] == "request16"
    assert boundary.header["issued_chunk_id"] == "chunk16"
    assert int(boundary.header["deadline_ns"]) == 1000
    assert boundary.header["clock_domain"] == "synthetic-clock"
    assert full.shape == (1, 48, 5, 12, 20) and cropped.shape == (1, 48, 5, 10, 20)
    assert not torch.count_nonzero(full[:, :, 1:])
    wrong = methods.robocasa_current_latent_to_visual96(cropped[0, :, 0])
    assert boundary.visual96 != struct.pack("<96f", *wrong.tolist())
    assert tuple(vars(model)) == model_keys
    assert parameter_ids == tuple(id(parameter) for parameter in model.tokenizer_vision_gen.model.model.parameters())
    for name, value in model.tokenizer_vision_gen.model.model.state_dict().items():
        assert torch.equal(value, state[name])


@pytest.mark.parametrize("batch,time", [(1, 1), (1, 17), (8, 1)])
def test_default_actual_hook_is_noop_for_legacy_training_and_native_b8(methods, batch, time):
    model, calls = _model(methods)
    pixels = torch.zeros(batch, 3, time, 192, 320)
    original = model.encode(pixels).contiguous().float()
    calls.clear()
    with torch.enable_grad():
        actual = model._encode_vision_item(pixels, num_views=1)
    assert len(calls) == 1 and torch.equal(actual, original)


def test_generation_abort_keeps_candidate_private_and_restores_noop_context(methods):
    model, calls = _model(methods)
    capture = _scope(model)
    sentinel = RuntimeError("native sampling failed")
    with pytest.raises(RuntimeError) as error, torch.no_grad():
        with capture:
            _policy_prefix(model)
            raise sentinel
    assert error.value is sentinel and len(calls) == 1
    with pytest.raises(RuntimeError, match="private/unavailable"):
        _ = capture.boundary
    with torch.no_grad():
        _policy_prefix(model)
    assert len(calls) == 2  # The expired observer is absent from the following legacy request.


def test_missing_or_second_event_fails_closed_even_if_observer_error_is_caught(methods):
    model, calls = _model(methods)
    missing = _scope(model)
    with pytest.raises(RuntimeError, match="exactly one"):
        with missing:
            pass
    with pytest.raises(RuntimeError, match="private/unavailable"):
        _ = missing.boundary
    duplicate = _scope(model)
    with pytest.raises(RuntimeError, match="capture failed"), torch.no_grad():
        with duplicate:
            _policy_prefix(model)
            with pytest.raises(RuntimeError, match="exactly one"):
                _policy_prefix(model)
    assert len(calls) == 2
    with pytest.raises(RuntimeError):
        _ = duplicate.boundary


@pytest.mark.parametrize("kind", ["source", "T17", "B8", "crop", "nan", "model", "grad"])
def test_wrong_current_event_cannot_become_boundary(methods, kind):
    model, calls = _model(methods)
    capture = _scope(model)
    with pytest.raises((ValueError, RuntimeError)), torch.no_grad():
        with capture:
            if kind == "source":
                _policy_prefix(model, _rgb(1))
            elif kind in ("T17", "B8"):
                pixels = torch.zeros(8 if kind == "B8" else 1, 3, 17 if kind == "T17" else 1, 192, 320)
                model._encode_vision_item(pixels, num_views=1)
            elif kind == "grad":
                with torch.enable_grad():
                    _policy_prefix(model)
            else:
                prepared = _prepared()
                pixels = prepared.padded_single_frame.unsqueeze(0).float().div_(127.5).sub_(1)
                latent = model.encode(pixels)
                if kind == "crop":
                    latent = latent[:, :, :, :10]
                elif kind == "nan":
                    latent[0, 0, 0, 0, 0] = float("nan")
                observe_policy_current_encoding(object() if kind == "model" else model, pixels, latent)
    with pytest.raises(RuntimeError):
        _ = capture.boundary


@pytest.mark.parametrize("when", ["before", "during", "after"])
def test_original_absolute_deadline_is_not_extended(methods, when):
    model, calls = _model(methods)
    clock = [1000 if when == "before" else 100]
    capture = _scope(model, clock=lambda: clock[0])
    with pytest.raises(TimeoutError, match="original Policy boundary"), torch.no_grad():
        with capture:
            if when == "during":
                clock[0] = 1000
            _policy_prefix(model)
            if when == "after":
                clock[0] = 1000
    with pytest.raises(RuntimeError):
        _ = capture.boundary
    with torch.no_grad():
        _policy_prefix(model)


def test_nested_reentry_and_thread_contexts_do_not_leak_between_sessions(methods):
    model, _ = _model(methods)
    capture = _scope(model)
    with torch.no_grad(), capture:
        with pytest.raises(RuntimeError, match="nested"):
            with _scope(model):
                pass
        with pytest.raises(RuntimeError, match="reentered"):
            with capture:
                pass
        _policy_prefix(model)
    with pytest.raises(RuntimeError, match="reentered"):
        with capture:
            pass
    barrier = threading.Barrier(2)

    def run(variant):
        separate, calls = _model(methods)
        with torch.no_grad(), _scope(separate, variant=variant) as candidate:
            barrier.wait(timeout=10)
            _policy_prefix(separate, _rgb(variant))
        assert len(calls) == 1
        return candidate.boundary

    with ThreadPoolExecutor(max_workers=2) as pool:
        boundaries = list(pool.map(run, [0, 1]))
    assert [item.header["lease"]["session_id"] for item in boundaries] == ["session0", "session1"]
    assert [item.header["source_rgb_sha256"] for item in boundaries] == [source_digest(_rgb(0)), source_digest(_rgb(1))]


@pytest.mark.parametrize("kind", ["stream", "cache", "identity", "step", "backend"])
def test_unsupported_capture_is_rejected_before_observing_or_encoding(methods, kind):
    model, calls = _model(methods)
    options = {}
    if kind == "stream":
        model.tokenizer_vision_gen.use_streaming_encode = True
    elif kind == "cache":
        model.tokenizer_vision_gen._keep_encoder_cache = True
    elif kind == "step":
        options["step"] = 1
    elif kind == "identity":
        identity = _identity()
        identity["preprocess_profile_sha256"] = "f" * 64
        options["identity"] = identity
    else:
        model.tokenizer_vision_gen.model.model._aot_chunk_fns = {(1, 96, 160, 0): object()}
    with pytest.raises((ValueError, RuntimeError)):
        with _scope(model, **options):
            pass
    assert not calls

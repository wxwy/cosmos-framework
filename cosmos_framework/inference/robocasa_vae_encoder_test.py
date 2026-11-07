# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU synthetic adapter checks; these do not establish actual Wan BF16 parity."""

from __future__ import annotations

import ast
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional
from torchvision.transforms import functional as transforms_f

from cosmos_framework.inference.robocasa_vae_encoder import (
    LoadedWanEncode1,
    PreparedCompositeFrame,
    current_latent_to_visual96,
    prepare_composite_frame,
    preprocess_profile_digest,
    visual96_contract_digest,
)
from cosmos_framework.inference.robocasa_vae_protocol import EncoderDrainUnproven, Frame, ProtocolError

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]
_ROOT = Path(__file__).resolve().parents[1]


def _identity(image_size=256, backend="eager"):
    return {
        "vae_weights_sha256": "1" * 64,
        "vae_config_sha256": "2" * 64,
        "preprocess_profile_sha256": preprocess_profile_digest(image_size),
        "visual96_contract_sha256": visual96_contract_digest(),
        "encoder_backend": backend,
        "encoder_dtype": "bfloat16",
    }


def _rgb(variant):
    y, x = np.indices((256, 512))
    return (
        np.stack(
            (
                np.full_like(x, 17 * variant % 256),
                (x + 3 * y + 13 * variant) % 256,
                (7 * x + y + 89 * (x >= 256) + 23 * variant) % 256,
            ),
            axis=-1,
        )
        .astype(np.uint8)
        .tobytes()
    )


def _frame(variant=0, *, identity=None):
    lease = {
        "server_epoch": "server-epoch",
        "run_lease": "run",
        "manifest_digest": "a" * 64,
        "slot_id": f"slot-{variant % 3}",
        "generation": 1,
        "task": "task",
        "episode_index": 0,
        "attempt_id": f"attempt-{variant % 3}",
        "session_id": f"session-{variant % 3}",
        "episode_id": f"episode-{variant % 3}",
    }
    return Frame.create(
        lease,
        f"chunk-{variant}",
        f"request-{variant}",
        variant * 3,
        _rgb(variant),
        struct.pack("<15f", *[0.01 * component for component in range(15)]),
        _identity() if identity is None else identity,
        episode_deadline_ns=10**18,
        job_deadline_ns=10**18,
        clock_domain="synthetic-cpu",
    )


def _pixels(frame):
    return (
        torch.from_numpy(np.frombuffer(frame.rgb, dtype=np.uint8).reshape(256, 512, 3).copy())
        .permute(2, 0, 1)
        .contiguous()
    )


@pytest.fixture(scope="module")
def golden():
    """Execute only reviewed math definitions extracted from the actual baseline.

    No imports from the dataset-owning modules are executed. This preserves a
    source-backed oracle instead of comparing a copied helper against itself.
    """
    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
    from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution

    namespace = {
        "torch": torch,
        "np": np,
        "Image": Image,
        "Any": Any,
        "F": functional,
        "transforms_F": transforms_f,
        "VIDEO_RES_SIZE_INFO": VIDEO_RES_SIZE_INFO,
        "get_vision_data_resolution": get_vision_data_resolution,
        "PreparedCompositeFrame": PreparedCompositeFrame,
    }
    definitions = (
        ("data/generator/action/utils/transforms.py", {"find_closest_target_size", "reflection_pad_to_target"}),
        ("model/generator/vision_encoder.py", {"normalize_uint8_item"}),
        ("model/generator/mot/robocasa_exact_window_local.py", {"robocasa_current_latent_to_visual96"}),
        (
            "inference/robocasa_composite_visual.py",
            {"prepare_robocasa_composite_frame", "encode_current_composite_visual96"},
        ),
    )
    for relative, names in definitions:
        path = _ROOT / relative
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class _SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, 48, dtype=torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, 48, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, state):
        # A deterministic, independent per-row fake. It intentionally does not
        # emulate Wan convolution kernels or make a real model-parity claim.
        pooled = functional.avg_pool2d(state[:, :, 0].float(), kernel_size=16)
        values = pooled.repeat(1, 16, 1, 1) * self.scale.float().view(1, 48, 1, 1)
        values = values + self.bias.float().view(1, 48, 1, 1)
        return values.bfloat16().unsqueeze(2)


class _Tokenizer:
    is_causal = True
    use_streaming_encode = False
    _keep_encoder_cache = False
    spatial_compression_factor = 16
    temporal_compression_factor = 4

    def __init__(self):
        self.model = SimpleNamespace(model=_SyntheticEncoder().eval(), dtype=torch.bfloat16, device="cpu")
        self.calls = []

    def encode(self, state):
        self.calls.append(state.detach().clone())
        return self.model.model(state.bfloat16()).to(state.dtype)


def test_import_and_contract_digests_require_no_torch_or_training_dataset_imports():
    script = """
import sys
from cosmos_framework.inference.robocasa_vae_encoder import preprocess_profile_digest, visual96_contract_digest
assert len(preprocess_profile_digest()) == len(visual96_contract_digest()) == 64
assert 'torch' not in sys.modules
assert not any(name.startswith('cosmos_framework.data.generator.action.datasets') for name in sys.modules)
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(_ROOT.parent)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_ROOT.parent, env=environment, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("image_size", [128, 192, 256, 320])
def test_pure_preprocess_matches_actual_baseline_math_without_dataset_imports(golden, image_size):
    before = {name for name in sys.modules if name.startswith("cosmos_framework.data.generator.action.datasets")}
    source = _pixels(_frame(2))
    expected = golden.prepare_robocasa_composite_frame(source, image_size)
    actual = prepare_composite_frame(source, image_size)
    for name in ("source_uint8", "padded_single_frame", "padded_image_size"):
        value, reference = getattr(actual, name), getattr(expected, name)
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert torch.equal(value, reference)
    assert actual.padded_single_frame.shape[:2] == (3, 1)
    after = {name for name in sys.modules if name.startswith("cosmos_framework.data.generator.action.datasets")}
    assert after == before


def test_pure_pool_preserves_complete_precrop_latent_and_gradient(golden):
    with torch.enable_grad():
        source = torch.arange(48 * 12 * 20, dtype=torch.float32).reshape(48, 12, 20).requires_grad_()
        actual = current_latent_to_visual96(source)
        expected = golden.robocasa_current_latent_to_visual96(source)
        assert torch.equal(actual, expected) and actual.shape == (96,)
        assert not torch.equal(actual, current_latent_to_visual96(source[:, :10]))
        (actual_grad,) = torch.autograd.grad(actual.sum(), source, retain_graph=True)
        (expected_grad,) = torch.autograd.grad(expected.sum(), source)
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_injected_batch_uses_only_batch_axis_and_matches_independent_encode1(golden, batch):
    tokenizer = _Tokenizer()
    adapter = LoadedWanEncode1(tokenizer, identity=_identity())
    variants = ([5, 1, 7, 0, 6, 2, 4, 3] if batch == 8 else [3, 0, 2, 1])[:batch]
    frames = [_frame(index) for index in variants]
    expected = []
    state = {name: value.clone() for name, value in tokenizer.model.model.state_dict().items()}
    identities = tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    for frame in frames:
        prepared = golden.prepare_robocasa_composite_frame(_pixels(frame), 256)
        _, visual = golden.encode_current_composite_visual96(tokenizer, prepared)
        expected.append(struct.pack("<96f", *visual.tolist()))
    independent_inputs = tuple(tokenizer.calls)
    tokenizer.calls.clear()  # Observation counter only; no model/cache reset.
    output = adapter(frames)
    assert output == expected and all(type(value) is bytes and len(value) == 384 for value in output)
    assert len(tokenizer.calls) == 1 and tokenizer.calls[0].shape[:3] == (batch, 3, 1)
    assert tokenizer.calls[0].dtype == torch.float32
    for index, prior_input in enumerate(independent_inputs):
        assert torch.equal(tokenizer.calls[0][index : index + 1], prior_input)
    assert tuple(state) == tuple(tokenizer.model.model.state_dict())
    assert identities == tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    for name, value in tokenizer.model.model.state_dict().items():
        assert torch.equal(value, state[name])


def test_profile_digests_bind_actual_image_size_and_full_canvas_contract():
    assert preprocess_profile_digest(128) != preprocess_profile_digest(256)
    identity = _identity(128)
    with pytest.raises(ValueError, match="preprocessing"):
        LoadedWanEncode1(_Tokenizer(), identity=identity, image_size=256)
    identity = _identity()
    identity["visual96_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Visual96"):
        LoadedWanEncode1(_Tokenizer(), identity=identity)


def test_identity_and_source_tampering_fail_before_encoding():
    tokenizer = _Tokenizer()
    adapter = LoadedWanEncode1(tokenizer, identity=_identity())
    foreign = _identity()
    foreign["vae_weights_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        adapter([_frame(identity=foreign)])
    damaged = _frame()
    object.__setattr__(damaged, "rgb", bytes([1]) + damaged.rgb[1:])
    with pytest.raises(ProtocolError):
        adapter([damaged])
    assert not tokenizer.calls
    with pytest.raises(TypeError):
        adapter.identity["vae_config_sha256"] = "f" * 64
    adapter.image_size = 128
    with pytest.raises(RuntimeError, match="configuration changed"):
        adapter([_frame()])
    assert not tokenizer.calls


@pytest.mark.parametrize("kind", ["stream", "cache", "noncausal", "training", "unfrozen", "dtype"])
def test_unsupported_loaded_tokenizer_is_rejected(kind):
    tokenizer = _Tokenizer()
    if kind == "stream":
        tokenizer.use_streaming_encode = True
    elif kind == "cache":
        tokenizer._keep_encoder_cache = True
    elif kind == "noncausal":
        tokenizer.is_causal = False
    elif kind == "training":
        tokenizer.model.model.train()
    elif kind == "unfrozen":
        next(tokenizer.model.model.parameters()).requires_grad_(True)
    else:
        tokenizer.model.dtype = torch.float32
    with pytest.raises(RuntimeError):
        LoadedWanEncode1(tokenizer, identity=_identity())
    assert not tokenizer.calls


def test_aot_b1_guard_and_backend_identity_are_not_silently_changed():
    tokenizer = _Tokenizer()
    tokenizer.model.model._aot_chunk_fns = {(1, 96, 160, 0): lambda *_args: None}
    eager = LoadedWanEncode1(tokenizer, identity=_identity())
    with pytest.raises(RuntimeError, match="eager identity"):
        eager([_frame()])
    aot_identity = _identity(backend="aot_b1")
    aot = LoadedWanEncode1(tokenizer, identity=aot_identity)
    with pytest.raises(ValueError, match="eager identity"):
        aot([_frame(index, identity=aot_identity) for index in range(4)])
    tokenizer.model.model._aot_chunk_fns.clear()
    with pytest.raises(RuntimeError, match="matching prime artifact"):
        aot([_frame(identity=aot_identity)])
    assert not tokenizer.calls


@pytest.mark.parametrize("kind", ["T17", "cropped", "channels", "dtype", "nan", "inf", "none"])
def test_malformed_latent_aborts_whole_batch_without_fallback(kind):
    tokenizer = _Tokenizer()
    original = tokenizer.encode

    def invalid(state):
        output = original(state)
        if kind == "T17":
            return output.repeat(1, 1, 17, 1, 1)
        if kind == "cropped":
            return output[:, :, :, :-2]
        if kind == "channels":
            return output[:, :-1]
        if kind == "dtype":
            return output.bfloat16()
        if kind == "none":
            return None
        output[0, 0, 0, 0, 0] = float("nan") if kind == "nan" else float("inf")
        return output

    tokenizer.encode = invalid
    adapter = LoadedWanEncode1(tokenizer, identity=_identity())
    with pytest.raises(ValueError, match="complete finite fp32"):
        adapter([_frame(0), _frame(1)])
    assert len(tokenizer.calls) == 1


@pytest.mark.parametrize("batch", [0, 9])
def test_batch_bounds_do_not_encode(batch):
    tokenizer = _Tokenizer()
    adapter = LoadedWanEncode1(tokenizer, identity=_identity())
    with pytest.raises(ValueError, match="1..8"):
        adapter([_frame(index) for index in range(batch)])
    assert not tokenizer.calls


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_device_failure_requires_explicit_unproven_drain_without_using_gpu(monkeypatch, device_type):
    adapter = LoadedWanEncode1(_Tokenizer(), identity=_identity())
    # CPU-only control test: the fake device label never reaches a tensor op.
    adapter._device = SimpleNamespace(type=device_type)

    def fail(_frames):
        raise ValueError("synthetic backend failure")

    monkeypatch.setattr(adapter, "_encode_batch", fail)
    expected = EncoderDrainUnproven if device_type == "cuda" else ValueError
    with pytest.raises(expected) as raised:
        adapter([])
    if device_type == "cuda":
        assert isinstance(raised.value.__cause__, ValueError)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict current-frame capture: real Wan parity and fail-closed scope ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference import robocasa_local_pipeline_parity_test as g0
from cosmos_framework.inference.robocasa_composite_visual import (
    CapturedCurrentVisual,
    capture_policy_current_visuals,
    composite_frame_digest,
    encode_current_composite_visual96,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import robocasa_current_latent_to_visual96
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]

# Reuse the frozen G0 fixture's real Wan2.2 constructor, official scale, 48D
# latent ABI and bf16 math. Do not duplicate or weaken its numeric assertions.
cpu_threads = g0.cpu_threads
tokenizer = g0.tokenizer
server = g0.server


def _prepared(variant=0, image_size=256):
    return prepare_robocasa_composite_frame(g0._composite(variant), image_size)


def _input(prepared):
    return normalize_uint8_item(prepared.padded_single_frame.unsqueeze(0), {"device": "cpu", "dtype": torch.float32})


def _latent(prepared):
    height, width = prepared.padded_single_frame.shape[-2:]
    return torch.arange(48 * (height // 16) * (width // 16), dtype=torch.float32).reshape(
        1, 48, 1, height // 16, width // 16
    )


def _stub_encode(monkeypatch, tokenizer, prepared):
    """Failure-mechanism double only; real numerical parity is tested below."""
    calls = []
    latent = _latent(prepared)

    def encode(value):
        calls.append(value.clone())
        return latent

    monkeypatch.setattr(tokenizer, "encode", encode)
    return encode, calls, latent


def _scope(tokenizer, *frames):
    return capture_policy_current_visuals(
        tokenizer, frames, image_sizes=[frame.source_uint8.shape[1] for frame in frames]
    )


def test_real_wan_policy_capture_matches_independent_precrop_without_extra_encode(monkeypatch, tokenizer, server):
    service = g0._service(server, tokenizer)
    frames = [g0._composite(0), g0._composite(1)]
    prepared = [prepare_robocasa_composite_frame(frame, 256) for frame in frames]
    preps = [service._prep_policy_item(g0._request(frame, 256)) for frame in frames]
    state = {name: value.clone() for name, value in tokenizer.model.model.state_dict().items()}
    identities = tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    original = tokenizer.encode
    calls = []

    def observe(value):
        calls.append(value.clone())
        return original(value)

    monkeypatch.setattr(tokenizer, "encode", observe)
    with torch.inference_mode():
        expected = [encode_current_composite_visual96(tokenizer, frame)[1] for frame in prepared]
        calls.clear()
        with capture_policy_current_visuals(tokenizer, prepared, image_sizes=[256, 256]) as captured:
            assert not calls  # Entering capture never encodes.
            with pytest.raises(RuntimeError, match="successfully completed"):
                _ = captured.current
            result = service.model.get_data_and_condition(
                g0._policy_batch(service, preps), vision_condition_indexes=[[0], [0]]
            )
            assert len(calls) == 2
    assert tokenizer.encode is observe
    assert len(calls) == 2  # Exiting capture never encodes either.
    assert len(captured.current) == 2
    for record, frame, visual, final_x0 in zip(
        captured.current, prepared, expected, result.x0_tokens_vision, strict=True
    ):
        assert record.source_digest == composite_frame_digest(frame.source_uint8)
        assert record.image_size == 256 and record.preprocess_profile == PREPROCESS_PROFILE
        g0._exact(record.visual96, visual.cpu())
        assert not record.visual96.requires_grad
        cropped_visual = robocasa_current_latent_to_visual96(final_x0[0, :, 0].float())
        assert not torch.equal(record.visual96, cropped_visual)
    assert tuple(state) == tuple(tokenizer.model.model.state_dict())
    assert identities == tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    for name, value in tokenizer.model.model.state_dict().items():
        g0._exact(value, state[name])


def test_digest_is_layout_independent_and_binds_dimensions_and_bytes():
    source = g0._composite(0)
    noncontiguous = source.transpose(1, 2).contiguous().transpose(1, 2)
    assert not noncontiguous.is_contiguous()
    assert composite_frame_digest(source) == composite_frame_digest(noncontiguous)
    assert composite_frame_digest(source) != composite_frame_digest(source.reshape(3, 512, 256))
    changed = source.clone()
    changed[0, 0, 0] ^= 1
    assert composite_frame_digest(source) != composite_frame_digest(changed)


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(3, 2, 2), torch.zeros(3, 0, 2, dtype=torch.uint8), torch.zeros(1, 2, 2, dtype=torch.uint8), "frame"],
)
def test_digest_rejects_noncanonical_frames(bad):
    with pytest.raises(ValueError, match="uint8"):
        composite_frame_digest(bad)


def test_captured_record_owns_its_cpu_visual():
    visual = torch.arange(96, dtype=torch.float32, requires_grad=True)
    record = CapturedCurrentVisual("a" * 64, visual, 256, PREPROCESS_PROFILE)
    assert record.visual96.data_ptr() != visual.data_ptr() and not record.visual96.requires_grad
    with torch.no_grad():
        visual.add_(2)
    assert record.visual96[0] == 0
    with pytest.raises(ValueError, match="identity"):
        CapturedCurrentVisual("bad", torch.zeros(96), 256, PREPROCESS_PROFILE)
    with pytest.raises(ValueError, match="CPU fp32"):
        CapturedCurrentVisual("a" * 64, torch.zeros(96, dtype=torch.bfloat16), 256, PREPROCESS_PROFILE)


def test_missing_encode_fails_and_restores_inherited_attribute(tokenizer):
    assert "encode" not in vars(tokenizer)
    original = tokenizer.encode.__func__
    scope = _scope(tokenizer, _prepared())
    with pytest.raises(RuntimeError, match="exactly one"):
        with scope:
            pass
    assert "encode" not in vars(tokenizer) and tokenizer.encode.__func__ is original
    with pytest.raises(RuntimeError, match="successfully completed"):
        _ = scope.current


def test_extra_encode_fails_before_forwarding_and_restores_exact_override(monkeypatch, tokenizer):
    frame = _prepared()
    original, calls, latent = _stub_encode(monkeypatch, tokenizer, frame)
    scope = _scope(tokenizer, frame)
    with pytest.raises(ValueError, match="call/count"), torch.no_grad():
        with scope:
            assert tokenizer.encode(_input(frame)) is latent
            tokenizer.encode(_input(frame))
    assert len(calls) == 1 and tokenizer.encode is original
    with pytest.raises(RuntimeError):
        _ = scope.current


@pytest.mark.parametrize("kind", ["value", "dtype", "T17", "order", "rank"])
def test_wrong_input_never_reaches_original_encode(monkeypatch, tokenizer, kind):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    value = _input(frame)
    if kind == "value":
        value[0, 0, 0, 0, 0] += 0.001
    elif kind == "dtype":
        value = value.bfloat16()
    elif kind == "T17":
        value = value.repeat(1, 1, 17, 1, 1)
    elif kind == "order":
        value = _input(_prepared(1))
    else:
        value = value.squeeze(0)
    with pytest.raises(ValueError, match="full fp32 T=1"), torch.no_grad():
        with _scope(tokenizer, frame):
            tokenizer.encode(value)
    assert not calls and tokenizer.encode is original


@pytest.mark.parametrize("kind", ["nan", "inf", "cropped", "dtype", "T5", "channels", "none"])
def test_wrong_latent_fails_closed_and_does_not_publish(monkeypatch, tokenizer, kind):
    frame = _prepared()
    latent = _latent(frame)
    if kind == "nan":
        latent[0, 0, 0, 0, 0] = float("nan")
    elif kind == "inf":
        latent[0, 0, 0, 0, 0] = float("inf")
    elif kind == "cropped":
        latent = latent[:, :, :, :-2]
    elif kind == "dtype":
        latent = latent.bfloat16()
    elif kind == "T5":
        latent = latent.repeat(1, 1, 5, 1, 1)
    elif kind == "channels":
        latent = latent[:, :-1]
    else:
        latent = None
    calls = []

    def original(value):
        calls.append(value)
        return latent

    monkeypatch.setattr(tokenizer, "encode", original)
    scope = _scope(tokenizer, frame)
    with pytest.raises(ValueError, match="complete pre-crop"), torch.no_grad():
        with scope:
            tokenizer.encode(_input(frame))
    assert len(calls) == 1 and tokenizer.encode is original
    with pytest.raises(RuntimeError):
        _ = scope.current


def test_nested_and_reentered_scopes_fail_without_disturbing_outer(monkeypatch, tokenizer):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    scope = _scope(tokenizer, frame)
    with torch.no_grad(), scope:
        active = tokenizer.encode
        with pytest.raises(RuntimeError, match="nested"):
            with _scope(tokenizer, frame):
                pass
        with pytest.raises(RuntimeError, match="reentered"):
            with scope:
                pass
        assert tokenizer.encode is active
        tokenizer.encode(_input(frame))
    assert len(calls) == 1 and tokenizer.encode is original and len(scope.current) == 1
    with pytest.raises(RuntimeError, match="reentered"):
        with scope:
            pass
    assert tokenizer.encode is original
    with pytest.raises(RuntimeError, match="no longer active"):
        active(_input(frame))
    assert len(calls) == 1


@pytest.mark.parametrize("from_encoder", [False, True])
def test_original_exception_is_preserved_and_scope_is_restored(monkeypatch, tokenizer, from_encoder):
    frame = _prepared()
    sentinel = RuntimeError("original model failure")

    def original(_value):
        raise sentinel

    monkeypatch.setattr(tokenizer, "encode", original)
    scope = _scope(tokenizer, frame)
    with pytest.raises(RuntimeError) as error, torch.no_grad():
        with scope:
            if from_encoder:
                tokenizer.encode(_input(frame))
            raise sentinel
    assert error.value is sentinel and tokenizer.encode is original
    with pytest.raises(RuntimeError):
        _ = scope.current


def test_swallowed_encode_error_still_aborts_capture(monkeypatch, tokenizer):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    scope = _scope(tokenizer, frame)
    with pytest.raises(RuntimeError, match="capture failed"), torch.no_grad():
        with scope:
            with pytest.raises(ValueError):
                tokenizer.encode(_input(frame).bfloat16())
    assert not calls and tokenizer.encode is original


@pytest.mark.parametrize("when", ["before_encode", "after_encode"])
def test_cache_mode_change_inside_scope_is_rejected(monkeypatch, tokenizer, when):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    scope = _scope(tokenizer, frame)
    with pytest.raises(RuntimeError, match="Wan2.2"), torch.no_grad():
        with scope:
            if when == "after_encode":
                tokenizer.encode(_input(frame))
            tokenizer._keep_encoder_cache = True
            if when == "before_encode":
                tokenizer.encode(_input(frame))
    assert len(calls) == (1 if when == "after_encode" else 0)
    assert tokenizer.encode is original
    with pytest.raises(RuntimeError):
        _ = scope.current


def test_training_grad_mode_never_reaches_original_encode(monkeypatch, tokenizer):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    with pytest.raises(RuntimeError, match="inference-only"), torch.enable_grad():
        with _scope(tokenizer, frame):
            tokenizer.encode(_input(frame))
    assert not calls and tokenizer.encode is original


@pytest.mark.parametrize("remove", [False, True])
def test_external_encode_replacement_is_rejected_and_original_attribute_restored(tokenizer, remove):
    frame = _prepared()
    assert "encode" not in vars(tokenizer)
    scope = _scope(tokenizer, frame)
    with pytest.raises(RuntimeError, match="replaced"):
        with scope:
            if remove:
                del tokenizer.encode
            else:
                tokenizer.encode = lambda _value: None
    assert "encode" not in vars(tokenizer)


@pytest.mark.parametrize("kind", ["streaming", "cached", "noncausal", "training", "unfrozen", "fake"])
def test_unsupported_tokenizer_fails_before_installing_wrapper(monkeypatch, tokenizer, kind):
    if kind == "streaming":
        tokenizer.use_streaming_encode = True
    elif kind == "cached":
        tokenizer._keep_encoder_cache = True
    elif kind == "noncausal":
        tokenizer._causal = False
    elif kind == "training":
        tokenizer.model.model.train()
    elif kind == "unfrozen":
        next(tokenizer.model.model.parameters()).requires_grad_(True)
    else:
        tokenizer = SimpleNamespace(encode=lambda _value: None, model=tokenizer.model)
    with pytest.raises(RuntimeError, match="Wan"):
        with _scope(tokenizer, _prepared()):
            pass
    assert "encode" not in vars(tokenizer) or kind == "fake"


@pytest.mark.parametrize("kind", ["source", "metadata", "image_size", "empty", "count"])
def test_unbound_prepared_source_or_cardinality_is_rejected_without_encoding(monkeypatch, tokenizer, kind):
    frame = _prepared()
    original, calls, _ = _stub_encode(monkeypatch, tokenizer, frame)
    sizes = [256]
    if kind == "source":
        frame = replace(frame, source_uint8=_prepared(1).source_uint8)
    elif kind == "metadata":
        metadata = frame.padded_image_size.clone()
        metadata[-1] -= 1
        frame = replace(frame, padded_image_size=metadata)
    elif kind == "image_size":
        sizes = [128]
    elif kind == "count":
        sizes = []
    frames = [] if kind == "empty" else [frame]
    with pytest.raises(ValueError):
        with capture_policy_current_visuals(tokenizer, frames, image_sizes=sizes):
            pass
    assert not calls and tokenizer.encode is original


def test_independent_encode_optional_timing_preserves_actual_math(tokenizer):
    frame = _prepared()
    events = []

    class Timing:
        @contextmanager
        def measure(self, name):
            events.append((name, "enter"))
            try:
                yield
            finally:
                events.append((name, "exit"))

    expected = encode_current_composite_visual96(tokenizer, frame)
    actual = encode_current_composite_visual96(tokenizer, frame, timing=Timing())
    assert events == [
        ("independent_encode1", "enter"),
        ("independent_encode1", "exit"),
        ("visual96", "enter"),
        ("visual96", "exit"),
    ]
    for observed, reference in zip(actual, expected, strict=True):
        g0._exact(observed, reference)


def test_actual_transaction_batch_captures_each_policy_current_frame(monkeypatch, tokenizer, server):
    """Exercise the real service's required-mode capture/timing/BatchOutput wiring."""
    service = g0._service(server, tokenizer)
    service.requires_state = True
    service.cfg.guidance = 1.25
    service.cfg.seed = 23
    service.cfg.num_steps = 2
    service.cfg.action_normalization = "minmax"
    service.action_normalization = "minmax"
    service.action_min = service.action_range = None
    prefix_calls = []
    updates = [object(), object()]

    def prefixes(update):
        prefix_calls.append(update)
        return (None,)

    service.local_memory_adapter = SimpleNamespace(pipeline_enabled=True, mode="required", prefixes=prefixes)
    media_room_calls = []

    def media_room():
        media_room_calls.append(True)
        return 0, 0

    service.transactions = SimpleNamespace(media_room=media_room)
    frames = [g0._composite(0), g0._composite(1)]
    prepared = [prepare_robocasa_composite_frame(frame, 256) for frame in frames]
    requests = [g0._request(frame, 256) for frame in frames]
    for index, request in enumerate(requests):
        request["state"] = [float(index * 20 + component) for component in range(15)]
    state = {name: value.clone() for name, value in tokenizer.model.model.state_dict().items()}
    identities = tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    with torch.inference_mode():
        expected_visuals = [encode_current_composite_visual96(tokenizer, frame)[1] for frame in prepared]
    # Observe only the following service call. No model/tokenizer cache is reset.
    original_encode = tokenizer.encode
    encoded = []

    def observe(value):
        result = original_encode(value)
        encoded.append((value.clone(), result.clone()))
        return result

    monkeypatch.setattr(tokenizer, "encode", observe)
    tokenizer_keys = tuple(vars(tokenizer))
    metadata_moves = []
    original_to = torch.Tensor.to

    def metadata_to(value, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            # _transaction_batch hardcodes only this metadata placement. Pixels,
            # VAE casts, latent tensors and all tensor math retain their real paths.
            assert value.shape == (2, 4) and value.dtype == torch.float32
            metadata_moves.append(value.clone())
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", metadata_to)
    generation = []
    expected_actions = []

    def generate(batch, **kwargs):
        assert not encoded
        assert len(batch["video"]) == 2
        assert kwargs == {
            "guidance": 1.25,
            "seed": [23, 23],
            "num_steps": 2,
            "has_negative_prompt": False,
            "_local_memory_prefixes": (None, None),
        }
        clean = service.model.get_data_and_condition(batch, vision_condition_indexes=[[0], [0]])
        generation.append(clean)
        assert len(encoded) == 2 and clean.batch_size == 2
        actions = []
        for index, row in enumerate(batch["action"]):
            g0._exact(row[0][0], torch.tensor(requests[index]["state"], dtype=torch.float32))
            assert row[0].shape == (17, 15) and not torch.count_nonzero(row[0][1:])
            action = torch.arange(16 * 15, dtype=torch.float32).reshape(16, 15) + index * 1000
            expected_actions.append(action.tolist())
            # Preserve the clean state as output row zero, which the service must
            # strip once when returning its 16 executed raw15 action rows.
            actions.append(torch.cat((row[0][:1], action)).unsqueeze(0))
        return {"action": actions, "vision": None}

    service.model.generate_samples_from_batch = generate
    output = service._transaction_batch(requests, updates)
    assert isinstance(output, server.BatchOutput)
    assert output.actions == expected_actions and len(output.actions) == 2
    assert all(len(action) == 16 and all(len(row) == 15 for row in action) for action in output.actions)
    assert output.media == (None, None) and media_room_calls == [True]
    assert prefix_calls == updates and len(generation) == 1
    assert len(encoded) == 2 and len(output.current_visuals) == 2
    assert tokenizer.encode is observe and tuple(vars(tokenizer)) == tokenizer_keys
    assert len(metadata_moves) == 1
    g0._exact(metadata_moves[0], torch.stack([frame.padded_image_size for frame in prepared]))
    for record, frame, reference, (value, raw_latent), cropped in zip(
        output.current_visuals, prepared, expected_visuals, encoded, generation[0].x0_tokens_vision, strict=True
    ):
        assert isinstance(record, CapturedCurrentVisual)
        assert record.source_digest == composite_frame_digest(frame.source_uint8)
        assert record.image_size == 256 and record.preprocess_profile == PREPROCESS_PROFILE
        assert value.shape[:3] == (1, 3, 1) and raw_latent.shape[:3] == (1, 48, 1)
        g0._exact(value, _input(frame))
        g0._exact(record.visual96, reference)
        g0._exact(record.visual96, robocasa_current_latent_to_visual96(raw_latent[0, :, 0]))
        assert not torch.equal(record.visual96, robocasa_current_latent_to_visual96(cropped[0, :, 0]))
    assert output.device_timing["timing_source"] == "cpu_perf_counter"
    assert set(output.device_timing["stages"]) == {"native_generation"}
    assert output.device_timing["stages"]["native_generation"]["count"] == 1
    assert output.device_timing["stages"]["native_generation"]["total_ms"] >= 0
    assert tuple(state) == tuple(tokenizer.model.model.state_dict())
    assert identities == tuple((name, id(value)) for name, value in tokenizer.model.model.named_parameters())
    for name, value in tokenizer.model.model.state_dict().items():
        g0._exact(value, state[name])

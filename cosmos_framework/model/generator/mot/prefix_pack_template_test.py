# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Issue31 Stage2: request-local native packing, independent of text-KV gain."""

import dataclasses
import weakref
from types import MethodType, SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, SequencePlan
from cosmos_framework.model.generator.mot.prefix_text_kv_memory_test import _generation_harness
from cosmos_framework.model.generator.mot.prefix_text_kv_memory_test import cpu_kernels as cpu_kernels
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if dataclasses.is_dataclass(value):
        return {field.name: _snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value)
    return value


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            _assert_same(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noisy_action", [False, True])
def test_native_velocity_template_matches_repack_every_step(monkeypatch, dtype, noisy_action):
    """Real packer/get_velocity/update; substitute device transfer and model only.

    All finalized dataclass fields are compared, not just the synthetic velocity:
    current Prefix, text/positions, conditioned/noisy masks, domain/raw15 validity,
    payloads and timesteps. This exercises 30 distinct noise/time values.
    """
    torch.manual_seed(71)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    prefix = torch.randn(4, 32, dtype=dtype)
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 16, 2, 2, 2, dtype=dtype)],
        x0_tokens_action=[torch.randn(17, 16, dtype=dtype)],
        x0_tokens_local_memory=[prefix],
        action_domain_id=[torch.full((17,), 2, dtype=torch.int64)],
        raw_action_dim=[15],
        action_valid_mask=[torch.tensor([True] * 14 + [False])],
    )
    plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=True,
            has_local_memory=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=[0] if noisy_action else list(range(17)),
        )
    ]
    packed_snapshots = []

    def pack(plans, text, data, time, **kwargs):
        return pack_input_sequence(
            plans,
            text,
            data,
            time,
            special_tokens={"eos_token_id": 1, "start_of_generation": 2, "end_of_generation": 3},
            latent_patch_size=1,
            action_dim=16,
            **kwargs,
        )

    def denoise(*, data_batch_packed, **kwargs):
        p = data_batch_packed
        packed_snapshots.append(_snapshot(p))
        assert p.local_memory_tokens[0] is prefix
        shift = prefix.mean() + p.vision.timesteps.float().mean().to(dtype) * 0.001
        return dict(
            preds_vision=[p.vision.tokens[0].squeeze(0) * 0.5 + shift], preds_action=[p.action.tokens[0] * 0.25 + shift]
        )

    obj = SimpleNamespace(
        config=SimpleNamespace(action_gen=True, sound_gen=False),
        tensor_kwargs=dict(device="cpu", dtype=dtype),
        _pack_input_sequence=pack,
        _derive_include_end_of_generation_token=lambda: False,
        denoise=denoise,
    )
    for name in ("_get_velocity", "_update_inference_pack_template", "_copy_timestep_to_template"):
        setattr(obj, name, MethodType(getattr(OmniMoTModel, name), obj))
    template = pack(plans, [[10, 20, 30]], clean, torch.zeros(1))
    original_prefix = prefix.clone()
    for step in range(30):
        noise = [torch.randn(128 + (272 if noisy_action else 0), dtype=dtype)]
        kwargs = dict(
            noise_x=noise,
            timestep=torch.tensor([[1000.0 - step * 30.0]]),
            text_tokens=[[10, 20, 30]],
            sequence_plans=plans,
            gen_data_clean=clean,
            has_noisy_actions=noisy_action,
        )
        reference = obj._get_velocity(**kwargs)
        actual = obj._get_velocity(packed_sequence_template=template, **kwargs)
        _assert_same(packed_snapshots[-2], packed_snapshots[-1])
        torch.testing.assert_close(reference[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(prefix, original_prefix, rtol=0, atol=0)
    assert template.local_memory_tokens[0] is prefix
    assert not hasattr(obj, "packed_sequence_template")


def test_generation_reuses_one_pack_and_keeps_text_cache_and_prefix():
    obj, batch = _generation_harness()
    kwargs = dict(guidance=1.0, num_steps=30, seed=[31], _local_memory_prefixes=(torch.randn(4, 32),))
    reference = obj.generate_samples_from_batch(batch, use_prefix_pack_templates=False, **kwargs)
    assert obj.seen == [(True, False, False)] + [(True, True, False)] * 29
    obj.seen.clear()
    actual = obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.templates == 1
    assert obj.seen == [(True, False, True)] + [(True, True, True)] * 29
    for key in actual:
        torch.testing.assert_close(actual[key][0], reference[key][0], rtol=0, atol=0)
    # Disabling text KV preserves the original Prefix-on path as before Stage2.
    obj.seen.clear()
    original = obj.generate_samples_from_batch(batch, use_prefix_text_kv_cache=False, **kwargs)
    assert obj.seen == [(False, False, False)] * 30
    for key in original:
        torch.testing.assert_close(original[key][0], reference[key][0], rtol=0, atol=0)


@pytest.mark.parametrize("setting", ["temporal", "three_way", "postprocess", "cp", "guidance"])
def test_prefix_pack_uses_only_the_approved_single_sample_cache_layout(setting):
    obj, batch = _generation_harness()
    kwargs = dict(guidance=1.0, num_steps=3, seed=[31], _local_memory_prefixes=(torch.randn(4, 32),))
    if setting == "temporal":
        obj.config.video_temporal_causal = True
    elif setting == "three_way":
        obj.config.joint_attn_implementation = "three_way"
    elif setting == "postprocess":
        kwargs["velocity_postprocess_builder"] = lambda **kw: None
    elif setting == "cp":
        obj.parallel_dims = SimpleNamespace(cp_enabled=True, cfgp_enabled=False, dp_shard_mesh=None)
        # A real CP generation needs a process group; check the existing CP
        # eligibility guard here without fabricating distributed execution.
        assert not obj._can_reuse_inference_text_kv(
            [SequencePlan(has_text=True)],
            GenerationDataClean(batch_size=1, is_image_batch=False),
            reuse_pack_templates=True,
            has_velocity_postprocess=False,
        )
        assert obj.templates == 0
        return
    else:
        kwargs["guidance"] = 2.0
        obj._run_classifier_free_guidance = MethodType(OmniMoTModel._run_classifier_free_guidance, obj)
    obj.generate_samples_from_batch(batch, **kwargs)
    assert obj.templates == 0
    assert all(not any(row) for row in obj.seen)


@pytest.mark.parametrize("fail_at", [None, 1, 3])
def test_pack_lifetime_request_changes_and_failure_retry(fail_at):
    obj, batch = _generation_harness()
    refs = []

    class Template:
        def to_cuda(self):
            pass

    def pack(*args, **kwargs):
        value = Template()
        refs.append(weakref.ref(value))
        return value

    obj._pack_input_sequence = pack
    prefix = torch.randn(4, 32)
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31])
    if fail_at is not None:
        obj.fail_at = fail_at
        with pytest.raises(RuntimeError, match="injected generation failure"):
            obj.generate_samples_from_batch(batch, _local_memory_prefixes=(prefix,), **kwargs)
        assert all(not cache.is_initialized for cache in obj.created[-1])
        obj.fail_at = None
    obj.seen.clear()
    a = obj.generate_samples_from_batch(batch, _local_memory_prefixes=(prefix,), **kwargs)
    assert obj.seen[0] == (True, False, True)
    changed = obj.generate_samples_from_batch(batch, _local_memory_prefixes=(prefix + 3,), **kwargs)
    assert not torch.equal(a["action"][0], changed["action"][0])
    obj.text += 2
    different_text = obj.generate_samples_from_batch(batch, _local_memory_prefixes=(prefix,), **kwargs)
    assert not torch.equal(a["action"][0], different_text["action"][0])
    obj.text -= 2
    # A cold-start request and a new prefixed Episode never share templates/KV.
    obj.generate_samples_from_batch(batch, **kwargs)
    again = obj.generate_samples_from_batch(batch, _local_memory_prefixes=(prefix,), **kwargs)
    torch.testing.assert_close(a["action"][0], again["action"][0], rtol=0, atol=1e-6)
    assert len(refs) == 5 + int(fail_at is not None)
    assert all(ref() is None for ref in refs)


def test_no_prefix_pack_option_preserves_native_fast_path():
    obj, batch = _generation_harness()
    kwargs = dict(guidance=1.0, num_steps=6, seed=[31])
    a = obj.generate_samples_from_batch(batch, use_prefix_pack_templates=False, **kwargs)
    obj.seen.clear()
    b = obj.generate_samples_from_batch(batch, use_prefix_pack_templates=True, **kwargs)
    assert obj.seen == [(True, False, True)] + [(True, True, True)] * 5
    assert obj.templates == 2
    torch.testing.assert_close(a["action"][0], b["action"][0], rtol=0, atol=0)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Actual Action decoder/compiler with generated CPU raw15/H16 fixtures."""

import copy
import json
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.mot.action_encode_compile_test import install_compile
from cosmos_framework.model.generator.mot.action_io_projector import ACTION_IO_PROJECTOR_TYPES
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def isolated_decode_compile_cache():
    # Reset only between independent test cases, never inside a request sequence.
    torch.compiler.reset()
    assert torch._dynamo.config.recompile_limit == 8
    yield
    torch.compiler.reset()


def tiny_decode_model(*, dtype=torch.float32, local_memory=False, projector="domain_aware", embedding=True):
    """Use the real VFM class, with tiny generated weights and action_dim=15."""
    torch.manual_seed(83)
    language_model = nn.Module()
    language_model.config = SimpleNamespace(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8, num_hidden_layers=1
    )
    config = Cosmos3VFMNetworkConfig(
        action_gen=True,
        action_dim=15,
        num_embodiment_domains=4,
        action_io_projector_type=projector,
        enable_action_modality_embedding=embedding,
        local_memory_enabled=local_memory,
        joint_attn_implementation="two_way",
        latent_channel_size=4,
        latent_patch_size=1,
        max_latent_h=1,
        max_latent_w=1,
        max_latent_t=1,
        vlm_config=language_model.config,
    )
    model = Cosmos3VFMNetwork(language_model, config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    model.action2llm.to(dtype)
    model.llm2action.to(dtype)
    if embedding:
        model.action_modality_embed.data = model.action_modality_embed.data.to(dtype)
    return model


def decode_case(batch, *, kind="noisy", ragged=False, framewise=False, variant=0, dtype=torch.float32):
    """Return finalized packing metadata and an independent leaf hidden-state input.

    Normal action samples are [16,15]; ragged samples vary around H16. Metadata
    includes non-contiguous sequence positions, different task/domain patterns,
    mixed clean samples, and reversed noisy-frame order.
    """
    if kind not in ("noisy", "clean", "none", "tokens_none"):
        raise ValueError(f"unknown decoder fixture kind: {kind}")
    generator = torch.Generator().manual_seed(197 + batch + variant)
    lengths = [14 + 2 * (i % 3) if ragged else 16 for i in range(batch)]
    tokens = [torch.randn(length, 15, generator=generator, requires_grad=True) for length in lengths]
    noisy_indexes = []
    for i, length in enumerate(lengths):
        if kind != "noisy" or (variant % 3 == 2 and i % 2 == 1):
            indexes = torch.empty(0, dtype=torch.long)
        elif variant % 2:
            indexes = torch.tensor([length - 1, 0], dtype=torch.long)
        else:
            indexes = torch.tensor([length - 1], dtype=torch.long)
        noisy_indexes.append(indexes)
    domains = [
        (torch.arange(length, dtype=torch.long) + i + variant).remainder(4)
        if framewise
        else torch.tensor((i + variant) % 4)
        for i, length in enumerate(lengths)
    ]
    total = sum(lengths)
    sequence_indexes = torch.arange(1, 2 * total + 1, 2)
    mse_indexes = torch.cat(
        [sequence_indexes[sum(lengths[:i]) : sum(lengths[: i + 1])][indexes] for i, indexes in enumerate(noisy_indexes)]
    )
    masks = [torch.ones(length, dtype=torch.bool) for length in lengths]
    for mask, indexes in zip(masks, noisy_indexes):
        mask[indexes] = False
    action = ModalityData(
        tokens=tokens,
        token_shapes=[(length,) for length in lengths],
        domain_id=domains,
        sequence_indexes=sequence_indexes,
        mse_loss_indexes=mse_indexes,
        noisy_frame_indexes=noisy_indexes,
        condition_mask=masks,
    )
    if kind == "tokens_none":
        action.tokens = None
    packed = PackedSequence(action=None if kind == "none" else action, sequence_length=2 * total + 3)
    hidden = torch.randn(packed.sequence_length, 32, generator=generator, dtype=dtype, requires_grad=True)
    return packed, hidden


def decode(model, packed, hidden):
    """Call the real production decoder; return its list of predicted tensors."""
    output = {}
    model._decode_action(packed, hidden, output)
    return output["preds_action"]


def _assert_outputs(actual, expected, *, dtype):
    assert isinstance(actual, list) and len(actual) == len(expected)
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    for value, reference in zip(actual, expected):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        torch.testing.assert_close(value, reference, **tolerance)


def _loss(values):
    return sum(value.float().square().sum() + value.float().sum() for value in values)


def _assert_gradient_match(actual_model, expected_model, actual_input, expected_input, dtype):
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    assert (actual_input.grad is None) == (expected_input.grad is None)
    if actual_input.grad is not None:
        torch.testing.assert_close(actual_input.grad, expected_input.grad, **tolerance)
    expected_parameters = dict(expected_model.named_parameters())
    assert dict(actual_model.named_parameters()).keys() == expected_parameters.keys()
    for name, parameter in actual_model.named_parameters():
        expected = expected_parameters[name]
        assert (parameter.grad is None) == (expected.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected.grad, **tolerance)


def _assert_branch_semantics(model, packed, hidden, values, kind):
    used = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("action2llm.", "llm2action.")) or name == "action_modality_embed"
    }
    if kind == "noisy":
        assert hidden.grad is not None
        selected = torch.zeros(hidden.shape[0], dtype=torch.bool)
        selected[packed.action.mse_loss_indexes] = True
        torch.testing.assert_close(hidden.grad[~selected], torch.zeros_like(hidden.grad[~selected]), rtol=0, atol=0)
        for value, shape, indexes in zip(values, packed.action.token_shapes, packed.action.noisy_frame_indexes):
            assert value.shape == (shape[0], 15)
            clean = torch.ones(shape[0], dtype=torch.bool)
            clean[indexes] = False
            torch.testing.assert_close(value[clean], torch.zeros_like(value[clean]), rtol=0, atol=0)
        for name, parameter in used.items():
            assert (parameter.grad is None) == (not name.startswith("llm2action.")), name
    else:
        # Dummy uses hidden only for dtype/device, not tensor values.
        assert hidden.grad is None
        assert all(parameter.grad is not None for parameter in used.values())
        if kind == "clean":
            for value, token in zip(values, packed.action.tokens):
                assert value.shape == token.shape
                torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
            for parameter in used.values():
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)
        else:
            # Absence returns the actual dummy projection, not a detached zero.
            assert len(values) == 1 and values[0].shape == (1, 15)
            if (
                model.action_io_projector_type != "shared_weight_no_bias"
                or model.config.enable_action_modality_embedding
            ):
                assert torch.count_nonzero(values[0]).item() > 0
                assert any(torch.count_nonzero(parameter.grad).item() > 0 for parameter in used.values())
    if packed.action is not None and packed.action.tokens is not None:
        assert all(token.grad is None for token in packed.action.tokens)


@pytest.mark.parametrize("local_memory", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_alternating_B1_to_B8_long_run_preserves_default_limit(tmp_path, local_memory, dtype):
    eager = tiny_decode_model(dtype=dtype, local_memory=local_memory)
    compiled = copy.deepcopy(eager)
    original_ids = {name: id(parameter) for name, parameter in compiled.named_parameters()}
    original_keys = tuple(compiled.state_dict())
    counter, targets = install_compile(compiled, "eager")
    count = 0
    started = time.monotonic()
    try:
        with torch.no_grad():
            for cycle in range(128):
                for batch in (1, 8, 2, 7, 3, 6, 4, 5):
                    packed, hidden = decode_case(
                        batch,
                        kind="clean" if cycle % 3 == 1 else "noisy",
                        ragged=cycle % 2 != 0,
                        framewise=cycle % 5 == 1,
                        variant=cycle,
                        dtype=dtype,
                    )
                    expected = decode(eager, packed, hidden)
                    actual = decode(compiled, packed, hidden)
                    assert len(actual) == len(expected)
                    for value, reference in zip(actual, expected):
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    count += 1
        assert count == 1024
        assert counter.frame_count <= 3
        assert torch._dynamo.config.recompile_limit == 8
        assert tuple(compiled.state_dict()) == original_keys
        assert {name: id(parameter) for name, parameter in compiled.named_parameters()} == original_ids
    finally:
        (tmp_path / "decode_compile_witness.json").write_text(
            json.dumps(
                dict(
                    local_memory=local_memory,
                    dtype=str(dtype),
                    action_dim=15,
                    normal_horizon=16,
                    backend="eager",
                    calls_completed=count,
                    graph_count=counter.frame_count,
                    compile_targets=targets,
                    recompile_limit=torch._dynamo.config.recompile_limit,
                    seconds=time.monotonic() - started,
                ),
                indent=2,
            )
        )


@pytest.mark.parametrize("projector", ACTION_IO_PROJECTOR_TYPES)
@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_compiled_outputs_and_complete_gradients_match_eager(projector, embedding, dtype):
    eager = tiny_decode_model(dtype=dtype, projector=projector, embedding=embedding, local_memory=True)
    compiled = copy.deepcopy(eager)
    original_ids = {name: id(parameter) for name, parameter in compiled.named_parameters()}
    original_keys = tuple(compiled.state_dict())
    install_compile(compiled, "aot_eager")
    for kind in ("noisy", "clean", "none", "tokens_none"):
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        packed, hidden = decode_case(5, kind=kind, ragged=True, framewise=True, variant=5, dtype=dtype)
        expected_packed, expected_hidden = copy.deepcopy(packed), copy.deepcopy(hidden)
        actual_packed, actual_hidden = copy.deepcopy(packed), copy.deepcopy(hidden)
        expected = decode(eager, expected_packed, expected_hidden)
        actual = decode(compiled, actual_packed, actual_hidden)
        _assert_outputs(actual, expected, dtype=dtype)
        _loss(actual).backward()
        _loss(expected).backward()
        _assert_gradient_match(compiled, eager, actual_hidden, expected_hidden, dtype)
        _assert_branch_semantics(compiled, actual_packed, actual_hidden, actual, kind)
        _assert_branch_semantics(eager, expected_packed, expected_hidden, expected, kind)
    assert tuple(compiled.state_dict()) == original_keys
    assert {name: id(parameter) for name, parameter in compiled.named_parameters()} == original_ids


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_cpu_inductor_outputs_and_gradients_with_dynamic_batches(tmp_path, dtype):
    eager = tiny_decode_model(dtype=dtype, local_memory=True)
    compiled = copy.deepcopy(eager)
    counter, targets = install_compile(compiled, "inductor")
    for step in range(32):
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        kind = "clean" if step % 3 == 0 else "noisy"
        packed, hidden = decode_case(1 + step % 8, kind=kind, ragged=True, framewise=True, variant=step, dtype=dtype)
        expected_packed, expected_hidden = copy.deepcopy(packed), copy.deepcopy(hidden)
        actual_packed, actual_hidden = copy.deepcopy(packed), copy.deepcopy(hidden)
        expected = decode(eager, expected_packed, expected_hidden)
        actual = decode(compiled, actual_packed, actual_hidden)
        _assert_outputs(actual, expected, dtype=dtype)
        _loss(actual).backward()
        _loss(expected).backward()
        _assert_gradient_match(compiled, eager, actual_hidden, expected_hidden, dtype)
        _assert_branch_semantics(compiled, actual_packed, actual_hidden, actual, kind)
    assert counter.frame_count <= 3
    assert torch._dynamo.config.recompile_limit == 8
    (tmp_path / "decode_inductor_witness.json").write_text(
        json.dumps(
            dict(
                dtype=str(dtype),
                action_dim=15,
                normal_horizon=16,
                backend="inductor",
                calls=32,
                graph_count=counter.frame_count,
                compile_targets=targets,
                recompile_limit=8,
            )
        )
    )


@pytest.mark.parametrize("kind", ["noisy", "clean", "none", "tokens_none"])
def test_decode_updates_output_dict_in_place_without_erasing_other_keys(kind):
    model = tiny_decode_model()
    install_compile(model, "eager")
    packed, hidden = decode_case(2, kind=kind, variant=1)
    sentinel = object()
    result = {"sentinel": sentinel, "preds_action": "old value"}
    assert model._decode_action(packed, hidden, result) is None
    assert result["sentinel"] is sentinel and isinstance(result["preds_action"], list)


@pytest.mark.parametrize("failure", ["unaligned_domain", "domain_range", "noisy_index", "empty_action_list"])
def test_decode_invalid_inputs_remain_fail_closed(failure):
    model = tiny_decode_model()
    install_compile(model, "eager")
    packed, hidden = decode_case(1, framewise=True)
    if failure == "unaligned_domain":
        packed.action.domain_id[0] = torch.tensor([0, 1])
        error = ValueError
    elif failure == "domain_range":
        packed.action.domain_id[0] = torch.full((16,), 4, dtype=torch.long)
        error = (IndexError, RuntimeError)
    elif failure == "noisy_index":
        packed.action.noisy_frame_indexes[0] = torch.tensor([16])
        error = (IndexError, RuntimeError)
    else:
        # Preserve the old unsupported empty-list behavior; no successful dummy shortcut.
        packed.action = ModalityData(tokens=[])
        error = IndexError
    sentinel = object()
    result = {"preds_action": sentinel}
    with pytest.raises(error):
        model._decode_action(packed, hidden, result)
    assert result["preds_action"] is sentinel

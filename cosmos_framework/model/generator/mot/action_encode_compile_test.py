# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real Action head + Dynamo, using tiny CPU weights and finalized packing data."""

import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.mot import parallelize_vfm_network
from cosmos_framework.model.generator.mot.action_io_projector import ACTION_IO_PROJECTOR_TYPES
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def isolated_dynamo_cache():
    # Independent test cases/models must not share a code object's guard cache.
    # No reset happens within a tested request sequence; keep the production cap.
    torch.compiler.reset()
    assert torch._dynamo.config.recompile_limit == 8
    yield
    torch.compiler.reset()


def tiny_model(*, dtype=torch.float32, local_memory=False, projector="domain_aware", embedding=True):
    torch.manual_seed(71)
    language_model = nn.Module()
    language_model.config = SimpleNamespace(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8, num_hidden_layers=1
    )
    config = Cosmos3VFMNetworkConfig(
        action_gen=True,
        action_dim=5,
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
    # The timestep embedder intentionally stays FP32, as its production contract requires.
    return model


def action_case(batch, *, noisy=True, ragged=False, framewise=False, single_timestep=False):
    generator = torch.Generator().manual_seed(113 + batch)
    lengths = [2 + i % 3 if ragged else 3 for i in range(batch)]
    # Include unused tail rows to test pack_action's per-sample slicing.
    tokens = [torch.randn(length + 1, 5, generator=generator, requires_grad=True) for length in lengths]
    indexes = [torch.tensor([length - 1]) if noisy else torch.empty(0, dtype=torch.long) for length in lengths]
    domains = [
        torch.arange(length, dtype=torch.long).remainder(4) if framewise else torch.tensor(i % 4)
        for i, length in enumerate(lengths)
    ]
    total = sum(lengths)
    sequence_indexes = torch.arange(1, 2 * total + 1, 2)
    noisy_sequence_indexes = torch.cat(
        [sequence_indexes[sum(lengths[:i]) : sum(lengths[: i + 1])][index] for i, index in enumerate(indexes)]
    )
    times = torch.full((batch,), 0.25) if single_timestep else torch.linspace(0.1, 0.9, batch)
    action = ModalityData(
        tokens=tokens,
        token_shapes=[(length,) for length in lengths],
        domain_id=domains,
        sequence_indexes=sequence_indexes,
        mse_loss_indexes=noisy_sequence_indexes,
        noisy_frame_indexes=indexes,
        timesteps=times if noisy else torch.empty(0),
    )
    return PackedSequence(action=action, uses_single_timestep=single_timestep, sequence_length=2 * total + 3)


def install_compile(model, backend):
    counter = CompileCounterWithBackend(backend)
    original = torch.compile
    targets = []

    def counted(function, **options):
        targets.append(function.__name__)
        return original(function, backend=counter, **options)

    # Use the actual production installer/options; only choose a CPU backend.
    with patch.object(parallelize_vfm_network.torch, "compile", counted):
        parallelize_vfm_network.apply_compile(model, CompileConfig(enabled=True))
    return counter, targets


def encode(model, packed, dtype):
    destination = torch.full((packed.sequence_length, 32), 0.125, dtype=dtype)
    model._encode_action(packed, destination, dtype)
    return destination


@pytest.mark.parametrize("local_memory", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_alternating_B1_to_B8_metadata_long_run_stays_below_default_compile_limit(tmp_path, local_memory, dtype):
    eager = tiny_model(dtype=dtype, local_memory=local_memory)
    compiled = copy.deepcopy(eager)
    keys = tuple(compiled.state_dict())
    counter, targets = install_compile(compiled, "eager")
    calls = 0
    started = time.monotonic()
    try:
        with torch.no_grad():
            for cycle in range(128):
                for batch in (1, 8, 2, 7, 3, 6, 4, 5):
                    packed = action_case(
                        batch,
                        noisy=cycle % 3 != 0,
                        ragged=cycle % 2 != 0,
                        framewise=cycle % 5 == 1,
                        single_timestep=cycle % 7 == 1,
                    )
                    expected = encode(eager, packed, dtype)
                    actual = encode(compiled, packed, dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    calls += 1
        assert calls == 1024
        assert counter.frame_count <= 2
        assert tuple(compiled.state_dict()) == keys
        assert torch._dynamo.config.recompile_limit == 8
    finally:
        (tmp_path / "action_compile_witness.json").write_text(
            json.dumps(
                dict(
                    local_memory=local_memory,
                    dtype=str(dtype),
                    backend="eager",
                    calls_completed=calls,
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
def test_compiled_action_output_and_gradients_match_eager(projector, embedding, dtype):
    eager = tiny_model(dtype=dtype, projector=projector, embedding=embedding)
    compiled = copy.deepcopy(eager)
    install_compile(compiled, "aot_eager")
    packed = action_case(5, ragged=True, framewise=True, single_timestep=True)
    expected_input = copy.deepcopy(packed)
    actual_input = copy.deepcopy(packed)
    expected = encode(eager, expected_input, dtype)
    actual = encode(compiled, actual_input, dtype)
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    for actual_token, expected_token in zip(actual_input.action.tokens, expected_input.action.tokens):
        torch.testing.assert_close(actual_token.grad, expected_token.grad, **tolerance)
    for name, parameter in compiled.named_parameters():
        reference = dict(eager.named_parameters())[name]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, **tolerance)
    assert tuple(compiled.state_dict()) == tuple(eager.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_inductor_alternating_batch_outputs_match_eager(tmp_path, dtype):
    eager = tiny_model(dtype=dtype)
    compiled = copy.deepcopy(eager)
    counter, targets = install_compile(compiled, "inductor")
    with torch.no_grad():
        for step in range(32):
            packed = action_case(1 + step % 8, noisy=step % 2 == 0, ragged=True, framewise=True)
            expected = encode(eager, packed, dtype)
            actual = encode(compiled, packed, dtype)
            tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(actual, expected, **tolerance)
    assert counter.frame_count <= 2
    (tmp_path / "inductor_witness.json").write_text(
        json.dumps(dict(dtype=str(dtype), backend="inductor", calls=32, graphs=counter.frame_count, targets=targets))
    )


def test_absent_action_and_invalid_domain_metadata_do_not_bypass_validation():
    model = tiny_model()
    counter, _ = install_compile(model, "eager")
    destination = torch.randn(3, 32)
    before = destination.clone()
    model._encode_action(PackedSequence(), destination, torch.float32)
    torch.testing.assert_close(destination, before)
    assert counter.frame_count == 0
    packed = action_case(1)
    packed.action.domain_id[0] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="one ID per action token"):
        model._encode_action(packed, destination, torch.float32)

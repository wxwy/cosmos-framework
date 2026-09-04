# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTFastState,
    ContinualTTTFastStateTransition,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
    RecurrentLocalMemoryBackend,
    StatelessLocalReplayReadout,
    TTTLocalMemoryBackend,
)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(8)
    return {
        "history_visual_summary": torch.randn(2, 4, 96),
        "local_history_action": torch.randn(2, 4, 10),
        "history_age_steps": torch.tensor([[4, 3, 2, 1], [0, 0, 2, 1]]),
        "history_dt_s": torch.tensor([[[0.4], [0.3], [0.2], [0.1]], [[0.0], [0.0], [0.2], [0.1]]]),
        "history_mask": torch.tensor([[True, True, True, True], [False, False, True, True]]),
        "history_state": torch.randn(2, 4, 8),
    }


@pytest.mark.L0
def test_local_evidence_encoder_is_stateless_masked_and_differentiable() -> None:
    encoder = LocalEvidenceEncoder(evidence_dim=32, state_mean=torch.zeros(8), state_std=torch.ones(8))
    inputs = _inputs()
    output = encoder(**inputs)

    assert output.shape == (2, 4, 32)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~inputs["history_mask"]]) == 0
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in encoder.parameters())
    assert not any("gru" in name.lower() or "recurrent" in name.lower() for name, _ in encoder.named_modules())


@pytest.mark.L0
def test_masked_evidence_is_inert_and_state_requires_explicit_stats() -> None:
    encoder = LocalEvidenceEncoder(evidence_dim=16)
    inputs = _inputs()
    inputs.pop("history_state")
    baseline = encoder(**inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["history_visual_summary"][1, :2] = 1e6
    changed["local_history_action"][1, :2] = -1e6
    changed["history_dt_s"][1, :2] = 1e6
    changed["history_age_steps"][1, :2] = 64
    torch.testing.assert_close(encoder(**changed), baseline)

    with pytest.raises(ValueError, match="state_mean/state_std"):
        encoder(**inputs, history_state=_inputs()["history_state"])


@pytest.mark.L0
def test_state_normalization_and_shape_checks() -> None:
    with pytest.raises(ValueError, match="provided together"):
        LocalEvidenceEncoder(state_mean=torch.zeros(8))

    encoder = LocalEvidenceEncoder(evidence_dim=16, state_mean=torch.zeros(8), state_std=torch.full((8,), 1e-9))
    assert torch.equal(encoder.state_std, torch.full((8,), 1e-6))
    inputs = _inputs()
    with pytest.raises(ValueError, match="local_history_action"):
        encoder(**{**inputs, "local_history_action": torch.zeros(2, 4, 9)})


@pytest.mark.L0
def test_stateless_local_replay_readout_masks_and_uses_latest_valid() -> None:
    readout = StatelessLocalReplayReadout(evidence_dim=3, local_dim=5, hidden_dim=4)
    evidence = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
            [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, False, False], [False, True, True]])
    mean, latest, has_valid = readout.summarize(evidence, mask)
    torch.testing.assert_close(mean, torch.tensor([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [5.5, 5.5, 5.5]]))
    torch.testing.assert_close(latest, torch.tensor([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [6.0, 6.0, 6.0]]))
    assert has_valid.tolist() == [True, False, True]

    output = readout(evidence, mask)
    assert output.shape == (3, 1, 5)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[1]) == 0
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in readout.parameters())


@pytest.mark.L0
def test_r09_recurrent_backend_presence_partial_reset_and_segments() -> None:
    torch.manual_seed(4)
    backend = RecurrentLocalMemoryBackend(evidence_dim=3, local_dim=5)
    evidence = torch.randn(3, 4, 3)
    mask = torch.tensor([[True, True, True, True], [False, False, False, False], [True, False, True, False]])
    tokens, state, present = backend.replay(evidence, mask)
    latent, _ = state
    assert present.tolist() == [True, False, True]
    assert torch.equal(tokens[1], torch.zeros_like(tokens[1]))
    reset = backend.reset_mask(state, torch.tensor([False, True, False]))
    assert torch.equal(reset[0][[0, 2]], latent[[0, 2]]) and torch.count_nonzero(reset[0][1]) == 0
    assert reset[1].tolist() == [True, False, True]
    reset_tokens, _, reset_present = backend.replay(evidence[:, :1], torch.zeros(3, 1, dtype=torch.bool), reset)
    assert reset_present.tolist() == [True, False, True]
    assert torch.count_nonzero(reset_tokens[1]) == 0
    _, first, _ = backend.replay(evidence[:, :2], mask[:, :2])
    _, second, _ = backend.replay(evidence[:, 2:], mask[:, 2:], (first[0].detach(), first[1]))
    torch.testing.assert_close(second[0], latent, rtol=0, atol=1e-6)
    carry_mask = torch.tensor([[True, False], [False, False], [False, False]])
    _, carried, _ = backend.replay(evidence[:, :2], carry_mask)
    carry_tokens, _, carry_present = backend.replay(evidence[:, 2:], torch.zeros_like(carry_mask), (carried[0].detach(), carried[1]))
    assert carry_present.tolist() == [True, False, False]
    torch.testing.assert_close(carry_tokens[0], carried[0][0, None], rtol=0, atol=1e-6)


@pytest.mark.L0
def test_r09_b0_ttt_backend_contract() -> None:
    torch.manual_seed(9)
    backend = TTTLocalMemoryBackend(evidence_dim=3, local_dim=2)
    evidence = torch.randn(2, 7, 3)
    mask = torch.tensor([[True] * 7, [False] * 7])
    token, state, present = backend.replay(evidence, mask)
    assert present.tolist() == [True, False]
    assert token.shape == (2, 1, 2) and torch.isfinite(token).all() and not token.requires_grad
    assert not list(backend.named_parameters())
    assert state[0].shape == (2, 2, 3) and state[1].shape == (2, 4, 3)
    assert state[4].tolist() == [3, 0]
    _, first, _ = backend.replay(evidence[:, :2], mask[:, :2])
    split_token, split_state, split_present = backend.replay(evidence[:, 2:], mask[:, 2:], first)
    torch.testing.assert_close(split_token, token, rtol=0, atol=0)
    assert split_present.tolist() == present.tolist()
    assert all(torch.equal(left, right) for left, right in zip(split_state, state, strict=True))


@pytest.mark.L0
def test_r09_b0_ttt_backend_isolates_samples_and_resets_selected_state() -> None:
    torch.manual_seed(10)
    backend = TTTLocalMemoryBackend(evidence_dim=3, local_dim=2)
    evidence = torch.randn(3, 6, 3)
    mask = torch.tensor([[True] * 6, [True, True, True, True, False, False], [False] * 6])
    token, state, present = backend.replay(evidence, mask)

    permutation = torch.tensor([1, 0, 2])
    permuted_token, permuted_state, permuted_present = backend.replay(evidence[permutation], mask[permutation])
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted_token[inverse], token)
    assert torch.equal(permuted_present[inverse], present)
    assert all(torch.equal(value[inverse], reference) for value, reference in zip(permuted_state, state, strict=True))

    changed = evidence.clone()
    changed[1] = 1e5
    isolated_token, isolated_state, _ = backend.replay(changed, mask)
    assert torch.equal(isolated_token[0], token[0])
    assert all(torch.equal(value[0], reference[0]) for value, reference in zip(isolated_state, state, strict=True))

    continuation_token, continuation_state, continuation_present = backend.replay(
        torch.zeros(3, 2, 3), torch.zeros(3, 2, dtype=torch.bool), state
    )
    assert torch.equal(continuation_token, token)
    assert torch.equal(continuation_present, present)
    assert all(torch.equal(value, reference) for value, reference in zip(continuation_state, state, strict=True))

    done = torch.tensor([False, True, False])
    partial = backend.reset_mask(state, done)
    assert all(torch.equal(value[~done], reference[~done]) for value, reference in zip(partial, state, strict=True))
    assert all(torch.count_nonzero(value[done]) == 0 for value in partial)
    assert partial[3].tolist() == [True, False, False]
    full = backend.reset_mask(state, torch.ones(3, dtype=torch.bool))
    assert all(torch.count_nonzero(value) == 0 for value in full)


@pytest.mark.L0
def test_r09_b1_ttt_rejects_outer_no_grad_without_mutating_state() -> None:
    backend = TTTLocalMemoryBackend(evidence_dim=3, local_dim=2)
    evidence = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    _, state, _ = backend.replay(evidence, mask)
    reference = tuple(value.clone() for value in state)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context, pytest.raises(RuntimeError, match="training grad-mode"):
            backend.replay(evidence, mask, state)
        assert all(torch.equal(value, expected) for value, expected in zip(state, reference, strict=True))


def _continual_ttt_core(*, ttt_tbptt_steps: int = 4, k_local: int = 1) -> ContinualTTTLocalMemoryCore:
    return ContinualTTTLocalMemoryCore(
        evidence_dim=5,
        local_dim=3,
        ttt_dim=4,
        fast_hidden_dim=6,
        inner_lr=0.2,
        ttt_tbptt_steps=ttt_tbptt_steps,
        k_local=k_local,
    )


def _continual_ttt_manual_read(value: torch.Tensor, state: ContinualTTTFastState) -> torch.Tensor:
    hidden = F.silu(F.linear(value, state.fast_in_weight, state.fast_in_bias))
    return F.linear(hidden, state.fast_out_weight, state.fast_out_bias)


@pytest.mark.L0
def test_continual_ttt_constructor_validation_and_configurable_tbptt() -> None:
    assert ContinualTTTLocalMemoryCore().ttt_tbptt_steps == 16
    actual_steps = [ContinualTTTLocalMemoryCore(ttt_tbptt_steps=value).ttt_tbptt_steps for value in (1, 7, 16)]
    assert actual_steps == [1, 7, 16]
    for field in ("evidence_dim", "local_dim", "ttt_dim", "fast_hidden_dim", "ttt_tbptt_steps", "k_local"):
        for value in (True, 0, -1):
            with pytest.raises(ValueError, match=field):
                ContinualTTTLocalMemoryCore(**{field: value})
    for value in (0, -1, float("nan"), float("inf"), "invalid"):
        with pytest.raises(ValueError, match="inner_lr"):
            ContinualTTTLocalMemoryCore(inner_lr=value)


@pytest.mark.L0
def test_continual_ttt_parameter_registry_and_element_count() -> None:
    core = ContinualTTTLocalMemoryCore()
    expected = {
        "key_proj.weight",
        "key_proj.bias",
        "query_proj.weight",
        "query_proj.bias",
        "value_proj.weight",
        "value_proj.bias",
        "slot_queries",
        "w0_fast_in_weight",
        "w0_fast_in_bias",
        "w0_fast_out_weight",
        "w0_fast_out_bias",
    }
    assert set(dict(core.named_parameters())) == expected
    assert set(core.state_dict()) == expected
    assert sum(parameter.numel() for parameter in core.parameters()) == 53_632
    assert "inner_lr" not in core.state_dict() and "ttt_tbptt_steps" not in core.state_dict()


@pytest.mark.L0
def test_continual_ttt_initial_state_shape_storage_and_w0_gradient() -> None:
    torch.manual_seed(20)
    core = _continual_ttt_core()
    state = core.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    expected_shapes = ((2, 6, 4), (2, 6), (2, 3, 6), (2, 3))
    for value, w0, shape in zip(state, core._w0, expected_shapes, strict=True):
        assert value.shape == shape and value.dtype == torch.float32 and value.device.type == "cpu"
        assert torch.equal(value[0], w0) and torch.equal(value[1], w0)
        assert value.stride(0) != 0 and value[0].data_ptr() != value[1].data_ptr()
    sum(value.sum() for value in state).backward()
    assert all(parameter.grad is not None for parameter in core._w0)
    with pytest.raises(ValueError, match="batch"):
        core.initial_state(False)


@pytest.mark.L0
def test_continual_ttt_step_matches_manual_kvb_and_is_functional() -> None:
    torch.manual_seed(21)
    core = _continual_ttt_core()
    state = core.initial_state(1)
    original = tuple(value.clone() for value in state)
    key = torch.randn(1, 4)
    query = torch.randn(1, 4)
    value = torch.randn(1, 3)
    work = ContinualTTTFastState(*(member[0] for member in state))
    loss = (_continual_ttt_manual_read(key[0], work) - value[0]).square().mean()
    gradients = torch.autograd.grad(loss, work, create_graph=True)
    expected_state = ContinualTTTFastState(
        *(member - core.inner_lr * gradient for member, gradient in zip(work, gradients, strict=True))
    )
    expected_token = _continual_ttt_manual_read(query[0], expected_state)
    token, state_out, present = core.step_projected(
        key_t=key, query_t=query, value_t=value, state_in=state, valid=torch.tensor([True]), create_graph=True
    )
    torch.testing.assert_close(token[0, 0], expected_token, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(state_out, expected_state, strict=True):
        torch.testing.assert_close(actual[0], expected, atol=1e-6, rtol=1e-5)
    assert present.tolist() == [True]
    assert all(torch.equal(member, reference) for member, reference in zip(state, original, strict=True))


@pytest.mark.L0
def test_continual_ttt_update_is_batch_independent() -> None:
    torch.manual_seed(22)
    core = _continual_ttt_core()
    key, query, value = (torch.randn(3, width) for width in (4, 4, 3))
    batched = core.step_projected(
        key_t=key,
        query_t=query,
        value_t=value,
        state_in=core.initial_state(3),
        valid=torch.tensor([True, True, False]),
        create_graph=True,
    )
    single = core.step_projected(
        key_t=key[:1],
        query_t=query[:1],
        value_t=value[:1],
        state_in=core.initial_state(1),
        valid=torch.tensor([True]),
        create_graph=True,
    )
    torch.testing.assert_close(batched[0][0], single[0][0], atol=1e-6, rtol=1e-5)
    for batch_member, single_member in zip(batched[1], single[1], strict=True):
        torch.testing.assert_close(batch_member[0], single_member[0], atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_invalid_rows_are_exactly_inert() -> None:
    torch.manual_seed(23)
    core = _continual_ttt_core()
    state = core.initial_state(2)
    key, query, value = (torch.randn(2, width) for width in (4, 4, 3))
    token, state_out, present = core.step_projected(
        key_t=key, query_t=query, value_t=value, state_in=state, valid=torch.tensor([True, False]), create_graph=True
    )
    assert present.tolist() == [True, False] and torch.count_nonzero(token[1]) == 0
    assert all(torch.equal(output[1], source[1]) for output, source in zip(state_out, state, strict=True))
    assert any(not torch.equal(output[0], source[0]) for output, source in zip(state_out, state, strict=True))


@pytest.mark.L0
def test_continual_ttt_reset_restores_complete_w0_per_sample() -> None:
    torch.manual_seed(24)
    core = _continual_ttt_core()
    state = ContinualTTTFastState(*(value + torch.randn_like(value) for value in core.initial_state(3)))
    done = torch.tensor([False, True, False])
    reset = core.reset_mask(state, done)
    for output, source, w0 in zip(reset, state, core._w0, strict=True):
        assert torch.equal(output[~done], source[~done]) and torch.equal(output[1], w0)
    full = core.reset_mask(state, torch.ones(3, dtype=torch.bool))
    assert all(
        torch.equal(output, w0.unsqueeze(0).expand_as(output))
        for output, w0 in zip(full, core._w0, strict=True)
    )


@pytest.mark.L0
def test_continual_ttt_detach_preserves_values_and_cuts_graph() -> None:
    core = _continual_ttt_core()
    state = core.initial_state(2)
    detached = core.detach_state(state)
    assert all(
        torch.equal(output, source) and output.grad_fn is None and not output.requires_grad
        for output, source in zip(detached, state, strict=True)
    )
    assert all(output.data_ptr() == source.data_ptr() for output, source in zip(detached, state, strict=True))


@pytest.mark.L0
def test_continual_ttt_scan_matches_timestep_steps() -> None:
    torch.manual_seed(25)
    core = _continual_ttt_core()
    evidence = torch.randn(2, 4, 5)
    valid = torch.tensor([[True, True, False, True], [False, True, True, False]])
    scan_token, scan_state, scan_present = core.scan_segment(evidence, valid)
    state = core.initial_state(2)
    tokens, presents = [], []
    for index in range(4):
        token, state, present = core.step(evidence[:, index], state, valid[:, index])
        tokens.append(token[:, 0])
        presents.append(present)
    torch.testing.assert_close(scan_token, torch.stack(tokens, dim=1), atol=1e-6, rtol=1e-5)
    assert torch.equal(scan_present, torch.stack(presents, dim=1))
    for actual, expected in zip(scan_state, state, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_detached_carry_rebuilds_graph_and_tbptt_does_not_change_values() -> None:
    torch.manual_seed(26)
    core = _continual_ttt_core(ttt_tbptt_steps=4)
    evidence = torch.randn(2, 4, 5)
    valid = torch.ones(2, 4, dtype=torch.bool)
    _, first_state, _ = core.scan_segment(evidence[:, :2], valid[:, :2])
    carry = core.detach_state(first_state)
    second_token, second_state, _ = core.scan_segment(evidence[:, 2:], valid[:, 2:], carry)
    assert all(value.requires_grad and value.grad_fn is not None for value in second_state)
    wider = _continual_ttt_core(ttt_tbptt_steps=7)
    wider.load_state_dict(core.state_dict())
    wider_token, wider_state, _ = wider.scan_segment(evidence[:, 2:], valid[:, 2:], wider.detach_state(first_state))
    torch.testing.assert_close(second_token, wider_token, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(second_state, wider_state, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_rejects_invalid_inputs_before_state_mutation() -> None:
    torch.manual_seed(27)
    core = _continual_ttt_core(ttt_tbptt_steps=2)
    state = core.initial_state(1)
    reference = tuple(value.clone() for value in state)
    nonfinite_state = ContinualTTTFastState(state[0] * float("nan"), state[1], state[2], state[3])
    bad_calls = (
        lambda: core.scan_segment(torch.empty(1, 0, 5), torch.empty(1, 0, dtype=torch.bool), state),
        lambda: core.scan_segment(torch.randn(1, 3, 5), torch.ones(1, 3, dtype=torch.bool), state),
        lambda: core.step(torch.full((1, 5), float("nan")), state, torch.ones(1, dtype=torch.bool)),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.full((1, 4), float("inf")),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.randn(1, 5),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4, dtype=torch.float64),
            query_t=torch.randn(1, 4),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.randn(1, 4),
            value_t=torch.randn(1, 3),
            state_in=nonfinite_state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.randn(1, 4),
            value_t=torch.randn(1, 3),
            state_in=ContinualTTTFastState(state[0], state[1].double(), state[2], state[3]),
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.randn(1, 4),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.empty(1, device="meta", dtype=torch.bool),
            create_graph=True,
        ),
        lambda: core.step_projected(
            key_t=torch.randn(1, 4),
            query_t=torch.randn(1, 4),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=1,
        ),
    )
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()
        assert all(torch.equal(value, expected) for value, expected in zip(state, reference, strict=True))


@pytest.mark.L0
def test_continual_ttt_outer_loss_reaches_kqv_and_learned_w0() -> None:
    torch.manual_seed(28)
    core = _continual_ttt_core()
    token, _, _ = core.scan_segment(torch.randn(2, 3, 5), torch.ones(2, 3, dtype=torch.bool), create_graph=True)
    token.square().sum().backward()
    groups = (
        (core.key_proj.weight, core.key_proj.bias),
        (core.query_proj.weight, core.query_proj.bias),
        (core.value_proj.weight, core.value_proj.bias),
        core._w0,
    )
    for group in groups:
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in group)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in group)


@pytest.mark.L0
def test_continual_ttt_create_graph_modes_are_numerically_equal() -> None:
    torch.manual_seed(29)
    core = _continual_ttt_core()
    key, query, value = (torch.randn(2, width) for width in (4, 4, 3))
    true_result = core.step_projected(
        key_t=key,
        query_t=query,
        value_t=value,
        state_in=core.initial_state(2),
        valid=torch.ones(2, dtype=torch.bool),
        create_graph=True,
    )
    false_result = core.step_projected(
        key_t=key,
        query_t=query,
        value_t=value,
        state_in=core.initial_state(2),
        valid=torch.ones(2, dtype=torch.bool),
        create_graph=False,
    )
    torch.testing.assert_close(true_result[0], false_result[0], atol=1e-6, rtol=1e-5)
    for left, right in zip(true_result[1], false_result[1], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_rejects_disabled_grad_before_mutation() -> None:
    torch.manual_seed(30)
    core = _continual_ttt_core()
    state = core.initial_state(1)
    reference = tuple(value.clone() for value in state)
    evidence = torch.randn(1, 5)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context, pytest.raises(RuntimeError, match="ordinary grad mode"):
            core.step(evidence, state, torch.tensor([True]))
        assert all(torch.equal(value, expected) for value, expected in zip(state, reference, strict=True))


@pytest.mark.L0
def test_continual_ttt_projected_and_evidence_step_are_identical() -> None:
    torch.manual_seed(31)
    core = _continual_ttt_core()
    evidence = torch.randn(2, 5)
    valid = torch.tensor([True, False])
    state = core.initial_state(2)
    projected = core.project_evidence(evidence)
    direct_result = core.step(evidence, state, valid)
    projected_result = core.step_projected(
        key_t=projected[0], query_t=projected[1], value_t=projected[2], state_in=state, valid=valid, create_graph=True
    )
    torch.testing.assert_close(direct_result[0], projected_result[0], atol=1e-6, rtol=1e-5)
    assert torch.equal(direct_result[2], projected_result[2])
    for left, right in zip(direct_result[1], projected_result[1], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_fast_state_payload_formula() -> None:
    core = ContinualTTTLocalMemoryCore()
    state = core.initial_state(1)
    elements = sum(value.numel() for value in state)
    assert elements == 12_448
    assert elements * torch.tensor([], dtype=torch.float32).element_size() == 49_792
    assert elements * torch.tensor([], dtype=torch.bfloat16).element_size() == 24_896


@pytest.mark.L0
def test_continual_ttt_multi_slot_construction_registry_and_checkpoint_identity() -> None:
    for k_local, count in ((1, 53_632), (4, 53_824), (8, 54_080)):
        core = ContinualTTTLocalMemoryCore(k_local=k_local)
        assert core.k_local == k_local and core.slot_queries.shape == (k_local, 64)
        assert sum(parameter.numel() for parameter in core.parameters()) == count
        assert sum(value.numel() for value in core.initial_state(1)) == 12_448
    assert torch.count_nonzero(ContinualTTTLocalMemoryCore(k_local=1).slot_queries) == 0
    with pytest.raises(RuntimeError, match="slot_queries"):
        ContinualTTTLocalMemoryCore(k_local=4).load_state_dict(ContinualTTTLocalMemoryCore(k_local=1).state_dict())


@pytest.mark.L0
def test_continual_ttt_multi_slot_post_update_read_is_pure_and_invalid_rows_are_inert() -> None:
    torch.manual_seed(32)
    core = _continual_ttt_core(k_local=4)
    state = core.initial_state(2)
    original = tuple(member.clone() for member in state)
    key, query_base, value = (torch.randn(2, width) for width in (4, 4, 3))
    queries = core.project_queries(query_base)
    reads = core.read_many(queries, state)
    for row in range(2):
        expected = torch.stack([_continual_ttt_manual_read(query, ContinualTTTFastState(*(member[row] for member in state))) for query in queries[row]])
        torch.testing.assert_close(reads[row], expected, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(member, reference) for member, reference in zip(state, original, strict=True))

    token, state_out, present = core.step_projected_many(
        key_t=key,
        query_base_t=query_base,
        value_t=value,
        state_in=state,
        valid=torch.tensor([True, False]),
        create_graph=True,
    )
    work = ContinualTTTFastState(*(member[0] for member in state))
    inner = (_continual_ttt_manual_read(key[0], work) - value[0]).square().mean()
    gradients = torch.autograd.grad(inner, work, create_graph=True)
    updated = ContinualTTTFastState(*(member - core.inner_lr * grad for member, grad in zip(work, gradients, strict=True)))
    expected = torch.stack([_continual_ttt_manual_read(query, updated) for query in queries[0]])
    torch.testing.assert_close(token[0], expected, atol=1e-6, rtol=1e-5)
    assert present.tolist() == [True, False] and torch.count_nonzero(token[1]) == 0
    assert all(torch.equal(output[1], source[1]) for output, source in zip(state_out, state, strict=True))


@pytest.mark.L0
def test_continual_ttt_multi_slot_one_write_per_valid_sample_and_legacy_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(33)
    core = _continual_ttt_core(k_local=4)
    key, query_base, value = (torch.randn(3, width) for width in (4, 4, 3))
    original_grad = torch.autograd.grad
    calls = 0

    def counted_grad(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    core.step_projected_many(
        key_t=key,
        query_base_t=query_base,
        value_t=value,
        state_in=core.initial_state(3),
        valid=torch.tensor([True, False, True]),
        create_graph=True,
    )
    assert calls == 2
    with pytest.raises(ValueError, match="step_projected_many"):
        core.step_projected(
            key_t=key,
            query_t=query_base,
            value_t=value,
            state_in=core.initial_state(3),
            valid=torch.ones(3, dtype=torch.bool),
            create_graph=True,
        )
    with pytest.raises(ValueError, match="step_many"):
        core.step(torch.randn(3, 5), core.initial_state(3), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="scan_segment_many"):
        core.scan_segment(torch.randn(3, 2, 5), torch.ones(3, 2, dtype=torch.bool))


@pytest.mark.L0
def test_continual_ttt_k1_wrapper_and_multi_slot_scan_gradients() -> None:
    torch.manual_seed(34)
    one = _continual_ttt_core(k_local=1)
    key, query, value = (torch.randn(2, width) for width in (4, 4, 3))
    legacy = one.step_projected(
        key_t=key, query_t=query, value_t=value, state_in=one.initial_state(2), valid=torch.ones(2, dtype=torch.bool), create_graph=True
    )
    many = one.step_projected_many(
        key_t=key, query_base_t=query, value_t=value, state_in=one.initial_state(2), valid=torch.ones(2, dtype=torch.bool), create_graph=True
    )
    torch.testing.assert_close(legacy[0], many[0], atol=1e-6, rtol=1e-5)
    for left, right in zip(legacy[1], many[1], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    core = _continual_ttt_core(k_local=4)
    evidence = torch.randn(2, 3, 5)
    token, state, present = core.scan_segment_many(evidence, torch.ones(2, 3, dtype=torch.bool), create_graph=True)
    assert token.shape == (2, 3, 4, 3) and present.shape == (2, 3)
    token.square().sum().backward()
    for parameter in (*core.query_proj.parameters(), core.slot_queries, *core.key_proj.parameters(), *core.value_proj.parameters(), *core._w0):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
    assert all(value.requires_grad for value in state)


@pytest.mark.L0
def test_continual_ttt_multi_slot_permutation_isolation_and_scan_equivalence() -> None:
    torch.manual_seed(35)
    core = _continual_ttt_core(k_local=4)
    state = core.initial_state(1)
    query_base = torch.randn(1, 4)
    reads = core.read_many(core.project_queries(query_base), state)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        core.slot_queries.copy_(core.slot_queries[permutation])
    permuted_reads = core.read_many(core.project_queries(query_base), state)
    torch.testing.assert_close(permuted_reads, reads[:, permutation], atol=1e-6, rtol=1e-5)
    changed_queries = core.project_queries(query_base).clone()
    changed_queries[:, 1] += 0.5
    changed_reads = core.read_many(changed_queries, state)
    assert torch.equal(changed_reads[:, [0, 2, 3]], permuted_reads[:, [0, 2, 3]])

    evidence = torch.randn(2, 3, 5)
    valid = torch.tensor([[True, True, False], [False, True, True]])
    scan_tokens, scan_state, scan_present = core.scan_segment_many(evidence, valid)
    state = core.initial_state(2)
    tokens, present = [], []
    for index in range(evidence.shape[1]):
        token, state, step_present = core.step_many(evidence[:, index], state, valid[:, index])
        tokens.append(token)
        present.append(step_present)
    torch.testing.assert_close(scan_tokens, torch.stack(tokens, dim=1), atol=1e-6, rtol=1e-5)
    assert torch.equal(scan_present, torch.stack(present, dim=1))
    for actual, expected in zip(scan_state, state, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.L0
def test_continual_ttt_multi_slot_invalid_rows_skip_fast_read_even_when_finite_values_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(36)
    core = _continual_ttt_core(k_local=4)
    state = core.initial_state(2)
    for member in state:
        member[1].fill_(1e20)
    original = tuple(member.clone() for member in state)
    original_fast_mlp = core._fast_mlp
    call_shapes: list[tuple[int, ...]] = []

    def counted_fast_mlp(value: torch.Tensor, fast_state: ContinualTTTFastState) -> torch.Tensor:
        call_shapes.append(tuple(value.shape))
        return original_fast_mlp(value, fast_state)

    monkeypatch.setattr(core, "_fast_mlp", counted_fast_mlp)
    token, state_out, present = core.step_projected_many(
        key_t=torch.randn(2, 4),
        query_base_t=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1e20, 1e20, 1e20, 1e20]]),
        value_t=torch.randn(2, 3),
        state_in=state,
        valid=torch.tensor([True, False]),
        create_graph=True,
    )
    assert call_shapes == [(4,), (4, 4)]
    assert present.tolist() == [True, False]
    assert torch.isfinite(token[1]).all() and torch.equal(token[1], torch.zeros_like(token[1]))
    assert all(torch.equal(output[1], source[1]) for output, source in zip(state_out, original, strict=True))


@pytest.mark.L0
def test_continual_ttt_multi_slot_public_validation_fails_before_any_update(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _continual_ttt_core(k_local=4)
    state = core.initial_state(1)
    reference = tuple(member.clone() for member in state)
    base = torch.randn(1, 4)
    for value in (
        torch.randn(1, 1, 4),
        torch.randn(1, 5),
        torch.randn(1, 4, dtype=torch.float64),
        torch.full((1, 4), float("nan")),
        torch.empty(1, 4, device="meta"),
    ):
        with pytest.raises(ValueError):
            core.project_queries(value)
    queries = core.project_queries(base)
    for value in (
        torch.randn(1, 3, 4),
        torch.randn(1, 4, 4, dtype=torch.float64),
        torch.full((1, 4, 4), float("inf")),
        torch.empty(1, 4, 4, device="meta"),
    ):
        with pytest.raises(ValueError):
            core.read_many(value, state)
    with pytest.raises(ValueError):
        core.read_many(queries, ContinualTTTFastState(state[0], state[1].double(), state[2], state[3]))

    def unexpected_grad(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("invalid multi-slot input must not update state")

    monkeypatch.setattr(torch.autograd, "grad", unexpected_grad)
    with pytest.raises(ValueError):
        core.step_projected_many(
            key_t=torch.randn(1, 4),
            query_base_t=torch.randn(1, 4, dtype=torch.float64),
            value_t=torch.randn(1, 3),
            state_in=state,
            valid=torch.ones(1, dtype=torch.bool),
            create_graph=True,
        )
    with pytest.raises(ValueError):
        core.step_many(torch.full((1, 5), float("nan")), state, torch.ones(1, dtype=torch.bool))
    assert all(torch.equal(member, expected) for member, expected in zip(state, reference, strict=True))


@pytest.mark.L0
def test_continual_ttt_transition_row_selective_detach_preserves_current_token_gradients() -> None:
    torch.manual_seed(37)
    core = _continual_ttt_core(ttt_tbptt_steps=2, k_local=2)
    runtime = ContinualTTTFastStateTransition(core)
    state1 = ContinualTTTFastState(
        *(value.detach().clone().requires_grad_(True) for value in core.initial_state(2))
    )
    counter1 = torch.tensor([1, 0], dtype=torch.int64)
    evidence2 = torch.randn(2, 5, requires_grad=True)
    token2, state2, present2, counter2 = runtime.step(
        evidence2,
        state1,
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        counter1,
    )
    _, updated2, _ = core.step_many(evidence2, state1, torch.tensor([True, True]), create_graph=True)

    assert token2.shape == (2, 2, 3) and present2.tolist() == [True, True] and counter2.tolist() == [0, 1]
    for output, updated in zip(state2, updated2, strict=True):
        torch.testing.assert_close(output, updated, atol=1e-6, rtol=1e-5)

    detached_gradients = torch.autograd.grad(
        sum(value[0].square().sum() for value in state2), tuple(state1), allow_unused=True, retain_graph=True
    )
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in detached_gradients)
    live_gradients = torch.autograd.grad(
        sum(value[1].square().sum() for value in state2), tuple(state1), allow_unused=True, retain_graph=True
    )
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient[1]) > 0
        for gradient in live_gradients
    )

    boundary_token, _, _, _ = runtime.step(
        evidence2[:1],
        core.initial_state(1),
        torch.tensor([True]),
        torch.tensor([False]),
        torch.tensor([1], dtype=torch.int64),
    )
    slow_gradients = torch.autograd.grad(boundary_token.square().sum(), tuple(core.parameters()), allow_unused=True)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in slow_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in slow_gradients if gradient is not None)
    assert {name for name, _ in runtime.named_parameters()} == {
        f"core.{name}" for name, _ in core.named_parameters()
    }


@pytest.mark.L0
def test_continual_ttt_transition_n1_returns_updated_read_and_detached_carry() -> None:
    torch.manual_seed(38)
    core = _continual_ttt_core(ttt_tbptt_steps=1, k_local=2)
    runtime = ContinualTTTFastStateTransition(core)
    evidence = torch.randn(1, 5)
    reference_token, reference_state, reference_present = core.step_many(
        evidence, core.initial_state(1), torch.tensor([True]), create_graph=True
    )
    token, state, present, counter = runtime.step(
        evidence,
        None,
        torch.tensor([True]),
        torch.tensor([False]),
        torch.zeros(1, dtype=torch.int64),
    )
    torch.testing.assert_close(token, reference_token, atol=1e-6, rtol=1e-5)
    assert present.tolist() == reference_present.tolist() == [True] and counter.tolist() == [0]
    for output, expected in zip(state, reference_state, strict=True):
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        gradient = torch.autograd.grad(output.square().sum(), core._w0, allow_unused=True, retain_graph=True)
        assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)


@pytest.mark.L0
def test_continual_ttt_transition_counter_progression_for_default_and_nondefault_lengths() -> None:
    torch.manual_seed(39)
    for steps, expected in ((3, [1, 2, 0, 1, 2]), (16, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])):
        core = _continual_ttt_core(ttt_tbptt_steps=steps)
        runtime = ContinualTTTFastStateTransition(core)
        state: ContinualTTTFastState | None = None
        counter = torch.zeros(1, dtype=torch.int64)
        observed = []
        for _ in expected:
            _, state, present, counter = runtime.step(
                torch.randn(1, 5), state, torch.tensor([True]), torch.tensor([False]), counter
            )
            assert present.tolist() == [True]
            observed.append(counter.item())
        assert observed == expected


@pytest.mark.L0
def test_continual_ttt_transition_reset_invalid_rows_are_exactly_isolated() -> None:
    torch.manual_seed(40)
    core = _continual_ttt_core(ttt_tbptt_steps=3)
    runtime = ContinualTTTFastStateTransition(core)
    state = ContinualTTTFastState(*(value + torch.randn_like(value) for value in core.initial_state(2)))
    token, state_out, present, counter = runtime.step(
        torch.randn(2, 5),
        state,
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        torch.tensor([1, 2], dtype=torch.int64),
    )
    assert torch.equal(token, torch.zeros_like(token)) and present.tolist() == [False, False] and counter.tolist() == [0, 2]
    for output, source, w0 in zip(state_out, state, core._w0, strict=True):
        torch.testing.assert_close(output[0], w0, atol=0, rtol=0)
        torch.testing.assert_close(output[1], source[1], atol=0, rtol=0)


@pytest.mark.L0
def test_continual_ttt_transition_counter_grammar_fails_before_core_update(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _continual_ttt_core(ttt_tbptt_steps=2)
    runtime = ContinualTTTFastStateTransition(core)
    calls = 0

    def unexpected_step_many(*args: object, **kwargs: object) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid counter grammar must not reach core.step_many")

    monkeypatch.setattr(core, "step_many", unexpected_step_many)
    evidence = torch.randn(1, 5)
    valid = torch.tensor([True])
    done = torch.tensor([False])
    state = core.initial_state(1)
    bad_calls = (
        lambda: runtime.step(evidence, state, valid, done, torch.tensor([-1], dtype=torch.int64)),
        lambda: runtime.step(evidence, state, valid, done, torch.tensor([2], dtype=torch.int64)),
        lambda: runtime.step(evidence, state, valid, done, torch.zeros(1, dtype=torch.int32)),
        lambda: runtime.step(evidence, state, valid, done, torch.zeros(1, 1, dtype=torch.int64)),
        lambda: runtime.step(evidence, state, valid, done, torch.empty(1, dtype=torch.int64, device="meta")),
        lambda: runtime.step(evidence, None, valid, done, torch.ones(1, dtype=torch.int64)),
    )
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()
    assert calls == 0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import LocalEvidenceEncoder, RecurrentLocalMemoryBackend, StatelessLocalReplayReadout


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
    _, first, _ = backend.replay(evidence[:, :2], mask[:, :2])
    _, second, _ = backend.replay(evidence[:, 2:], mask[:, 2:], (first[0].detach(), first[1]))
    torch.testing.assert_close(second[0], latent, rtol=0, atol=1e-6)
    carry_mask = torch.tensor([[True, False], [False, False], [False, False]])
    _, carried, _ = backend.replay(evidence[:, :2], carry_mask)
    carry_tokens, _, carry_present = backend.replay(evidence[:, 2:], torch.zeros_like(carry_mask), (carried[0].detach(), carried[1]))
    assert carry_present.tolist() == [True, False, False]
    torch.testing.assert_close(carry_tokens[0], carried[0][0, None], rtol=0, atol=1e-6)

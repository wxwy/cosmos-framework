# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import LocalEvidenceEncoder


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

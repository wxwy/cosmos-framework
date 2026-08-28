import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    LocalEvidenceEncoder,
    LocalHistoryRuntime,
    StatelessLocalReplayReadout,
)


def _runtime() -> LocalHistoryRuntime:
    return LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=5, hidden_dim=8),
    )


def _inputs(batch: int = 3, horizon: int = 4):
    mask = torch.zeros(batch, horizon, dtype=torch.bool)
    if horizon:
        mask[0, : min(2, horizon)] = True
        mask[2, :] = True
    return dict(
        history_visual_summary=torch.randn(batch, horizon, 4),
        local_history_action=torch.randn(batch, horizon, 3),
        history_age_steps=torch.arange(horizon).repeat(batch, 1),
        history_dt_s=torch.ones(batch, horizon, 1),
        history_mask=mask,
    )


def test_runtime_gates_mixed_batch_and_trace_shapes() -> None:
    tokens, present, evidence = _runtime()(**_inputs())
    assert tuple(tokens.shape) == (3, 1, 5)
    assert present.tolist() == [True, False, True]
    assert tuple(evidence.shape) == (3, 4, 8)
    assert torch.equal(tokens[1], torch.zeros_like(tokens[1]))
    assert torch.isfinite(tokens).all()


def test_runtime_h0_control_is_local_absent() -> None:
    inputs = _inputs(horizon=1)
    inputs["history_mask"] = torch.zeros(3, 1, dtype=torch.bool)
    tokens, present, _ = _runtime()(**inputs)
    assert not present.any()
    assert torch.equal(tokens, torch.zeros_like(tokens))


def test_runtime_rejects_nonfinite_dt() -> None:
    inputs = _inputs()
    inputs["history_dt_s"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="history_dt_s must be finite"):
        _runtime()(**inputs)

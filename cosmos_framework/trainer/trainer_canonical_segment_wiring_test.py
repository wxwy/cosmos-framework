from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.production_segment_wiring_test import _fixture
from cosmos_framework.trainer import ImaginaireTrainer


def test_canonical_trainer_delegates_then_commits_exact_result() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    trainer = object.__new__(ImaginaireTrainer)
    output = {
        "canonical_segment_forward": forward, "canonical_wiring": wiring,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": forward.result.local_tokens.sum(),
        "auxiliary_loss": torch.zeros((), requires_grad=True), "actual_n_valid": 1,
    }
    trainer._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == (identity,)


def test_canonical_trainer_rejects_mismatched_result_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    other_wiring, other_segment, other_identity, other_transaction = _fixture()
    other_forward = other_wiring.prepare(other_segment, other_identity, other_transaction)
    output = {
        "canonical_segment_forward": other_forward, "canonical_wiring": wiring,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": torch.ones((), requires_grad=True),
        "auxiliary_loss": torch.zeros((), requires_grad=True), "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()

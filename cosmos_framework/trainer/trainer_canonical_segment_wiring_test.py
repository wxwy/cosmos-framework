from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.mot.production_segment_wiring import run_native_forward_for_test
from cosmos_framework.model.generator.mot.production_segment_wiring_test import _fixture
from cosmos_framework.trainer import ImaginaireTrainer


def _model_marker_output(wiring, segment, identity, transaction):
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(local_ttt_enabled=True)
    return model.training_step(
        {
            "canonical_local_memory_segment": True,
            "canonical_segment": segment,
            "canonical_identity": identity,
            "canonical_transaction": transaction,
            "canonical_wiring": wiring,
            "canonical_member_index": 0,
        },
        0,
    )


def test_canonical_trainer_delegates_then_commits_exact_result() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    trainer = object.__new__(ImaginaireTrainer)
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


def test_canonical_trainer_rejects_same_adapter_substitute_wiring_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    true_local = wiring.local_slow_parameters[0]
    substitute_local = torch.nn.Parameter(torch.ones(()))
    substitute = type(wiring)(wiring.adapter, (substitute_local,))
    true_local.grad = torch.ones(())
    substitute_local.grad = torch.ones(())
    primary, auxiliary = run_native_forward_for_test(
        forward.payloads, forward.locals, forward.result.local_tokens, wiring.local_slow_parameters
    )
    output = {
        "canonical_segment_forward": forward, "canonical_wiring": substitute,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": primary,
        "auxiliary_loss": auxiliary, "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()
    torch.testing.assert_close(true_local.grad, torch.ones(()))
    torch.testing.assert_close(substitute_local.grad, torch.ones(()))


def test_canonical_trainer_rejects_external_plan_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    output["canonical_plan"] = transaction.plan
    with pytest.raises(RuntimeError, match="external plan"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()


def test_canonical_trainer_rejects_missing_capability_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    del output["canonical_identity"]
    with pytest.raises(RuntimeError, match="capability is incomplete"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()


def test_canonical_trainer_rejects_stale_result_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    stale = wiring.prepare(segment, identity, transaction)
    current = wiring.prepare(segment, identity, transaction)
    primary, auxiliary = run_native_forward_for_test(
        stale.payloads, stale.locals, stale.result.local_tokens, wiring.local_slow_parameters
    )
    output = {
        "canonical_segment_forward": stale, "canonical_wiring": wiring,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": primary,
        "auxiliary_loss": auxiliary, "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert current.result is wiring.adapter.pending_scan[2]
    assert transaction.snapshot().completed_members == ()


@pytest.mark.parametrize(
    ("primary", "auxiliary", "code"),
    (
        (torch.tensor(float("nan"), requires_grad=True), torch.zeros((), requires_grad=True), "LOCAL_MEM_NUMERICAL_FAILURE"),
        (torch.ones(()), torch.zeros(()), "LOCAL_MEM_OUTER_FAILURE"),
    ),
)
def test_canonical_trainer_failure_regression_clears_grad_without_commit(primary, auxiliary, code: str) -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    parameter = wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    output["primary_consumer_mean"] = primary
    output["auxiliary_loss"] = auxiliary
    with pytest.raises(RuntimeError, match=code):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    snapshot = transaction.snapshot()
    assert snapshot.completed_members == () and snapshot.terminal_failure_code == code
    assert snapshot.slow_grads_cleared and parameter.grad is None
    assert wiring.adapter.committed_snapshot() == ()


def test_two_step_marker_to_trainer_keeps_visible_local_primary_exactly_once() -> None:
    wiring, segment, identity, transaction = _fixture(two_steps=True)
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    expected = sum(token.sum() for token in output["canonical_segment_forward"].locals if token is not None)
    assert expected.abs() > 1e-6
    torch.testing.assert_close(output["primary_consumer_mean"], expected)
    object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == (identity,)

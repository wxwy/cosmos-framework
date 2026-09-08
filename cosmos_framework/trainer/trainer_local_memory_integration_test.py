import pytest
import torch

from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan
from cosmos_framework.trainer import ImaginaireTrainer


def test_local_memory_segment_backward_owns_single_primary_aux_scaling() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    primary = torch.tensor(6.0, requires_grad=True)
    auxiliary = torch.tensor(4.0, requires_grad=True)
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(2,))

    events: list[str] = []
    loss = trainer._run_local_memory_segment_backward(
        plan, 0, primary, auxiliary, 2, identity_valid=True,
        commit_fast=lambda: events.append("commit"), clear_slow_grads=lambda: events.append("clear"),
    )

    assert loss.item() == 10.0
    assert primary.grad.item() == 1.0
    assert auxiliary.grad.item() == 1.0
    assert events == ["commit"]


def test_local_memory_segment_terminal_failures_clear_without_fast_commit() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(1,))
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            identity_valid=False, commit_fast=lambda: events.append("commit"), clear_slow_grads=lambda: events.append("clear"),
        )
    assert events == ["clear"]


def test_local_memory_segment_planned_actual_mismatch_is_identity_terminal() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(2,))
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            identity_valid=True, commit_fast=lambda: events.append("commit"), clear_slow_grads=lambda: events.append("clear"),
        )
    assert events == ["clear"]

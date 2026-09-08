import torch

from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan
from cosmos_framework.trainer import ImaginaireTrainer


def test_local_memory_segment_backward_owns_single_primary_aux_scaling() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    primary = torch.tensor(6.0, requires_grad=True)
    auxiliary = torch.tensor(4.0, requires_grad=True)
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(2,))

    loss = trainer._run_local_memory_segment_backward(plan, 0, primary, auxiliary, 2)

    assert loss.item() == 10.0
    assert primary.grad.item() == 1.0
    assert auxiliary.grad.item() == 1.0

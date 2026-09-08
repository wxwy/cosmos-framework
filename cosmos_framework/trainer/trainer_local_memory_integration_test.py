import pytest
import torch

from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentIdentity,
)
from cosmos_framework.trainer import ImaginaireTrainer


def test_local_memory_segment_backward_owns_single_primary_aux_scaling() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    primary = torch.tensor(6.0, requires_grad=True)
    auxiliary = torch.tensor(4.0, requires_grad=True)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(2,))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    loss = trainer._run_local_memory_segment_backward(
        plan, 0, primary, auxiliary, 2, transaction=transaction, identity=identity, clear_slow_grads=lambda: None,
    )

    assert loss.item() == 10.0
    assert primary.grad.item() == 1.0
    assert auxiliary.grad.item() == 1.0
    assert transaction.snapshot().completed_members == (identity,)


def test_local_memory_segment_terminal_failures_clear_without_fast_commit() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(1,))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            transaction=transaction, identity=SegmentIdentity(1, "other", "suite", 0, 0, "digest"), clear_slow_grads=lambda: events.append("clear"),
        )
    assert events == ["clear"]
    snapshot = transaction.snapshot()
    assert snapshot.terminal_failure_code == "LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"
    assert snapshot.remaining_members_suppressed


def test_local_memory_segment_planned_actual_mismatch_is_identity_terminal() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(members=((0, "episode", 0),), planned_n_valid=(2,))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            transaction=transaction, identity=identity, clear_slow_grads=lambda: events.append("clear"),
        )
    assert events == ["clear"]


def test_local_memory_segment_transient_recovers_exact_suffix_and_preserves_prior_fast_commit() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identities = (
        SegmentIdentity(0, "episode", "suite", 0, 0, "digest"),
        SegmentIdentity(0, "episode", "suite", 1, 0, "digest"),
    )
    plan = GAWindowPlan(tuple((item.slot_id, item.episode_id, item.cursor) for item in identities), (2, 3), plan_chain_id="chain")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    for identity in identities:
        scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    trainer._run_local_memory_segment_backward(
        plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 2,
        transaction=transaction, identity=identities[0], clear_slow_grads=lambda: None,
    )
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_SUFFIX_RECOVERY"):
        trainer._run_local_memory_segment_backward(
            plan, 1, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 3,
            transaction=transaction, identity=identities[1], clear_slow_grads=lambda: events.append("clear"),
            failure_kind="LOAD_DECODE_TRANSIENT",
        )
    snapshot = transaction.snapshot()
    assert events == ["clear"]
    assert snapshot.completed_members == (identities[0],)
    assert snapshot.suffix_recovery is not None and snapshot.suffix_recovery.members == (plan.members[1],)
    assert snapshot.terminal_failure_code is None and not snapshot.remaining_members_suppressed
    first_suffix = snapshot.suffix_recovery
    with pytest.raises(RuntimeError, match="closed"):
        transaction.recover_transient(1)
    with pytest.raises(RuntimeError, match="closed"):
        transaction.successful_backward(1, identities[1], 3)
    assert transaction.snapshot().suffix_recovery is first_suffix


@pytest.mark.parametrize(
    ("failure_kind", "expected"),
    [("LOAD_DECODE_TRANSIENT", "LOCAL_MEM_RETRY_EXHAUSTED"), ("NUMERICAL", "LOCAL_MEM_NUMERICAL_FAILURE"), ("OUTER", "LOCAL_MEM_OUTER_FAILURE")],
)
def test_local_memory_segment_terminal_taxonomy_clears_and_suppresses(failure_kind: str, expected: str) -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(((0, "episode", 0),), (1,), attempt=1 if failure_kind == "LOAD_DECODE_TRANSIENT" else 0)
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    events: list[str] = []
    with pytest.raises(RuntimeError, match=expected):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            transaction=transaction, identity=identity, clear_slow_grads=lambda: events.append("clear"), failure_kind=failure_kind,
        )
    snapshot = transaction.snapshot()
    assert events == ["clear"] and snapshot.completed_members == ()
    assert snapshot.terminal_failure_code == expected and snapshot.remaining_members_suppressed


def test_local_memory_segment_grad_scaler_skip_clears_slow_side_without_fast_commit() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity])
    transaction = LocalMemoryTransaction(plan, scheduler)
    events: list[str] = []
    with pytest.raises(RuntimeError, match="LOCAL_MEM_GRAD_SCALER_SKIP"):
        trainer._run_local_memory_segment_backward(
            plan, 0, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 1,
            transaction=transaction, identity=identity, clear_slow_grads=lambda: events.append("clear"), grad_scaler_skip=True,
        )
    snapshot = transaction.snapshot()
    assert events == ["clear"] and snapshot.completed_members == ()
    assert snapshot.slow_grads_cleared and snapshot.slow_optimizer_steps == snapshot.slow_lr_scheduler_steps == 0
    with pytest.raises(RuntimeError, match="closed"):
        transaction.slow_optimizer_step_succeeded()


def test_local_memory_segment_numerical_and_backward_failures_are_terminal() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    for primary, expected in ((torch.tensor(float("nan"), requires_grad=True), "LOCAL_MEM_NUMERICAL_FAILURE"),):
        scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
        scheduler.admit([identity]); transaction = LocalMemoryTransaction(plan, scheduler)
        with pytest.raises(RuntimeError, match=expected):
            trainer._run_local_memory_segment_backward(plan, 0, primary, torch.tensor(0.0), 1, transaction=transaction, identity=identity, clear_slow_grads=lambda: None)
        with pytest.raises(RuntimeError, match="closed"):
            transaction.validate_success(0, identity, 1)

    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit([identity]); transaction = LocalMemoryTransaction(plan, scheduler)
    primary = torch.tensor(1.0, requires_grad=True)
    primary.register_hook(lambda _: (_ for _ in ()).throw(RuntimeError("backward boom")))
    with pytest.raises(RuntimeError, match="LOCAL_MEM_OUTER_FAILURE"):
        trainer._run_local_memory_segment_backward(plan, 0, primary, torch.tensor(0.0), 1, transaction=transaction, identity=identity, clear_slow_grads=lambda: None)


def test_local_memory_segment_retry_executes_suffix_and_skip_retains_fast_history() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identities = (SegmentIdentity(0, "episode", "suite", 0, 0, "digest"), SegmentIdentity(0, "episode", "suite", 1, 0, "digest"))
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (2, 3))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    for identity in identities:
        scheduler.admit([identity])
    original = LocalMemoryTransaction(plan, scheduler)
    trainer._run_local_memory_segment_backward(plan, 0, torch.tensor(2.0, requires_grad=True), torch.tensor(1.0, requires_grad=True), 2, transaction=original, identity=identities[0], clear_slow_grads=lambda: None)
    with pytest.raises(RuntimeError, match="LOCAL_MEM_SUFFIX_RECOVERY"):
        trainer._run_local_memory_segment_backward(plan, 1, torch.tensor(5.0, requires_grad=True), torch.tensor(2.0, requires_grad=True), 3, transaction=original, identity=identities[1], clear_slow_grads=lambda: None, failure_kind="LOAD_DECODE_TRANSIENT")
    retry = LocalMemoryTransaction(original.suffix_recovery, scheduler)
    loss = trainer._run_local_memory_segment_backward(retry.plan, 0, torch.tensor(5.0, requires_grad=True), torch.tensor(2.0, requires_grad=True), 3, transaction=retry, identity=identities[1], clear_slow_grads=lambda: None)
    assert loss.item() == 7.0 and retry.plan.ga_effective == 1
    exposure = dict(scheduler.cumulative_valid_consumer_exposure)
    with pytest.raises(RuntimeError, match="LOCAL_MEM_GRAD_SCALER_SKIP"):
        trainer._run_local_memory_segment_backward(retry.plan, 1, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0), 3, transaction=retry, identity=identities[1], clear_slow_grads=lambda: None, grad_scaler_skip=True)
    assert scheduler.cumulative_valid_consumer_exposure == exposure
    assert retry.slow_grads_cleared and retry.slow_optimizer_steps == retry.slow_lr_scheduler_steps == 0
    with pytest.raises(RuntimeError, match="closed"):
        retry.slow_optimizer_step_succeeded()


def test_local_memory_segment_recovery_objective_has_one_ga_division() -> None:
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (2, 3), attempt=1)
    first = plan.objective(0, torch.tensor(2.0), torch.tensor(4.0), 2)
    second = plan.objective(1, torch.tensor(5.0), torch.tensor(4.0), 3)
    torch.testing.assert_close(first + second, torch.tensor(7.8))


def test_local_memory_segment_two_member_suffix_uses_trainer_scaling_once() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    identities = tuple(SegmentIdentity(0, "episode", "suite", index, 0, "digest") for index in range(3))
    original_plan = GAWindowPlan(tuple((item.slot_id, item.episode_id, item.cursor) for item in identities), (2, 3, 4))
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    for identity in identities:
        scheduler.admit([identity])
    original = LocalMemoryTransaction(original_plan, scheduler)
    trainer._run_local_memory_segment_backward(original_plan, 0, torch.tensor(2.0, requires_grad=True), torch.tensor(1.0, requires_grad=True), 2, transaction=original, identity=identities[0], clear_slow_grads=lambda: None)
    with pytest.raises(RuntimeError, match="LOCAL_MEM_SUFFIX_RECOVERY"):
        trainer._run_local_memory_segment_backward(original_plan, 1, torch.tensor(0.0, requires_grad=True), torch.tensor(0.0), 3, transaction=original, identity=identities[1], clear_slow_grads=lambda: None, failure_kind="LOAD_DECODE_TRANSIENT")
    retry = LocalMemoryTransaction(original.suffix_recovery, scheduler)
    losses = [trainer._run_local_memory_segment_backward(retry.plan, index, torch.tensor(primary, requires_grad=True), torch.tensor(auxiliary, requires_grad=True), valid, transaction=retry, identity=identity, clear_slow_grads=lambda: None) for index, (identity, primary, auxiliary, valid) in enumerate(((identities[1], 5.0, 2.0, 3), (identities[2], 7.0, 4.0, 4)))]
    torch.testing.assert_close(sum(losses), torch.tensor((3 / 7) * 5 + 2 / 2 + (4 / 7) * 7 + 4 / 2))

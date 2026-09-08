# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)


def _batch() -> SegmentBatch:
    return SegmentBatch(
        consumer_visual_summary=torch.randn(1, 3, 96),
        consumer_payload=(("s0", "s1", None),),
        consumer_valid=torch.tensor([[True, True, False]]),
        consumer_step=torch.tensor([[0, 1, -1]]),
        evidence_visual_summary_prev=torch.randn(1, 3, 96),
        evidence_executed_action_prev=torch.randn(1, 3, 10),
        evidence_valid=torch.tensor([[False, True, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1]]),
        slot_id=torch.tensor([4]),
        episode_id=("episode",),
        category=("category",),
        segment_provenance=SegmentProvenance("m", "c", "s", 0),
    )


def test_segment_batch_shifted_validation_and_common_gather() -> None:
    batch = _batch()
    batch.validate(16)
    tokens = torch.randn(1, 3, 1, 32)
    payloads, local, identities = batch.gather_consumers(tokens, torch.tensor([[False, True, False]]))
    assert payloads == ["s0", "s1"]
    assert local[0] is None and torch.equal(local[1], tokens[0, 1])
    assert identities == [(4, "episode", 0), (4, "episode", 1)]


def test_segment_batch_rejects_step0_evidence_and_plan_counts() -> None:
    batch = _batch()
    invalid = SegmentBatch(
        consumer_visual_summary=batch.consumer_visual_summary,
        consumer_payload=batch.consumer_payload,
        consumer_valid=batch.consumer_valid,
        consumer_step=batch.consumer_step,
        evidence_visual_summary_prev=batch.evidence_visual_summary_prev,
        evidence_executed_action_prev=batch.evidence_executed_action_prev,
        evidence_valid=torch.tensor([[True, True, False]]),
        evidence_source_step=torch.tensor([[0, 0, -1]]),
        slot_id=batch.slot_id,
        episode_id=batch.episode_id,
        category=batch.category,
        segment_provenance=batch.segment_provenance,
    )
    with pytest.raises(ValueError, match="step0"):
        invalid.validate(16)
    plan = GAWindowPlan(((4, "episode", 0), (4, "episode", 1)), (1, 3))
    assert torch.equal(plan.objective(1, torch.tensor(4.0), torch.tensor(2.0), 3), torch.tensor(4.0))
    with pytest.raises(ValueError, match="actual"):
        plan.objective(0, torch.tensor(1.0), torch.tensor(1.0), 2)


def test_segment_batch_rejects_non_s0_missing_evidence_or_local() -> None:
    batch = _batch()
    missing_previous = SegmentBatch(
        consumer_visual_summary=batch.consumer_visual_summary,
        consumer_payload=batch.consumer_payload,
        consumer_valid=batch.consumer_valid,
        consumer_step=batch.consumer_step,
        evidence_visual_summary_prev=batch.evidence_visual_summary_prev,
        evidence_executed_action_prev=batch.evidence_executed_action_prev,
        evidence_valid=torch.tensor([[False, False, False]]),
        evidence_source_step=torch.tensor([[-1, -1, -1]]),
        slot_id=batch.slot_id,
        episode_id=batch.episode_id,
        category=batch.category,
        segment_provenance=batch.segment_provenance,
    )
    with pytest.raises(ValueError, match="non-S0"):
        missing_previous.validate(16)
    with pytest.raises(ValueError, match="only valid S0"):
        batch.gather_consumers(torch.randn(1, 3, 1, 32), torch.tensor([[False, False, False]]))


def test_rank_local_scheduler_is_deterministic_and_commits_exposure() -> None:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0, "b": 1.0})
    candidates = (
        SegmentIdentity(0, "a0", "a", 0, 0, "x"),
        SegmentIdentity(1, "b0", "b", 0, 0, "x"),
    )
    first = scheduler.admit(candidates)
    assert first.category == "b"
    scheduler.commit(first, 3)
    second = scheduler.admit(candidates)
    assert second.category == "a"
    scheduler.commit(second, 1)
    snapshot = scheduler.snapshot()
    assert snapshot["cumulative_valid_consumer_exposure"] == {"a": 1, "b": 3}
    with pytest.raises(ValueError, match="num_workers"):
        RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0}, num_workers=1)


def test_scheduler_terminal_rebind_is_per_slot_and_snapshot_safe() -> None:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0, "b": 1.0})
    provenance = SegmentProvenance("manifest", "config", "source", 3)
    scheduler.configure_queue(seed=7, epoch=2, permutation=(3, 1, 2), provenance=provenance)
    first = SegmentIdentity(0, "episode", "a", 0, 3, "source")
    other = SegmentIdentity(1, "other", "b", 0, 3, "source")
    assert scheduler.admit((first,)) == first
    scheduler.commit(first, 1)
    assert scheduler.admit((other,)) == other
    scheduler.commit(other, 1)
    terminal = SegmentIdentity(0, "episode", "a", 1, 4, "source", training_stream_end=True)
    assert scheduler.admit((terminal,)) == terminal
    scheduler.commit(terminal, 1)  # tail may contain a single valid consumer; PAD is absent from this count.
    rebuilt = RankLocalSegmentScheduler.rebuild(scheduler.snapshot())
    assert rebuilt.snapshot() == scheduler.snapshot()
    with pytest.raises(ValueError, match="admissible"):
        rebuilt.admit((terminal,))
    other_continuation = SegmentIdentity(1, "other", "b", 1, 4, "source")
    assert rebuilt.admit((other_continuation,)) == other_continuation
    rebuilt.commit(other_continuation, 1)
    replacement = SegmentIdentity(0, "next-a", "a", 0, 4, "source")
    scheduled = SegmentIdentity(0, "next-b", "b", 0, 4, "source")
    rebuilt.terminal_rebind(terminal)
    assert 0 not in rebuilt.terminal_slots and 0 not in rebuilt.stable_slots
    with pytest.raises(ValueError, match="only one admitted"):
        rebuilt.commit(replacement, 1)
    assert rebuilt.admit((replacement, scheduled)) == scheduled
    rebuilt.commit(scheduled, 1)
    rebound = RankLocalSegmentScheduler.rebuild(rebuilt.snapshot())
    assert rebound.snapshot() == rebuilt.snapshot()


def test_scheduler_rejects_slot_switches_and_noncontiguous_cursor() -> None:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0, "b": 1.0})
    first = SegmentIdentity(0, "episode", "a", 0, 0, "source")
    assert scheduler.admit((first,)) == first
    scheduler.commit(first, 1)
    switched = SegmentIdentity(0, "other", "b", 1, 1, "source")
    skipped = SegmentIdentity(0, "episode", "a", 2, 1, "source")
    with pytest.raises(ValueError, match="admissible"):
        scheduler.admit((switched, skipped))


def test_ga_transaction_suffix_retry_and_grad_scaler_skip_semantics() -> None:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0})
    identities = (
        SegmentIdentity(0, "episode", "a", 0, 0, "source"),
        SegmentIdentity(0, "episode", "a", 1, 0, "source"),
    )
    for identity in identities:
        assert scheduler.admit((identity,)) == identity
    plan = GAWindowPlan(tuple((item.slot_id, item.episode_id, item.cursor) for item in identities), (2, 3), plan_chain_id="chain")
    transaction = LocalMemoryTransaction(plan, scheduler)
    transaction.successful_backward(0, identities[0], 2)
    retry = transaction.fail_transient(1)
    assert retry.attempt == 1 and retry.members == (plan.members[1],)
    assert transaction.slow_grads_cleared
    with pytest.raises(RuntimeError, match="LOCAL_MEM_RETRY_EXHAUSTED"):
        retry.suffix_after_failure(0)
    # A GradScaler skip retains the committed episode exposure but takes no slow/LR step.
    exposure = dict(scheduler.cumulative_valid_consumer_exposure)
    transaction.grad_scaler_skip()
    assert scheduler.cumulative_valid_consumer_exposure == exposure
    assert transaction.slow_optimizer_steps == transaction.slow_lr_scheduler_steps == 0
    transaction.slow_optimizer_step_succeeded()
    assert transaction.slow_optimizer_steps == transaction.slow_lr_scheduler_steps == 1


@pytest.mark.parametrize("failed_index, expected_members", [(0, 2), (1, 1)])
def test_ga_plan_first_and_later_failure_have_one_immutable_suffix_retry(
    failed_index: int, expected_members: int
) -> None:
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (2, 3), plan_chain_id="A-B-C-D")
    retry = plan.suffix_after_failure(failed_index)
    assert retry.attempt == 1
    assert retry.plan_chain_id == plan.plan_chain_id
    assert retry.suffix_snapshot == retry.members
    assert len(retry.members) == expected_members
    # A/B/C/D terminal: a retry plan never gets a second retry, regardless of
    # whether its first member is the former first or former later member.
    with pytest.raises(RuntimeError, match="LOCAL_MEM_RETRY_EXHAUSTED"):
        retry.suffix_after_failure(0)


def test_ga_window_full_valid_objective_matches_unpartitioned_consumer_loss() -> None:
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (2, 3))
    consumer = (torch.tensor(2.0), torch.tensor(5.0))
    auxiliary = (torch.tensor(1.5), torch.tensor(1.5))
    actual = sum(plan.objective(index, consumer[index], auxiliary[index], plan.planned_n_valid[index]) for index in range(2))
    expected = (2 / 5) * consumer[0] + (3 / 5) * consumer[1] + auxiliary[0]
    torch.testing.assert_close(actual, expected)

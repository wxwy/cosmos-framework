from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalGAWindowPlan,
    CanonicalSegmentContractError,
    ChronologyCountRecord,
    MicrobatchPlanMember,
    NativeConsumerBatch,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
    queue_digest_preimage,
    queue_permutation,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance


def _provenance() -> SegmentProvenance:
    return SegmentProvenance("manifest", "config", "source", 0)


def _batch() -> SegmentBatch:
    return SegmentBatch(
        consumer_visual_summary=torch.zeros(2, 3, 96),
        consumer_payload=(("s0", "s1", None), ("s2", None, None)),
        consumer_valid=torch.tensor([[True, True, False], [True, False, False]]),
        consumer_step=torch.tensor([[0, 1, -1], [2, -1, -1]]),
        evidence_visual_summary_prev=torch.zeros(2, 3, 96),
        evidence_executed_action_prev=torch.zeros(2, 3, 10),
        evidence_valid=torch.tensor([[False, True, False], [True, False, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1], [1, -1, -1]]),
        slot_id=torch.tensor([0, 1]),
        episode_id=("episode-a", "episode-b"),
        category=("a", "b"),
        segment_provenance=_provenance(),
    )


def _member(*, index: int = 0, exposure: tuple[tuple[str, int], ...] = (("a", 0), ("b", 0))) -> MicrobatchPlanMember:
    identities = (
        SegmentIdentity(0, "episode-a", "a", 0, 0, "source"),
        SegmentIdentity(1, "episode-b", "b", 2, 0, "source"),
    )
    records = (
        ChronologyCountRecord(0, "episode-a", "a", "source", 0, 2, False, "manifest"),
        ChronologyCountRecord(1, "episode-b", "b", "source", 2, 3, False, "manifest"),
    )
    return MicrobatchPlanMember(
        index, identities, (_provenance(), _provenance()), records, (2, 1), 3,
        QueueEpochSnapshot(7, 2, "catalog", (("a", 0), ("b", 0))), exposure,
    )


def test_s0_non_s0_pad_have_one_native_count_and_stream_major_gather() -> None:
    batch = _batch()
    batch.validate(16)
    member = _member()
    tokens = torch.randn(2, 3, 1, 32)
    gathered = NativeConsumerBatch.from_segment(batch, member, tokens, torch.tensor([[False, True, False], [True, False, False]]))
    assert gathered.item_count == member.planned_n_valid == int(batch.consumer_valid.sum()) == 3
    assert gathered.identities == ((0, "episode-a", 0), (0, "episode-a", 1), (1, "episode-b", 2))
    assert gathered.local_prefixes[0] is None
    assert gathered.local_prefixes[1] is not None and gathered.local_prefixes[2] is not None


def test_member_rejects_s0_exclusion_or_foreign_row_before_native_forward() -> None:
    batch = _batch()
    with pytest.raises(ValueError, match="identity/chronology"):
        MicrobatchPlanMember(
            0, _member().row_identities, (_provenance(), _provenance()), _member().row_chronology, (1, 1), 2,
            _member().queue_snapshot, _member().projected_exposure_before,
        )
    foreign = _member()
    bad_batch = SegmentBatch(
        **{**batch.__dict__, "episode_id": ("foreign", "episode-b")}
    )
    with pytest.raises(CanonicalSegmentContractError, match="identity/provenance"):
        foreign.validate_batch(bad_batch)


def test_unequal_count_ga_objective_and_first_member_only_retry() -> None:
    first = _member()
    second = _member(index=1, exposure=(("a", 2), ("b", 1)))
    plan = CanonicalGAWindowPlan((first, second), 6, 2, "chain")
    actual = plan.objective(0, torch.tensor(4.0), torch.tensor(2.0), 3) + plan.objective(1, torch.tensor(2.0), torch.tensor(2.0), 3)
    torch.testing.assert_close(actual, torch.tensor(5.0))
    retry = plan.retry_first_member_pre_backward(0)
    assert retry.attempt == 1 and retry.members == plan.members and retry.original_n_valid_window == plan.original_n_valid_window
    with pytest.raises(CanonicalSegmentContractError, match="only attempt-0 first"):
        plan.retry_first_member_pre_backward(1)


def test_full_valid_ga_consumer_term_degenerates_to_one_over_ga() -> None:
    first = _member()
    second = _member(index=1, exposure=(("a", 2), ("b", 1)))
    plan = CanonicalGAWindowPlan((first, second), 6, 2, "full")
    for index in range(2):
        consumer_only = plan.objective(index, torch.tensor(6.0), torch.tensor(0.0), 3)
        torch.testing.assert_close(consumer_only, torch.tensor(3.0))


def test_later_member_retry_is_fail_closed() -> None:
    first = _member()
    second = _member(index=1, exposure=(("a", 2), ("b", 1)))
    plan = CanonicalGAWindowPlan((first, second), 6, 2, "later")
    with pytest.raises(CanonicalSegmentContractError, match="only attempt-0 first"):
        plan.retry_first_member_pre_backward(1)


def test_projected_planning_is_pure_and_all_row_commit_is_atomic() -> None:
    initial = ProjectedSchedulerState(_member().queue_snapshot, (("a", 0), ("b", 0)))
    scheduler = CanonicalBatchScheduler(initial)
    member = _member()
    projected = scheduler.project((member,))
    assert scheduler.snapshot == initial
    assert projected[-1].exposure == (("a", 2), ("b", 1))
    with pytest.raises(CanonicalSegmentContractError, match="actual count"):
        scheduler.reconcile_after_backward(member, 2)
    assert scheduler.snapshot == initial
    scheduler.reconcile_after_backward(member, 3)
    assert scheduler.snapshot.exposure == (("a", 2), ("b", 1))
    assert tuple(slot for slot, _ in scheduler.snapshot.stable_slots) == (0, 1)


def test_queue_preimage_permutation_and_rollover_preserve_exposure() -> None:
    preimage = queue_digest_preimage(queue_seed=7, epoch=2, category="a", canonical_index=0)
    assert preimage == b"PSM-WMA/queue/v1\0" + b"7\0" + b"2\0a\0" + b"0"
    assert preimage != queue_digest_preimage(queue_seed=8, epoch=2, category="a", canonical_index=0)
    assert preimage != queue_digest_preimage(queue_seed=7, epoch=3, category="a", canonical_index=0)
    assert preimage != queue_digest_preimage(queue_seed=7, epoch=2, category="b", canonical_index=0)
    assert preimage != queue_digest_preimage(queue_seed=7, epoch=2, category="a", canonical_index=1)
    assert queue_permutation(queue_seed=7, epoch=2, category="a", catalog_size=4) == queue_permutation(queue_seed=7, epoch=2, category="a", catalog_size=4)
    initial = ProjectedSchedulerState(QueueEpochSnapshot(7, 2, "catalog", (("a", 1),)), (("a", 9),))
    scheduler = CanonicalBatchScheduler(initial)
    scheduler.rollover_if_exhausted({"a": 1})
    assert scheduler.snapshot.queue_snapshot.epoch == 3
    assert scheduler.snapshot.exposure == (("a", 9),)


def test_bound_continuation_blocks_rollover() -> None:
    identity = SegmentIdentity(0, "episode-a", "a", 0, 0, "source")
    state = ProjectedSchedulerState(
        QueueEpochSnapshot(7, 2, "catalog", (("a", 1),)), (("a", 9),), ((0, identity),)
    )
    with pytest.raises(CanonicalSegmentContractError, match="bound continuation"):
        CanonicalBatchScheduler(state).rollover_if_exhausted({"a": 1})

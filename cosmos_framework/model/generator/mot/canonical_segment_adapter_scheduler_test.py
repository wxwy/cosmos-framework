from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    CanonicalSegmentContractError,
    CatalogRow,
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
        SegmentIdentity(1, "episode-b", "b", 0, 0, "source"),
    )
    records = (
        ChronologyCountRecord(0, "episode-a", "a", "source", 0, 2, False, "manifest"),
        ChronologyCountRecord(1, "episode-b", "b", "source", 2, 3, False, "manifest"),
    )
    return MicrobatchPlanMember(
        index, identities, (_provenance(), _provenance()), records, (2, 1), 3,
        QueueEpochSnapshot(7, 2, "catalog", (("a", 0), ("b", 0))), exposure,
    )


def _state() -> ProjectedSchedulerState:
    member = _member()
    catalog = tuple(
        CatalogRow(identity, chronology, provenance)
        for identity, chronology, provenance in zip(member.row_identities, member.row_chronology, member.row_provenances, strict=True)
    )
    snapshot = QueueEpochSnapshot(
        7,
        2,
        "catalog",
        (("a", 0), ("b", 0)),
        (("a", queue_permutation(queue_seed=7, epoch=2, category="a", catalog_size=1)), ("b", queue_permutation(queue_seed=7, epoch=2, category="b", catalog_size=1))),
    )
    return ProjectedSchedulerState(snapshot, (("a", 0), ("b", 0)), target_distribution=(("a", .5), ("b", .5)), catalog=catalog)


def _single_member(*, index: int, stop: int, exposure: tuple[tuple[str, int], ...]) -> MicrobatchPlanMember:
    identity = SegmentIdentity(0, "episode-a", "a", 0, 0, "source")
    record = ChronologyCountRecord(0, "episode-a", "a", "source", 0, stop, False, "manifest")
    return MicrobatchPlanMember(
        index, (identity,), (_provenance(),), (record,), (stop,), stop,
        QueueEpochSnapshot(7, 2, "catalog", (("a", 0),)), exposure,
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
    first = _single_member(index=0, stop=3, exposure=(("a", 0),))
    second = _single_member(index=1, stop=1, exposure=(("a", 3),))
    plan = CanonicalGAWindowPlan((first, second), 4, 2, "chain")
    actual = plan.objective(0, torch.tensor(4.0), torch.tensor(2.0), 3) + plan.objective(1, torch.tensor(8.0), torch.tensor(2.0), 1)
    torch.testing.assert_close(actual, torch.tensor(7.0))
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


def test_weighted_deficit_selects_the_underexposed_free_category() -> None:
    rows = (
        CatalogRow(
            SegmentIdentity(0, "episode-a", "a", 0, 0, "source"),
            ChronologyCountRecord(0, "episode-a", "a", "source", 0, 1, False, "manifest"),
            _provenance(),
        ),
        CatalogRow(
            SegmentIdentity(0, "episode-b", "b", 0, 0, "source"),
            ChronologyCountRecord(0, "episode-b", "b", "source", 0, 1, False, "manifest"),
            _provenance(),
        ),
    )
    state = ProjectedSchedulerState(
        QueueEpochSnapshot(
            7,
            2,
            "catalog",
            (("a", 0), ("b", 0)),
            (("a", queue_permutation(queue_seed=7, epoch=2, category="a", catalog_size=1)), ("b", queue_permutation(queue_seed=7, epoch=2, category="b", catalog_size=1))),
        ),
        (("a", 3), ("b", 0)), target_distribution=(("a", .75), ("b", .25)), catalog=rows,
    )
    plan = CanonicalBatchScheduler(state).freeze_plan(slot_groups=((0,),), plan_chain_id="deficit")
    assert plan.members[0].row_identities[0].category == "b"


def test_later_member_retry_is_fail_closed() -> None:
    first = _member()
    second = _member(index=1, exposure=(("a", 2), ("b", 1)))
    plan = CanonicalGAWindowPlan((first, second), 6, 2, "later")
    with pytest.raises(CanonicalSegmentContractError, match="only attempt-0 first"):
        plan.retry_first_member_pre_backward(1)


def test_batch_window_retry_lifecycle_rejects_post_backward_and_terminalizes_later_failure() -> None:
    first, second = _member(), _member(index=1, exposure=(("a", 2), ("b", 1)))
    plan = CanonicalGAWindowPlan((first, second), 6, 2, "lifecycle")
    transaction = CanonicalBatchWindowTransaction(plan)
    transaction.mark_backward_started(0)
    transaction.mark_reconciled(0)
    with pytest.raises(CanonicalSegmentContractError, match="unstarted"):
        transaction.retry_first_member_pre_backward()
    transaction.terminalize(1, "LOCAL_MEM_RETRY_AFTER_MEMBER")
    snapshot = transaction.snapshot()
    assert snapshot.slow_grads_cleared and snapshot.remaining_members_suppressed


def test_projected_planning_is_pure_and_all_row_commit_is_atomic() -> None:
    initial = _state()
    scheduler = CanonicalBatchScheduler(initial)
    plan = scheduler.freeze_plan(slot_groups=((0, 1),), plan_chain_id="pure")
    member = plan.members[0]
    assert scheduler.snapshot == initial
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


def test_terminal_projection_reconcile_and_rollover_release_are_atomic() -> None:
    initial = _state()
    member = _member()
    terminal_identities = tuple(replace(identity, training_stream_end=True) for identity in member.row_identities)
    terminal_records = tuple(replace(record, training_stream_end=True) for record in member.row_chronology)
    terminal = replace(member, row_identities=terminal_identities, row_chronology=terminal_records)
    initial = replace(initial, catalog=tuple(CatalogRow(identity, record, _provenance()) for identity, record in zip(terminal_identities, terminal_records, strict=True)))
    scheduler = CanonicalBatchScheduler(initial)
    plan = scheduler.freeze_plan(slot_groups=((0, 1),), plan_chain_id="terminal")
    frozen = plan.members[0]
    assert initial.terminal_slots == ()
    scheduler.reconcile_after_backward(frozen, 3)
    assert tuple(slot for slot, _ in scheduler.snapshot.terminal_slots) == (0, 1)
    scheduler.rollover_if_exhausted({"a": 1, "b": 1})
    assert scheduler.snapshot.terminal_slots == () and scheduler.snapshot.exposure == (("a", 2), ("b", 1))


def test_reconcile_rejects_reconstructed_or_out_of_order_member_before_mutation() -> None:
    initial = _state()
    scheduler = CanonicalBatchScheduler(initial)
    plan = scheduler.freeze_plan(slot_groups=((0, 1),), plan_chain_id="identity")
    reconstructed = replace(plan.members[0])
    with pytest.raises(CanonicalSegmentContractError, match="foreign, stale"):
        scheduler.reconcile_after_backward(reconstructed, reconstructed.planned_n_valid)
    assert scheduler.snapshot == initial


def test_shared_backward_precedes_exact_atomic_reconcile() -> None:
    initial = _state()
    scheduler = CanonicalBatchScheduler(initial)
    member = scheduler.freeze_plan(slot_groups=((0, 1),), plan_chain_id="backward").members[0]
    scalar = torch.nn.Parameter(torch.tensor(2.0))
    (scalar * 3.0).backward()
    torch.testing.assert_close(scalar.grad, torch.tensor(3.0))
    assert scheduler.snapshot == initial
    scheduler.reconcile_after_backward(member, member.planned_n_valid)
    assert scheduler.snapshot.exposure == (("a", 2), ("b", 1))


def test_terminal_rebind_advances_frozen_queue_and_fresh_states_match() -> None:
    permutation = queue_permutation(queue_seed=7, epoch=2, category="a", catalog_size=2)
    old = CatalogRow(
        SegmentIdentity(0, "episode-old", "a", 0, 0, "source", True),
        ChronologyCountRecord(0, "episode-old", "a", "source", 0, 1, True, "manifest"),
        _provenance(),
    )
    fresh = CatalogRow(
        SegmentIdentity(0, "episode-fresh", "a", 0, 0, "source"),
        ChronologyCountRecord(0, "episode-fresh", "a", "source", 0, 1, False, "manifest"),
        _provenance(),
    )
    ordered: list[CatalogRow | None] = [None, None]
    ordered[permutation[0]], ordered[permutation[1]] = old, fresh
    state = ProjectedSchedulerState(
        QueueEpochSnapshot(7, 2, "catalog", (("a", 0),), (("a", permutation),)),
        (("a", 0),), target_distribution=(("a", 1.0),), catalog=tuple(ordered),  # type: ignore[arg-type]
    )
    first, second = CanonicalBatchScheduler(state), CanonicalBatchScheduler(state)
    first_plan, second_plan = (
        first.freeze_plan(slot_groups=((0,),), plan_chain_id="rebind"),
        second.freeze_plan(slot_groups=((0,),), plan_chain_id="rebind"),
    )
    assert first_plan.members == second_plan.members
    first.reconcile_after_backward(first_plan.members[0], 1)
    rebound = first.freeze_plan(slot_groups=((0,),), plan_chain_id="rebind-next")
    assert rebound.members[0].row_identities[0].episode_id == "episode-fresh"
    assert first.snapshot.queue_snapshot.positions == (("a", 1),)
    first.reconcile_after_backward(rebound.members[0], 1)
    assert first.snapshot.queue_snapshot.positions == (("a", 2),)

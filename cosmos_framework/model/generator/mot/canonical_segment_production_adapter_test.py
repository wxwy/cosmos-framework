from __future__ import annotations

import torch
import pytest

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    CanonicalSegmentContractError,
    CatalogRow,
    ChronologyCountRecord,
    MicrobatchPlanMember,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalProductionAdapter,
    CanonicalProductionFastStateFrontier,
    CanonicalProductionSegmentRequest,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance


def test_adapter_scan_derives_stream_major_gather_and_fp32_state() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    batch = SegmentBatch(
        torch.zeros(1, 2, 96), (("s0", "s1"),), torch.tensor([[True, True]]), torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96), torch.zeros(1, 2, 10), torch.tensor([[False, True]]), torch.tensor([[-1, 0]]),
        torch.tensor([0]), ("episode",), ("category",), provenance,
    )
    member = MicrobatchPlanMember(
        0, (SegmentIdentity(0, "episode", "category", 0, 0, "source"),), (provenance,),
        (ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest"),), (2,), 2,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)), (),
    )
    plan = CanonicalGAWindowPlan((member,), 2, 1, "chain")
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    request = CanonicalProductionSegmentRequest(scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, batch)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core)
    result = adapter.scan(request)
    assert result.gathered.item_count == 2
    assert result.gathered.local_prefixes[0] is None and result.gathered.local_prefixes[1] is not None
    assert all(value.dtype is torch.float32 for value in result.candidate_state_out)
    assert result.slot_chain == ((0, "episode", "source", 0),)
    with pytest.raises(Exception, match="already has a scan capability"):
        adapter.scan(request)


def test_adapter_commit_is_exact_once_and_preflights_before_frontier_mutation() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identity = SegmentIdentity(0, "episode", "category", 0, 0, "source")
    record = ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest")
    state = ProjectedSchedulerState(
        QueueEpochSnapshot(
            1,
            0,
            "catalog",
            (("category", 0),),
            (("category", (0,)),),
        ),
        (("category", 0),),
        target_distribution=(("category", 1.0),),
        catalog=(CatalogRow(identity, record, provenance),),
    )
    scheduler = CanonicalBatchScheduler(state)
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="commit")
    member = plan.members[0]
    batch = SegmentBatch(
        torch.zeros(1, 2, 96), (("s0", "s1"),), torch.tensor([[True, True]]), torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96), torch.zeros(1, 2, 10), torch.tensor([[False, True]]), torch.tensor([[-1, 0]]),
        torch.tensor([0]), ("episode",), ("category",), provenance,
    )
    transaction = CanonicalBatchWindowTransaction(plan)
    request = CanonicalProductionSegmentRequest(scheduler, plan, transaction, member, 0, batch)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core)
    result = adapter.scan(request)
    capability = adapter.prepare_commit(request, result)
    scheduler_before = scheduler.snapshot
    with pytest.raises(CanonicalSegmentContractError, match="current backward member"):
        adapter.commit_success(capability)
    assert scheduler.snapshot == scheduler_before
    continuation = MicrobatchPlanMember(
        0,
        (SegmentIdentity(0, "episode", "category", 1, 1, "source"),),
        (provenance,),
        (ChronologyCountRecord(0, "episode", "category", "source", 2, 3, False, "manifest"),),
        (1,),
        1,
        member.queue_snapshot,
        member.projected_exposure_before,
    )
    with pytest.raises(CanonicalSegmentContractError, match="lacks exact committed"):
        adapter.frontier.state_for(continuation)
    transaction.mark_backward_started(0)
    adapter.commit_success(capability)
    assert transaction.snapshot().completed_members == (0,)
    with pytest.raises(CanonicalSegmentContractError, match="foreign or already consumed"):
        adapter.commit_success(capability)


def test_fast_state_frontier_preserves_w0_gradients_and_slot_isolation() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identities = (
        SegmentIdentity(0, "episode-0", "category", 0, 0, "source-0"),
        SegmentIdentity(1, "episode-1", "category", 0, 0, "source-1"),
    )
    member = MicrobatchPlanMember(
        0,
        identities,
        (provenance, provenance),
        (
            ChronologyCountRecord(0, "episode-0", "category", "source-0", 0, 1, False, "manifest"),
            ChronologyCountRecord(1, "episode-1", "category", "source-1", 0, 1, False, "manifest"),
        ),
        (1, 1),
        2,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)),
        (),
    )
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    frontier = CanonicalProductionFastStateFrontier(core)
    fresh = frontier.state_for(member)
    assert all(value.dtype is torch.float32 for value in fresh)
    sum(value.sum() for value in fresh).backward()
    assert all(parameter.grad is not None for parameter in core._w0)
    # Give each slot a distinct numeric state without introducing a shared storage alias.
    committed = type(fresh)(*(value + torch.arange(2, dtype=torch.float32).reshape(2, *([1] * (value.ndim - 1))) for value in fresh))
    frontier.commit(member, committed)
    continued = MicrobatchPlanMember(
        0,
        (
            SegmentIdentity(0, "episode-0", "category", 1, 1, "source-0"),
            SegmentIdentity(1, "episode-1", "category", 1, 1, "source-1"),
        ),
        (provenance, provenance),
        (
            ChronologyCountRecord(0, "episode-0", "category", "source-0", 1, 2, False, "manifest"),
            ChronologyCountRecord(1, "episode-1", "category", "source-1", 1, 2, False, "manifest"),
        ),
        (1, 1),
        2,
        member.queue_snapshot,
        (),
    )
    state = frontier.state_for(continued)
    for value, expected in zip(state, committed, strict=True):
        torch.testing.assert_close(value, expected)
        assert value[0].data_ptr() != value[1].data_ptr()

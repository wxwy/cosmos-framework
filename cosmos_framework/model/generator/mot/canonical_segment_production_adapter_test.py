from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    ChronologyCountRecord,
    MicrobatchPlanMember,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalProductionAdapter,
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

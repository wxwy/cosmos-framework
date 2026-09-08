from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring


def _fixture() -> tuple[CanonicalSegmentWiring, SegmentBatch, SegmentIdentity, LocalMemoryTransaction]:
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((identity,))
    transaction = LocalMemoryTransaction(GAWindowPlan(((0, "episode", 0),), (1,)), scheduler)
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar(),
    )
    local = torch.nn.Parameter(torch.ones(()))
    segment = SegmentBatch(
        torch.zeros(1, 1, 96), ((object(),),), torch.tensor([[True]]), torch.tensor([[0]]),
        torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")),
        torch.tensor([[False]]), torch.tensor([[-1]]), torch.tensor([0]), ("episode",), ("suite",),
        SegmentProvenance("m", "c", "digest", 0),
    )
    return CanonicalSegmentWiring(adapter, (local,)), segment, identity, transaction


def test_wiring_preserves_exact_pending_result_and_local_only_grads() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    assert wiring.adapter.pending_scan == (identity, transaction, forward.result)
    unrelated = torch.nn.Parameter(torch.ones(())); unrelated.grad = torch.ones(())
    wiring.local_slow_parameters[0].grad = torch.ones(())
    wiring.clear_local_slow_grads()
    assert wiring.local_slow_parameters[0].grad is None
    torch.testing.assert_close(unrelated.grad, torch.ones(()))
    with pytest.raises(ValueError, match="cardinality"):
        from cosmos_framework.model.generator.mot.production_segment_wiring import run_native_forward_for_test
        run_native_forward_for_test((object(),), ())

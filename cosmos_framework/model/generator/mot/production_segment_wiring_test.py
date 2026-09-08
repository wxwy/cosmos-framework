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
from cosmos_framework.model.generator.mot.production_segment_wiring import (
    CanonicalSegmentWiring,
    run_native_forward_for_test,
)


def _fixture(two_steps: bool = False) -> tuple[CanonicalSegmentWiring, SegmentBatch, SegmentIdentity, LocalMemoryTransaction]:
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "digest")
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    scheduler.admit((identity,))
    n_valid = 2 if two_steps else 1
    transaction = LocalMemoryTransaction(GAWindowPlan(((0, "episode", 0),), (n_valid,)), scheduler)
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar(),
    )
    local = torch.nn.Parameter(torch.ones(()))
    steps = 2 if two_steps else 1
    evidence_visual = torch.full((1, steps, 96), float("nan"))
    evidence_action = torch.full((1, steps, 10), float("nan"))
    evidence_valid = torch.zeros(1, steps, dtype=torch.bool)
    evidence_source_step = torch.full((1, steps), -1)
    if two_steps:
        evidence_visual[:, 1].zero_()
        evidence_action[:, 1].zero_()
        evidence_valid[:, 1] = True
        evidence_source_step[:, 1] = 0
    segment = SegmentBatch(
        torch.zeros(1, steps, 96), (tuple(object() for _ in range(steps)),), torch.ones(1, steps, dtype=torch.bool),
        torch.arange(steps).unsqueeze(0), evidence_visual, evidence_action,
        evidence_valid, evidence_source_step, torch.tensor([0]), ("episode",), ("suite",),
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
        run_native_forward_for_test((object(),), (), forward.result.local_tokens, wiring.local_slow_parameters)


def test_s0_test_spy_remains_graph_bearing() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    primary, auxiliary = run_native_forward_for_test(
        forward.payloads, forward.locals, forward.result.local_tokens, wiring.local_slow_parameters
    )
    assert primary.requires_grad
    assert auxiliary.shape == torch.Size([])


def test_forward_binds_the_exact_wiring_capability() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    substitute = CanonicalSegmentWiring(wiring.adapter, (torch.nn.Parameter(torch.ones(())),))
    assert forward.wiring is wiring
    assert forward.wiring is not substitute


def test_non_s0_spy_counts_visible_local_exactly_once() -> None:
    wiring, segment, identity, transaction = _fixture(two_steps=True)
    forward = wiring.prepare(segment, identity, transaction)
    primary, _ = run_native_forward_for_test(
        forward.payloads, forward.locals, forward.result.local_tokens, wiring.local_slow_parameters
    )
    expected = sum(token.sum() for token in forward.locals if token is not None)
    torch.testing.assert_close(primary, expected)

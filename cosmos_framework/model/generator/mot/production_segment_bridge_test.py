from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner, RuntimePhase
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_segment_bridge import NativeBatchOutcome, NativeBatchResult, OpenMemberCapability, RetryMemberCapability, run_member
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring


def _owner() -> CanonicalSegmentRuntimeOwner:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    adapter = CanonicalLocalMemorySegmentAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar())
    return CanonicalSegmentRuntimeOwner(scheduler, CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),)))


def _identity(cursor: int) -> SegmentIdentity:
    return SegmentIdentity(0, "episode", "suite", cursor, cursor, "source")


def _segment(identity: SegmentIdentity) -> SegmentBatch:
    return SegmentBatch(torch.zeros(1, 1, 96), ((object(),),), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long), torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")), torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1), -1, dtype=torch.long), torch.tensor([0]), ("episode",), ("suite",), SegmentProvenance("manifest", "config", identity.source_digest, identity.segment_id))


def _success(payloads, locals) -> NativeBatchOutcome:
    assert len(payloads) == len(locals)
    return NativeBatchOutcome(result=NativeBatchResult(torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True), 1))


def test_initial_continuation_finishes_inside_bridge_and_requires_resolution() -> None:
    owner, first, second = _owner(), _identity(0), _identity(1)
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 1))
    prior = run_member(owner, first, _segment(first), _success, initial_plan=plan)
    assert isinstance(prior, OpenMemberCapability)
    completed = run_member(owner, second, _segment(second), _success, prior=prior)
    assert owner.phase is RuntimePhase.SLOW_RESOLUTION_PENDING
    owner.resolve_local_memory_slow_window(completed, scaler_skipped=False)
    assert owner.phase is RuntimePhase.IDLE


def test_attempt_zero_source_failure_retries_exact_suffix_once() -> None:
    owner, identity = _owner(), _identity(0)
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    retry = run_member(owner, identity, _segment(identity), lambda *_: NativeBatchOutcome(source_failure="LOAD_DECODE_TRANSIENT"), initial_plan=plan)
    assert isinstance(retry, RetryMemberCapability)
    completed = run_member(owner, identity, _segment(identity), _success, retry=retry)
    owner.resolve_local_memory_slow_window(completed, scaler_skipped=True)
    assert owner.phase is RuntimePhase.IDLE

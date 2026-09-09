from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner, RuntimePhase
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentIdentity
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring


def _owner() -> tuple[CanonicalSegmentRuntimeOwner, SegmentIdentity]:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    adapter = CanonicalLocalMemorySegmentAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar())
    return CanonicalSegmentRuntimeOwner(scheduler, CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),))), SegmentIdentity(0, "episode", "suite", 0, 0, "source")


def test_skip_resume_reuses_exact_admission_and_plan() -> None:
    owner, identity = _owner()
    owner.admit((identity,))
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    transaction = owner.begin(plan)
    # The full prepared path is covered by adapter integration fixtures; this
    # CPU owner fixture proves only retained admission/plan authority.
    owner.phase = RuntimePhase.SKIP_READY
    owner._skipped_identity, owner._skipped_plan = identity, plan
    resumed = owner.resume_skipped()
    assert resumed.plan is plan
    assert owner.identity is identity
    assert owner.phase is RuntimePhase.MEMBER_READY


def test_skip_resume_rejects_replacement_or_committed_identity() -> None:
    owner, identity = _owner()
    owner.admit((identity,))
    owner.phase, owner._skipped_identity = RuntimePhase.SKIP_READY, identity
    owner._skipped_plan = GAWindowPlan(((0, "episode", 0),), (1,), attempt=1)
    with pytest.raises(RuntimeError, match="retained exact authority"):
        owner.resume_skipped()
    assert owner.phase is RuntimePhase.SKIP_READY

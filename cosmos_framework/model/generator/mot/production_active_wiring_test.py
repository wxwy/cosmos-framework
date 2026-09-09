from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_active_wiring import ProductionActiveWiringRegistry
from cosmos_framework.model.generator.mot.production_segment_bridge import NativeBatchResult
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.trainer import ImaginaireTrainer


def _fixture():
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar()
    )
    owner = CanonicalSegmentRuntimeOwner(scheduler, CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),)))
    identity = SegmentIdentity(0, "episode", "suite", 0, 0, "source")
    segment = SegmentBatch(
        torch.zeros(1, 1, 96), (({"opaque": "payload"},),), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
        torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")), torch.zeros(1, 1, dtype=torch.bool),
        torch.full((1, 1), -1, dtype=torch.long), torch.tensor([0]), ("episode",), ("suite",),
        SegmentProvenance("manifest", "config", "source", 0),
    )
    return owner, identity, segment, GAWindowPlan(((0, "episode", 0),), (1,))


def test_active_registry_consumes_exact_prepared_and_forward_capabilities_once() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    assert prepared.actual_n_valid == 1 and prepared.inputs.locals == (None,)
    consumed = registry.consume_prepared_for_model(prepared)
    active = registry.publish_active_forward(consumed, NativeBatchResult(torch.ones(()), torch.zeros(())))
    assert registry.consume_active_forward(active) is active
    with pytest.raises(RuntimeError, match="stale or foreign"):
        registry.consume_active_forward(active)


def test_active_registry_rejects_double_prepare_and_foreign_consume_without_new_prepare() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    with pytest.raises(RuntimeError, match="not idle"):
        registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    other_owner, _, _, _ = _fixture()
    other = ProductionActiveWiringRegistry(other_owner)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        other.consume_prepared_for_model(prepared)
    pending = owner.adapter.pending()
    assert owner.phase.name == "PREPARED" and pending is not None and pending[2] is prepared.forward.result


def test_active_model_and_trainer_consume_one_capability_and_complete_window() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._psm_active_wiring_registry = registry

    output, _ = model.training_step(
        {"psm_local_memory_active": True, "psm_local_memory_prepared": prepared}, 0
    )
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_active_local_memory_backward(
        model, output, torch.amp.GradScaler("cuda", enabled=False), grad_accum_iter=0
    )

    assert owner.phase.name == "SLOW_RESOLUTION_PENDING"
    assert prepared.transaction.snapshot().completed_members == (identity,)
    assert trainer._psm_active_completed_window.owner is owner

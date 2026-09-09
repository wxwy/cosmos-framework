from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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


def test_active_initial_preflight_and_post_prepare_contract_failure_leave_no_pending_scan() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    wrong_first = GAWindowPlan(((0, "other", 0),), (1,))
    with pytest.raises(RuntimeError, match="exact first member"):
        registry.prepare_initial(identity, segment, wrong_first, trainer_grad_accum_iter=0)
    assert owner.phase.name == "IDLE" and owner.adapter.pending() is None

    bad_payload = replace(segment, consumer_payload=((object(),),))
    with pytest.raises(TypeError, match="non-callable mappings"):
        registry.prepare_initial(identity, bad_payload, plan, trainer_grad_accum_iter=0)
    assert owner.phase.name == "ABORTED" and owner.adapter.pending() is None
    assert owner.wiring.local_slow_parameters[0].grad is None


def test_active_post_prepare_count_failure_is_terminal_and_clears_pending() -> None:
    owner, identity, segment, _ = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    plan = GAWindowPlan(((0, "episode", 0),), (2,))
    with pytest.raises(RuntimeError, match="gathered count"):
        registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    assert owner.phase.name == "ABORTED" and owner.adapter.pending() is None


def test_active_initial_arm_rejects_counter_or_ga_mismatch_before_owner_mutation() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._psm_active_wiring_registry = registry
    trainer = object.__new__(ImaginaireTrainer)
    trainer._psm_active_wiring_registry = registry
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=1))

    with pytest.raises(RuntimeError, match="native accumulation boundary"):
        trainer.arm_active_local_memory_initial(model, identity, segment, plan, grad_accum_iter=1)
    assert owner.phase.name == "IDLE" and owner.adapter.pending() is None

    wrong_ga = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 1))
    with pytest.raises(RuntimeError, match="native accumulation boundary"):
        trainer.arm_active_local_memory_initial(model, identity, segment, wrong_ga, grad_accum_iter=0)
    assert owner.phase.name == "IDLE" and owner.adapter.pending() is None


def test_active_tagged_transient_retries_only_the_first_member() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    retry_plan = registry.abort_source_transient(prepared)
    assert owner.phase.name == "RETRY_READY" and owner.adapter.pending() is None
    retried = registry.prepare_retry(identity, segment, retry_plan, trainer_grad_accum_iter=0)
    assert retried.transaction.plan is retry_plan and owner.phase.name == "PREPARED"


def test_active_tagged_transient_after_a_member_is_terminal() -> None:
    owner, identity, segment, _ = _fixture()
    next_identity = SegmentIdentity(0, "episode", "suite", 1, 1, "source")
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 1))
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._psm_active_wiring_registry = registry
    output, _ = model.training_step({"psm_local_memory_active": True, "psm_local_memory_prepared": prepared}, 0)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_active_local_memory_backward(
        model, output, torch.amp.GradScaler("cuda", enabled=False), grad_accum_iter=0
    )
    continuation = registry.prepare_continuation(
        next_identity, segment, prepared.transaction, trainer_grad_accum_iter=1
    )
    with pytest.raises(RuntimeError, match="LOCAL_MEM_RETRY_AFTER_MEMBER"):
        registry.abort_source_transient(continuation)
    assert owner.phase.name == "ABORTED" and owner.adapter.pending() is None


def test_active_optimizer_refuses_unknown_enabled_scaler_outcome_before_step() -> None:
    class Scaler:
        _per_optimizer_states: dict[int, object] = {}

        def __init__(self) -> None:
            self.unscaled = self.stepped = False

        def is_enabled(self) -> bool:
            return True

        def unscale_(self, optimizer: object) -> None:
            self.unscaled = True

        def step(self, optimizer: object) -> None:
            self.stepped = True

        def update(self) -> None:
            raise AssertionError("unknown scaler outcome must not update")

    class Owner:
        def resolve_preflighted_slow_window(self, *, scaler_skipped: bool) -> None:
            raise AssertionError("unknown scaler outcome must not resolve")

    scaler = Scaler()
    trainer = object.__new__(ImaginaireTrainer)
    with pytest.raises(RuntimeError, match="scaler outcome is unavailable"):
        trainer._optimizer_step(
            torch.nn.Identity(), object(), object(), scaler, iteration=0, active_seal=SimpleNamespace(owner=Owner())
        )
    assert scaler.unscaled and not scaler.stepped


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

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_active_wiring import ActiveSourceTransientError, ProductionActiveWiringRegistry
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
    retried = registry.prepare_retry(segment, retry_plan, trainer_grad_accum_iter=0)
    assert retried.transaction.plan is retry_plan and owner.phase.name == "PREPARED"


def test_active_trainer_retry_arm_consumes_exact_retained_plan_at_counter_zero() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    trainer = object.__new__(ImaginaireTrainer)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    trainer._psm_active_wiring_registry = registry
    model._psm_active_wiring_registry = registry
    trainer._psm_active_retry_registry = registry
    trainer._psm_active_retry_plan = registry.abort_source_transient(prepared)

    trainer.arm_active_local_memory_retry(model, segment, grad_accum_iter=0)

    assert trainer._psm_active_armed_prepared.identity is identity
    assert owner.phase.name == "PREPARED"
    assert trainer._psm_active_retry_plan is None and trainer._psm_active_retry_registry is None


def test_active_trainer_tagged_exception_retains_exact_first_member_retry_authority() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._psm_active_armed_prepared = prepared

    trainer._handle_active_forward_exception(prepared, ActiveSourceTransientError("source transient"))

    assert owner.phase.name == "RETRY_READY"
    assert trainer._psm_active_retry_registry is registry
    assert trainer._psm_active_retry_plan is not plan
    assert trainer._psm_active_retry_plan.attempt == 1
    assert trainer._psm_active_retry_plan.members == plan.members
    assert trainer._psm_active_armed_prepared is None


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


@pytest.mark.parametrize(("found_inf", "scheduler_steps"), ((False, 1), (True, 0)))
def test_active_optimizer_resolves_enabled_scaler_success_and_skip(found_inf: bool, scheduler_steps: int) -> None:
    class Scaler:
        def __init__(self) -> None:
            self._per_optimizer_states: dict[int, object] = {}
            self.unscaled = self.stepped = self.updated = False

        def is_enabled(self) -> bool:
            return True

        def unscale_(self, optimizer: object) -> None:
            self.unscaled = True
            self._per_optimizer_states[id(optimizer)] = {
                "found_inf_per_device": {"cpu": torch.tensor(float(found_inf))}
            }

        def step(self, optimizer: object) -> None:
            self.stepped = True

        def update(self) -> None:
            self.updated = True

    class Owner:
        def __init__(self) -> None:
            self.resolutions: list[bool] = []

        def resolve_preflighted_slow_window(self, *, scaler_skipped: bool) -> None:
            self.resolutions.append(scaler_skipped)

    class Scheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    optimizer, scaler, owner, scheduler = object(), Scaler(), Owner(), Scheduler()
    trainer = object.__new__(ImaginaireTrainer)
    trainer._psm_active_completed_window = object()
    trainer._psm_active_registry = object()
    trainer._optimizer_step(torch.nn.Identity(), optimizer, scheduler, scaler, iteration=0, active_seal=SimpleNamespace(owner=owner))

    assert scaler.unscaled and scaler.stepped and scaler.updated
    assert owner.resolutions == [found_inf] and scheduler.steps == scheduler_steps
    assert trainer._psm_active_completed_window is None and trainer._psm_active_registry is None


def test_active_backward_rejects_wrong_identity_before_scaled_backward_and_clears_pending() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    prepared = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._psm_active_wiring_registry = registry
    output, _ = model.training_step({"psm_local_memory_active": True, "psm_local_memory_prepared": prepared}, 0)
    object.__setattr__(prepared, "identity", SegmentIdentity(0, "other", "suite", 0, 0, "source"))
    trainer = object.__new__(ImaginaireTrainer)

    with pytest.raises(RuntimeError, match="LOCAL_MEM_IDENTITY_CONTRACT_FAILURE"):
        trainer._run_active_local_memory_backward(
            model, output, torch.amp.GradScaler("cuda", enabled=False), grad_accum_iter=0
        )
    assert owner.phase.name == "ABORTED" and owner.adapter.pending() is None
    assert owner.wiring.local_slow_parameters[0].grad is None


@pytest.mark.parametrize(
    "data_batch, error",
    (
        ({"psm_local_memory_active": True}, "keys are incomplete"),
        (
            {
                "psm_local_memory_active": True,
                "psm_local_memory_prepared": object(),
                "psm_local_memory_unexpected": True,
            },
            "keys are incomplete",
        ),
        ({"psm_local_memory_active": True, "psm_local_memory_prepared": object()}, "invalid type"),
    ),
)
def test_active_model_marker_schema_rejects_missing_extra_and_bad_capability(data_batch, error: str) -> None:
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    with pytest.raises((TypeError, ValueError), match=error):
        model.training_step(data_batch, 0)


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


def test_active_two_member_window_keeps_one_registry_token_until_completion() -> None:
    owner, identity, segment, _ = _fixture()
    next_identity = SegmentIdentity(0, "episode", "suite", 1, 1, "source")
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 1))
    registry = ProductionActiveWiringRegistry(owner)
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model._psm_active_wiring_registry = registry
    trainer = object.__new__(ImaginaireTrainer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    first = registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    output, _ = model.training_step({"psm_local_memory_active": True, "psm_local_memory_prepared": first}, 0)
    trainer._run_active_local_memory_backward(model, output, scaler, grad_accum_iter=0)
    assert owner.phase.name == "MEMBER_COMMITTED" and not hasattr(trainer, "_psm_active_completed_window")

    second = registry.prepare_continuation(next_identity, segment, first.transaction, trainer_grad_accum_iter=1)
    assert second.ga_window_token is first.ga_window_token
    output, _ = model.training_step({"psm_local_memory_active": True, "psm_local_memory_prepared": second}, 1)
    trainer._run_active_local_memory_backward(model, output, scaler, grad_accum_iter=1)
    assert owner.phase.name == "SLOW_RESOLUTION_PENDING"
    assert trainer._psm_active_completed_window.transaction.snapshot().completed_members == (identity, next_identity)


def test_active_open_window_rejects_untagged_trainer_interleave_before_forward() -> None:
    owner, identity, segment, plan = _fixture()
    registry = ProductionActiveWiringRegistry(owner)
    registry.prepare_initial(identity, segment, plan, trainer_grad_accum_iter=0)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._psm_active_wiring_registry = registry

    with pytest.raises(RuntimeError, match="open active Local window forbids no-marker interleaving"):
        trainer.training_step(object(), object(), object(), object(), {}, grad_accum_iter=0)

    assert owner.phase.name == "PREPARED" and owner.adapter.pending() is not None


def test_active_optimizer_boundary_rejects_open_or_foreign_window_before_callbacks() -> None:
    trainer = object.__new__(ImaginaireTrainer)
    open_owner = SimpleNamespace(phase=SimpleNamespace(name="PREPARED"), transaction=object())
    trainer._psm_active_wiring_registry = SimpleNamespace(owner=open_owner)
    with pytest.raises(RuntimeError, match="without exact completion"):
        trainer._preflight_active_optimizer_boundary(1)

    expected_transaction = SimpleNamespace(plan=SimpleNamespace(ga_effective=1))
    bound_owner = SimpleNamespace(phase=SimpleNamespace(name="SLOW_RESOLUTION_PENDING"), transaction=expected_transaction)
    bound_registry = SimpleNamespace(owner=bound_owner)
    trainer._psm_active_wiring_registry = bound_registry
    trainer._psm_active_registry = bound_registry
    trainer._psm_active_completed_window = SimpleNamespace(owner=bound_owner, transaction=object())
    with pytest.raises(RuntimeError, match="does not match optimizer boundary"):
        trainer._preflight_active_optimizer_boundary(1)

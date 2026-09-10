from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CatalogRow,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalProductionAdapter,
    CanonicalProductionCommitCapability,
    CanonicalProductionSegmentRequest,
    build_canonical_native_loss_split,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_integration_test import (
    _bound_request_and_carrier,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.production_segment_wiring import run_native_forward_for_test
from cosmos_framework.model.generator.mot.production_segment_wiring_test import _fixture
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.trainer import ImaginaireTrainer


def _model_marker_output(wiring, segment, identity, transaction):
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(local_ttt_enabled=True)
    return model.training_step(
        {
            "canonical_local_memory_segment": True,
            "canonical_segment": segment,
            "canonical_identity": identity,
            "canonical_transaction": transaction,
            "canonical_wiring": wiring,
            "canonical_member_index": 0,
        },
        0,
    )


def test_canonical_trainer_delegates_then_commits_exact_result() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    trainer = object.__new__(ImaginaireTrainer)
    trainer._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == (identity,)


def test_canonical_trainer_rejects_mismatched_result_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    other_wiring, other_segment, other_identity, other_transaction = _fixture()
    other_forward = other_wiring.prepare(other_segment, other_identity, other_transaction)
    output = {
        "canonical_segment_forward": other_forward, "canonical_wiring": wiring,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": torch.ones((), requires_grad=True),
        "auxiliary_loss": torch.zeros((), requires_grad=True), "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()


def test_canonical_trainer_rejects_same_adapter_substitute_wiring_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    forward = wiring.prepare(segment, identity, transaction)
    true_local = wiring.local_slow_parameters[0]
    substitute_local = torch.nn.Parameter(torch.ones(()))
    substitute = type(wiring)(wiring.adapter, (substitute_local,))
    true_local.grad = torch.ones(())
    substitute_local.grad = torch.ones(())
    primary, auxiliary = run_native_forward_for_test(
        forward.payloads, forward.locals, forward.result.local_tokens, wiring.local_slow_parameters
    )
    output = {
        "canonical_segment_forward": forward, "canonical_wiring": substitute,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": primary,
        "auxiliary_loss": auxiliary, "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()
    torch.testing.assert_close(true_local.grad, torch.ones(()))
    torch.testing.assert_close(substitute_local.grad, torch.ones(()))


def test_canonical_trainer_rejects_external_plan_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    output["canonical_plan"] = transaction.plan
    with pytest.raises(RuntimeError, match="external plan"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()


def test_canonical_trainer_rejects_missing_capability_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    del output["canonical_identity"]
    with pytest.raises(RuntimeError, match="capability is incomplete"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == ()


def test_canonical_trainer_rejects_stale_result_before_backward() -> None:
    wiring, segment, identity, transaction = _fixture()
    stale = wiring.prepare(segment, identity, transaction)
    current = wiring.prepare(segment, identity, transaction)
    primary, auxiliary = run_native_forward_for_test(
        stale.payloads, stale.locals, stale.result.local_tokens, wiring.local_slow_parameters
    )
    output = {
        "canonical_segment_forward": stale, "canonical_wiring": wiring,
        "canonical_transaction": transaction, "canonical_member_index": 0,
        "canonical_identity": identity, "primary_consumer_mean": primary,
        "auxiliary_loss": auxiliary, "actual_n_valid": 1,
    }
    with pytest.raises(RuntimeError, match="capability identity"):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert current.result is wiring.adapter.pending_scan[2]
    assert transaction.snapshot().completed_members == ()


@pytest.mark.parametrize(
    ("primary", "auxiliary", "code"),
    (
        (torch.tensor(float("nan"), requires_grad=True), torch.zeros((), requires_grad=True), "LOCAL_MEM_NUMERICAL_FAILURE"),
        (torch.ones(()), torch.zeros(()), "LOCAL_MEM_OUTER_FAILURE"),
    ),
)
def test_canonical_trainer_failure_regression_clears_grad_without_commit(primary, auxiliary, code: str) -> None:
    wiring, segment, identity, transaction = _fixture()
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    parameter = wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    output["primary_consumer_mean"] = primary
    output["auxiliary_loss"] = auxiliary
    with pytest.raises(RuntimeError, match=code):
        object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    snapshot = transaction.snapshot()
    assert snapshot.completed_members == () and snapshot.terminal_failure_code == code
    assert snapshot.slow_grads_cleared and parameter.grad is None
    assert wiring.adapter.committed_snapshot() == ()


def test_two_step_marker_to_trainer_keeps_visible_local_primary_exactly_once() -> None:
    wiring, segment, identity, transaction = _fixture(two_steps=True)
    output, _ = _model_marker_output(wiring, segment, identity, transaction)
    expected = sum(token.sum() for token in output["canonical_segment_forward"].locals if token is not None)
    assert expected.abs() > 1e-6
    torch.testing.assert_close(output["primary_consumer_mean"], expected)
    object.__new__(ImaginaireTrainer)._run_canonical_segment_backward(output)
    assert transaction.snapshot().completed_members == (identity,)


def test_canonical_native_scaler_rejection_disposes_before_backward() -> None:
    request, carrier = _bound_request_and_carrier()
    adapter = CanonicalProductionAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(evidence_dim=256),
    )
    result = adapter.scan(request)
    prepared = adapter.prepare_native_inputs(
        request, result, carrier, input_image_key="images", input_video_key="video"
    )
    prepared = adapter.attach_native_preparation(
        prepared,
        input_text_indexes=[[], []],
        sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)],
        gen_data_clean=object(),
        memory_info={},
        data_resolutions=None,
        vae_pixel_shapes=[],
    )
    anchor = torch.nn.Parameter(torch.ones(()))
    split = build_canonical_native_loss_split(
        consumer_identities=prepared.traversal.identities,
        modalities={},
        sample_level_scale=torch.ones(()),
        auxiliary_loss=anchor * 0.0,
        graph_anchor=anchor,
    )
    capability = adapter.bind_native_forward(prepared, split)
    with pytest.raises(RuntimeError, match="CANONICAL_NATIVE_SCALER_UNSUPPORTED"):
        object.__new__(ImaginaireTrainer)._run_canonical_native_backward(
            {"psm_canonical_native_forward": capability},
            SimpleNamespace(is_enabled=lambda: True),
        )
    assert all(parameter.grad is None for parameter in capability.slow_parameters)
    assert adapter._native_forward_capabilities == {}
    assert adapter._scan_requests == set() and adapter._scan_results == {}
    assert request.transaction.snapshot().terminal_failure_code == "CANONICAL_NATIVE_SCALER_UNSUPPORTED"


@pytest.mark.parametrize("scaler_enabled", (True, False))
def test_canonical_training_step_rejects_before_model_forward(scaler_enabled: bool) -> None:
    trainer = object.__new__(ImaginaireTrainer)
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = object() if scaler_enabled else torch.optim.SGD((parameter,), lr=0.1)
    scaler = SimpleNamespace(is_enabled=lambda: scaler_enabled)
    with pytest.raises(RuntimeError, match="rejects scaler or optimizer before scan"):
        trainer.training_step(
            object(), optimizer, None, scaler, {"canonical_production_segment_mode": True}
        )


def test_canonical_native_rejects_batch_slow_parameter_authority_before_backward() -> None:
    request, carrier = _bound_request_and_carrier()
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(evidence_dim=256))
    result = adapter.scan(request)
    prepared = adapter.attach_native_preparation(adapter.prepare_native_inputs(request, result, carrier, input_image_key="images", input_video_key="video"), input_text_indexes=[[], []], sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)], gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[])
    anchor = torch.ones((), requires_grad=True)
    capability = adapter.bind_native_forward(prepared, build_canonical_native_loss_split(consumer_identities=prepared.traversal.identities, modalities={}, sample_level_scale=torch.ones(()), auxiliary_loss=anchor * 0.0, graph_anchor=anchor))
    foreign = torch.nn.Parameter(torch.ones(()))
    foreign.grad = torch.ones(())
    with pytest.raises(RuntimeError, match="CANONICAL_NATIVE_SLOW_PARAMETER_AUTHORITY"):
        object.__new__(ImaginaireTrainer)._run_canonical_native_backward({"psm_canonical_native_forward": capability, "psm_canonical_native_slow_parameters": (foreign,)}, SimpleNamespace(is_enabled=lambda: False, scale=lambda value: value))
    torch.testing.assert_close(foreign.grad, torch.ones(()))
    assert all(parameter.grad is None for parameter in capability.slow_parameters)
    assert request.transaction.snapshot().terminal_failure_code == "CANONICAL_NATIVE_SLOW_PARAMETER_AUTHORITY"


def test_canonical_native_post_mutation_failure_preserves_trainer_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    request, carrier = _bound_request_and_carrier()
    identity = request.member.row_identities[0]
    chronology = request.member.row_chronology[0]
    provenance = request.member.row_provenances[0]
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(
            QueueEpochSnapshot(1, 0, "catalog", (("category", 0),), (("category", (0,)),)),
            (("category", 0),),
            target_distribution=(("category", 1.0),),
            catalog=(CatalogRow(identity, chronology, provenance),),
        )
    )
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="post-mutation")
    member = plan.members[0]
    request = CanonicalProductionSegmentRequest(
        scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, request.segment_batch
    )
    carrier = replace(
        carrier,
        request=request,
        member=member,
        row_identities=member.row_identities,
        row_chronology=member.row_chronology,
    )
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(evidence_dim=256))
    result = adapter.scan(request)
    prepared = adapter.attach_native_preparation(adapter.prepare_native_inputs(request, result, carrier, input_image_key="images", input_video_key="video"), input_text_indexes=[[], []], sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)], gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[])
    anchor = torch.ones((), requires_grad=True)
    capability = adapter.bind_native_forward(prepared, build_canonical_native_loss_split(consumer_identities=prepared.traversal.identities, modalities={}, sample_level_scale=torch.ones(()), auxiliary_loss=anchor * 0.0, graph_anchor=anchor))
    prepared_capabilities: list[CanonicalProductionCommitCapability] = []
    production_prepare_commit = adapter.prepare_commit
    production_frontier_commit = adapter.frontier.commit

    def capture_production_prepare_commit(request, result):
        commit_capability = production_prepare_commit(request, result)
        prepared_capabilities.append(commit_capability)
        return commit_capability

    def fail_after_real_frontier_mutation(member, state):
        production_frontier_commit(member, state)
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(adapter, "prepare_commit", capture_production_prepare_commit)
    monkeypatch.setattr(adapter.frontier, "commit", fail_after_real_frontier_mutation)
    for parameter in capability.slow_parameters:
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="CANONICAL_NATIVE_POST_MUTATION_FAILURE"):
        object.__new__(ImaginaireTrainer)._run_canonical_native_backward(
            {"psm_canonical_native_forward": capability}, SimpleNamespace(is_enabled=lambda: False, scale=lambda value: value)
    )
    assert len(prepared_capabilities) == 1
    assert id(prepared_capabilities[0]) in adapter._commit_capabilities
    assert all(parameter.grad is not None for parameter in capability.slow_parameters)
    assert adapter._commit_capabilities and adapter._scan_requests and adapter.frontier._states
    assert request.transaction.snapshot().terminal_failure_code is None


@pytest.mark.parametrize("phase", ("backward", "prepare", "commit"))
def test_canonical_native_dispatcher_failure_disposes_exact_authority(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    request, carrier = _bound_request_and_carrier()
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(evidence_dim=256))
    result = adapter.scan(request)
    prepared = adapter.attach_native_preparation(adapter.prepare_native_inputs(request, result, carrier, input_image_key="images", input_video_key="video"), input_text_indexes=[[], []], sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)], gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[])
    anchor = torch.ones((), requires_grad=True)
    capability = adapter.bind_native_forward(prepared, build_canonical_native_loss_split(consumer_identities=prepared.traversal.identities, modalities={}, sample_level_scale=torch.ones(()), auxiliary_loss=anchor * 0.0, graph_anchor=anchor))
    for parameter in capability.slow_parameters:
        parameter.grad = torch.ones_like(parameter)
    scheduler_before = request.scheduler.snapshot
    if phase == "backward":
        scaler = SimpleNamespace(is_enabled=lambda: False, scale=lambda value: SimpleNamespace(backward=lambda: (_ for _ in ()).throw(RuntimeError("injected backward failure"))))
    else:
        scaler = SimpleNamespace(is_enabled=lambda: False, scale=lambda value: value)
        target = adapter.prepare_commit if phase == "prepare" else adapter.commit_success
        monkeypatch.setattr(adapter, target.__name__, lambda *args: (_ for _ in ()).throw(RuntimeError(f"injected {phase} failure")))
    with pytest.raises(RuntimeError, match="CANONICAL_NATIVE_(BACKWARD|COMMIT)_FAILURE"):
        object.__new__(ImaginaireTrainer)._run_canonical_native_backward({"psm_canonical_native_forward": capability}, scaler)
    assert all(parameter.grad is None for parameter in capability.slow_parameters)
    assert adapter._native_forward_capabilities == {}
    assert adapter._scan_requests == set() and adapter._scan_results == {}
    assert request.scheduler.snapshot == scheduler_before
    assert request.transaction.snapshot().completed_members == ()
    assert request.transaction.snapshot().terminal_failure_code in {"CANONICAL_NATIVE_BACKWARD_FAILURE", "CANONICAL_NATIVE_COMMIT_FAILURE"}

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
    ChronologyCountRecord,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
    queue_permutation,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalNativeModalityTerms,
    CanonicalProductionAdapter,
    CanonicalProductionCommitCapability,
    CanonicalProductionSegmentRequest,
    CanonicalRawRowCarrier,
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
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.production_segment_wiring import run_native_forward_for_test
from cosmos_framework.model.generator.mot.production_segment_wiring_test import _fixture
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.trainer import ImaginaireTrainer


def _model_marker_output(wiring, segment, identity, transaction):
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    return model._canonical_local_memory_segment_forward(
        {
            "canonical_segment": segment,
            "canonical_identity": identity,
            "canonical_transaction": transaction,
            "canonical_wiring": wiring,
            "canonical_member_index": 0,
        },
        0,
    )


def _synthetic_carrier_for_request(request: CanonicalProductionSegmentRequest) -> CanonicalRawRowCarrier:
    """Build an in-memory canonical carrier for one frozen synthetic member."""
    member, batch = request.member, request.segment_batch
    count = member.planned_n_valid
    identity = member.row_identities[0]
    raw = tuple({"canonical_identity": (identity.slot_id, identity.episode_id, step)} for step in range(count))
    samples = tuple({"canonical_identity": value["canonical_identity"], "images": value, "sequence_plan": SequencePlan(has_text=False)} for value in raw)
    return CanonicalRawRowCarrier(
        request, member, batch, member.row_identities, member.row_chronology, (raw,), (samples,),
        {"images": [sample["images"] for sample in samples], "sequence_plan": [sample["sequence_plan"] for sample in samples]},
        raw_row_source_identities=(tuple((identity.slot_id, identity.episode_id, identity.source_digest, step) for step in range(count)),),
        row_model_source_rows=(raw,),
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


def test_canonical_native_dispatcher_scales_and_backwards_exactly_once() -> None:
    request, carrier = _bound_request_and_carrier()
    identity = request.member.row_identities[0]
    chronology = request.member.row_chronology[0]
    provenance = request.member.row_provenances[0]
    second_identity = replace(identity, cursor=1, segment_id=1)
    second_chronology = replace(chronology, consumer_step_stop_exclusive=5, consumer_step_start=0)
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(
            QueueEpochSnapshot(1, 0, "catalog", (("category", 0),), (("category", (0,)),)),
            (("category", 0),), target_distribution=(("category", 1.0),),
            catalog=(CatalogRow(identity, chronology, provenance), CatalogRow(second_identity, second_chronology, provenance)),
        )
    )
    plan = scheduler.freeze_plan(slot_groups=((0,), (0,)), plan_chain_id="normal-non-degenerate-dispatch")
    assert (tuple(member.planned_n_valid for member in plan.members), plan.original_n_valid_window, plan.original_ga_effective) == ((2, 5), 7, 2)
    request = CanonicalProductionSegmentRequest(
        scheduler, plan, CanonicalBatchWindowTransaction(plan), plan.members[0], 0, request.segment_batch
    )
    carrier = replace(carrier, request=request, member=plan.members[0], row_identities=plan.members[0].row_identities, row_chronology=plan.members[0].row_chronology)
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(evidence_dim=256))
    result = adapter.scan(request)
    prepared = adapter.attach_native_preparation(adapter.prepare_native_inputs(request, result, carrier, input_image_key="images", input_video_key="video"), input_text_indexes=[[], []], sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)], gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[])
    anchor = torch.nn.Parameter(torch.ones(()))
    capability = adapter.bind_native_forward(prepared, build_canonical_native_loss_split(consumer_identities=prepared.traversal.identities, modalities={"vision": CanonicalNativeModalityTerms(torch.tensor([7.0, 7.0], requires_grad=True), (0, 1), 1.0)}, sample_level_scale=torch.ones(()), auxiliary_loss=anchor * 3.0, graph_anchor=anchor))

    class CountingScaler:
        def __init__(self) -> None:
            self.scaled: list[torch.Tensor] = []
            self.backward_calls = 0

        def is_enabled(self) -> bool:
            return False

        def scale(self, value: torch.Tensor):
            self.scaled.append(value)
            owner = self

            class CountedTensor:
                def backward(self) -> None:
                    owner.backward_calls += 1
                    value.backward()

            return CountedTensor()

    scaler = CountingScaler()
    objective = object.__new__(ImaginaireTrainer)._run_canonical_native_backward(
        {"psm_canonical_native_forward": capability}, scaler
    )
    assert scaler.scaled == [objective] and scaler.backward_calls == 1
    torch.testing.assert_close(objective, torch.tensor(2 / 7 * 7 + 3 / 2))
    assert request.transaction.snapshot().completed_members == (0,)


def test_canonical_native_dispatcher_recovery_scales_and_commits_exact_suffix_once() -> None:
    """Exercise the frozen ``(5, 3)`` recovery plan through the real dispatcher seam."""
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    counts = (2, 5, 3)
    rows = tuple(
        CatalogRow(
            SegmentIdentity(0, "episode", "category", index, index, "source", index == 2),
            ChronologyCountRecord(0, "episode", "category", "source", 0, count, index == 2, "manifest"),
            provenance,
        )
        for index, count in enumerate(counts)
    )
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(
            QueueEpochSnapshot(
                1, 0, "catalog", (("category", 0),),
                (("category", queue_permutation(queue_seed=1, epoch=0, category="category", catalog_size=1)),),
            ),
            (("category", 0),), target_distribution=(("category", 1.0),), catalog=rows,
        )
    )
    plan = scheduler.freeze_plan(slot_groups=((0,), (0,), (0,)), plan_chain_id="recovery-native-dispatch")
    assert (tuple(member.planned_n_valid for member in plan.members), plan.original_n_valid_window, plan.original_ga_effective) == ((2, 5, 3), 10, 3)

    def batch(member) -> SegmentBatch:
        count = member.planned_n_valid
        identity = member.row_identities[0]
        return SegmentBatch(
            torch.zeros(1, count, 96), (tuple(f"s{index}" for index in range(count)),),
            torch.ones(1, count, dtype=torch.bool), torch.arange(count).reshape(1, count),
            torch.zeros(1, count, 96), torch.zeros(1, count, 10),
            torch.tensor([[False, *([True] * (count - 1))]]), torch.tensor([[-1, *range(count - 1)]]),
            torch.tensor([identity.slot_id]), (identity.episode_id,), (identity.category,), provenance,
        )

    class CountingScaler:
        def __init__(self) -> None:
            self.scaled: list[torch.Tensor] = []
            self.backward_calls = 0

        def is_enabled(self) -> bool:
            return False

        def scale(self, value: torch.Tensor):
            self.scaled.append(value)
            owner = self

            class CountedTensor:
                def backward(self) -> None:
                    owner.backward_calls += 1
                    value.backward()

            return CountedTensor()

    adapter = CanonicalProductionAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(evidence_dim=256),
    )
    original = CanonicalBatchWindowTransaction(plan)
    prefix = CanonicalProductionSegmentRequest(scheduler, plan, original, plan.members[0], 0, batch(plan.members[0]))
    prefix_result = adapter.scan(prefix)
    prefix_commit = adapter.prepare_commit(prefix, prefix_result)
    original.mark_backward_started(0)
    adapter.commit_success(prefix_commit)
    prefix_frontier = dict(adapter.frontier._states)
    failed = CanonicalProductionSegmentRequest(scheduler, plan, original, plan.members[1], 1, batch(plan.members[1]))
    recovery_capability = adapter.derive_suffix_recovery(
        failed, source_transient=adapter.declare_retryable_source_transient(failed)
    )
    recovered = adapter.consume_suffix_recovery(
        recovery_capability, segment_batches=tuple(batch(member) for member in plan.members[1:])
    )
    assert (tuple(request.member.planned_n_valid for request in recovered), recovered[0].plan.original_n_valid_window, recovered[0].plan.original_ga_effective) == ((5, 3), 8, 2)
    scaler = CountingScaler()
    objectives = []
    for request, primary_value in zip(recovered, (13.0, 17.0), strict=True):
        result = adapter.scan(request)
        prepared = adapter.attach_native_preparation(
            adapter.prepare_native_inputs(
                request, result, _synthetic_carrier_for_request(request), input_image_key="images", input_video_key="video"
            ),
            input_text_indexes=[[] for _ in range(request.member.planned_n_valid)],
            sequence_plans=[SequencePlan(has_text=False, has_local_memory=prefix is not None) for prefix in result.gathered.local_prefixes],
            gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[],
        )
        anchor = torch.nn.Parameter(torch.ones(()))
        capability = adapter.bind_native_forward(
            prepared,
            build_canonical_native_loss_split(
                consumer_identities=prepared.traversal.identities,
                modalities={"vision": CanonicalNativeModalityTerms(torch.full((request.member.planned_n_valid,), primary_value, requires_grad=True), tuple(range(request.member.planned_n_valid)), 1.0)},
                sample_level_scale=torch.ones(()), auxiliary_loss=anchor * 5.0, graph_anchor=anchor,
            ),
        )
        objectives.append(object.__new__(ImaginaireTrainer)._run_canonical_native_backward({"psm_canonical_native_forward": capability}, scaler))
    torch.testing.assert_close(torch.stack(objectives), torch.tensor((10.625, 8.875)))
    assert scaler.scaled == objectives and scaler.backward_calls == 2
    adapter.complete_suffix_recovery(recovery_capability.recovery)
    assert prefix_frontier and original.snapshot().slow_grads_cleared
    assert original.snapshot().suffix_recovery_reconciled and scheduler._frozen_transitions == []
    assert not adapter._suffix_recovery_requests and not adapter._suffix_recovery_scans


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

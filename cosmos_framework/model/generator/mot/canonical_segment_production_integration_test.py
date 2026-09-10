from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
    compute_flow_matching_loss,
    compute_flow_matching_loss_terms,
)
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
    CanonicalRawRowCarrier,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.omni_mot_model import (
    OmniMoTModel,
    _canonical_production_adapter_from_model,
    _canonical_production_request_from_batch,
)


def _request() -> CanonicalProductionSegmentRequest:
    return CanonicalProductionSegmentRequest.__new__(CanonicalProductionSegmentRequest)


class _UnitTimeWeight:
    def train_time_weight(self, timesteps: torch.Tensor, tensor_kwargs_fp32: dict) -> torch.Tensor:
        return torch.ones_like(timesteps, **tensor_kwargs_fp32)


def test_flow_matching_terms_preserve_legacy_wrapper_and_weighted_population() -> None:
    prediction = [torch.tensor([[1.0], [3.0]], requires_grad=True), torch.tensor([[2.0], [4.0]], requires_grad=True)]
    target = [torch.zeros_like(value) for value in prediction]
    masks = [torch.zeros(2, 1), torch.zeros(2, 1)]
    kwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
    terms = compute_flow_matching_loss_terms(
        prediction, target, masks, torch.ones(2, 2), True, _UnitTimeWeight(), kwargs
    )
    legacy_mean, legacy_unweighted = compute_flow_matching_loss(
        prediction, target, masks, torch.ones(2, 2), True, _UnitTimeWeight(), kwargs
    )
    torch.testing.assert_close(terms.weighted_mean, terms.weighted_per_instance.mean())
    torch.testing.assert_close(legacy_mean, terms.weighted_mean)
    torch.testing.assert_close(legacy_unweighted, terms.unweighted_per_instance)


def _bound_request_and_carrier() -> tuple[CanonicalProductionSegmentRequest, CanonicalRawRowCarrier]:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    segment_batch = SegmentBatch(
        torch.zeros(1, 2, 96),
        (("s0", "s1"),),
        torch.tensor([[True, True]]),
        torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96),
        torch.zeros(1, 2, 10),
        torch.tensor([[False, True]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([0]),
        ("episode",),
        ("category",),
        provenance,
    )
    member = MicrobatchPlanMember(
        0,
        (SegmentIdentity(0, "episode", "category", 0, 0, "source"),),
        (provenance,),
        (ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest"),),
        (2,),
        2,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)),
        (),
    )
    plan = CanonicalGAWindowPlan((member,), 2, 1, "integration")
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    request = CanonicalProductionSegmentRequest(
        scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, segment_batch
    )
    raw_rows = ((
        {"canonical_identity": (0, "episode", 0)},
        {"canonical_identity": (0, "episode", 1)},
    ),)
    samples = ((
        {"canonical_identity": (0, "episode", 0), "images": raw_rows[0][0]},
        {"canonical_identity": (0, "episode", 1), "images": raw_rows[0][1]},
    ),)
    raw_plans = [SequencePlan(has_text=False), SequencePlan(has_text=False)]
    for sample, plan_item in zip(samples[0], raw_plans, strict=True):
        sample["sequence_plan"] = plan_item
    carrier = CanonicalRawRowCarrier(
        request,
        member,
        segment_batch,
        member.row_identities,
        member.row_chronology,
        raw_rows,
        samples,
        {
            "images": [sample["images"] for sample in samples[0]],
            "sequence_plan": raw_plans,
        },
        raw_row_source_identities=(((0, "episode", "source", 0), (0, "episode", "source", 1)),),
        row_model_source_rows=raw_rows,
    )
    return request, carrier


def test_activation_matrix_fails_before_legacy_routes() -> None:
    assert _canonical_production_request_from_batch(local_ttt_enabled=False, data_batch={}) is None
    with pytest.raises(ValueError, match="require local_ttt_enabled"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=False, data_batch={"canonical_production_segment_mode": True}
        )
    with pytest.raises(ValueError, match="require local_ttt_enabled"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=False, data_batch={"canonical_production_segment_carrier": object()}
        )
    with pytest.raises(ValueError, match="exact canonical-production mode"):
        _canonical_production_request_from_batch(local_ttt_enabled=True, data_batch={})
    with pytest.raises(TypeError, match="CanonicalProductionSegmentRequest"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=True,
            data_batch={"canonical_production_segment_mode": True, "canonical_production_segment_input": object()},
        )
    request = _request()
    carrier = CanonicalRawRowCarrier.__new__(CanonicalRawRowCarrier)
    with pytest.raises(TypeError, match="CanonicalRawRowCarrier"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=True,
            data_batch={"canonical_production_segment_mode": True, "canonical_production_segment_input": request},
        )
    assert _canonical_production_request_from_batch(
        local_ttt_enabled=True,
        data_batch={
            "canonical_production_segment_mode": True,
            "canonical_production_segment_input": request,
            "canonical_production_segment_carrier": carrier,
        },
    ) is request
    with pytest.raises(ValueError, match="conflicts"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=True,
            data_batch={
                "canonical_production_segment_mode": True,
                "canonical_production_segment_input": request,
                "canonical_production_segment_carrier": carrier,
                "canonical_local_memory_segment": True,
            },
        )


def test_adapter_binds_only_the_exact_registered_canonical_modules() -> None:
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    model = SimpleNamespace(net=SimpleNamespace(local_history_runtime=SimpleNamespace(encoder=encoder, recurrent_backend=core)))
    adapter = _canonical_production_adapter_from_model(model)
    assert adapter.encoder is encoder and adapter.core is core
    assert _canonical_production_adapter_from_model(model) is adapter
    model._canonical_production_adapter = object()
    with pytest.raises(RuntimeError, match="not bound"):
        _canonical_production_adapter_from_model(model)


def test_canonical_branch_rejects_context_parallelism_before_adapter_lookup() -> None:
    request = _request()
    model = SimpleNamespace(parallel_dims=SimpleNamespace(cp_enabled=True))
    with pytest.raises(RuntimeError, match="rejects context parallelism before scan"):
        OmniMoTModel._canonical_production_segment_forward(model, request, object(), 1)


def test_canonical_safe_preparation_adapts_only_gathered_prefixes() -> None:
    calls: list[str] = []
    plans = [SequencePlan(has_text=False), SequencePlan(has_text=False)]
    clean = SimpleNamespace(x0_tokens_local_memory=None, raw_state_vision=[])
    model = SimpleNamespace(
        _load_and_tokenize_text_data=lambda batch, iteration: calls.append("text") or [[1], [2]],
        input_video_key="video",
        input_image_key="image",
        get_data_and_condition=lambda batch, iteration: calls.append("clean") or clean,
        memory_init_training=lambda value, batch, indexes: (calls.append("memory") or value, {}),
        _get_vae_pixel_shapes=lambda raw: [],
    )
    carrier = SimpleNamespace(model_data_batch={"text_token_ids": []})
    result = SimpleNamespace(gathered=SimpleNamespace(item_count=2, local_prefixes=(None, "prefix")))
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: calls.append("plan") or plans
    try:
        prepared = OmniMoTModel._prepare_canonical_production_inputs(model, carrier, result, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    assert calls == ["text", "plan", "clean", "memory"]
    assert [plan.has_local_memory for plan in plans] == [False, False]
    assert [plan.has_local_memory for plan in prepared[1]] == [False, True]
    assert clean.x0_tokens_local_memory == ["prefix"]


def test_canonical_forward_preflights_foreign_batch_before_adapter_creation() -> None:
    request, carrier = _bound_request_and_carrier()
    foreign = CanonicalRawRowCarrier(
        carrier.request,
        carrier.member,
        carrier.segment_batch,
        carrier.row_identities,
        carrier.row_chronology,
        carrier.raw_rows,
        carrier.row_model_samples,
        {"images": [dict(value) for value in carrier.model_data_batch["images"]]},
        raw_row_source_identities=carrier.raw_row_source_identities,
        row_model_source_rows=carrier.row_model_source_rows,
    )
    model = SimpleNamespace(parallel_dims=None, input_image_key="images", input_video_key="video")
    with pytest.raises(Exception, match="model batch source is foreign"):
        OmniMoTModel._canonical_production_segment_forward(model, request, foreign, 1)
    assert not hasattr(model, "_canonical_production_adapter")


def test_canonical_forward_rejects_foreign_raw_source_before_adapter_creation() -> None:
    request, carrier = _bound_request_and_carrier()
    foreign_source = CanonicalRawRowCarrier(
        carrier.request,
        carrier.member,
        carrier.segment_batch,
        carrier.row_identities,
        carrier.row_chronology,
        carrier.raw_rows,
        carrier.row_model_samples,
        carrier.model_data_batch,
        raw_row_source_identities=(((0, "episode", "foreign-source", 0), (0, "episode", "source", 1)),),
        row_model_source_rows=carrier.row_model_source_rows,
    )
    model = SimpleNamespace(parallel_dims=None, input_image_key="images", input_video_key="video")
    with pytest.raises(Exception, match="model sample source is foreign"):
        OmniMoTModel._canonical_production_segment_forward(model, request, foreign_source, 1)
    assert not hasattr(model, "_canonical_production_adapter")


def test_canonical_forward_aborts_real_pending_scan_before_hard_stop() -> None:
    request, carrier = _bound_request_and_carrier()
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    calls: list[str] = []
    clean = SimpleNamespace(x0_tokens_local_memory=None, raw_state_vision=[])

    model = SimpleNamespace(
        parallel_dims=None,
        input_image_key="images",
        input_video_key="video",
        net=SimpleNamespace(local_history_runtime=SimpleNamespace(encoder=encoder, recurrent_backend=core)),
        _load_and_tokenize_text_data=lambda batch, iteration: calls.append("text") or [[1], [2]],
        get_data_and_condition=lambda batch, iteration: calls.append("clean") or clean,
        memory_init_training=lambda value, batch, indexes: (calls.append("memory") or value, {}),
        _get_vae_pixel_shapes=lambda raw: [],
    )
    model._prepare_canonical_production_inputs = (
        lambda prepared_carrier, result, iteration: OmniMoTModel._prepare_canonical_production_inputs(
            model, prepared_carrier, result, iteration
        )
    )
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: calls.append("plan") or kwargs["data_batch"]["sequence_plan"]
    try:
        with pytest.raises(RuntimeError, match="native forward seam is unavailable"):
            OmniMoTModel._canonical_production_segment_forward(model, request, carrier, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    assert calls == ["text", "plan", "clean", "memory"]
    assert [plan.has_local_memory for plan in carrier.model_data_batch["sequence_plan"]] == [False, False]
    assert model._canonical_production_adapter._scan_requests == set()
    assert model._canonical_production_adapter._scan_results == {}


def _production_model(*, memory_init_training) -> tuple[SimpleNamespace, list[str]]:
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    calls: list[str] = []
    clean = SimpleNamespace(x0_tokens_local_memory=None, raw_state_vision=[])
    model = SimpleNamespace(
        parallel_dims=None,
        input_image_key="images",
        input_video_key="video",
        net=SimpleNamespace(local_history_runtime=SimpleNamespace(encoder=encoder, recurrent_backend=core)),
        _load_and_tokenize_text_data=lambda batch, iteration: calls.append("text") or [[1], [2]],
        get_data_and_condition=lambda batch, iteration: calls.append("clean") or clean,
        memory_init_training=memory_init_training,
        _get_vae_pixel_shapes=lambda raw: [],
    )
    model._prepare_canonical_production_inputs = (
        lambda prepared_carrier, result, iteration: OmniMoTModel._prepare_canonical_production_inputs(
            model, prepared_carrier, result, iteration
        )
    )
    return model, calls


def _assert_aborted_without_commit(
    request: CanonicalProductionSegmentRequest, adapter: CanonicalProductionAdapter, scheduler_before: object
) -> None:
    assert adapter._scan_requests == set()
    assert adapter._scan_results == {}
    assert adapter._commit_capabilities == set()
    assert adapter.frontier._states == {}
    assert request.scheduler.snapshot == scheduler_before
    assert request.transaction.completed_members == []
    assert request.transaction.backward_started is False


def test_canonical_forward_aborts_gather_mismatch_without_commit() -> None:
    request, carrier = _bound_request_and_carrier()
    model, _ = _production_model(memory_init_training=lambda value, batch, indexes: (value, {}))
    adapter = CanonicalProductionAdapter(
        model.net.local_history_runtime.encoder, model.net.local_history_runtime.recurrent_backend
    )
    model._canonical_production_adapter = adapter
    scan = adapter.scan

    def scan_with_mismatched_gather(current_request):
        result = scan(current_request)
        gathered = dataclasses.replace(
            result.gathered,
            payloads=result.gathered.payloads[:-1],
            local_prefixes=result.gathered.local_prefixes[:-1],
            identities=result.gathered.identities[:-1],
        )
        mismatch = dataclasses.replace(result, gathered=gathered)
        adapter._scan_results.pop(id(result))
        adapter._scan_results[id(mismatch)] = current_request
        return mismatch

    adapter.scan = scan_with_mismatched_gather
    scheduler_before = request.scheduler.snapshot
    with pytest.raises(RuntimeError, match="gathered result differs"):
        OmniMoTModel._canonical_production_segment_forward(model, request, carrier, 1)
    _assert_aborted_without_commit(request, adapter, scheduler_before)


def test_canonical_forward_aborts_memory_initialization_exception_without_commit() -> None:
    request, carrier = _bound_request_and_carrier()
    calls: list[str] = []
    model, _ = _production_model(
        memory_init_training=lambda value, batch, indexes: calls.append("memory")
        or (_ for _ in ()).throw(RuntimeError("injected memory initialization failure"))
    )
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: kwargs["data_batch"]["sequence_plan"]
    scheduler_before = request.scheduler.snapshot
    try:
        with pytest.raises(RuntimeError, match="injected memory initialization failure"):
            OmniMoTModel._canonical_production_segment_forward(model, request, carrier, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    _assert_aborted_without_commit(request, model._canonical_production_adapter, scheduler_before)
    assert calls == ["memory"]


def test_canonical_forward_never_calls_ordinary_or_legacy_preparation() -> None:
    request, carrier = _bound_request_and_carrier()
    model, _ = _production_model(memory_init_training=lambda value, batch, indexes: (value, {}))
    for name in (
        "_prepare_training_data",
        "_get_training_inputs",
        "_inject_local_history",
        "_ttt_local_memory_tokens",
    ):
        setattr(model, name, lambda *args, _name=name, **kwargs: pytest.fail(f"unexpected {_name} call"))
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: kwargs["data_batch"]["sequence_plan"]
    try:
        with pytest.raises(RuntimeError, match="native forward seam is unavailable"):
            OmniMoTModel._canonical_production_segment_forward(model, request, carrier, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original


def test_canonical_safe_preparation_rejects_post_clean_ordinary_local_memory() -> None:
    calls: list[str] = []
    plans = [SequencePlan(has_text=False), SequencePlan(has_text=False)]
    clean = SimpleNamespace(x0_tokens_local_memory=None, raw_state_vision=[])

    def materialize(batch, iteration):
        calls.append("clean")
        batch["local_memory"] = [object()]
        return clean

    model = SimpleNamespace(
        _load_and_tokenize_text_data=lambda batch, iteration: calls.append("text") or [[1], [2]],
        input_video_key="video",
        input_image_key="image",
        get_data_and_condition=materialize,
        memory_init_training=lambda *args: pytest.fail("memory init must not run"),
        _get_vae_pixel_shapes=lambda raw: [],
    )
    carrier = SimpleNamespace(model_data_batch={"text_token_ids": []})
    result = SimpleNamespace(gathered=SimpleNamespace(item_count=2, local_prefixes=(None, "prefix")))
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: calls.append("plan") or plans
    try:
        with pytest.raises(RuntimeError, match="introduced an ordinary Local payload"):
            OmniMoTModel._prepare_canonical_production_inputs(model, carrier, result, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    assert calls == ["text", "plan", "clean"]
    assert [plan.has_local_memory for plan in plans] == [False, False]


def test_training_step_no_local_falls_through_without_canonical_construction() -> None:
    calls: list[str] = []
    model = SimpleNamespace(
        config=SimpleNamespace(local_ttt_enabled=False),
        _get_training_inputs=lambda batch, iteration: calls.append("ordinary")
        or (_ for _ in ()).throw(RuntimeError("ordinary no-local path reached")),
        _canonical_production_segment_forward=lambda *args: pytest.fail("canonical branch must not run"),
    )
    with pytest.raises(RuntimeError, match="ordinary no-local path reached"):
        OmniMoTModel.training_step(model, {}, 1)
    assert calls == ["ordinary"]
    assert not hasattr(model, "_canonical_production_adapter")

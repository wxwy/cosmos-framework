from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
    for raw, sample in zip(raw_rows[0], samples[0], strict=True):
        raw["canonical_model_sample"] = sample
    carrier = CanonicalRawRowCarrier(
        request,
        member,
        segment_batch,
        member.row_identities,
        member.row_chronology,
        raw_rows,
        samples,
        {"images": [sample["images"] for sample in samples[0]]},
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
    plans = [SimpleNamespace(has_local_memory=False), SimpleNamespace(has_local_memory=False)]
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
        OmniMoTModel._prepare_canonical_production_inputs(model, carrier, result, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    assert calls == ["text", "plan", "clean", "memory"]
    assert [plan.has_local_memory for plan in plans] == [False, True]
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
    )
    model = SimpleNamespace(parallel_dims=None, input_image_key="images", input_video_key="video")
    with pytest.raises(Exception, match="model batch source is foreign"):
        OmniMoTModel._canonical_production_segment_forward(model, request, foreign, 1)
    assert not hasattr(model, "_canonical_production_adapter")


def test_canonical_forward_aborts_real_pending_scan_before_hard_stop() -> None:
    request, carrier = _bound_request_and_carrier()
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    calls: list[str] = []

    def hard_stop(*_args) -> None:
        calls.append("helper")
        raise RuntimeError("controlled pre-packer stop")

    model = SimpleNamespace(
        parallel_dims=None,
        input_image_key="images",
        input_video_key="video",
        net=SimpleNamespace(local_history_runtime=SimpleNamespace(encoder=encoder, recurrent_backend=core)),
        _prepare_canonical_production_inputs=hard_stop,
    )
    with pytest.raises(RuntimeError, match="controlled pre-packer stop"):
        OmniMoTModel._canonical_production_segment_forward(model, request, carrier, 1)
    assert calls == ["helper"]
    assert model._canonical_production_adapter._scan_requests == set()
    assert model._canonical_production_adapter._scan_results == {}

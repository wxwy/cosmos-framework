from __future__ import annotations

import pytest
from types import SimpleNamespace

from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import CanonicalProductionSegmentRequest
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.omni_mot_model import (
    OmniMoTModel,
    _canonical_production_adapter_from_model,
    _canonical_production_request_from_batch,
)


def _request() -> CanonicalProductionSegmentRequest:
    return CanonicalProductionSegmentRequest.__new__(CanonicalProductionSegmentRequest)


def test_activation_matrix_fails_before_legacy_routes() -> None:
    assert _canonical_production_request_from_batch(local_ttt_enabled=False, data_batch={}) is None
    with pytest.raises(ValueError, match="require local_ttt_enabled"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=False, data_batch={"canonical_production_segment_mode": True}
        )
    with pytest.raises(ValueError, match="exact canonical-production mode"):
        _canonical_production_request_from_batch(local_ttt_enabled=True, data_batch={})
    with pytest.raises(TypeError, match="CanonicalProductionSegmentRequest"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=True,
            data_batch={"canonical_production_segment_mode": True, "canonical_production_segment_input": object()},
        )
    request = _request()
    assert _canonical_production_request_from_batch(
        local_ttt_enabled=True,
        data_batch={"canonical_production_segment_mode": True, "canonical_production_segment_input": request},
    ) is request
    with pytest.raises(ValueError, match="conflicts"):
        _canonical_production_request_from_batch(
            local_ttt_enabled=True,
            data_batch={
                "canonical_production_segment_mode": True,
                "canonical_production_segment_input": request,
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
    object.__setattr__(request, "carrier", object())
    model = SimpleNamespace(parallel_dims=SimpleNamespace(cp_enabled=True))
    with pytest.raises(RuntimeError, match="rejects context parallelism before scan"):
        OmniMoTModel._canonical_production_segment_forward(model, request, 1)


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
    request = SimpleNamespace(carrier=SimpleNamespace(model_data_batch={"text_token_ids": []}))
    result = SimpleNamespace(gathered=SimpleNamespace(item_count=2, local_prefixes=(None, "prefix")))
    import cosmos_framework.model.generator.omni_mot_model as module

    original = module.build_sequence_plans_from_data_batch
    module.build_sequence_plans_from_data_batch = lambda **kwargs: calls.append("plan") or plans
    try:
        OmniMoTModel._prepare_canonical_production_inputs(model, request, result, 1)
    finally:
        module.build_sequence_plans_from_data_batch = original
    assert calls == ["text", "plan", "clean", "memory"]
    assert [plan.has_local_memory for plan in plans] == [False, True]
    assert clean.x0_tokens_local_memory == ["prefix"]

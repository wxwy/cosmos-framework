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

from __future__ import annotations

import pytest

from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import CanonicalProductionSegmentRequest
from cosmos_framework.model.generator.omni_mot_model import _canonical_production_request_from_batch


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

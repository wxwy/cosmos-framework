from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)


def _state() -> ContinualTTTFastState:
    return ContinualTTTFastState(*(torch.ones(1, 1) for _ in range(4)))


def test_sidecar_uses_canonical_cursor_and_terminal_reset() -> None:
    sidecar = LocalMemorySegmentSidecar()
    first = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    sidecar.commit(first, _state())
    carried = sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))
    assert carried is not None and not carried[0].requires_grad
    with pytest.raises(ValueError, match="canonical continuation"):
        sidecar.read(SegmentIdentity(2, "episode", "suite", 2, 2, "source"))
    terminal = SegmentIdentity(2, "episode", "suite", 1, 1, "source", training_stream_end=True)
    sidecar.commit(terminal, _state())
    assert sidecar.read(SegmentIdentity(2, "replacement", "suite", 0, 2, "source")) is None


def test_adapter_scans_masked_segment_and_preserves_gather_identity() -> None:
    payload0, payload1 = object(), object()
    segment = SegmentBatch(
        consumer_visual_summary=torch.randn(1, 3, 96),
        consumer_payload=((payload0, payload1, None),),
        consumer_valid=torch.tensor([[True, True, False]]),
        consumer_step=torch.tensor([[0, 1, -1]]),
        evidence_visual_summary_prev=torch.randn(1, 3, 96),
        evidence_executed_action_prev=torch.randn(1, 3, 10),
        evidence_valid=torch.tensor([[False, True, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1]]),
        slot_id=torch.tensor([2]), episode_id=("episode",), category=("suite",),
        segment_provenance=SegmentProvenance("m", "c", "source", 0),
    )
    encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    core = ContinualTTTLocalMemoryCore(ttt_tbptt_steps=3)
    adapter = CanonicalLocalMemorySegmentAdapter(encoder, core, LocalMemorySegmentSidecar())
    identity = SegmentIdentity(2, "episode", "suite", 0, 0, "source")
    result = adapter.scan(segment, identity=identity)
    assert result.payloads == (payload0, payload1)
    assert result.locals[0] is None and result.locals[1] is not None
    assert result.identities == ((2, "episode", 0), (2, "episode", 1))
    assert result.state_out.fast_in_weight.requires_grad
    adapter.commit(identity, result)
    carried = adapter.sidecar.read(SegmentIdentity(2, "episode", "suite", 1, 1, "source"))
    assert carried is not None and not carried.fast_in_weight.requires_grad

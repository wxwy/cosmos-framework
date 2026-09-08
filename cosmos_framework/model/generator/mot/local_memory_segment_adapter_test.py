from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTFastState
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentIdentity
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import LocalMemorySegmentSidecar


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

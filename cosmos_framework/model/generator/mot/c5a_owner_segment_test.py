import torch

from .c5a_owner_segment import AdmissionCapability, C5AOwnerSegmentCPU
from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _runtime() -> C5AOwnerSegmentCPU:
    return C5AOwnerSegmentCPU(LocalEvidenceEncoder(evidence_dim=256, visual_dim=3, action_dim=2), ContinualTTTLocalMemoryCore())


def test_source_key_replay_precedes_new_binding() -> None:
    runtime = _runtime()
    cap = AdmissionCapability("o", "s", 0, b"x")
    runtime.begin()
    binding = runtime.admit(cap, encoded=torch.zeros(256))
    assert binding == (0, 0, 0, 0)
    runtime._pending.replay[cap.source_key] = torch.ones(1, 1, 32)
    cached = runtime.admit(cap, encoded=torch.zeros(256))
    assert torch.equal(cached, torch.ones(1, 1, 32))
    assert len(runtime._pending.rows) == 1


def test_committed_replay_is_detached_and_zero_write() -> None:
    runtime = _runtime()
    cap = AdmissionCapability("o", "s", 0, b"x")
    runtime.begin()
    runtime.admit(cap, encoded=torch.zeros(256))
    runtime._pending.replay[cap.source_key] = torch.ones(1, 1, 32, requires_grad=True)
    runtime.commit()
    runtime.begin()
    value = runtime.admit(cap, encoded=torch.zeros(256))
    assert value.grad_fn is None and runtime.c5_write_count == 0


def test_forged_capability_is_rejected() -> None:
    runtime = _runtime()
    runtime.begin()
    forged = AdmissionCapability("o", "s", 0, bytearray(b"x"))  # type: ignore[arg-type]
    try:
        runtime.admit(forged, encoded=torch.zeros(256))
    except ValueError:
        pass
    else:
        raise AssertionError("non-bytes source must be rejected")

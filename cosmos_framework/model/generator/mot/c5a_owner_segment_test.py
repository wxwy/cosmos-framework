from dataclasses import replace

import pytest
import torch

from .c5a_owner_segment import AdmissionAuthority, AdmissionCapability, C5AOwnerSegmentCPU
from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _source() -> dict[str, torch.Tensor]:
    return {"history_visual_summary": torch.zeros(1, 1, 96), "local_history_action": torch.zeros(1, 1, 10),
            "history_age_steps": torch.zeros(1, 1, dtype=torch.long), "history_dt_s": torch.zeros(1, 1, 1),
            "history_mask": torch.ones(1, 1, dtype=torch.bool)}


def _runtime() -> tuple[AdmissionAuthority, C5AOwnerSegmentCPU, dict[str, torch.Tensor]]:
    source = _source(); authority = AdmissionAuthority()
    return authority, C5AOwnerSegmentCPU(authority, LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore()), source


def test_materialize_then_commit_and_replay_without_second_write() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); tokens = runtime.materialize("a"); runtime.commit("a")
    writes = runtime.c5_write_count; runtime.begin("a"); replay = runtime.admit(cap, source=source)
    assert torch.equal(replay, tokens[0].detach()) and runtime.c5_write_count == writes


def test_conflicting_digest_and_forged_capability_fail_before_c5() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source)
    conflict = replace(cap, source_bytes=b"bad")
    with pytest.raises(ValueError, match="conflicting"):
        runtime.admit(conflict, source=source)
    forged = replace(cap, _seal=object())
    with pytest.raises(ValueError, match="untrusted"):
        runtime.admit(forged, source=source)


def test_abort_discards_pending_candidate() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); runtime.materialize("a"); runtime.abort("a")
    assert "a" not in runtime._state_by_owner and runtime.c5_write_count == 1


def test_committed_chronology_rejects_skip_and_reset_restarts_epoch() -> None:
    authority, runtime, source = _runtime()
    cap0 = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap0, source=source); runtime.commit("a")
    cap2 = authority.issue(owner_key="a", source_identity="s", source_timestep=2, source=source)
    runtime.begin("a")
    with pytest.raises(ValueError, match="contiguous"):
        runtime.admit(cap2, source=source)
    runtime.abort("a"); runtime.reset("a")
    cap0b = authority.issue(owner_key="a", source_identity="s2", source_timestep=0, source=source)
    runtime.begin("a")
    assert runtime.admit(cap0b, source=source) == (0, 0)

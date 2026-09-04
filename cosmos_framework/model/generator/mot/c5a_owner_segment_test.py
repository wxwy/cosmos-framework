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
    runtime.begin("a"); runtime.admit(cap, source=source); tokens = runtime.materialize("a"); runtime.mark_backward_done("a"); runtime.commit("a")
    writes = runtime.c5_write_count; runtime.begin("a"); replay = runtime.admit(cap, source=source)
    assert torch.equal(replay, tokens[0].detach()) and runtime.c5_write_count == writes


def test_conflicting_digest_and_forged_capability_fail_before_c5() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source)
    conflict = replace(cap, source_bytes=b"bad")
    with pytest.raises(ValueError, match="untrusted"):
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
    runtime.begin("a"); runtime.admit(cap0, source=source); runtime.materialize("a"); runtime.mark_backward_done("a"); runtime.commit("a")
    cap2 = authority.issue(owner_key="a", source_identity="s", source_timestep=2, source=source)
    runtime.begin("a")
    with pytest.raises(ValueError, match="contiguous"):
        runtime.admit(cap2, source=source)
    runtime.abort("a"); runtime.reset("a")
    cap0b = authority.issue(owner_key="a", source_identity="s2", source_timestep=0, source=source, epoch=1)
    runtime.begin("a")
    assert runtime.admit(cap0b, source=source) == (0, 0)


def test_finish_rejects_nonterminal_short_and_accepts_terminal_remainder() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source)
    with pytest.raises(ValueError, match="segment length"):
        runtime.finish("a", terminal=False)
    runtime.finish("a", terminal=True)
    assert "a" not in runtime._state_by_owner


def test_materialize_many_permutation_preserves_owner_rows() -> None:
    def run(order: list[str]) -> dict[str, torch.Tensor]:
        torch.manual_seed(7)
        authority, runtime, source = _runtime()
        sources = {"a": source, "b": {**source, "history_visual_summary": torch.ones(1, 1, 96)}}
        for owner in ("a", "b"):
            cap = authority.issue(owner_key=owner, source_identity="s", source_timestep=0, source=sources[owner])
            runtime.begin(owner); runtime.admit(cap, source=sources[owner])
        result = runtime.materialize_many(order)
        return {owner: result[index].detach() for index, owner in enumerate(order)}

    first, second = run(["a", "b"]), run(["b", "a"])
    torch.testing.assert_close(first["a"], second["a"])
    torch.testing.assert_close(first["b"], second["b"])


def test_materialize_requires_explicit_backward_and_rejects_second_materialize() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); runtime.materialize("a")
    with pytest.raises(RuntimeError, match="already completed"):
        runtime.materialize("a")
    with pytest.raises(RuntimeError, match="backward"):
        runtime.commit("a")
    runtime.mark_backward_done("a"); runtime.commit("a")


def test_reset_rejects_capability_from_previous_epoch() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.reset("a"); runtime.begin("a")
    with pytest.raises(ValueError, match="epoch"):
        runtime.admit(cap, source=source)

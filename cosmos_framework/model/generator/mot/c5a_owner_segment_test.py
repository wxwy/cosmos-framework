from dataclasses import replace

import pytest
import torch

from .c5a_owner_segment import AdmissionAuthority, AdmissionCapability, C5AOwnerSegmentCPU
from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _source() -> dict[str, torch.Tensor]:
    return {"history_visual_summary": torch.zeros(1, 1, 96), "local_history_action": torch.zeros(1, 1, 10),
            "history_age_steps": torch.zeros(1, 1, dtype=torch.long), "history_dt_s": torch.zeros(1, 1, 1),
            "history_mask": torch.ones(1, 1, dtype=torch.bool)}


def _runtime(*, segment_steps: int = 16) -> tuple[AdmissionAuthority, C5AOwnerSegmentCPU, dict[str, torch.Tensor]]:
    source = _source(); authority = AdmissionAuthority()
    return authority, C5AOwnerSegmentCPU(authority, LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore(ttt_tbptt_steps=segment_steps)), source


def test_materialize_then_commit_and_replay_without_second_write() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); tokens = runtime.materialize("a"); runtime.backward_and_mark("a", tokens.float().sum()); runtime.commit("a")
    writes = runtime.c5_write_count; runtime.begin("a"); replay = runtime.admit(cap, source=source)
    assert torch.equal(replay.value, tokens[0].detach()) and replay.present and runtime.c5_write_count == writes


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
    runtime.begin("a"); runtime.admit(cap0, source=source); token0 = runtime.materialize("a"); runtime.backward_and_mark("a", token0.float().sum()); runtime.commit("a")
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
    runtime.materialize("a"); runtime.backward_and_mark("a", runtime._pending_by_owner["a"].witness.float().sum() * 0)
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
    runtime.backward_and_mark("a", runtime._pending_by_owner["a"].witness.float().sum() * 0); runtime.commit("a")


def test_reset_rejects_capability_from_previous_epoch() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.reset("a"); runtime.begin("a")
    with pytest.raises(ValueError, match="epoch"):
        runtime.admit(cap, source=source)


def test_materialize_many_valid_done_and_row_mismatch_are_explicit() -> None:
    torch.manual_seed(11)
    authority, runtime, source = _runtime()
    seen: list[tuple[tuple[int, ...], torch.Tensor]] = []
    original = runtime.core.scan_segment_many

    def spy(evidence: torch.Tensor, valid: torch.Tensor, state=None, *, create_graph=True):
        seen.append((tuple(evidence.shape), valid.detach().clone()))
        return original(evidence, valid, state, create_graph=create_graph)

    runtime.core.scan_segment_many = spy
    for owner in ("a", "b"):
        owner_source = {**source, "history_visual_summary": torch.full((1, 1, 96), float(owner == "b"))}
        cap = authority.issue(owner_key=owner, source_identity="s", source_timestep=0, source=owner_source)
        runtime.begin(owner); runtime.admit(cap, source=owner_source)
    runtime.materialize_many(["a", "b"], valid=torch.tensor([[True], [False]]), done_before=torch.tensor([False, True]))
    assert seen[0][0] == (2, 1, 256) and seen[0][1].tolist() == [[True], [False]]
    with pytest.raises(RuntimeError, match="backward"):
        runtime.commit("a")
    batch_loss = sum(pending.witness.float().sum() for pending in runtime._pending_by_owner.values()) * 0
    runtime.backward_and_mark_many(["a", "b"], batch_loss)
    for owner in ("a", "b"):
        runtime.commit(owner)

    authority2, runtime2, source2 = _runtime()
    cap_a = authority2.issue(owner_key="a", source_identity="s", source_timestep=0, source=source2)
    cap_b = authority2.issue(owner_key="b", source_identity="s", source_timestep=0, source=source2)
    runtime2.begin("a"); runtime2.admit(cap_a, source=source2)
    runtime2.begin("b"); runtime2.admit(cap_b, source=source2)
    cap_b1 = authority2.issue(owner_key="b", source_identity="s", source_timestep=1, source=source2)
    runtime2.admit(cap_b1, source=source2)
    with pytest.raises(ValueError, match="equal segment lengths"):
        runtime2.materialize_many(["a", "b"])


def test_outer_backward_gradients_and_abort_rollback_are_observable() -> None:
    torch.manual_seed(19)
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source)
    token = runtime.materialize("a")[0]
    loss = token.float().square().mean(); runtime.backward_and_mark("a", loss)
    grads = [parameter.grad for parameter in list(runtime.encoder.parameters()) + list(runtime.core.parameters())]
    assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in grads)
    runtime.commit("a")
    with pytest.raises(RuntimeError, match="no pending"):
        runtime.commit("a")

    cap2 = authority.issue(owner_key="a", source_identity="s2", source_timestep=1, source=source)
    runtime.begin("a"); runtime.admit(cap2, source=source); runtime.materialize("a"); runtime.abort("a")
    assert "a" not in runtime._pending_by_owner and all(key[0] == "a" for key in runtime._identity_index)


@pytest.mark.parametrize("segment_steps", [1, 3, 16])
def test_segment_lengths_and_terminal_remainders(segment_steps: int) -> None:
    authority, runtime, source = _runtime(segment_steps=segment_steps)
    for timestep in range(segment_steps):
        cap = authority.issue(owner_key="a", source_identity="s", source_timestep=timestep, source=source)
        if timestep == 0:
            runtime.begin("a")
        runtime.admit(cap, source=source)
    runtime.materialize("a"); runtime.backward_and_mark("a", runtime._pending_by_owner["a"].witness.float().sum() * 0); runtime.finish("a", terminal=False)
    assert runtime._last_timestep["a"] == segment_steps - 1

    authority2, runtime2, source2 = _runtime(segment_steps=segment_steps)
    runtime2.begin("a")
    for timestep in range(max(0, segment_steps - 1)):
        cap = authority2.issue(owner_key="a", source_identity="s", source_timestep=timestep, source=source2)
        runtime2.admit(cap, source=source2)
    if segment_steps == 1:
        runtime2.finish("a", terminal=True)
    else:
        runtime2.materialize("a"); runtime2.backward_and_mark("a", runtime2._pending_by_owner["a"].witness.float().sum() * 0); runtime2.finish("a", terminal=True)
    assert "a" not in runtime2._state_by_owner


def test_fast_state_is_not_registered_as_slow_parameters() -> None:
    _, runtime, _ = _runtime()
    names = {name for name, _ in runtime.core.named_parameters()}
    assert names == {
        "slot_queries", "w0_fast_in_weight", "w0_fast_in_bias", "w0_fast_out_weight", "w0_fast_out_bias",
        "key_proj.weight", "key_proj.bias", "query_proj.weight", "query_proj.bias", "value_proj.weight", "value_proj.bias",
    }


def test_materialized_segment_rejects_unseen_admission_but_allows_exact_replay() -> None:
    authority, runtime, source = _runtime(); cap0 = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap0, source=source); token = runtime.materialize("a")
    writes = runtime.c5_write_count
    assert torch.equal(runtime.admit(cap0, source=source).value, token[0].detach())
    cap1 = authority.issue(owner_key="a", source_identity="s", source_timestep=1, source=source)
    with pytest.raises(RuntimeError, match="unseen admission"):
        runtime.admit(cap1, source=source)
    assert len(runtime._pending_by_owner["a"].rows) == 1 and runtime.c5_write_count == writes


def test_backward_failure_does_not_open_commit_phase() -> None:
    authority, runtime, source = _runtime(); cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); runtime.materialize("a")
    loss = runtime._pending_by_owner["a"].witness.float().sum()
    loss.register_hook(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        runtime.backward_and_mark("a", loss)
    with pytest.raises(RuntimeError, match="backward"):
        runtime.commit("a")


def test_done_before_requires_owner_epoch_reset() -> None:
    authority, runtime, source = _runtime(segment_steps=1)
    cap = authority.issue(owner_key="a", source_identity="s", source_timestep=0, source=source)
    runtime.begin("a"); runtime.admit(cap, source=source); token = runtime.materialize("a")
    runtime.backward_and_mark("a", token.float().sum()); runtime.commit("a")
    cap_next = authority.issue(owner_key="a", source_identity="s2", source_timestep=1, source=source)
    runtime.begin("a"); runtime.admit(cap_next, source=source)
    with pytest.raises(RuntimeError, match="explicit owner reset"):
        runtime.materialize_many(["a"], done_before=torch.tensor([True]))

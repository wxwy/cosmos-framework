import pytest
import torch

from .c5a_owner_segment import AdmissionAuthority
from .c6_runtime_adapter import C6SyntheticRuntimeAdapter
from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder


def _source(value: float = 0.0) -> dict[str, torch.Tensor]:
    return {"history_visual_summary": torch.full((1, 1, 96), value), "local_history_action": torch.zeros(1, 1, 10),
            "history_age_steps": torch.zeros(1, 1, dtype=torch.long), "history_dt_s": torch.zeros(1, 1, 1),
            "history_mask": torch.ones(1, 1, dtype=torch.bool)}


def _adapter(segment_steps: int = 1) -> tuple[AdmissionAuthority, C6SyntheticRuntimeAdapter]:
    authority = AdmissionAuthority()
    return authority, C6SyntheticRuntimeAdapter(authority, LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore(ttt_tbptt_steps=segment_steps))


def _commit(adapter: C6SyntheticRuntimeAdapter, authority: AdmissionAuthority, source: dict[str, torch.Tensor], timestep: int, *, epoch: int = 0, owner: str = "ep/0") -> None:
    segment_id = f"s{timestep}"
    cap = authority.issue(owner_key=owner, source_identity=f"{segment_id}:{timestep}", source_timestep=timestep, source=source, epoch=epoch)
    adapter.admit(cap, source=source, segment_id=segment_id, row_index=0)
    token = adapter.materialize(owner)
    adapter.backward_and_mark(owner, token.float().sum() * 0)
    adapter.finish(owner, terminal=False)


def _snapshot(adapter: C6SyntheticRuntimeAdapter, owner: str) -> tuple[object, object, dict, dict, dict, int]:
    delegate = adapter._c5a
    state = delegate._state_by_owner.get(owner)
    state_copy = None if state is None else tuple(value.clone() for value in state)
    committed = {key: (record.value.clone(), record.shape, record.present) for key, record in delegate._committed.items()}
    pending = delegate._pending_by_owner.get(owner)
    pending_sig = None if pending is None else (pending.phase, tuple(cap.source_key for cap, _ in pending.rows), tuple(pending.valid_rows or ()), None if pending.witness is None else tuple(pending.witness.shape))
    return state_copy, delegate._last_timestep.get(owner), committed, dict(delegate._identity_index), dict(delegate._epoch_by_owner), delegate.c5_write_count, pending_sig


def _assert_snapshot_equal(before: tuple, after: tuple) -> None:
    assert before[1] == after[1] and before[3:] == after[3:] and set(before[2]) == set(after[2])
    assert all(torch.equal(before[2][key][0], after[2][key][0]) and before[2][key][1:] == after[2][key][1:] for key in before[2])
    assert (before[0] is None) == (after[0] is None)
    if before[0] is not None and after[0] is not None:
        assert all(torch.equal(x, y) for x, y in zip(before[0], after[0], strict=True))


def test_adapter_delegates_segment_and_preserves_prefix_shape() -> None:
    authority, adapter = _adapter()
    source = _source(); adapter.begin_segment("ep/0")
    cap = authority.issue(owner_key="ep/0", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    token = adapter.materialize("ep/0")
    assert tuple(token.shape) == (1, 1, 32)
    adapter.backward_and_mark("ep/0", token.float().sum() * 0); adapter.finish("ep/0", terminal=False)


def test_two_segments_use_owner_global_timestep() -> None:
    authority, adapter = _adapter()
    source = _source(); adapter.begin_segment("ep/0"); _commit(adapter, authority, source, 0)
    adapter.begin_segment("ep/0"); _commit(adapter, authority, source, 1)


def test_reset_rejects_pending_then_explicit_abort_opens_new_epoch() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("ep/0")
    cap = authority.issue(owner_key="ep/0", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    with pytest.raises(RuntimeError, match="pending"):
        adapter.reset("ep/0")
    adapter.abort("ep/0"); adapter.reset("ep/0")
    stale = authority.issue(owner_key="ep/0", source_identity="s0:0", source_timestep=0, source=source, epoch=0)
    adapter.begin_segment("ep/0")
    with pytest.raises(ValueError, match="epoch"):
        adapter.admit(stale, source=source, segment_id="s0", row_index=0)


def test_done_without_pending_is_reset_alias() -> None:
    _, adapter = _adapter(); adapter.done("ep/0"); adapter.reset("ep/0")


def test_batch_permutation_is_owner_keyed() -> None:
    authority, adapter = _adapter()
    for owner, value in (("a", 0.0), ("b", 1.0)):
        source = _source(value); adapter.begin_segment(owner)
        cap = authority.issue(owner_key=owner, source_identity="s0:0", source_timestep=0, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    output = adapter.materialize_many(["b", "a"])
    assert tuple(output.shape) == (2, 1, 32)
    adapter.backward_and_mark_many(["a", "b"], output.float().sum() * 0)
    adapter.finish("a", terminal=False); adapter.finish("b", terminal=False)


def test_public_finish_enforces_short_and_terminal_lengths() -> None:
    authority, adapter = _adapter(segment_steps=3); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0); token = adapter.materialize("a")
    adapter.backward_and_mark("a", token.float().sum() * 0)
    with pytest.raises(ValueError, match="segment length"):
        adapter.finish("a", terminal=False)
    adapter.finish("a", terminal=True)
    _, empty = _adapter(segment_steps=3); empty.begin_segment("z"); empty.finish("z", terminal=True)


def test_segment_identity_coordinates_and_invalid_coordinates_fail_closed() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    with pytest.raises(ValueError, match="segment coordinates"):
        adapter.admit(cap, source=source, segment_id="wrong", row_index=0)
    with pytest.raises(ValueError, match="invalid segment coordinates"):
        adapter.admit(cap, source=source, segment_id="s0", row_index=-1)


def test_skip_duplicate_and_changed_bytes_are_rejected_or_replayed() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap0 = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap0, source=source, segment_id="s0", row_index=0); token = adapter.materialize("a")
    adapter.backward_and_mark("a", token.float().sum() * 0); adapter.finish("a", terminal=False)
    adapter.begin_segment("a")
    skip = authority.issue(owner_key="a", source_identity="s2:2", source_timestep=2, source=source)
    with pytest.raises(ValueError, match="contiguous"):
        adapter.admit(skip, source=source, segment_id="s2", row_index=0)
    replay = adapter.admit(cap0, source=source, segment_id="s0", row_index=0)
    assert replay.present
    changed = _source(1.0)
    changed_cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=changed)
    with pytest.raises(ValueError, match="conflicting source digest"):
        adapter.admit(changed_cap, source=changed, segment_id="s0", row_index=0)
    adapter.abort("a")


def test_batch_permutation_preserves_public_owner_values() -> None:
    def run(order: list[str]) -> dict[str, torch.Tensor]:
        torch.manual_seed(9); authority, adapter = _adapter()
        for owner, value in (("a", 0.0), ("b", 1.0)):
            source = _source(value); adapter.begin_segment(owner)
            cap = authority.issue(owner_key=owner, source_identity="s0:0", source_timestep=0, source=source)
            adapter.admit(cap, source=source, segment_id="s0", row_index=0)
        output = adapter.materialize_many(order)
        adapter.backward_and_mark_many(order, output.float().sum() * 0)
        for owner in order:
            adapter.finish(owner, terminal=False)
        return {owner: output[index].detach() for index, owner in enumerate(order)}
    first, second = run(["a", "b"]), run(["b", "a"])
    torch.testing.assert_close(first["a"], second["a"])
    torch.testing.assert_close(first["b"], second["b"])


def test_batch_row_mismatch_is_delegated() -> None:
    authority, adapter = _adapter(); source = _source()
    for owner in ("a", "b"):
        adapter.begin_segment(owner)
        cap = authority.issue(owner_key=owner, source_identity="s0:0", source_timestep=0, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    adapter.admit(authority.issue(owner_key="a", source_identity="s1:1", source_timestep=1, source=source), source=source, segment_id="s1", row_index=1)
    with pytest.raises(ValueError, match="equal segment lengths"):
        adapter.materialize_many(["a", "b"])
    adapter.abort("a"); adapter.abort("b")


@pytest.mark.parametrize("steps", [1, 3, 16])
def test_public_n_matrix_uses_one_segment_closure(steps: int) -> None:
    authority, adapter = _adapter(segment_steps=steps); source = _source(); adapter.begin_segment("a")
    for timestep in range(steps):
        cap = authority.issue(owner_key="a", source_identity=f"s0:{timestep}", source_timestep=timestep, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=timestep)
    output = adapter.materialize("a")
    assert tuple(output.shape) == (steps, 1, 32)
    adapter.backward_and_mark("a", output.float().sum() * 0); adapter.finish("a", terminal=False)


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_public_terminal_remainder_matrix(rows: int) -> None:
    authority, adapter = _adapter(segment_steps=3); source = _source(); adapter.begin_segment("a")
    for timestep in range(rows):
        cap = authority.issue(owner_key="a", source_identity=f"s0:{timestep}", source_timestep=timestep, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=timestep)
    if rows == 0:
        adapter.finish("a", terminal=True)
        return
    output = adapter.materialize("a"); adapter.backward_and_mark("a", output.float().sum() * 0); adapter.finish("a", terminal=True)


def test_pending_done_rejects_without_mutation_and_fresh_epoch_replays() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    writes = adapter.c5_write_count
    with pytest.raises(RuntimeError, match="pending"):
        adapter.done("a")
    assert adapter.c5_write_count == writes
    adapter.abort("a"); adapter.done("a"); adapter.begin_segment("a")
    changed = _source(2.0)
    fresh = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=changed, epoch=1)
    assert adapter.admit(fresh, source=changed, segment_id="s0", row_index=0) == (0, 0)


def test_public_segment_loss_negative_fixtures_are_fail_closed() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0); output = adapter.materialize("a")
    with pytest.raises(RuntimeError, match="unrelated"):
        adapter.backward_and_mark("a", torch.ones((), requires_grad=True) * 2)
    with pytest.raises(RuntimeError, match="scalar graph"):
        adapter.backward_and_mark("a", torch.zeros(()))
    adapter.abort("a")
    authority, adapter = _adapter(); adapter.begin_segment("a"); adapter.begin_segment("b")
    for owner in ("a", "b"):
        cap = authority.issue(owner_key=owner, source_identity="s0:0", source_timestep=0, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    output_a = adapter.materialize("a"); adapter.materialize("b")
    with pytest.raises(RuntimeError, match="unrelated"):
        adapter.backward_and_mark_many(["a", "b"], output_a.float().sum())
    adapter.abort("a"); adapter.abort("b")


def test_public_backward_failure_and_abort_leave_no_committed_candidate() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0); output = adapter.materialize("a")
    loss = output.float().sum(); loss.register_hook(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.backward_and_mark("a", loss)
    with pytest.raises(RuntimeError, match="backward"):
        adapter.finish("a", terminal=False)
    adapter.abort("a")


def test_pending_done_preserves_complete_public_seam_snapshot() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    before = _snapshot(adapter, "a")
    with pytest.raises(RuntimeError, match="pending"):
        adapter.done("a")
    after = _snapshot(adapter, "a")
    _assert_snapshot_equal(before, after)
    adapter.abort("a")


def test_public_backward_failure_abort_preserves_committed_snapshot() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("a")
    cap0 = authority.issue(owner_key="a", source_identity="s0:0", source_timestep=0, source=source)
    adapter.admit(cap0, source=source, segment_id="s0", row_index=0); token0 = adapter.materialize("a")
    adapter.backward_and_mark("a", token0.float().sum() * 0); adapter.finish("a", terminal=False)
    committed_baseline = _snapshot(adapter, "a")
    adapter.begin_segment("a")
    cap1 = authority.issue(owner_key="a", source_identity="s1:1", source_timestep=1, source=source)
    adapter.admit(cap1, source=source, segment_id="s1", row_index=0); token1 = adapter.materialize("a")
    before = _snapshot(adapter, "a")
    loss = token1.float().sum(); loss.register_hook(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.backward_and_mark("a", loss)
    after = _snapshot(adapter, "a")
    _assert_snapshot_equal(before, after)
    adapter.abort("a")
    after_abort = _snapshot(adapter, "a")
    assert after_abort[0] is not None and after_abort[1] == committed_baseline[1] and after_abort[3] == committed_baseline[3] and after_abort[4] == committed_baseline[4] and after_abort[6] is None
    assert set(after_abort[2]) == set(committed_baseline[2])
    assert all(torch.equal(after_abort[2][key][0], committed_baseline[2][key][0]) for key in committed_baseline[2])


def test_local_disabled_parity_is_zero_write_no_memory_path() -> None:
    _, adapter = _adapter()
    assert adapter.c5_write_count == 0 and not adapter._c5a._pending_by_owner
    sample = torch.tensor([[1.0, 2.0]])
    disabled_packed = (sample.clone(), None)
    baseline_packed = (sample.clone(), None)
    disabled_loss = disabled_packed[0].square().mean()
    baseline_loss = baseline_packed[0].square().mean()
    assert torch.equal(disabled_packed[0], baseline_packed[0]) and disabled_packed[1] is baseline_packed[1]
    assert torch.equal(disabled_loss, baseline_loss)
    adapter.done("disabled-owner")
    assert adapter.c5_write_count == 0 and not adapter._c5a._state_by_owner and not adapter._c5a._pending_by_owner

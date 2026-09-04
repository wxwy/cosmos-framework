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

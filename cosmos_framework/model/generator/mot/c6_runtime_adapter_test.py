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
    cap = authority.issue(owner_key=owner, source_identity="seg", source_timestep=timestep, source=source, epoch=epoch)
    adapter.admit(cap, source=source, segment_id=f"s{timestep}", row_index=0)
    token = adapter.materialize(owner)
    adapter.backward_and_mark(owner, token.float().sum() * 0)
    adapter.commit(owner)


def test_adapter_delegates_segment_and_preserves_prefix_shape() -> None:
    authority, adapter = _adapter()
    source = _source(); adapter.begin_segment("ep/0")
    cap = authority.issue(owner_key="ep/0", source_identity="seg", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    token = adapter.materialize("ep/0")
    assert tuple(token.shape) == (1, 1, 32)
    adapter.backward_and_mark("ep/0", token.float().sum() * 0); adapter.commit("ep/0")


def test_two_segments_use_owner_global_timestep() -> None:
    authority, adapter = _adapter()
    source = _source(); adapter.begin_segment("ep/0"); _commit(adapter, authority, source, 0)
    adapter.begin_segment("ep/0"); _commit(adapter, authority, source, 1)


def test_reset_rejects_pending_then_explicit_abort_opens_new_epoch() -> None:
    authority, adapter = _adapter(); source = _source(); adapter.begin_segment("ep/0")
    cap = authority.issue(owner_key="ep/0", source_identity="seg", source_timestep=0, source=source)
    adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    with pytest.raises(RuntimeError, match="pending"):
        adapter.reset("ep/0")
    adapter.abort("ep/0"); adapter.reset("ep/0")
    stale = authority.issue(owner_key="ep/0", source_identity="seg", source_timestep=0, source=source, epoch=0)
    adapter.begin_segment("ep/0")
    with pytest.raises(ValueError, match="epoch"):
        adapter.admit(stale, source=source, segment_id="s1", row_index=0)


def test_done_without_pending_is_reset_alias() -> None:
    _, adapter = _adapter(); adapter.done("ep/0"); adapter.reset("ep/0")


def test_batch_permutation_is_owner_keyed() -> None:
    authority, adapter = _adapter()
    for owner, value in (("a", 0.0), ("b", 1.0)):
        source = _source(value); adapter.begin_segment(owner)
        cap = authority.issue(owner_key=owner, source_identity="seg", source_timestep=0, source=source)
        adapter.admit(cap, source=source, segment_id="s0", row_index=0)
    output = adapter.materialize_many(["b", "a"])
    assert tuple(output.shape) == (2, 1, 32)
    adapter.backward_and_mark_many(["a", "b"], sum(adapter._c5a._pending_by_owner[o].witness.float().sum() * 0 for o in ("a", "b")))
    adapter.commit("a"); adapter.commit("b")

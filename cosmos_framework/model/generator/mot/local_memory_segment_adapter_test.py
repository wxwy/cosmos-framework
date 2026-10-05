"""B0 pending capability、提交原子性和跨段连续性测试。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    LocalMemoryTransaction,
    RankLocalSegmentScheduler,
    SegmentIdentity,
    SegmentProvenance,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    BatchedLocalMemorySegmentAdapter,
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)
from cosmos_framework.model.generator.mot.local_memory_segment_test import make_segment


def setup_adapter():
    torch.manual_seed(3)
    return (
        CanonicalLocalMemorySegmentAdapter(
            LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar()
        ),
        RankLocalSegmentScheduler(),
    )


def prepare(adapter, scheduler, cursor=0, slot=0, episode="episode", terminal=False):
    identity = SegmentIdentity(slot, episode, "robocasa", cursor, cursor, "source", terminal)
    transaction = LocalMemoryTransaction(GAWindowPlan((identity.member,), (16,)), scheduler)
    segment = make_segment(cursor, slot, episode)
    result = adapter.scan(segment, identity=identity, transaction=transaction)
    return identity, transaction, segment, result


def succeed(adapter, identity, transaction, result):
    result.local_tokens.square().sum().backward()
    transaction.successful_backward(identity, result)
    adapter.commit(identity, result, transaction=transaction)


def assert_snapshot_equal(left, right):
    assert len(left) == len(right)
    for old, new in zip(left, right, strict=True):
        assert old[:2] == new[:2]
        for first, second in zip(old[2], new[2], strict=True):
            assert torch.equal(first, second)


def test_commit_requires_success_and_detaches():
    adapter, scheduler = setup_adapter()
    identity, transaction, _, result = prepare(adapter, scheduler)
    assert adapter.sidecar.snapshot() == ()
    assert scheduler._committed == {}
    assert result.locals[0] is None
    with pytest.raises(RuntimeError):
        adapter.commit(identity, result, transaction=transaction)
    result.local_tokens.square().sum().backward()
    transaction.successful_backward(identity, result)
    assert adapter.sidecar.snapshot() == ()
    assert scheduler._committed == {}
    adapter.commit(identity, result, transaction=transaction)
    saved = adapter.sidecar.snapshot()[0][2]
    for value, candidate in zip(saved, result.state_out, strict=True):
        assert value.dtype == torch.float32
        assert not value.requires_grad and value.grad_fn is None
        assert value.data_ptr() != candidate.data_ptr()
        torch.testing.assert_close(value, candidate)
    with pytest.raises(RuntimeError):
        adapter.commit(identity, result, transaction=transaction)


@pytest.mark.parametrize("failure", ["abort", "skip", "discard"])
@pytest.mark.parametrize("after_backward", [False, True])
def test_failed_pending_cannot_commit_or_pollute_live(failure, after_backward):
    adapter, scheduler = setup_adapter()
    identity, transaction, _, result = prepare(adapter, scheduler)
    succeed(adapter, identity, transaction, result)
    before = adapter.sidecar.snapshot()
    frontier = dict(scheduler._committed)
    identity, transaction, _, result = prepare(adapter, scheduler, cursor=1)
    if after_backward:
        result.local_tokens.square().sum().backward()
        transaction.successful_backward(identity, result)
    if failure == "abort":
        transaction.terminal_failure("outer_failure")
    elif failure == "skip":
        transaction.grad_scaler_skip()
    else:
        adapter.discard_pending(identity, result, transaction=transaction)
    with pytest.raises(RuntimeError):
        adapter.commit(identity, result, transaction=transaction)
    assert_snapshot_equal(before, adapter.sidecar.snapshot())
    assert scheduler._committed == frontier
    if failure != "discard":
        adapter.discard_pending(identity, result, transaction=transaction)
    # 丢弃后可以从原 committed frontier 重试同一段。
    retry_id, retry_tx, _, retry = prepare(adapter, scheduler, cursor=1)
    assert retry_id == identity
    succeed(adapter, retry_id, retry_tx, retry)


def test_continuation_uses_committed_state_and_not_w0():
    adapter, scheduler = setup_adapter()
    identity, transaction, _, result = prepare(adapter, scheduler)
    succeed(adapter, identity, transaction, result)
    committed = adapter.sidecar.snapshot()[0][2]
    identity, transaction, segment, result = prepare(adapter, scheduler, cursor=1)
    expected, expected_state, _ = adapter.core.scan_segment_masked_encoded_many(
        adapter.encoder,
        segment.evidence_visual_summary_prev,
        segment.evidence_executed_action_prev,
        segment.evidence_valid,
        committed,
    )
    fresh, _, _ = adapter.core.scan_segment_masked_encoded_many(
        adapter.encoder,
        segment.evidence_visual_summary_prev,
        segment.evidence_executed_action_prev,
        segment.evidence_valid,
    )
    torch.testing.assert_close(result.local_tokens, expected)
    assert not torch.allclose(result.local_tokens, fresh)
    assert result.local_present.all()
    for actual, reference in zip(result.state_out, expected_state, strict=True):
        torch.testing.assert_close(actual, reference)
    succeed(adapter, identity, transaction, result)


def test_exact_capability_and_no_pending_overwrite():
    adapter, scheduler = setup_adapter()
    identity, transaction, segment, result = prepare(adapter, scheduler)
    with pytest.raises(RuntimeError):
        adapter.scan(segment, identity=identity, transaction=transaction)
    with pytest.raises(RuntimeError):
        transaction.successful_backward(replace(identity), result)
    with pytest.raises(RuntimeError):
        transaction.successful_backward(identity, replace(result))
    transaction.successful_backward(identity, result)
    with pytest.raises(RuntimeError):
        adapter.commit(replace(identity), result, transaction=transaction)
    with pytest.raises(RuntimeError):
        adapter.discard_pending(identity, replace(result), transaction=transaction)
    adapter.commit(identity, result, transaction=transaction)


def test_action_width_checked_by_adapter_before_scan():
    adapter, scheduler = setup_adapter()
    identity = SegmentIdentity(0, "episode", "robocasa", 0, 0, "source")
    transaction = LocalMemoryTransaction(GAWindowPlan((identity.member,), (16,)), scheduler)
    segment = make_segment(action_dim=64)
    segment.validate(16)
    with pytest.raises(ValueError, match="action"):
        adapter.scan(segment, identity=identity, transaction=transaction)
    assert adapter.sidecar.snapshot() == ()


def test_terminal_and_explicit_reset_only_clear_selected_slot():
    adapter, scheduler = setup_adapter()
    for slot in (0, 1):
        identity, transaction, _, result = prepare(adapter, scheduler, slot=slot)
        succeed(adapter, identity, transaction, result)
    identity, transaction, _, result = prepare(adapter, scheduler, cursor=1, terminal=True)
    succeed(adapter, identity, transaction, result)
    assert [record[0].slot_id for record in adapter.sidecar.snapshot()] == [1]
    identity, transaction, _, result = prepare(adapter, scheduler, episode="new")
    assert result.locals[0] is None
    with pytest.raises(RuntimeError):
        adapter.reset(1, scheduler=scheduler)
    succeed(adapter, identity, transaction, result)
    adapter.reset(0, scheduler=scheduler)
    assert [record[0].slot_id for record in adapter.sidecar.snapshot()] == [1]
    assert 0 not in scheduler._committed


def test_stale_source_and_nonfinite_candidate_do_not_commit():
    adapter, scheduler = setup_adapter()
    identity, transaction, _, result = prepare(adapter, scheduler)
    transaction.successful_backward(identity, result)
    with torch.no_grad():
        result.state_out.fast_in_weight[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        adapter.commit(identity, result, transaction=transaction)
    assert adapter.sidecar.snapshot() == ()
    assert scheduler._committed == {}
    adapter.discard_pending(identity, result, transaction=transaction)
    identity, transaction, _, result = prepare(adapter, scheduler)
    succeed(adapter, identity, transaction, result)
    bad = SegmentIdentity(0, "episode", "robocasa", 1, 1, "other-source")
    with pytest.raises(ValueError):
        scheduler.validate(bad)


def test_batched_adapter_mixed_rows_keep_independent_capabilities():
    torch.manual_seed(41)
    encoder = LocalEvidenceEncoder(evidence_dim=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=16, k_local=1
    )
    sidecar, scheduler = LocalMemorySegmentSidecar(), RankLocalSegmentScheduler()
    previous = SegmentIdentity(0, "episode", "robocasa", 0, 0, "source")
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    sidecar._records[0] = previous, provenance, core.detach_state(core.initial_state(1))
    scheduler._committed[0] = previous
    identities = (
        SegmentIdentity(0, "episode", "robocasa", 1, 1, "source"),
        SegmentIdentity(1, "fresh", "robocasa", 0, 0, "source"),
    )
    segments = (make_segment(cursor=1, slot=0), make_segment(cursor=0, slot=1, episode="fresh"))
    transactions = tuple(
        LocalMemoryTransaction(GAWindowPlan((identity.member,), (16,)), scheduler) for identity in identities
    )
    net = SimpleNamespace(
        config=SimpleNamespace(local_memory_enabled=True),
        local_memory_runtime=SimpleNamespace(encoder=encoder, core=core),
    )
    calls = []

    def scan(visual, action, valid, state, *, continuation_mask):
        calls.append((visual.shape, action.shape, valid.shape, continuation_mask.clone()))
        return Cosmos3VFMNetwork.scan_local_memory(
            net, visual, action, valid, state, continuation_mask=continuation_mask
        )

    adapter = BatchedLocalMemorySegmentAdapter(encoder, core, sidecar, scan)
    results = adapter.scan(segments, identities=identities, transactions=transactions)
    assert len(calls) == 1
    assert calls[0][:3] == ((2, 16, 96), (2, 16, 15), (2, 16))
    assert torch.equal(calls[0][3], torch.tensor([True, False]))
    assert len(results) == 2 and results[0].identities[0] != results[1].identities[0]
    assert results[0].locals[0] is not None and results[1].locals[0] is None
    with pytest.raises(RuntimeError):
        adapter.commit(0, identities[1], results[0], transaction=transactions[0])
    for row, (identity, transaction, result) in enumerate(zip(identities, transactions, results, strict=True)):
        transaction.successful_backward(identity, result)
        adapter.commit(row, identity, result, transaction=transaction)
    assert [record[0].slot_id for record in sidecar.snapshot()] == [0, 1]
    assert scheduler._committed == {0: identities[0], 1: identities[1]}


def test_batched_adapter_all_fresh_and_all_continuation_never_read_w0(monkeypatch):
    encoder = LocalEvidenceEncoder(evidence_dim=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=16, k_local=1
    )
    sidecar, scheduler = LocalMemorySegmentSidecar(), RankLocalSegmentScheduler()
    seen = []

    def forbidden(batch):
        raise AssertionError("batched adapter 不得调用 core.initial_state")

    monkeypatch.setattr(core, "initial_state", forbidden)

    def scan(visual, action, valid, state, *, continuation_mask):
        seen.append((state, continuation_mask))
        candidate = state or ContinualTTTFastState(
            torch.zeros(2, 8, 6), torch.zeros(2, 8), torch.zeros(2, 4, 8), torch.zeros(2, 4)
        )
        return torch.zeros(2, 16, 1, 4), candidate, valid

    for cursor in (0, 1):
        adapter = BatchedLocalMemorySegmentAdapter(encoder, core, sidecar, scan)
        identities = tuple(
            SegmentIdentity(slot, f"episode{slot}", "robocasa", cursor, cursor, "source") for slot in (0, 1)
        )
        segments = tuple(make_segment(cursor=cursor, slot=slot, episode=f"episode{slot}") for slot in (0, 1))
        transactions = tuple(
            LocalMemoryTransaction(GAWindowPlan((identity.member,), (16,)), scheduler) for identity in identities
        )
        results = adapter.scan(segments, identities=identities, transactions=transactions)
        for row, (identity, transaction, result) in enumerate(zip(identities, transactions, results, strict=True)):
            transaction.successful_backward(identity, result)
            adapter.commit(row, identity, result, transaction=transaction)
    assert len(seen) == 2
    assert seen[0] == (None, None)
    assert seen[1][0] is not None and seen[1][1] is None
    assert all(not value.requires_grad for value in seen[1][0])

"""Sparse H16 admission over the real CPU Local scan and Worker state machine.

Synthetic RGB/Visual96, actions and an injected CPU encoder; no Wan/GPU or real
env.step claim. The native Sim producer and Root IPC have separate integration
tests. Full completion authority, not feature counts, admits the sparse subset.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
import torch

from cosmos_framework.inference.local_memory_online import OnlineMemoryRequest
from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
from cosmos_framework.inference.robocasa_feature_evidence import validate_post_step_receipt
from cosmos_framework.inference.robocasa_feature_evidence_test import CLOCK, KEY, TOKEN
from cosmos_framework.inference.robocasa_local_memory_policy_test import (
    _commit_feature_cold,
    _feature_adapter,
    _prepare_feature,
)
from cosmos_framework.inference.robocasa_vae_protocol import (
    Frame,
    canonical_json,
    selected_history_step,
    validate_history_k,
)
from cosmos_framework.inference.robocasa_vae_worker import SharedVAEWorker, WorkerConfig

KS = (1, 2, 4, 8, 16)
SELECTED = {1: (0,), 2: (0, 8), 4: (0, 4, 8, 12), 8: tuple(range(0, 16, 2)), 16: tuple(range(16))}


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("entry", ["owner", "memory"])
def test_sparse_partial_tail_can_End_but_cannot_prepare_a_new_Policy_chunk(k, entry):
    h, adapter, model = _feature_adapter(history_k=k)
    _commit_feature_cold(adapter, h)
    rows = h.rows(count=8)
    original = adapter.snapshot
    with pytest.raises(ValueError, match="H16"):
        if entry == "owner":
            _prepare_feature(adapter, h, 8, rows)
        else:
            sources = tuple(range(0, 8, 16 // k))
            adapter.memory.prepare(
                OnlineMemoryRequest(
                    h.lease["session_id"],
                    h.lease["episode_id"],
                    8,
                    sources,
                    torch.zeros(len(sources), 96),
                    torch.zeros(len(sources), 15),
                    stream_t1=True,
                    history_k=k,
                )
            )
    assert adapter.snapshot is original and not model.scan_calls
    assert not adapter.memory._pending and not adapter._visual_pending
    adapter.reset(h.lease["session_id"])
    h.owner.end_session(TOKEN, h.lease)


@pytest.mark.parametrize("value", [0, 3, 5, 32, True, 2.0, "2"])
def test_history_k_rejects_coercion_and_unsupported_values(value):
    with pytest.raises(ValueError):
        validate_history_k(value, 16)


@pytest.mark.parametrize("k", KS)
def test_real_worker_and_policy_two_chunks_select_original_steps_only(k):
    torch.manual_seed(23)
    h, adapter, model = _feature_adapter(history_k=k)
    calls = []

    def encoder(frames):
        calls.extend(frame.header["source_step"] for frame in frames)
        return [h.visual(frame.header["source_step"]) for frame in frames]

    worker = SharedVAEWorker(
        encoder,
        h.identity,
        KEY,
        "key",
        "worker",
        worker_epoch="worker-epoch",
        clock_domain=CLOCK,
        clock_ns=lambda: h.clock[0],
        config=WorkerConfig(history_k=k),
    )
    sid, cap = h.lease["session_id"], "private-test-session-capability"
    worker.owner_register(h.lease, cap, 10000)
    _commit_feature_cold(adapter, h)
    assert model.scan_calls == []  # First current Policy frame is not past evidence.
    try:
        for start in (0, 16):
            rows = h.rows(start)
            boundary = h.boundary(start)
            worker.owner_boundary_skip(
                h.lease,
                f"request-{start}",
                f"request-{start}",
                start,
                rows[0]["post_step_receipt"]["source_rgb_sha256"],
                rows[0]["post_step_receipt"]["raw15_sha256"],
                hashlib.sha256(canonical_json(boundary.record())).hexdigest(),
                rows[0]["post_step_receipt"],
            )
            worker.owner_issue(sid, f"request-{start}", f"request-{start}", start + 1, 15)
            selected = tuple(start + j for j in SELECTED[k])
            # Force the real asynchronous race: every completion/skip precedes
            # the earlier selected Frame submission. No sleep/timing lottery.
            for row in rows[1:]:
                skip = row["source_step"] not in selected
                worker.owner_complete(
                    sid,
                    f"request-{start}",
                    row["source_step"] + 1,
                    **({"skipped_receipt": row["post_step_receipt"]} if skip else {}),
                )
            session = worker._sessions[sid]
            assert session.completed == start + 16
            assert session.submitted == (start + 16 if k == 1 else selected[1])
            jobs = []
            for step in selected[1:]:
                frame = Frame.create(
                    h.lease,
                    f"request-{start}",
                    f"request-{start}",
                    step,
                    h.rgb(step),
                    h.action(step),
                    h.identity,
                    10000,
                    9500,
                    CLOCK,
                )
                key = worker.submit(frame, cap)
                assert worker.submit(frame, cap) == key
                with worker._changed:
                    assert worker._changed.wait_for(lambda: worker.poll(sid, key, cap)["status"] != "pending", 3)
                result = worker.poll(sid, key, cap)
                assert result["status"] == "ready"
                rows[step - start]["feature"] = result["feature"]
                jobs.append(key)
            before = adapter.snapshot
            update = _prepare_feature(adapter, h, start + 16, rows)
            assert adapter.snapshot is before  # Nothing published before commit.
            assert update.memory.replacement.consumer_step == start + 16
            assert update.visual_replacement.last_source_steps == selected
            status = adapter.status(update)
            assert status["adapted_steps"] == k and status["encoded_steps"] == 0
            if k != 16:
                assert status["selected_source_steps"] == list(selected)
                assert status["skipped_source_steps"] == [s for s in range(start, start + 16) if s not in selected]
                assert status["verified_worker_features"] == k - 1
            assert len(model.scan_calls) == (start // 16 + 1) * k
            assert all(count == 1 for count, _ in model.scan_calls)
            adapter.stage_boundary(update, h.boundary(start + 16))
            response = CachedMemoryResponse(
                f"request-{start + 16}", "f" * 64, canonical_json({"local_memory": adapter.status(update)})
            )
            adapter.commit(update, response)
            snapshot = adapter.snapshot
            replay = _prepare_feature(adapter, h, start + 16, deepcopy(rows))
            assert replay.memory.replay and adapter.status(replay)["adapted_steps"] == 0
            adapter.commit(replay, response)
            assert adapter.snapshot.fast[sid] is snapshot.fast[sid]
            changed = deepcopy(rows)
            changed[-1]["post_step_receipt"]["raw15_sha256"] = "f" * 64
            with pytest.raises(ValueError, match="replay"):
                _prepare_feature(adapter, h, start + 16, changed)
            for key in jobs:
                assert worker.ack(sid, key, cap)["acknowledged"]
            assert session.issued == session.completed == session.submitted == session.consumed == start + 16
            assert not session.jobs and worker.info()["items"] == 0
            assert calls == [s for base in range(0, start + 1, 16) for s in (base + j for j in SELECTED[k][1:])]
            h.commit_chunk(start + 16)
        assert len(calls) == 2 * (k - 1)
        assert len(h.owner._sessions[sid].completed) == 0  # New chunk owns a fresh completion ledger.
    finally:
        worker.end(sid, cap, h.clock[0] + 3_000_000_000)
        worker.close(h.clock[0] + 3_000_000_000)
        adapter.reset(sid)
        h.owner.end_session(TOKEN, h.lease)
    assert worker.info()["sessions"] == 0 and not adapter.snapshot.fast


@pytest.mark.parametrize("k", KS)
def test_sparse_abort_cannot_advance_and_retry_reuses_original_source_ids(k):
    h, adapter, model = _feature_adapter(history_k=k)
    _commit_feature_cold(adapter, h)
    rows = h.rows()
    original = adapter.snapshot
    update = _prepare_feature(adapter, h, 16, rows)
    state = tuple(t.detach().clone() for t in update.memory.replacement.state)
    token = update.memory.replacement.token.detach().clone()
    adapter.stage_boundary(update, h.boundary(16))
    adapter.abort(update)
    assert adapter.snapshot is original and adapter.committed_boundary(h.lease["session_id"], 16) is None
    retry = _prepare_feature(adapter, h, 16, rows)
    assert retry.visual_replacement.last_source_steps == SELECTED[k]
    assert all(torch.equal(a, b) for a, b in zip(state, retry.memory.replacement.state, strict=True))
    assert torch.equal(token, retry.memory.replacement.token)
    adapter.abort(retry)
    assert len(model.scan_calls) == 2 * k


@pytest.mark.parametrize(
    "mutation", ["missing_skip", "forged_skip", "skip_feature", "wrong_k", "future", "changed_action"]
)
def test_subset_never_bypasses_full_completion_authority(mutation):
    h, adapter, model = _feature_adapter(history_k=4)
    _commit_feature_cold(adapter, h)
    rows = h.rows()
    if mutation == "missing_skip":
        rows.pop(1)
    elif mutation == "forged_skip":
        rows[1]["post_step_receipt"]["raw15_sha256"] = "f" * 64
    elif mutation == "skip_feature":
        rows[1]["feature"] = h.feature(1)
    elif mutation == "wrong_k":
        rows[1]["post_step_receipt"]["history_k"] = 2
    elif mutation == "future":
        rows[-1]["source_step"] = 16
    else:
        rows[4]["feature"]["header"]["raw15_sha256"] = "f" * 64
    before = adapter.snapshot
    with pytest.raises(ValueError):
        _prepare_feature(adapter, h, 16, rows)
    assert adapter.snapshot is before and not model.scan_calls


def test_skip_metadata_must_have_no_RGB_and_cannot_mark_selected_source_skipped():
    h, adapter, _ = _feature_adapter(history_k=4)
    _commit_feature_cold(adapter, h)
    zero = h.complete(0)
    with pytest.raises(ValueError, match="RGB"):
        h.complete(1, source_rgb_sha256="a" * 64)
    forged = deepcopy(zero)
    forged["selected"] = False
    forged["source_rgb_sha256"] = None
    with pytest.raises(ValueError, match="selection"):
        validate_post_step_receipt(forged)


def test_frozen_k_and_expired_original_deadline_fail_before_scan():
    h, adapter, model = _feature_adapter(history_k=2)
    _commit_feature_cold(adapter, h)
    rows = h.rows()
    from cosmos_framework.inference.robocasa_local_memory_policy_test import _feature_request

    req = _feature_request(h, 16, rows)
    req["local_memory"]["local_ttt_history_k"] = 4
    with pytest.raises(ValueError, match="frozen"):
        adapter.prepare(req, lease=h.lease, deadline_ns=9000, clock_domain=CLOCK, request_key="request-16")
    h.clock[0] = 9001
    with pytest.raises(ValueError, match="expiry"):
        _prepare_feature(adapter, h, 16, rows)
    assert not model.scan_calls


def test_worker_skip_replay_forgery_and_late_frame_cannot_advance_or_encode():
    h, adapter, _ = _feature_adapter(history_k=2)
    _commit_feature_cold(adapter, h)
    rows = h.rows()
    encoded = []
    worker = SharedVAEWorker(
        lambda frames: encoded.extend(frames),
        h.identity,
        KEY,
        "key",
        "worker",
        worker_epoch="worker-epoch",
        clock_domain=CLOCK,
        clock_ns=lambda: h.clock[0],
        config=WorkerConfig(history_k=2),
        autostart=False,
    )
    sid, cap = h.lease["session_id"], "private-worker-test-session-capability"
    worker.owner_register(h.lease, cap, 10000)
    try:
        zero = rows[0]["post_step_receipt"]
        worker.owner_boundary_skip(
            h.lease,
            "request-0",
            "request-0",
            0,
            zero["source_rgb_sha256"],
            zero["raw15_sha256"],
            hashlib.sha256(canonical_json(h.boundary().record())).hexdigest(),
            zero,
        )
        worker.owner_issue(sid, "request-0", "request-0", 1, 15)
        one = rows[1]["post_step_receipt"]
        worker.owner_complete(sid, "request-0", 2, skipped_receipt=one)
        worker.owner_complete(sid, "request-0", 2, skipped_receipt=deepcopy(one))
        previous = (worker._sessions[sid].completed, worker._sessions[sid].submitted)
        for field, value in (
            ("raw15_sha256", "f" * 64),
            ("history_k", 4),
            ("source_rgb_sha256", "f" * 64),
            ("step_completed_ns", "9999"),
        ):
            changed = {**one, field: value}
            with pytest.raises(ValueError):
                worker.owner_complete(sid, "request-0", 2, skipped_receipt=changed)
            assert (worker._sessions[sid].completed, worker._sessions[sid].submitted) == previous
        unselected = Frame.create(
            h.lease, "request-0", "request-0", 2, h.rgb(2), h.action(2), h.identity, 10000, 9500, CLOCK
        )
        with pytest.raises(ValueError, match="unselected"):
            worker.submit(unselected, cap)
        for row in rows[2:9]:
            worker.owner_complete(
                sid,
                "request-0",
                row["source_step"] + 1,
                **({"skipped_receipt": row["post_step_receipt"]} if row["source_step"] != 8 else {}),
            )
        h.clock[0] = 9600
        late = Frame.create(h.lease, "request-0", "request-0", 8, h.rgb(8), h.action(8), h.identity, 10000, 9500, CLOCK)
        with pytest.raises(TimeoutError, match="deadline"):
            worker.submit(late, cap)
        assert not encoded and worker.info()["items"] == worker.info()["queued"] == 0
    finally:
        worker.close(h.clock[0] + 3_000_000_000)


@pytest.mark.parametrize("start", [0, 16, 32])
@pytest.mark.parametrize("k", KS)
def test_selection_uses_actual_chunk_relative_time_without_renumbering(start, k):
    assert tuple(s for s in range(start, start + 16) if selected_history_step(s, start, k, 16)) == tuple(
        start + j for j in SELECTED[k]
    )
    for invalid in (start - 1, start + 16):
        with pytest.raises(ValueError):
            selected_history_step(invalid, start, k, 16)

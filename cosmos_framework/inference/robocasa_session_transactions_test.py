from __future__ import annotations

import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.local_memory_state import CachedMemoryResponse, MemorySnapshot
from cosmos_framework.inference.robocasa_local_memory_contract import (
    EVIDENCE_FORMAT,
    EVIDENCE_VERSION,
    PREPROCESS_PROFILE,
)
from cosmos_framework.inference.robocasa_local_memory_policy import RoboCasaLocalMemoryPolicyAdapter
from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions, json_bytes, strict_json
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime


class RealLocalModel(nn.Module):
    """Actual CPU Local encoder/core/scan; only Wan/diffusion boundaries are doubles."""

    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.local_memory_runtime = LocalMemoryRuntime(
            evidence_dim=16, local_dim=32, ttt_dim=8, fast_hidden_dim=16, k_local=4
        )
        self.scan_calls = 0

        def scan(visual, action, valid, state, *, create_graph=False):
            self.scan_calls += 1
            runtime = self.net.local_memory_runtime
            return runtime.core.scan_segment_masked_encoded_many(
                runtime.encoder, visual, action, valid, state, create_graph=create_graph
            )

        self.net.scan_local_memory = scan
        self.encoded = 0

        def encode(value):
            self.encoded += 1
            return torch.ones(1, 48, 1, 12, 20) * value.float().mean()

        self.tokenizer_vision_gen = SimpleNamespace(
            encode=encode,
            model=SimpleNamespace(model=nn.Linear(1, 1)),
            use_streaming_encode=False,
            _keep_encoder_cache=False,
        )


def make_adapter(slots=2):
    torch.manual_seed(19)
    model = RealLocalModel()
    adapter = RoboCasaLocalMemoryPolicyAdapter(
        SimpleNamespace(model=model),
        mode="required",
        max_sessions=slots,
        decode_image=lambda value: torch.full((3, 256, 512), int(value), dtype=torch.uint8),
    )
    return adapter, model


def registration(slot="slot0", generation=1, **changes):
    return {
        "run_lease": "run",
        "manifest_digest": "a" * 64,
        "slot_id": slot,
        "generation": generation,
        "task": "CloseFridge",
        "episode_index": 0,
        "attempt_id": f"attempt-{generation}",
        **changes,
    }


def register(actor, slot="slot0", generation=1, **changes):
    return actor.register(registration(slot, generation, **changes), "cpu-owner").json()["lease"]


def payload(lease, step=0, start=0, value=1):
    return {
        "image": "0",
        "prompt": "close the fridge",
        "domain_name": "robocasa",
        "image_size": 256,
        "state": [0.0] * 15,
        "local_memory": {
            "session_id": lease["session_id"],
            "episode_id": lease["episode_id"],
            "consumer_step": step,
            "reset": step == 0,
            "image_size": 256,
            "preprocess_profile": PREPROCESS_PROFILE,
            "evidence_version": EVIDENCE_VERSION,
            "evidence_format": EVIDENCE_FORMAT,
            "evidence": [
                {"source_step": i, "composite_image": str(value), "executed_action": [value * 0.01] * 15}
                for i in range(start, step)
            ],
        },
    }


def wait_result(actor, lease, pending, *, expected=200):
    value = pending.json()
    if pending.status == 200:
        return pending
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            result = actor.result(lease, value["request_key"], value["payload_sha256"])
        except RuntimeError:
            if expected != 409:
                raise
            time.sleep(0.005)
            continue
        if result.status != 202:
            if result.status == 409 and not result.json().get("quiescent") and not actor.info()["quarantined"]:
                time.sleep(0.005)
                continue
            assert result.status == expected, result.json()
            return result
        time.sleep(0.005)
    raise AssertionError(f"bounded CPU actor did not finish: {actor.info()}")


def ack(actor, lease, result):
    value = result.json()
    return actor.acknowledge(lease, value["request_key"], value["payload_sha256"])


@pytest.fixture
def runtime():
    adapter, model = make_adapter()
    calls = []

    def generate(request, update):
        calls.append((request, adapter.prefixes(update)))
        token = adapter.prefixes(update)[0]
        value = 0.0 if token is None else float(token.sum())
        return [[value] * 15 for _ in range(16)]

    actor = SessionTransactions(
        adapter,
        generate,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0", "slot1"),
    )
    yield actor, adapter, model, calls
    assert actor.close(), actor.info()


def test_real_fast_weights_isolated_w0_readonly_reset_a_preserves_b(runtime):
    actor, adapter, model, calls = runtime
    slow = {key: value.detach().clone() for key, value in model.net.named_parameters()}
    a, b = register(actor), register(actor, "slot1")
    for lease in (a, b):
        result = wait_result(actor, lease, actor.submit(lease, payload(lease)))
        assert result.json()["local_memory"]["prefix_present"] is False
        ack(actor, lease, result)
    for lease, value in ((a, 1), (b, 7)):
        result = wait_result(actor, lease, actor.submit(lease, payload(lease, 2, 0, value)))
        assert result.json()["local_memory"]["prefix_present"] is True
        ack(actor, lease, result)
    snapshot = adapter.snapshot
    wa, wb = (snapshot.fast[lease["session_id"]].state for lease in (a, b))
    assert any(not torch.equal(x, y) for x, y in zip(wa, wb, strict=True))
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(wa, wb, strict=True))
    for key, parameter in model.net.named_parameters():
        torch.testing.assert_close(parameter, slow[key], rtol=0, atol=0)
        assert parameter.grad is None
    assert actor.end(a).json()["acknowledged"] is True
    assert adapter.snapshot.fast[b["session_id"]] is snapshot.fast[b["session_id"]]
    assert adapter.snapshot.visual[b["session_id"]] is snapshot.visual[b["session_id"]]
    assert adapter.snapshot.responses[b["session_id"]] is snapshot.responses[b["session_id"]]
    fresh = register(actor, generation=2)
    assert fresh["session_id"] != a["session_id"]
    result = wait_result(actor, fresh, actor.submit(fresh, payload(fresh)))
    assert result.json()["local_memory"]["prefix_present"] is False
    assert adapter.snapshot.fast[fresh["session_id"]].state is None
    assert len(calls) == 5 and model.scan_calls == 2


def test_response_replay_before_after_ack_never_reserves_or_generates_twice(runtime):
    actor, adapter, model, calls = runtime
    lease = register(actor)
    request = payload(lease)
    result = wait_result(actor, lease, actor.submit(lease, request))
    size = actor.info()["encoded_bytes"]
    assert size > 0
    assert actor.submit(lease, request).body == result.body
    assert actor.info()["encoded_bytes"] == size
    ack(actor, lease, result)
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    assert actor.submit(lease, request).body == result.body
    ack(actor, lease, result)
    assert len(calls) == 1 and model.scan_calls == 0
    changed = deepcopy(request)
    changed["prompt"] = "different instruction"
    with pytest.raises(ValueError, match="changed payload"):
        actor.submit(lease, changed)
    assert actor.info()["tickets"] == 0
    next_result = wait_result(actor, lease, actor.submit(lease, payload(lease, 1, 0)))
    with pytest.raises(ValueError, match="stale or evicted"):
        actor.submit(lease, request)
    assert len(calls) == 2 and adapter.snapshot.fast[lease["session_id"]].consumer_step == 1
    ack(actor, lease, next_result)


@pytest.mark.parametrize("fault", ["generation", "nan", "shape", "serialize", "publication"])
def test_failures_do_not_publish_partial_fast_visual_response(runtime, monkeypatch, fault):
    actor, adapter, _, _ = runtime
    lease = register(actor)
    cold = wait_result(actor, lease, actor.submit(lease, payload(lease)))
    ack(actor, lease, cold)
    original = adapter.snapshot
    if fault == "generation":

        def broken(*_):
            raise RuntimeError("injected generation failure")

        actor.generate = broken
    elif fault == "nan":
        actor.generate = lambda *_: [[float("nan")] * 15 for _ in range(16)]
    elif fault == "shape":
        actor.generate = lambda *_: [[0.0] * 15] * 15
    elif fault == "serialize":
        monkeypatch.setattr(adapter, "status", lambda _: {"invalid": object()})
    else:
        original_build = MemorySnapshot.build
        monkeypatch.setattr(MemorySnapshot, "build", lambda *_, **__: (_ for _ in ()).throw(MemoryError("publish")))
    pending = actor.submit(lease, payload(lease, 1, 0))
    result = wait_result(actor, lease, pending, expected=409)
    assert result.json()["error_type"] in {"RuntimeError", "ValueError", "TypeError", "MemoryError"}
    assert result.json()["traceback"] and adapter.snapshot is original
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    assert not adapter.memory._pending and not adapter._visual_pending
    with pytest.raises(RuntimeError, match="failed"):
        actor.submit(lease, payload(lease, 1, 0))
    if fault == "publication":
        monkeypatch.setattr(MemorySnapshot, "build", original_build)
    assert actor.end(lease).json()["acknowledged"] is True


@pytest.mark.parametrize("during", ["generation", "quiescence"])
def test_end_waits_for_owned_executor_and_releases_only_after_drain(runtime, during):
    actor, adapter, _, _ = runtime
    lease = register(actor)
    entered, release = threading.Event(), threading.Event()
    original_generate = actor.generate

    def barrier():
        entered.set()
        assert release.wait(5), "test barrier must be released"

    if during == "generation":
        actor.generate = lambda *args: (barrier(), original_generate(*args))[1]
    else:
        actor.quiesce = barrier
    original = adapter.snapshot
    pending = actor.submit(lease, payload(lease))
    try:
        assert entered.wait(5)
        assert actor.end(lease).status == 202
        assert actor.end(lease).json()["acknowledged"] is False
        assert actor.info()["encoded_bytes"] > 0
        with pytest.raises(ValueError, match="previous physical Slot"):
            register(actor, generation=2)
        assert adapter.snapshot is original
    finally:
        release.set()
    wait_result(actor, lease, pending, expected=409)
    deadline = time.monotonic() + 5
    while actor.info()["busy"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert actor.end(lease).json()["acknowledged"] is True
    assert actor.end(lease).json()["acknowledged"] is True
    fresh = register(actor, generation=2)
    with pytest.raises(ValueError, match="stale"):
        actor.end(lease)
    assert fresh["session_id"] != lease["session_id"]


def test_unproven_gpu_drain_quarantines_without_state_commit_or_end_ack():
    adapter, _ = make_adapter(1)

    def no_drain():
        raise RuntimeError("unproven device drain")

    actor = SessionTransactions(
        adapter,
        lambda *_: [[0.0] * 15] * 16,
        no_drain,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
    )
    lease = register(actor)
    original = adapter.snapshot
    pending = actor.submit(lease, payload(lease))
    wait_result(actor, lease, pending, expected=409)
    deadline = time.monotonic() + 5
    while not actor.info()["quarantined"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert actor.info()["quarantined"]
    assert actor.end(lease).json()["acknowledged"] is False
    assert actor.info()["tickets"] == 1 and actor.info()["encoded_bytes"] > 0
    assert adapter.snapshot is original
    assert actor.close(timeout_s=0.01) is False


def test_pair_owner_prevents_fast_only_commit_reset_and_snapshot_is_coherent(runtime):
    actor, adapter, _, _ = runtime
    lease = register(actor)
    update = adapter.prepare(payload(lease))
    with pytest.raises(RuntimeError, match="transaction owner"):
        adapter.memory.commit(update.memory)
    with pytest.raises(RuntimeError, match="transaction owner"):
        adapter.memory.reset_session(lease["session_id"])
    old = adapter.snapshot
    record = CachedMemoryResponse("key", "sha", b"{}")
    adapter.commit(update, record)
    assert old.fast == old.visual == old.responses == {}
    assert set(adapter.snapshot.fast) == set(adapter.snapshot.visual) == set(adapter.snapshot.responses)
    assert adapter.snapshot.responses[lease["session_id"]] is record


@pytest.mark.parametrize("kind", ["owner", "run", "manifest", "slot", "generation", "frontier", "payload", "json"])
def test_identity_and_input_negatives_do_not_create_tickets(runtime, kind):
    actor, _, _, calls = runtime
    if kind == "owner":
        with pytest.raises(ValueError):
            actor.register(registration(), "wrong-owner")
        return
    if kind in ("run", "manifest", "slot", "generation"):
        change = {
            "run": {"run_lease": "other"},
            "manifest": {"manifest_digest": "b" * 64},
            "slot": {"slot_id": "foreign"},
            "generation": {"generation": True},
        }[kind]
        with pytest.raises(ValueError):
            actor.register(registration(**change), "cpu-owner")
        return
    lease = register(actor)
    request = payload(lease)
    if kind == "frontier":
        request["local_memory"]["consumer_step"] = True
    elif kind == "payload":
        request["items"] = [request.copy()]
    else:
        request["state"][0] = float("nan")
    with pytest.raises(ValueError):
        actor.submit(lease, request)
    assert actor.info()["tickets"] == 0 and not calls


def test_strict_json_rejects_duplicates_invalid_utf8_and_nonfinite():
    for raw in (b'{"a":1,"a":2}', b"\xff", b'{"a":NaN}'):
        with pytest.raises((ValueError, UnicodeError)):
            strict_json(raw)
    assert strict_json(json_bytes({"instruction": "关冰箱"})) == {"instruction": "关冰箱"}


def test_inflight_bytes_stay_reserved_through_commit_until_delivery_ack(runtime):
    actor, _, _, _ = runtime
    a, b = register(actor), register(actor, "slot1")
    first, second = payload(a), payload(b)
    actor.max_bytes = len(json_bytes(first)) + len(json_bytes(second)) - 1
    result = wait_result(actor, a, actor.submit(a, first))
    before = actor.info()["encoded_bytes"]
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        actor.submit(b, second)
    assert actor.info()["tickets"] == 1 and actor.info()["encoded_bytes"] == before
    ack(actor, a, result)
    wait_result(actor, b, actor.submit(b, second))


def test_deadline_and_mutable_caller_input_cannot_change_owned_work(runtime):
    actor, adapter, _, calls = runtime
    lease = register(actor)
    clock = [0.0]
    actor.clock, actor.timeout_s = lambda: clock[0], 1.0
    entered, release = threading.Event(), threading.Event()
    original_generate = actor.generate
    seen = []

    def generate(request, update):
        seen.append(request["prompt"])
        entered.set()
        assert release.wait(5)
        return original_generate(request, update)

    actor.generate = generate
    request = payload(lease)
    original = adapter.snapshot
    pending = actor.submit(lease, request)
    try:
        assert entered.wait(5)
        request["prompt"] = "mutation after admission"
        clock[0] = 2.0
        with pytest.raises(RuntimeError, match="cancelled/expired"):
            actor.result(lease, pending.json()["request_key"], pending.json()["payload_sha256"])
        assert actor.end(lease).json()["acknowledged"] is False
        assert actor.info()["encoded_bytes"] > 0 and adapter.snapshot is original
    finally:
        release.set()
    wait_result(actor, lease, pending, expected=409)
    assert seen == ["close the fridge"] and len(calls) == 1
    assert adapter.snapshot is original and actor.info()["encoded_bytes"] == 0
    assert actor.end(lease).json()["acknowledged"] is True


def test_model_scan_receives_clones_not_committed_fast_storage(runtime):
    actor, adapter, _, _ = runtime
    lease = register(actor)
    for request in (payload(lease), payload(lease, 1, 0)):
        result = wait_result(actor, lease, actor.submit(lease, request))
        ack(actor, lease, result)
    original = adapter.snapshot
    saved = [value.clone() for value in original.fast[lease["session_id"]].state]
    scan = adapter.memory.scan_local_memory

    def mutate_then_fail(visual, action, valid, state, **kwargs):
        state[0].add_(100)
        raise RuntimeError("injected scan failure after in-place candidate modification")

    adapter.memory.scan_local_memory = mutate_then_fail
    try:
        with pytest.raises(RuntimeError, match="candidate modification"):
            adapter.prepare(payload(lease, 2, 1))
    finally:
        adapter.memory.scan_local_memory = scan
    assert adapter.snapshot is original
    for actual, expected in zip(original.fast[lease["session_id"]].state, saved, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prior_response_replay_and_ack_do_not_release_new_inflight_frontier(runtime):
    actor, _, _, calls = runtime
    lease = register(actor)
    first_request = payload(lease)
    old = wait_result(actor, lease, actor.submit(lease, first_request))
    ack(actor, lease, old)
    entered, release = threading.Event(), threading.Event()
    original_generate = actor.generate

    def held(request, update):
        entered.set()
        assert release.wait(5)
        return original_generate(request, update)

    actor.generate = held
    next_request = payload(lease, 1, 0)
    next_pending = actor.submit(lease, next_request)
    try:
        assert entered.wait(5)
        owned_bytes = actor.info()["encoded_bytes"]
        assert actor.submit(lease, first_request).body == old.body
        ack(actor, lease, old)
        assert actor.info()["tickets"] == 1 and actor.info()["encoded_bytes"] == owned_bytes
    finally:
        release.set()
    latest = wait_result(actor, lease, next_pending)
    ack(actor, lease, latest)
    assert len(calls) == 2


def test_abort_cleanup_failure_cannot_be_reported_as_clean(monkeypatch):
    adapter, _ = make_adapter(1)
    actor = SessionTransactions(
        adapter,
        lambda *_: [[float("nan")] * 15] * 16,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
    )
    lease = register(actor)
    monkeypatch.setattr(adapter, "abort", lambda _: (_ for _ in ()).throw(RuntimeError("abort failed")))
    pending = actor.submit(lease, payload(lease))
    result = wait_result(actor, lease, pending, expected=409)
    assert result.json()["state"] == "QUARANTINED"
    assert actor.end(lease).json()["acknowledged"] is False
    assert actor.info()["tickets"] == 1
    assert actor.close(timeout_s=0.01) is False


def test_huge_frontier_gap_rejected_before_adapter_range_materialization(runtime):
    actor, _, model, _ = runtime
    lease = register(actor)
    cold = wait_result(actor, lease, actor.submit(lease, payload(lease)))
    ack(actor, lease, cold)
    bad = payload(lease, 1, 0)
    bad["local_memory"]["consumer_step"] = 10**18
    with pytest.raises(ValueError, match="bounded contiguous"):
        actor.submit(lease, bad)
    assert actor.info()["tickets"] == 0 and model.scan_calls == model.encoded == 0

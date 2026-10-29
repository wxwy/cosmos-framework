"""Real CPU Local state/native carriers; diffusion, device and HTTP boundaries doubled."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from cosmos_framework.inference import robocasa_batch_transactions as module
from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions
from cosmos_framework.inference.robocasa_session_transactions_test import (
    ack,
    make_adapter,
    payload,
    register,
    submit,
    wait_result,
)
from cosmos_framework.inference.robocasa_session_transactions_test import clock_kernel as clock_kernel
from cosmos_framework.inference.robocasa_transaction_client import TransactionPolicyClient
from cosmos_framework.model.generator.mot.memory_prefix import attach_local_prefixes, build_memory_prefix_context
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.robocasa_transactions_b1_test import DirectTransport, service


@pytest.fixture
def make_native(monkeypatch):
    actors = []

    def make(slots=3, cap=3, wait=80):
        obj = service(monkeypatch)
        obj.local_memory_adapter, obj.model = make_adapter(slots)
        obj.model.eval()
        obj.native_calls = []

        def generation(batch, **kwargs):
            n = len(batch["prompt"])
            prefixes = kwargs["_local_memory_prefixes"]
            plans, clean = attach_local_prefixes(batch["sequence_plan"], GenerationDataClean(n, False), prefixes)
            projector = torch.nn.Linear(32, 6, bias=False)
            context = build_memory_prefix_context(prefixes, projector, torch.zeros(6), target_dtype=torch.float32)
            obj.native_calls.append(
                {
                    "batch": batch,
                    "kwargs": kwargs,
                    "has_local": [p.has_local_memory for p in plans],
                    "compact": clean.x0_tokens_local_memory,
                    "context": context,
                }
            )
            actions = []
            for i, token in enumerate(prefixes):
                offset = 0 if token is None else float(token.square().sum()) * 0.0001
                value = float(batch["action"][i][0][0, 0]) + offset
                actions.append(torch.full((1, 17, 64), value))
            return {"action": actions, "vision": [torch.zeros(1, 3, 1, 8, 8) for _ in range(n)]}

        obj.model.generate_samples_from_batch = generation
        actor = BatchedSessionTransactions(
            obj.local_memory_adapter,
            obj._transaction_batch,
            lambda: None,
            decode_media=obj._transaction_decode_media,
            max_batch_size=cap,
            batch_wait_ms=wait,
            owner_token="cpu-owner",
            run_lease="run",
            manifest_digest="a" * 64,
            slot_ids=tuple(f"slot{i}" for i in range(slots)),
            model_lock=obj._lock,
            timeout_s=5,
        )
        obj.transactions = actor
        actors.append(actor)
        return obj, actor

    yield make
    for actor in actors:
        if actor.info()["quarantined"]:
            assert actor.close() is False  # Never pretend unknown device cleanup succeeded.
            actor._executor.shutdown(wait=True)
        else:
            assert actor.close()


def group(actor, leases, requests):
    pending = [submit(actor, lease, request) for lease, request in zip(leases, requests, strict=True)]
    return [wait_result(actor, lease, p) for lease, p in zip(leases, pending, strict=True)]


@pytest.mark.parametrize("n,cap", [(1, 1), (2, 2), (3, 3), (2, 8), (1, 4), (3, 2)])
def test_one_native_call_per_full_or_partial_batch_mapping_and_carriers(make_native, n, cap):
    obj, actor = make_native(n, cap)
    leases = [register(actor, f"slot{i}") for i in range(n)]
    reqs = [payload(l) for l in leases]
    for i, r in enumerate(reqs):
        r["state"][0] = i * 0.1
        r["prompt"] = f"real instruction {i}"
    results = group(actor, leases, reqs)
    assert len(obj.native_calls) == (n + cap - 1) // cap
    assert [r.json()["assignment"]["session_id"] for r in results] == [l["session_id"] for l in leases]
    for call in obj.native_calls:
        size = len(call["batch"]["prompt"])
        assert size <= cap and call["kwargs"]["seed"] == [7] * size
        assert len(call["batch"]["sequence_plan"]) == size
        assert all(a[0].shape == (17, 64) for a in call["batch"]["action"])
        assert call["has_local"] == [False] * size and call["context"] is None
    for lease, result in zip(leases, results, strict=True):
        ack(actor, lease, result)
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    assert actor.info()["batches"] == len(obj.native_calls)


def test_mixed_hot_cold_hot_native_prefix_offsets_and_fast_state_reset_isolation(make_native):
    obj, actor = make_native()
    leases = [register(actor, f"slot{i}") for i in range(3)]
    slow = {k: p.detach().clone() for k, p in obj.model.named_parameters()}
    for i in (0, 2):
        r = wait_result(actor, leases[i], submit(actor, leases[i], payload(leases[i])))
        ack(actor, leases[i], r)
    results = group(actor, leases, [payload(leases[0], 2, 0, 1), payload(leases[1]), payload(leases[2], 2, 0, 9)])
    call = obj.native_calls[-1]
    assert call["has_local"] == [True, False, True]
    assert len(call["compact"]) == 2
    assert call["context"].sample_offsets.tolist() == [0, 4, 4, 8]
    assert call["context"].present.tolist() == [True, False, True]
    a = obj.local_memory_adapter.snapshot.fast[leases[0]["session_id"]]
    b = obj.local_memory_adapter.snapshot.fast[leases[2]["session_id"]]
    assert any(not torch.equal(x, y) for x, y in zip(a.state, b.state, strict=True))
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a.state, b.state, strict=True))
    for k, p in obj.model.named_parameters():
        torch.testing.assert_close(p, slow[k], rtol=0, atol=0)
        assert p.grad is None
    for lease, result in zip(leases, results, strict=True):
        ack(actor, lease, result)
    assert actor.end(leases[0]).json()["acknowledged"]
    assert obj.local_memory_adapter.snapshot.fast[leases[2]["session_id"]] is b
    fresh = register(actor, generation=2)
    cold = wait_result(actor, fresh, submit(actor, fresh, payload(fresh)))
    assert cold.json()["local_memory"]["prefix_present"] is False


@pytest.mark.parametrize(
    "fault", ["prepare0", "prepare1", "generate", "cardinality", "nan", "shape", "serialize", "publication", "drain"]
)
def test_sealed_member_failures_abort_entire_bundle_without_partial_state(make_native, monkeypatch, fault):
    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    cold = group(actor, leases, [payload(l) for l in leases])
    for l, r in zip(leases, cold, strict=True):
        ack(actor, l, r)
    before = obj.local_memory_adapter.snapshot
    original = actor.generate_many
    if fault.startswith("prepare"):
        prepare = actor.adapter.prepare
        fail_at = int(fault[-1])
        count = [0]

        def broken(req):
            current = count[0]
            count[0] += 1
            if current == fail_at:
                raise RuntimeError("injected member prepare failure")
            return prepare(req)

        monkeypatch.setattr(actor.adapter, "prepare", broken)
    elif fault in ("generate", "cardinality", "nan", "shape"):

        def bad(reqs, updates):
            if fault == "generate":
                raise RuntimeError("injected native Generation failure")
            out = original(reqs, updates)
            if fault == "cardinality":
                return replace(out, actions=out.actions[:1])
            if fault == "nan":
                out.actions[1][0][0] = float("nan")
            else:
                out.actions[1].pop()
            return out

        actor.generate_many = bad
    elif fault == "serialize":
        encode = module.json_bytes
        count = [0]

        def fail_second(value):
            if "native_batch" in value:
                count[0] += 1
                if count[0] == 2:
                    raise ValueError("injected serialization failure")
            return encode(value)

        monkeypatch.setattr(module, "json_bytes", fail_second)
    elif fault == "publication":
        monkeypatch.setattr(
            actor.adapter, "commit_many", lambda _: (_ for _ in ()).throw(MemoryError("allocation failure"))
        )
    else:
        actor.quiesce = lambda: (_ for _ in ()).throw(RuntimeError("unproven device drain"))
    pending = [submit(actor, l, payload(l, 1, 0, 1 + i)) for i, l in enumerate(leases)]
    for l, p in zip(leases, pending, strict=True):
        reply = wait_result(actor, l, p, expected=409)
        assert reply.status == 409 and reply.json()["error_type"]
    assert obj.local_memory_adapter.snapshot is before
    if fault != "drain":
        assert not actor.adapter._visual_pending and not actor.adapter.memory._pending
        assert actor.info()["tickets"] == 0
    else:
        assert actor.info()["quarantined"] and actor.info()["tickets"] == 2
        assert actor.end(leases[0]).json()["acknowledged"] is False


def test_atomic_publication_validates_all_pending_capabilities_before_any_write(make_native):
    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    updates = [obj.local_memory_adapter.prepare(payload(l)) for l in leases]
    before = obj.local_memory_adapter.snapshot
    obj.local_memory_adapter.abort(updates[1])
    with pytest.raises(RuntimeError, match="pending capability"):
        obj.local_memory_adapter.commit_many([(updates[0], None), (updates[1], None)])
    assert obj.local_memory_adapter.snapshot is before
    obj.local_memory_adapter.abort(updates[0])


def test_End_during_sealed_generation_aborts_all_members_and_waits_for_executor(make_native):
    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    entered, release = threading.Event(), threading.Event()
    original = actor.generate_many

    def blocked(reqs, updates):
        entered.set()
        assert release.wait(5)
        return original(reqs, updates)

    actor.generate_many = blocked
    before = obj.local_memory_adapter.snapshot
    pending = [submit(actor, l, payload(l)) for l in leases]
    try:
        assert entered.wait(5)
        assert actor.end(leases[0]).json()["acknowledged"] is False
        assert actor.info()["tickets"] == 2
    finally:
        release.set()
    for l, p in zip(leases, pending, strict=True):
        wait_result(actor, l, p, expected=409)
    assert obj.local_memory_adapter.snapshot is before
    assert actor.end(leases[0]).json()["acknowledged"] is True


def test_original_deadline_failure_after_shared_generation_aborts_every_member(make_native):
    obj, actor = make_native(2, 2)
    clock = [0.0]
    actor.clock = lambda: clock[0]
    leases = [register(actor, f"slot{i}") for i in range(2)]
    original = actor.generate_many

    def late(reqs, updates):
        value = original(reqs, updates)
        clock[0] = 6.0
        return value

    actor.generate_many = late
    before = obj.local_memory_adapter.snapshot
    pending = [submit(actor, l, payload(l)) for l in leases]
    for l, p in zip(leases, pending, strict=True):
        wait_result(actor, l, p, expected=409)
    assert len(obj.native_calls) == 1 and obj.local_memory_adapter.snapshot is before


@pytest.mark.parametrize("drop", ["/v2/inference", "/v2/inference/ack"])
def test_real_client_dispatch_B2_replay_and_delivery_loss_never_regenerates(make_native, drop):
    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    clients = []
    for lease in leases:
        c = TransactionPolicyClient(
            "http://cpu", timeout_s=5, transport=DirectTransport(obj, (drop,)), batch_profile=True, max_batch_size=2
        )
        c.bind_assignment({"lease": lease, "prediction_clock_domain": actor.clock_domain})
        clients.append(c)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda pair: pair[0].predict(payload(pair[1])), zip(clients, leases, strict=True)))
    assert len(obj.native_calls) == 1 and actor.info()["tickets"] == 0
    assert results[0]["native_batch"]["id"] == results[1]["native_batch"]["id"]
    assert sorted(r["native_batch"]["index"] for r in results) == [0, 1]
    for c, l, r in zip(clients, leases, results, strict=True):
        snapshot = obj.local_memory_adapter.snapshot
        assert c.predict(payload(l)) == r
        assert obj.local_memory_adapter.snapshot is snapshot
        c.end()
    assert len(obj.native_calls) == 1


@pytest.mark.parametrize("fault", ["decode", "spool", "none"])
def test_optional_media_cannot_block_action_commit_or_SR_and_is_bounded(make_native, monkeypatch, fault):
    obj, actor = make_native(3, 3)
    leases = [register(actor, f"slot{i}") for i in range(3)]
    if fault == "decode":
        actor.decode_media = lambda _: (_ for _ in ()).throw(ValueError("bad video"))
    elif fault == "spool":
        actor.decode_media = None
    else:
        actor.decode_media = lambda _: ["synthetic-png-diagnostic"]
    reqs = [{**payload(l), "media": {"replan": 0}} for l in leases]
    results = group(actor, leases, reqs)
    assert len(obj.native_calls) == 1 and all(r.status == 200 for r in results)
    assert actor.info()["media_jobs"] <= 2 and actor.info()["media_cpu_bytes"] <= 128 * 1024**2
    for l, r in zip(leases, results, strict=True):
        ack(actor, l, r)
        data = r.json()
        if data["media"]["status"] == "pending":
            limit = time.monotonic() + 5
            while time.monotonic() < limit:
                media = actor.media_result(l, data["request_key"], data["payload_sha256"])
                if media.status == 200:
                    break
                time.sleep(0.005)
            assert media.json()["media"]["status"] == ("error" if fault == "decode" else "ready")
        assert actor.end(l).json()["acknowledged"] is True
    assert actor.info()["media_jobs"] == 0


def test_new_batch_profile_B1_matches_existing_legacy_action_and_fast_state(make_native):
    modern, actor = make_native(1, 1)
    legacy, _ = make_native(1, 1)
    legacy.transactions = None
    legacy.model.decode = lambda _: torch.zeros(1, 3, 1, 16, 16)
    lease = register(actor)
    for request in (payload(lease), payload(lease, 2, 0, 7)):
        current = wait_result(actor, lease, submit(actor, lease, request))
        reference = legacy.predict_policy(request)
        torch.testing.assert_close(
            torch.tensor(current.json()["action"]), torch.tensor(reference["action"]), rtol=0, atol=0
        )
        a = modern.local_memory_adapter.snapshot.fast[lease["session_id"]]
        b = legacy.local_memory_adapter.snapshot.fast[lease["session_id"]]
        if a.state is not None:
            for left, right in zip(a.state, b.state, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        ack(actor, lease, current)


@pytest.mark.parametrize("short_member", [0, 1])
def test_one_member_deadline_expires_and_other_live_member_still_rolls_back_atomically(make_native, short_member):
    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    clock = [0.0]
    actor.clock = lambda: clock[0]
    original = actor.generate_many

    def later(reqs, updates):
        output = original(reqs, updates)
        clock[0] = 1.0
        return output

    actor.generate_many = later
    before = obj.local_memory_adapter.snapshot
    pending = [
        actor.submit(
            l,
            payload(l),
            budget={"clock_domain": actor.clock_domain, "started_at": 0, "deadline": 0.5 if i == short_member else 5.0},
        )
        for i, l in enumerate(leases)
    ]
    for l, p in zip(leases, pending, strict=True):
        wait_result(actor, l, p, expected=409)
    assert obj.local_memory_adapter.snapshot is before and len(obj.native_calls) == 1


def test_End_before_seal_faults_only_queued_assignment_and_preserves_peer(make_native):
    obj, actor = make_native(2, 2, wait=200)
    a, b = register(actor), register(actor, "slot1")
    p = submit(actor, a, payload(a))
    assert actor.end(a).json()["acknowledged"] is False
    q = submit(actor, b, payload(b))
    wait_result(actor, a, p, expected=409)
    good = wait_result(actor, b, q)
    assert good.status == 200 and len(obj.native_calls) == 1
    assert a["session_id"] not in obj.local_memory_adapter.snapshot.fast
    assert actor.end(a).json()["acknowledged"] is True
    ack(actor, b, good)


@pytest.mark.parametrize("field,value", [("index", 8), ("size", 9), ("id", "wrong-batch"), ("size", True)])
def test_client_rejects_malformed_batch_mapping_before_delivery_ack(make_native, field, value):
    from cosmos_framework.inference.robocasa_session_transactions import Reply, json_bytes
    from cosmos_framework.scripts.robocasa_transactions_b1_test import Response

    obj, actor = make_native(1, 1)
    lease = register(actor)

    class WrongMapping(DirectTransport):
        def request(self, *a, **k):
            r = super().request(*a, **k)
            body = r.json()
            if "native_batch" in body:
                body["native_batch"][field] = value
                return Response(Reply(r.status_code, json_bytes(body)))
            return r

    client = TransactionPolicyClient(
        "http://cpu", timeout_s=5, transport=WrongMapping(obj), batch_profile=True, max_batch_size=1
    )
    client.bind_assignment({"lease": lease, "prediction_clock_domain": actor.clock_domain})
    with pytest.raises(ValueError, match="action/Local status"):
        client.predict(payload(lease))
    assert actor.info()["tickets"] == 1
    client.end()


def test_real_optional_decoder_runs_after_atomic_action_commit_and_outputs_bounded_PNG(make_native, monkeypatch):
    import base64
    import io

    from PIL import Image

    from cosmos_framework.scripts.robocasa_transactions_b1_test import server

    obj, actor = make_native(1, 1)
    lease = register(actor)

    def decode(value):
        assert lease["session_id"] in obj.local_memory_adapter.snapshot.responses
        return torch.zeros(1, 3, 2, 8, 8)

    obj.model.decode = decode
    monkeypatch.setattr(server, "_video_tensor_to_pil_images", lambda _: [Image.new("RGB", (8, 8))] * 2)
    current = wait_result(actor, lease, submit(actor, lease, {**payload(lease), "media": {"replan": 0}}))
    ack(actor, lease, current)
    data = current.json()
    limit = time.monotonic() + 5
    while time.monotonic() < limit:
        reply = actor.media_result(lease, data["request_key"], data["payload_sha256"])
        if reply.status == 200:
            break
        time.sleep(0.005)
    assert reply.json()["media"]["status"] == "ready"
    frames = reply.json()["media"]["video"]
    assert len(frames) == 2 and Image.open(io.BytesIO(base64.b64decode(frames[0]))).size == (8, 8)
    before = obj.local_memory_adapter.snapshot
    assert actor.media_result(lease, data["request_key"], data["payload_sha256"]).body == reply.body
    actor.media_ack(lease, data["request_key"], data["payload_sha256"])
    actor.media_ack(lease, data["request_key"], data["payload_sha256"])
    assert actor.info()["media_jobs"] == 0 and obj.local_memory_adapter.snapshot is before


def test_End_waits_until_optional_decoder_releases_model_ownership(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    entered, release = threading.Event(), threading.Event()

    def decode(value):
        entered.set()
        assert release.wait(5)
        return ["diagnostic"]

    actor.decode_media = decode
    current = wait_result(actor, lease, submit(actor, lease, {**payload(lease), "media": {"replan": 0}}))
    ack(actor, lease, current)
    try:
        assert entered.wait(5)
        assert actor.end(lease).json()["acknowledged"] is False
        assert lease["session_id"] in obj.local_memory_adapter.snapshot.fast
    finally:
        release.set()
    limit = time.monotonic() + 5
    while time.monotonic() < limit and actor.info()["busy"]:
        time.sleep(0.005)
    assert actor.end(lease).json()["acknowledged"] is True
    assert actor.info()["media_jobs"] == 0


@pytest.mark.parametrize("fault", ["spool_bytes", "spool_copy", "media_mapping"])
def test_optional_spool_failure_or_mapping_error_preserves_valid_actions(make_native, fault):
    from dataclasses import replace

    obj, actor = make_native(2, 2)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    generate = obj.model.generate_samples_from_batch

    class Unavailable:
        def numel(self):
            return 128 * 1024**2 if fault == "spool_bytes" else 1

        def element_size(self):
            return 4

        def detach(self):
            raise OSError("injected CPU spool copy failure")

    def broken(batch, **kwargs):
        out = generate(batch, **kwargs)
        out["vision"] = [Unavailable(), Unavailable()]
        return out

    if fault != "media_mapping":
        obj.model.generate_samples_from_batch = broken
    else:
        native = actor.generate_many
        actor.generate_many = lambda r, u: replace(native(r, u), media=(None,))
    results = group(actor, leases, [{**payload(l), "media": {"replan": 0}} for l in leases])
    assert len(obj.native_calls) == 1 and all(r.status == 200 for r in results)
    assert all(r.json()["media"]["status"] == "dropped" for r in results)
    assert actor.info()["media_jobs"] == 0


def test_batch_options_cannot_be_ignored_by_legacy_or_B1_preflight(tmp_path):
    from types import SimpleNamespace

    from cosmos_framework.scripts.robocasa_transactions_b1_test import server

    token = tmp_path / "owner"
    token.write_text("cpu-owner")
    token.chmod(0o600)
    args = SimpleNamespace(
        transaction_profile="batch",
        transaction_owner_token_file=token,
        transaction_run_lease="run",
        transaction_manifest_digest="a" * 64,
        transaction_slot_id=None,
        transaction_slot_ids="a,b",
        transaction_max_batch_size=2,
        transaction_batch_wait_ms=20,
        local_memory_mode="required",
        local_memory_max_sessions=2,
        action_chunk_size=16,
    )
    options = server._transaction_options(args)
    assert options["slot_ids"] == ("a", "b") and options["max_batch_size"] == 2
    for profile in ("b1", "legacy"):
        args.transaction_profile = profile
        with pytest.raises(ValueError):
            server._transaction_options(args)

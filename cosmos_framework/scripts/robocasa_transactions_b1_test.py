from __future__ import annotations

import io
import json
import sys
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import requests
import torch

from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions
from cosmos_framework.inference.robocasa_session_transactions_test import (
    ack,
    clock_kernel,  # noqa: F401 - shared explicit kernel fixture used by pytest
    make_adapter,
    payload,
    register,
    registration,
    submit,
    wait_result,
)
from cosmos_framework.inference.robocasa_transaction_client import TransactionPolicyClient

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    sys.modules.pop("cosmos_framework.scripts.action_policy_server_robocasa", None)
    from cosmos_framework.scripts import action_policy_server_robocasa as server


def service(monkeypatch):
    obj = server.ActionModelService.__new__(server.ActionModelService)
    obj.cfg = SimpleNamespace(
        action_chunk_size=16, max_action_dim=64, fps=20, guidance=1.0, seed=7, num_steps=2, dump_dir=None, dump_every=1
    )
    obj.requires_state, obj.raw_action_dim = True, 15
    obj._prompt_json_formatter = None
    obj.append_duration_fps = obj.append_resolution_info = False
    obj._lock, obj._req_id_lock, obj._req_id = threading.Lock(), threading.Lock(), 0
    obj._input_video_key = lambda: "video"
    obj.action_normalization, obj.action_min, obj.action_range = "minmax", torch.zeros(15), torch.ones(15)
    obj._should_dump = lambda _: False
    obj.local_memory_adapter, obj.model = make_adapter(1)
    obj.model.eval()
    obj.native_calls = []

    def generate(batch, **kwargs):
        obj.native_calls.append((batch, kwargs))
        prefix = kwargs["_local_memory_prefixes"][0]
        offset = 0.0 if prefix is None else float(prefix.square().sum()) * 0.001
        action = torch.arange(17 * 64).reshape(1, 17, 64).float() * 0.0001 + offset
        return {"action": [action], "vision": [torch.empty(0)]}

    obj.model.generate_samples_from_batch = generate
    obj.model.decode = lambda _: torch.zeros(1, 3, 1, 16, 16)
    monkeypatch.setattr(
        server, "_decode_base64_png_to_rgb_uint8", lambda _: torch.zeros(3, 256, 512, dtype=torch.uint8)
    )
    monkeypatch.setattr(server, "remove_reflection_padding", lambda video, _: video)
    monkeypatch.setattr(server, "_video_tensor_to_pil_images", lambda _: [])
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    obj.transactions = None
    return obj


def test_native_b1_action_and_real_fast_state_match_legacy_inputs(monkeypatch):
    modern, legacy = service(monkeypatch), service(monkeypatch)
    actor = SessionTransactions(
        modern.local_memory_adapter,
        modern._transaction_action,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
        model_lock=modern._lock,
    )
    modern.transactions = actor
    try:
        lease = register(actor)
        for request in (payload(lease), payload(lease, 2, 0, 7)):
            current = wait_result(actor, lease, submit(actor, lease, request))
            reference = legacy.predict_policy(request)
            torch.testing.assert_close(
                torch.tensor(current.json()["action"]), torch.tensor(reference["action"]), rtol=0, atol=0
            )
            actual, expected = modern.native_calls[-1], legacy.native_calls[-1]
            assert actual[1]["seed"] == expected[1]["seed"] == [7]
            assert actual[1]["num_steps"] == expected[1]["num_steps"] == 2
            torch.testing.assert_close(actual[0]["action"][0][0], expected[0]["action"][0][0], rtol=0, atol=0)
            assert actual[0]["action"][0][0].shape == (17, 64)
            assert actual[0]["video"][0][0].shape[1] == 17
            assert actual[0]["sequence_plan"] == expected[0]["sequence_plan"]
            modern_state = modern.local_memory_adapter.snapshot.fast[lease["session_id"]]
            legacy_state = legacy.local_memory_adapter.snapshot.fast[lease["session_id"]]
            if modern_state.state is not None:
                for left, right in zip(modern_state.state, legacy_state.state, strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                assert actual[1]["_local_memory_prefixes"][0].shape == (4, 32)
            ack(actor, lease, current)
        assert len(modern.native_calls) == len(legacy.native_calls) == 2
        with pytest.raises(ValueError, match="owned /v2"):
            modern.predict_policy(payload(lease))
        with pytest.raises(ValueError, match="serial /predict only"):
            modern.predict_policy_batch([payload(lease), payload(lease)])
    finally:
        assert actor.close()


class Response:
    def __init__(self, reply):
        self.status_code, self.body = reply.status, reply.body

    def json(self):
        return json.loads(self.body)


class DirectTransport:
    """Actual production dispatcher/client; no socket, server process or model load."""

    def __init__(self, obj, drop=()):
        self.obj, self.drop, self.dropped = obj, set(drop), set()
        self.calls = []

    def request(self, method, url, *, timeout, json=None, headers=None):
        path = urlsplit(url).path
        self.calls.append((method, path, timeout))
        reply = server.transaction_dispatch(self.obj, method, path, record=json, headers=headers)
        if path in self.drop and path not in self.dropped:
            self.dropped.add(path)
            raise requests.ConnectionError("injected lost HTTP response after server operation")
        return Response(reply)


@pytest.mark.parametrize("drop", ["/v2/session/register", "/v2/inference", "/v2/inference/ack", "/v2/session/end"])
def test_real_dispatch_client_retries_lost_response_without_double_generation(monkeypatch, drop):
    obj = service(monkeypatch)
    actor = SessionTransactions(
        obj.local_memory_adapter,
        obj._transaction_action,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
        model_lock=obj._lock,
    )
    obj.transactions = actor
    transport = DirectTransport(obj, (drop,))
    client = TransactionPolicyClient("http://cpu", timeout_s=5, transport=transport)
    try:
        lease = client.register(registration(), "cpu-owner")
        result = client.predict(payload(lease))
        assert result["local_memory"]["session_id"] == lease["session_id"]
        assert actor.info()["tickets"] == 0
        # Re-POST after ACK traverses production admission, not only cached().
        replay = client.predict(payload(lease))
        assert replay == result and actor.info()["tickets"] == 0
        assert len(obj.native_calls) == 1
        client.end()
        assert client.lease is None and actor.info()["sessions"] == 0
        assert drop in transport.dropped
    finally:
        assert actor.close()


def test_client_retry_deadline_is_absolute_not_reset():
    now, remaining = [0.0], []

    class AlwaysLost:
        def request(self, *_, timeout, **__):
            remaining.append(timeout)
            raise requests.ConnectionError("lost")

    client = TransactionPolicyClient(
        "http://cpu",
        timeout_s=0.05,
        transport=AlwaysLost(),
        clock=lambda: now[0],
        pause=lambda amount: now.__setitem__(0, now[0] + amount),
    )
    with pytest.raises(TimeoutError, match="original transaction deadline"):
        client.register(registration(), "synthetic-owner")
    assert 2 <= len(remaining) <= 4 and remaining == sorted(remaining, reverse=True)
    assert now[0] == pytest.approx(0.05)
    with pytest.raises(RuntimeError, match="ownership unresolved"):
        client.end()


@pytest.mark.parametrize(
    "change", ["mode", "sessions", "horizon", "missing", "permissions", "manifest", "developer", "timeout"]
)
def test_transaction_profile_fail_closed_preflight_before_model_load(tmp_path, change):
    path = tmp_path / "capability"
    path.write_text("synthetic-test-owner")
    path.chmod(0o600)
    args = SimpleNamespace(
        transaction_profile="b1",
        local_memory_mode="required",
        local_memory_max_sessions=1,
        action_chunk_size=16,
        transaction_owner_token_file=path,
        transaction_run_lease="run",
        transaction_manifest_digest="a" * 64,
        transaction_slot_id="slot0",
    )
    assert server._transaction_options(args)["slot_ids"] == ("slot0",)
    if change == "mode":
        args.local_memory_mode = "auto"
    elif change == "sessions":
        args.local_memory_max_sessions = 2
    elif change == "horizon":
        args.action_chunk_size = 32
    elif change == "missing":
        args.transaction_run_lease = None
    elif change == "permissions":
        path.chmod(0o644)
    elif change == "manifest":
        args.transaction_manifest_digest = "not-a-digest"
    elif change == "developer":
        args.run_validation = True
    else:
        args.transaction_timeout_s = float("nan")
    with pytest.raises(ValueError):
        server._transaction_options(args)


@pytest.mark.parametrize(
    "body,content_type,length,status",
    [
        (b"{}", "text/plain", "2", 415),
        (b"{}", "application/json", "0", 413),
        (b"{}", "application/json", str(33 * 1024**2), 413),
        (b"{}", "application/json", "3", 400),
        (b'{"lease":1,"lease":2}', "application/json", None, 400),
        (b"\xff", "application/json", "1", 400),
    ],
)
def test_actual_http_handler_bounds_and_strict_json_without_binding_socket(body, content_type, length, status):
    handler = server._ActionHandler.__new__(server._ActionHandler)
    handler.server = SimpleNamespace(
        service=SimpleNamespace(transactions=SimpleNamespace(timeout_s=600, http_slots=threading.BoundedSemaphore(2)))
    )
    handler.path = "/v2/inference"
    handler.headers = {"Content-Type": content_type, "Content-Length": str(len(body)) if length is None else length}
    handler.rfile = io.BytesIO(body)
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    replies = []
    handler._send_json = lambda code, value: replies.append((code, value))
    handler.do_POST()
    assert replies[0][0] == status


def test_v2_legacy_profile_and_numeric_request_id_do_not_bypass_identity():
    assert server.transaction_dispatch(SimpleNamespace(transactions=None), "POST", "/v2/inference").status == 404
    actor = SimpleNamespace(timeout_s=600)
    service_obj = SimpleNamespace(transactions=actor)
    with pytest.raises(ValueError, match="operation/schema"):
        server.transaction_dispatch(service_obj, "POST", "/v2/inference", {"items": [{}, {}]})


def test_http_capacity_refusal_does_not_parse_or_schedule():
    guard = threading.BoundedSemaphore(1)
    guard.acquire()
    handler = server._ActionHandler.__new__(server._ActionHandler)
    handler.server = SimpleNamespace(service=SimpleNamespace(transactions=SimpleNamespace(http_slots=guard)))
    handler.path = "/v2/inference"
    replies = []
    handler._send_json = lambda code, value: replies.append((code, value))
    handler.do_POST()
    assert replies[0][0] == 503


def test_lost_device_cannot_skip_required_gpu_quiescence(monkeypatch):
    obj = server.ActionModelService.__new__(server.ActionModelService)
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def failed_drain():
        called.append(True)
        raise RuntimeError("injected lost-device drain failure")

    monkeypatch.setattr(torch.cuda, "synchronize", failed_drain)
    with pytest.raises(RuntimeError, match="lost-device drain failure"):
        obj._transaction_quiesce()
    assert called == [True]


class BudgetClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def pause(self, amount):
        self.now += amount


def settled(actor):
    with actor._changed:
        assert actor._changed.wait_for(
            lambda: not actor._busy and all(t.quiescent for t in actor._tickets.values()), timeout=5
        ), actor.info()


@pytest.mark.parametrize("hot", [False, True])
def test_original_budget_delayed_upload_cannot_commit_after_client_timeout(monkeypatch, hot):
    obj, clock = service(monkeypatch), BudgetClock()
    entered, release = threading.Event(), threading.Event()
    original_generate = obj._transaction_action
    delayed = [False]

    def generate(request, update):
        if delayed[0]:
            entered.set()
            assert release.wait(5)
            clock.now = 1.2
        return original_generate(request, update)

    actor = SessionTransactions(
        obj.local_memory_adapter,
        generate,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
        model_lock=obj._lock,
        timeout_s=1,
        clock=clock,
    )
    obj.transactions = actor

    class DelayedUpload(DirectTransport):
        def request(self, method, url, *, timeout, json=None, headers=None):
            path = urlsplit(url).path
            if delayed[0] and path == "/v2/inference":
                clock.now = 0.8
            reply = super().request(method, url, timeout=timeout, json=json, headers=headers)
            if path == "/v2/inference":
                if delayed[0]:
                    assert entered.wait(5)
                    clock.now = 1.0
                else:
                    settled(actor)
            return reply

    client = TransactionPolicyClient(
        "http://cpu", timeout_s=1, clock=clock, pause=clock.pause, transport=DelayedUpload(obj)
    )
    try:
        lease = client.register(registration(), "cpu-owner")
        if hot:
            client.predict(payload(lease))
            clock.now = 0.0
        before = obj.local_memory_adapter.snapshot
        delayed[0] = True
        try:
            with pytest.raises(TimeoutError, match="original transaction deadline"):
                client.predict(payload(lease, 1, 0) if hot else payload(lease))
        finally:
            release.set()
            settled(actor)
        assert obj.local_memory_adapter.snapshot is before
        assert actor.info()["tickets"] == 0 and actor.info()["encoded_bytes"] == 0
        assert actor._failed["slot0"]["error_type"] == "RuntimeError"
        client.end()
        assert client.lease is None
    finally:
        release.set()
        assert actor.close()


@pytest.fixture
def budget_rig(monkeypatch):
    obj, clock = service(monkeypatch), BudgetClock()
    actor = SessionTransactions(
        obj.local_memory_adapter,
        obj._transaction_action,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
        model_lock=obj._lock,
        timeout_s=1,
        clock=clock,
    )
    obj.transactions = actor

    class Transport(DirectTransport):
        before = None
        after = None

        def __init__(self):
            super().__init__(obj)
            self.records = []

        def request(self, method, url, *, timeout, json=None, headers=None):
            path = urlsplit(url).path
            # Preserve the actual wire envelope, not references to mutable kwargs.
            self.records.append((path, deepcopy(json)))
            if self.before:
                self.before(path, json)
            try:
                reply = super().request(method, url, timeout=timeout, json=json, headers=headers)
            finally:
                if path == "/v2/inference":
                    settled(actor)
            if self.after:
                self.after(path, reply)
            return reply

    transport = Transport()
    client = TransactionPolicyClient("http://cpu", timeout_s=1, clock=clock, pause=clock.pause, transport=transport)
    lease = client.register(registration(), "cpu-owner")
    yield obj, actor, clock, transport, client, lease
    assert actor.close()


@pytest.mark.parametrize("lost", ["before_accept", "after_accept", "ack", "none"])
def test_original_budget_delay_retry_ack_and_postack_replay(budget_rig, lost):
    obj, actor, clock, transport, client, lease = budget_rig
    seen = []

    def before(path, body):
        if path == "/v2/inference" and not seen:
            seen.append(body["budget"].copy())
            clock.now += 0.25
            if lost == "before_accept":
                raise requests.ConnectionError("injected request lost before first arrival")

    transport.before = before
    if lost in ("after_accept", "ack"):
        transport.drop.add("/v2/inference" if lost == "after_accept" else "/v2/inference/ack")
    result = client.predict(payload(lease))
    assert 0.25 <= clock.now < 1
    posted = [body["budget"] for path, body in transport.records if path == "/v2/inference"]
    assert posted and all(b == seen[0] for b in posted)
    assert seen[0] == {"clock_domain": actor.clock_domain, "started_at": 0.0, "deadline": 1.0}
    assert len(obj.native_calls) == 1 and actor.info()["tickets"] == 0
    snapshot = obj.local_memory_adapter.snapshot
    clock.now = 3  # A NEW read budget permits replay, never new execution/commit.
    assert client.predict(payload(lease)) == result
    assert obj.local_memory_adapter.snapshot is snapshot
    assert len(obj.native_calls) == 1 and actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    client.end()


@pytest.mark.parametrize("stage", ["upload", "generation", "device_drain"])
def test_original_budget_exhaustion_rejects_before_publication(budget_rig, stage):
    obj, actor, clock, transport, client, lease = budget_rig
    snapshot = obj.local_memory_adapter.snapshot
    if stage == "upload":
        transport.before = lambda path, body: setattr(clock, "now", 1.1) if path == "/v2/inference" else None
    elif stage == "generation":
        original = actor.generate

        def late_generation(request, update):
            value = original(request, update)
            clock.now = 1.1
            return value

        actor.generate = late_generation
    else:
        actor.quiesce = lambda: setattr(clock, "now", 1.1)
    with pytest.raises(TimeoutError, match="original .*deadline"):
        client.predict(payload(lease))
    assert obj.local_memory_adapter.snapshot is snapshot
    assert len(obj.native_calls) == (0 if stage == "upload" else 1)
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0
    assert actor.info()["quarantined"] is False
    client.end()


@pytest.mark.parametrize("stage", ["response", "ack"])
def test_response_or_ack_after_original_deadline_is_not_client_success(budget_rig, stage):
    obj, actor, clock, transport, client, lease = budget_rig

    def delay(path, reply):
        if path == ("/v2/inference" if stage == "response" else "/v2/inference/ack"):
            clock.now = 1.1

    transport.after = delay
    with pytest.raises(TimeoutError, match="original transaction deadline"):
        client.predict(payload(lease))
    assert len(obj.native_calls) == 1  # Commit happened in budget; do not fake rollback of a committed result.
    snapshot = obj.local_memory_adapter.snapshot
    transport.after = None
    clock.now = 2
    if stage == "ack":
        client.predict(payload(lease))  # ACK already released Ticket; cached replay is a new read budget.
        assert actor.info()["tickets"] == 0
    else:
        # Undelivered expired work cannot be renewed by a new predict call.
        with pytest.raises(RuntimeError, match="cancelled/expired"):
            client.predict(payload(lease))
    assert obj.local_memory_adapter.snapshot is snapshot and len(obj.native_calls) == 1
    client.end()


@pytest.mark.parametrize(
    "change", ["missing", "domain", "future", "expired", "nonfinite", "huge_int", "bool", "too_large", "extra"]
)
def test_dispatcher_rejects_missing_or_invalid_original_budget_without_allocation(budget_rig, change):
    obj, actor, clock, transport, client, lease = budget_rig
    record = {
        "lease": lease,
        "payload": payload(lease),
        "budget": {
            "clock_domain": actor.clock_domain,
            "started_at": 0,
            "deadline": 1,
        },
    }
    if change == "missing":
        del record["budget"]
    elif change == "domain":
        record["budget"]["clock_domain"] = "foreign-boot-or-time-namespace"
    elif change == "future":
        record["budget"]["started_at"] = 0.5
    elif change == "expired":
        clock.now = 1
    elif change == "nonfinite":
        record["budget"]["deadline"] = float("nan")
    elif change == "huge_int":
        record["budget"]["deadline"] = 10**1000
    elif change == "bool":
        record["budget"]["deadline"] = True
    elif change == "too_large":
        record["budget"]["deadline"] = 2
    else:
        record["budget"]["remaining_timeout"] = 1
    snapshot = obj.local_memory_adapter.snapshot
    with pytest.raises((ValueError, TimeoutError)):
        server.transaction_dispatch(obj, "POST", "/v2/inference", record)
    assert obj.local_memory_adapter.snapshot is snapshot and len(obj.native_calls) == 0
    assert actor.info()["tickets"] == actor.info()["encoded_bytes"] == 0


def test_registration_foreign_clock_domain_cannot_be_used_for_prediction(budget_rig):
    obj, actor, clock, transport, client, lease = budget_rig
    client.end()
    client.clock_domain = "foreign-host"
    with pytest.raises(ValueError, match="clock domain mismatch"):
        client.register(registration(generation=2), "cpu-owner")
    assert not obj.native_calls
    with pytest.raises(RuntimeError, match="ownership unresolved"):
        client.end()


@pytest.mark.parametrize("which", ["boot", "offset", "clock"])
def test_missing_clock_comparability_evidence_fails_closed(monkeypatch, which):
    from cosmos_framework.inference import robocasa_session_transactions as module

    if which == "clock":
        monkeypatch.setattr(
            module.time, "get_clock_info", lambda _: SimpleNamespace(monotonic=True, implementation="other")
        )
    else:
        original = module.Path.open

        def missing(path, *args, **kwargs):
            if (which == "boot" and path.name == "boot_id") or (which == "offset" and path.name == "timens_offsets"):
                raise OSError("kernel clock identity unavailable")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(module.Path, "open", missing)
    with pytest.raises((ValueError, OSError)):
        module.monotonic_clock_domain()


def test_encoding_time_is_part_of_original_prediction_budget(budget_rig, monkeypatch):
    from cosmos_framework.inference import robocasa_transaction_client as module

    obj, actor, clock, transport, client, lease = budget_rig
    encode = module.json_bytes

    def slow_encode(value):
        if "image" in value:
            clock.now += 0.8
        return encode(value)

    monkeypatch.setattr(module, "json_bytes", slow_encode)
    client.predict(payload(lease))
    posted = [body["budget"] for path, body in transport.records if path == "/v2/inference"]
    assert posted == [{"clock_domain": actor.clock_domain, "started_at": 0, "deadline": 1}]
    assert len(obj.native_calls) == 1
    client.end()


def test_expired_original_budget_has_explicit_http_408(budget_rig):
    obj, actor, clock, transport, client, lease = budget_rig
    body = json.dumps(
        {
            "lease": lease,
            "payload": payload(lease),
            "budget": {
                "clock_domain": actor.clock_domain,
                "started_at": 0,
                "deadline": 1,
            },
        }
    ).encode()
    clock.now = 1.1
    handler = server._ActionHandler.__new__(server._ActionHandler)
    handler.server = SimpleNamespace(service=obj)
    handler.path = "/v2/inference"
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    replies = []
    handler._send_json = lambda code, value: replies.append((code, value))
    handler.do_POST()
    assert len(replies) == 1 and replies[0][0] == 408
    assert "original prediction deadline" in replies[0][1]["error"]
    assert actor.info()["tickets"] == 0 and not obj.native_calls


def test_lost_first_response_inflight_retry_retains_original_ticket_deadline(monkeypatch):
    obj, clock = service(monkeypatch), BudgetClock()
    entered, release = threading.Event(), threading.Event()
    native = obj._transaction_action

    def generate(request, update):
        entered.set()
        assert release.wait(5)
        clock.now = 1.1
        return native(request, update)

    actor = SessionTransactions(
        obj.local_memory_adapter,
        generate,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0",),
        model_lock=obj._lock,
        timeout_s=1,
        clock=clock,
    )
    obj.transactions = actor
    budgets, original_ticket = [], []

    class LostInflight(DirectTransport):
        def request(self, method, url, *, timeout, json=None, headers=None):
            path = urlsplit(url).path
            if path == "/v2/inference":
                budgets.append(deepcopy(json["budget"]))
                if len(budgets) == 1:
                    clock.now = 0.4
                    super().request(method, url, timeout=timeout, json=json, headers=headers)
                    assert entered.wait(5)
                    with actor._lock:
                        original_ticket.append(next(iter(actor._tickets.values())))
                    raise requests.ConnectionError("first response lost while Generation inflight")
                reply = super().request(method, url, timeout=timeout, json=json, headers=headers)
                with actor._lock:
                    assert next(iter(actor._tickets.values())) is original_ticket[0]
                    assert original_ticket[0].deadline == 1.0
                release.set()
                settled(actor)
                return reply
            return super().request(method, url, timeout=timeout, json=json, headers=headers)

    client = TransactionPolicyClient(
        "http://cpu", timeout_s=1, clock=clock, pause=clock.pause, transport=LostInflight(obj)
    )
    try:
        lease = client.register(registration(), "cpu-owner")
        before = obj.local_memory_adapter.snapshot
        with pytest.raises(TimeoutError, match="original transaction deadline"):
            client.predict(payload(lease))
        assert len(budgets) == 2 and budgets[0] == budgets[1]
        assert obj.local_memory_adapter.snapshot is before
        assert len(obj.native_calls) == 1 and actor.info()["tickets"] == 0
        client.end()
    finally:
        release.set()
        assert actor.close()


@pytest.mark.parametrize("change", ["boot", "offset", "malformed_boot", "duplicate_offset", "bad_nanos"])
def test_clock_identity_distinguishes_host_and_namespace_and_rejects_invalid_kernel_records(monkeypatch, change):
    from cosmos_framework.inference import robocasa_session_transactions as module

    baseline = module.monotonic_clock_domain()
    original = module.Path.open

    def changed(path, *args, **kwargs):
        if path.name == "boot_id":
            if change == "boot":
                return io.BytesIO(b"2cad22db-2388-4a3d-a652-785ed624fb60\n")
            if change == "malformed_boot":
                return io.BytesIO(b"invalid\n")
        if path.name == "timens_offsets":
            if change == "offset":
                return io.BytesIO(b"monotonic 10 0\nboottime 0 0\n")
            if change == "duplicate_offset":
                return io.BytesIO(b"monotonic 0 0\nmonotonic 0 0\n")
            if change == "bad_nanos":
                return io.BytesIO(b"monotonic 0 1000000000\n")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.Path, "open", changed)
    if change in ("boot", "offset"):
        assert module.monotonic_clock_domain() != baseline
    else:
        with pytest.raises(ValueError):
            module.monotonic_clock_domain()

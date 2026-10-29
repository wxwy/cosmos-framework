from __future__ import annotations

import io
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import requests
import torch

from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions
from cosmos_framework.inference.robocasa_session_transactions_test import (
    ack,
    make_adapter,
    payload,
    register,
    registration,
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
            current = wait_result(actor, lease, actor.submit(lease, request))
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
        args.local_memory_mode = "off"
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

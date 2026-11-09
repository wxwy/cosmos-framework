# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Actual Owner dispatcher/HTTP contracts with real CPU native transaction state.

Diffusion and the committed-boundary lookup are controlled synthetic boundaries.
No server process, CUDA, VAE/DCP loading, simulation or hardware claims are made.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import struct
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference import robocasa_boundary_capture as capture_module
from cosmos_framework.inference import robocasa_boundary_capture_test as capture_tests
from cosmos_framework.inference import robocasa_session_transactions_test as sessions
from cosmos_framework.inference import robocasa_vae_daemon as daemon
from cosmos_framework.inference.robocasa_feature_evidence import FeatureEvidenceOwner, PolicyBoundary
from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions, json_bytes
from cosmos_framework.inference.robocasa_vae_encoder import preprocess_profile_digest, visual96_contract_digest
from cosmos_framework.inference.robocasa_vae_protocol import ProtocolError, source_digest
from cosmos_framework.scripts import robocasa_transactions_b1_test as legacy
from cosmos_framework.simulation.robocasa.eval_utils import canonicalize_raw15_for_env

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]
clock_kernel = sessions.clock_kernel
capture_methods = capture_tests.methods
server = legacy.server
TOKEN = "synthetic-private-owner-control"
KEY = bytes(range(32))


def _identity():
    return {
        "vae_weights_sha256": "1" * 64,
        "vae_config_sha256": "2" * 64,
        "preprocess_profile_sha256": preprocess_profile_digest(256),
        "visual96_contract_sha256": visual96_contract_digest(),
        "encoder_backend": "eager",
        "encoder_dtype": "bfloat16",
    }


def _dispatch(rig, operation, record, *, token=TOKEN):
    headers = {} if token is None else {"X-Cosmos-Owner": token}
    return server.transaction_dispatch(rig.service, "POST", "/v2/feature/" + operation, record, headers)


def _enroll_record(rig):
    return {
        "worker_id": "worker",
        "worker_epoch": "worker-epoch",
        "key_id": "key",
        "key_hex": KEY.hex(),
        "identity": _identity(),
        "clock_domain": rig.actor.clock_domain,
        "bmax": 1,
        "wait_ms": 0,
    }


def _register_record(rig, lease=None):
    return {
        "lease": rig.lease if lease is None else lease,
        "worker_id": "worker",
        "actor_id": "actor",
        "actor_epoch": "actor-epoch",
        "episode_deadline_ns": str(110 * 10**9),
    }


@pytest.fixture
def wire_rig(monkeypatch):
    service, clock = legacy.service(monkeypatch), legacy.BudgetClock()
    clock.now = 100.0
    actor = SessionTransactions(
        service.local_memory_adapter,
        service._transaction_action,
        lambda: None,
        owner_token=TOKEN,
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=("slot0", "slot1"),
        model_lock=service._lock,
        timeout_s=5,
        clock=clock,
    )
    service.transactions = actor
    owner = FeatureEvidenceOwner(
        TOKEN, policy_identity=_identity(), max_sessions=2, clock_ns=lambda: int(clock.now * 10**9)
    )
    boundaries = {}
    owner.bind_boundary_lookup(lambda session, step: boundaries.get((session, step)))
    service.feature_evidence_owner = owner
    lease = actor.register(sessions.registration(), TOKEN).json()["lease"]
    service.cfg.experiment_name = "synthetic-cpu"
    service.cfg.checkpoint_dir = "/synthetic/not-loaded-dcp"
    service.cfg.action_stats_path = None
    service.raw_action_dim_source = "synthetic-explicit"
    service.setup_args = SimpleNamespace(config_file="/synthetic/not-loaded-config", config_file_type="module")
    service.local_feature_info = None
    rig = SimpleNamespace(
        service=service, actor=actor, owner=owner, lease=lease, clock=clock, boundaries=boundaries, release=[]
    )
    yield rig
    for gate in rig.release:
        gate.set()
    assert actor.close(timeout_s=5)
    actor._executor.shutdown(wait=True)


def _ready(rig):
    assert _dispatch(rig, "enroll", _enroll_record(rig)).status == 200
    assert _dispatch(rig, "register", _register_record(rig)).json() == {"registered": True, "duplicate": False}


def _committed(rig):
    request = sessions.payload(rig.lease)
    pending = sessions.submit(rig.actor, rig.lease, request)
    response = sessions.wait_result(rig.actor, rig.lease, pending)
    body = response.json()
    public = {name: value for name, value in rig.lease.items() if name != "capability"}
    boundary = PolicyBoundary.create(
        lease=public,
        request_key=body["request_key"],
        chunk_id=body["request_key"],
        source_step=body["frontier"],
        source_rgb_sha256=source_digest(bytes(256 * 512 * 3)),
        identity=_identity(),
        deadline_ns=105 * 10**9,
        clock_domain=rig.actor.clock_domain,
        visual96=struct.pack("<96f", *([0.25] * 96)),
    )
    rig.boundaries[(public["session_id"], body["frontier"])] = boundary
    assert sessions.ack(rig.actor, rig.lease, response).json()["acknowledged"]
    return body, boundary


def _issue_record(rig, body):
    return {"lease": rig.lease, "request_key": body["request_key"], "payload_sha256": body["payload_sha256"]}


def _complete_record(rig, body):
    canonical = canonicalize_raw15_for_env(body["action"][0], False)[1]
    return {
        "lease": rig.lease,
        "chunk_id": body["request_key"],
        "request_key": body["request_key"],
        "source_step": 0,
        "source_rgb_sha256": source_digest(bytes(256 * 512 * 3)),
        "raw15_f32le_b64": base64.b64encode(struct.pack("<15f", *canonical)).decode("ascii"),
        "step_completed_ns": str(100 * 10**9),
        "actor_id": "actor",
        "actor_epoch": "actor-epoch",
    }


@pytest.mark.parametrize("operation", ["enroll", "register", "issue", "complete", "boundary", "end", "revoke"])
def test_every_feature_endpoint_requires_owner_header_before_state_changes(wire_rig, operation):
    for token in (None, "untrusted-sim-capability-not-owner"):
        with pytest.raises(PermissionError):
            _dispatch(wire_rig, operation, {}, token=token)
    assert not wire_rig.owner._workers and not wire_rig.owner._sessions
    assert not wire_rig.service.native_calls


def test_actual_http_handler_reports_unauthorized_feature_control_without_secret_leak(wire_rig):
    body = json_bytes(_enroll_record(wire_rig))
    handler = server._ActionHandler.__new__(server._ActionHandler)
    handler.server = SimpleNamespace(service=wire_rig.service)
    handler.path = "/v2/feature/enroll"
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.connection = SimpleNamespace(settimeout=lambda _timeout: None)
    replies = []
    handler._send_json = lambda status, value: replies.append((status, json_bytes(value)))
    handler._send_bytes = lambda status, value: replies.append((status, value))
    handler.do_POST()
    assert len(replies) == 1 and replies[0][0] == 403
    assert TOKEN.encode() not in replies[0][1] and KEY.hex().encode() not in replies[0][1]
    assert not wire_rig.owner._workers


def test_enrollment_and_session_registration_match_loaded_policy_and_actual_actor_lease(wire_rig):
    rig = wire_rig
    record = _enroll_record(rig)
    mismatched = deepcopy(record)
    mismatched["identity"]["vae_weights_sha256"] = "f" * 64
    for bad in ({**record, "bmax": 8}, mismatched, {**record, "key_hex": "ab" * 31}):
        with pytest.raises((ValueError, ProtocolError)):
            _dispatch(rig, "enroll", bad)
    assert not rig.owner._workers
    public = _dispatch(rig, "enroll", record).json()
    assert public == {"worker_id": "worker", "worker_epoch": "worker-epoch", "key_id": "key"}
    foreign = deepcopy(rig.lease)
    foreign["episode_id"] = "forged-episode"
    with pytest.raises(ValueError, match="foreign"):
        _dispatch(rig, "register", _register_record(rig, foreign))
    assert not rig.owner._sessions
    result = _dispatch(rig, "register", _register_record(rig)).json()
    assert result == {"registered": True, "duplicate": False}
    assert "capability" not in rig.owner._sessions[rig.lease["session_id"]].lease
    assert _dispatch(rig, "register", _register_record(rig)).json()["duplicate"]


def test_issue_reads_actual_retained_native_response_and_canonicalizes_its_actions(wire_rig):
    rig = wire_rig
    _ready(rig)
    body, boundary = _committed(rig)
    record = _issue_record(rig, body)
    assert _dispatch(rig, "issue", record).json() == {"committed": True, "duplicate": False}
    chunk = rig.owner._sessions[rig.lease["session_id"]].chunk
    expected = tuple(struct.pack("<15f", *canonicalize_raw15_for_env(row, False)[1]) for row in body["action"])
    assert chunk.actions == expected and chunk.start == 0
    assert chunk.deadline == int(boundary.header["deadline_ns"])
    assert any(value != struct.pack("<15f", *row) for value, row in zip(expected, body["action"], strict=True))
    assert _dispatch(rig, "issue", record).json()["duplicate"]
    with pytest.raises(ValueError, match="exact Owner"):
        _dispatch(rig, "issue", {**record, "actions": body["action"]})


@pytest.mark.parametrize("change", ["request_key", "payload_sha256", "foreign_lease"])
def test_issue_cannot_select_forged_or_foreign_native_response(wire_rig, change):
    rig = wire_rig
    _ready(rig)
    body, _ = _committed(rig)
    record = deepcopy(_issue_record(rig, body))
    if change == "foreign_lease":
        record["lease"]["slot_id"] = "slot1"
    else:
        record[change] = "f" * 64
    with pytest.raises(ValueError):
        _dispatch(rig, "issue", record)
    assert rig.owner._sessions[rig.lease["session_id"]].chunk is None


def test_issue_rejects_native_ticket_while_generation_is_still_uncommitted(wire_rig):
    rig = wire_rig
    _ready(rig)
    entered, release = threading.Event(), threading.Event()
    rig.release.append(release)
    original = rig.actor.generate

    def blocked(request, update):
        entered.set()
        assert release.wait(5)
        return original(request, update)

    rig.actor.generate = blocked
    pending = sessions.submit(rig.actor, rig.lease, sessions.payload(rig.lease))
    assert entered.wait(2)
    record = {
        "lease": rig.lease,
        "request_key": pending.json()["request_key"],
        "payload_sha256": pending.json()["payload_sha256"],
    }
    with pytest.raises(ValueError, match="already committed"):
        _dispatch(rig, "issue", record)
    assert rig.owner._sessions[rig.lease["session_id"]].chunk is None
    release.set()
    sessions.wait_result(rig.actor, rig.lease, pending)


@pytest.mark.parametrize("change", ["raw15", "encoding", "timestamp", "actor", "actor_epoch", "rgb"])
def test_complete_wire_bytes_timestamp_and_actor_bind_exact_executed_action(wire_rig, change):
    rig = wire_rig
    _ready(rig)
    body, _ = _committed(rig)
    _dispatch(rig, "issue", _issue_record(rig, body))
    record = _complete_record(rig, body)
    bad = deepcopy(record)
    if change == "raw15":
        bad["raw15_f32le_b64"] = base64.b64encode(struct.pack("<15f", *([0] * 15))).decode()
    elif change == "encoding":
        bad["raw15_f32le_b64"] += "="
    elif change == "timestamp":
        bad["step_completed_ns"] = str(101 * 10**9)
    elif change == "rgb":
        bad["source_rgb_sha256"] = "f" * 64
    else:
        bad["actor_id" if change == "actor" else "actor_epoch"] = "foreign"
    with pytest.raises((ValueError, ProtocolError)):
        _dispatch(rig, "complete", bad)
    assert not rig.owner._sessions[rig.lease["session_id"]].completed
    receipt = _dispatch(rig, "complete", record).json()
    assert receipt["source_step"] == 0 and receipt["success"] is True
    raw = base64.b64decode(record["raw15_f32le_b64"])
    assert receipt["raw15_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["actor_id"] == "actor" and receipt["actor_epoch"] == "actor-epoch"
    assert "capability" not in receipt["lease"] and TOKEN not in json.dumps(receipt)
    assert _dispatch(rig, "complete", record).json() == receipt


def test_boundary_end_revoke_are_owner_only_and_do_not_leak_stale_or_foreign_cache(wire_rig):
    rig = wire_rig
    _ready(rig)
    body, boundary = _committed(rig)
    query = {"lease": rig.lease, "source_step": 0}
    assert _dispatch(rig, "boundary", query).json() == boundary.record()
    foreign = deepcopy(query)
    foreign["lease"]["session_id"] = "foreign-session"
    with pytest.raises(ValueError):
        _dispatch(rig, "boundary", foreign)
    with pytest.raises(ValueError):
        server.transaction_dispatch(rig.service, "GET", "/v2/feature/boundary", None, {})
    assert _dispatch(rig, "end", {"lease": rig.lease}).json() == {"ended": True, "duplicate": False}
    assert _dispatch(rig, "end", {"lease": rig.lease}).json()["duplicate"]
    with pytest.raises(ProtocolError):
        _dispatch(rig, "boundary", query)
    assert _dispatch(rig, "revoke", {"worker_id": "worker", "worker_epoch": "worker-epoch"}).json() == {}
    assert not rig.owner._workers
    assert rig.actor.end(rig.lease).json()["acknowledged"]
    with pytest.raises(ValueError):
        _dispatch(rig, "boundary", query)


def _constructor_feature_block():
    tree = ast.parse(Path(server.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActionModelService")
    initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignments = [
        node
        for node in initialize.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr in ("feature_evidence_owner", "local_feature_info")
            for target in node.targets
        )
    ]
    branches = [
        node for node in initialize.body if isinstance(node, ast.If) and "history_source" in ast.unparse(node.test)
    ]
    assert len(assignments) == 2 and len(branches) == 1
    return compile(ast.Module(body=[*assignments, branches[0]], type_ignores=[]), str(server.__file__), "exec")


def test_c_constructor_attests_actual_model_object_and_info_exposes_only_public_effective_config(wire_rig, monkeypatch):
    rig = wire_rig
    loaded_tokenizer = object()
    rig.service.model.config = SimpleNamespace(tokenizer={"bucket_name": "", "vae_path": "/synthetic/wan-weights.bin"})
    rig.service.model.tokenizer_vision_gen = loaded_tokenizer
    calls = []

    def constructor(tokenizer, *, vae_path=None):
        assert tokenizer is loaded_tokenizer
        value = {"chunk_duration": 17, "use_streaming_encode": False, "causal": True}
        if vae_path is not None:
            value.update(bucket_name="", object_store_credential_path_pretrained="", vae_path=vae_path)
        calls.append(("constructor", deepcopy(value)))
        return value

    def loaded_identity(tokenizer, config):
        assert tokenizer is loaded_tokenizer and config == calls[0][1]
        calls.append(("identity", tokenizer))
        return _identity()

    monkeypatch.setattr(daemon, "worker_constructor_config", constructor)
    monkeypatch.setattr(daemon, "loaded_vae_identity", loaded_identity)
    args = SimpleNamespace(
        history_source="sim_shared_vae",
        local_memory_max_sessions=2,
        identity={**_identity(), "vae_weights_sha256": "f" * 64},
    )
    exec(
        _constructor_feature_block(), {"self": rig.service, "args": args, "transaction_options": {"owner_token": TOKEN}}
    )
    assert [call[0] for call in calls] == ["constructor", "identity", "constructor"]
    assert rig.service.feature_evidence_owner.identity == _identity()
    info = rig.service.get_info()
    assert info["local_feature"]["identity"] == _identity() != args.identity
    assert info["local_feature"]["worker_constructor_config"] == calls[-1][1]
    assert "vae_path" not in info["local_feature"]["worker_constructor_config"]
    encoded = json_bytes(info)
    assert TOKEN.encode() not in encoded and KEY.hex().encode() not in encoded
    assert rig.lease["capability"].encode() not in encoded
    handler = server._ActionHandler.__new__(server._ActionHandler)
    handler.server, handler.path = SimpleNamespace(service=rig.service), "/info"
    replies = []
    handler._send_json = lambda status, value: replies.append((status, value))
    handler.do_GET()
    assert replies == [(200, info)]


def test_c_policy_initialization_accepts_real_omegaconf_loaded_constructor(wire_rig):
    from omegaconf import OmegaConf

    from cosmos_framework.inference.robocasa_vae_daemon import loaded_vae_identity, worker_constructor_config
    from cosmos_framework.inference.robocasa_vae_daemon_test import _tokenizer

    rig = wire_rig
    tokenizer = _tokenizer(omega=True)
    rig.service.model.config = SimpleNamespace(
        tokenizer=OmegaConf.create({"bucket_name": "", "vae_path": "/cpu/synthetic-vae.bin"})
    )
    rig.service.model.tokenizer_vision_gen = tokenizer
    args = SimpleNamespace(history_source="sim_shared_vae", local_memory_max_sessions=2)
    exec(
        _constructor_feature_block(), {"self": rig.service, "args": args, "transaction_options": {"owner_token": TOKEN}}
    )
    info = rig.service.get_info()["local_feature"]
    assert info["identity"] == loaded_vae_identity(tokenizer)
    assert info["worker_constructor_config"] == worker_constructor_config(tokenizer)
    assert type(info["worker_constructor_config"]["encode_exact_durations"]) is list
    assert type(info["worker_constructor_config"]["encode_chunk_frames"]) is dict
    assert info["history_source"] == "sim_shared_vae"
    assert info["policy_bmax"] == 1 and info["policy_wait_ms"] == 0


def test_real_service_c_batch_uses_one_existing_precrop_hook_then_stages_private_boundary(
    wire_rig, capture_methods, monkeypatch
):
    rig = wire_rig
    model, calls = capture_tests._model(capture_methods)
    rig.service.model = model
    context = {
        "lease": {name: value for name, value in rig.lease.items() if name != "capability"},
        "request_key": "native-current",
        "chunk_id": "native-current",
        "source_step": 0,
        "identity": _identity(),
        "deadline_ns": 105 * 10**9,
        "clock_domain": rig.actor.clock_domain,
    }
    pending, staged = object(), []
    rig.service.local_memory_adapter = SimpleNamespace(
        mode="required",
        prefixes=lambda update: (None,),
        boundary_context=lambda update: deepcopy(context),
        stage_boundary=lambda update, boundary: staged.append((update, boundary)),
    )
    rig.service.transactions = SimpleNamespace(media_room=lambda: (0, 0))
    original_capture = capture_module.capture_policy_boundary

    def capture_with_synthetic_clock(*args, **kwargs):
        return original_capture(*args, **kwargs, clock_ns=lambda: int(rig.clock.now * 10**9))

    monkeypatch.setattr(capture_module, "capture_policy_boundary", capture_with_synthetic_clock)

    def generate(batch, **kwargs):
        assert kwargs["_local_memory_prefixes"] == (None,)
        model._normalize_video_databatch_inplace(batch)
        full = model._encode_vision_x0_tokens_prefix(batch["video"], [[0]])
        cropped = model._remove_padding_from_latent(full, batch["image_size"])
        assert full[0].shape[-2:] == (12, 20) and cropped[0].shape[-2:] == (10, 20)
        return {"action": [torch.zeros(1, 17, 64)], "vision": None}

    model.generate_samples_from_batch = generate
    before = dict(rig.boundaries)
    output = rig.service._transaction_batch([sessions.payload(rig.lease)], [pending])
    assert len(calls) == 1 and calls[0].shape == (1, 3, 1, 192, 320)
    assert len(staged) == 1 and staged[0][0] is pending and isinstance(staged[0][1], PolicyBoundary)
    assert staged[0][1].header["source_rgb_sha256"] == source_digest(bytes(256 * 512 * 3))
    assert staged[0][1].header["policy_request_key"] == "native-current"
    assert len(output.actions) == 1 and len(output.actions[0]) == 16
    assert all(len(row) == 15 for row in output.actions[0])
    assert rig.boundaries == before  # Staging is not an independent cache publication.

"""Dual-mode public transactions/native Sim; CPU diffusion and physics doubles."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions
from cosmos_framework.inference.robocasa_local_memory_policy import RoboCasaLocalMemoryPolicyAdapter
from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions
from cosmos_framework.inference.robocasa_session_transactions_test import (
    ack,
    make_adapter,
    payload,
    register,
    registration,
    submit,
    wait_result,
)
from cosmos_framework.inference.robocasa_session_transactions_test import (
    clock_kernel as clock_kernel,
)
from cosmos_framework.inference.robocasa_transaction_client import TransactionPolicyClient
from cosmos_framework.inference.robocasa_transaction_mode import validate_model_mode
from cosmos_framework.scripts.robocasa_transactions_b1_test import DirectTransport, server, service


def forbidden(*args, **kwargs):
    raise AssertionError("No-TTT must not access Local lifecycle/scan/prefix")


def no_ttt_payload(lease, index=0):
    value = payload(lease)
    del value["local_memory"]
    value["inference"] = {"session_id": lease["session_id"], "episode_id": lease["episode_id"], "request_index": index}
    return value


@pytest.fixture
def make_dual(monkeypatch):
    actors = []

    def make(mode, n=2, batch=True):
        obj = service(monkeypatch)
        if mode == "required":
            obj.local_memory_adapter, obj.model = make_adapter(n)
        else:
            obj.model = nn.Module()
            obj.model.net = nn.Module()
            obj.model.net.config = SimpleNamespace(local_memory_enabled=False)
            obj.model.net.scan_local_memory = forbidden
            obj.model.weight = nn.Parameter(torch.zeros(1))
            obj.local_memory_adapter = RoboCasaLocalMemoryPolicyAdapter(
                obj, mode="off", decode_image=forbidden, max_sessions=n
            )
            for name in ("prepare", "commit", "commit_many", "abort", "reset", "status", "prefixes"):
                monkeypatch.setattr(obj.local_memory_adapter, name, forbidden)
        obj.native_calls = []

        def generate(batch, **kwargs):
            count = len(batch["prompt"])
            if mode == "off":
                assert "_local_memory_prefixes" not in kwargs
                assert not hasattr(obj.model.net, "local_memory_runtime")
            else:
                assert len(kwargs["_local_memory_prefixes"]) == count
            obj.native_calls.append((batch, kwargs))
            raw = torch.tensor([0.2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5])
            result = torch.zeros(1, 17, 64)
            result[:, 1:, :15] = raw * 2 - 1
            return {
                "action": [result.clone() for _ in range(count)],
                "vision": [torch.zeros(1, 3, 1, 8, 8) for _ in range(count)],
            }

        obj.model.generate_samples_from_batch = generate
        opts = dict(
            owner_token="cpu-owner",
            run_lease="run",
            manifest_digest="a" * 64,
            slot_ids=tuple(f"slot{i}" for i in range(n)),
            model_lock=obj._lock,
            timeout_s=5,
        )
        actor = (
            BatchedSessionTransactions(
                obj.local_memory_adapter,
                obj._transaction_batch,
                lambda: None,
                max_batch_size=n,
                batch_wait_ms=1000,
                **opts,
            )
            if batch
            else SessionTransactions(obj.local_memory_adapter, obj._transaction_action, lambda: None, **opts)
        )
        obj.transactions = actor
        actors.append(actor)
        return obj, actor

    yield make
    for actor in actors:
        assert actor.close(timeout_s=5)
        actor._executor.shutdown(wait=True)
        assert actor._responses == {} and actor.info()["sessions"] == actor.info()["tickets"] == 0


@pytest.mark.parametrize("mode", ["off", "required"])
@pytest.mark.parametrize("n,batch", [(1, False), (1, True), (2, True)])
def test_public_native_generation_replay_session_isolation_and_real_Local(mode, n, batch, make_dual):
    obj, actor = make_dual(mode, n, batch)
    leases = [register(actor, f"slot{i}") for i in range(n)]

    def request(lease, index):
        return no_ttt_payload(lease, index) if mode == "off" else payload(lease, index, 0)

    for index in (0, 1):
        requests = [request(lease, index) for lease in leases]
        pending = [submit(actor, lease, value) for lease, value in zip(leases, requests, strict=True)]
        results = [wait_result(actor, lease, p) for lease, p in zip(leases, pending, strict=True)]
        assert len(obj.native_calls) == index + 1
        for lease, value, result in zip(leases, requests, results, strict=True):
            body = result.json()
            assert body["native_batch"]["size"] == n
            assert body["assignment"]["session_id"] == lease["session_id"]
            assert ("local_memory" in body) == (mode == "required")
            ack(actor, lease, result)
            assert submit(actor, lease, value).body == result.body
            changed = dict(value, prompt="changed")
            with pytest.raises(ValueError, match="changed payload"):
                submit(actor, lease, changed)
        assert len(obj.native_calls) == index + 1 and actor.info()["tickets"] == 0
    if mode == "required":
        assert obj.model.scan_calls == n
        states = [obj.local_memory_adapter.snapshot.fast[l["session_id"]].state for l in leases]
        assert all(state is not None for state in states)
        if n == 2:
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(*states, strict=True))
    else:
        assert obj.local_memory_adapter.memory is None
        with pytest.raises(RuntimeError, match="off"):
            _ = obj.local_memory_adapter.snapshot
    preserved = actor._responses[leases[-1]["session_id"]]
    assert actor.end(leases[0]).json()["acknowledged"]
    if n == 2:
        assert actor._responses[leases[-1]["session_id"]] is preserved
    fresh = register(actor, generation=2)
    assert fresh["session_id"] != leases[0]["session_id"]
    with pytest.raises(ValueError, match="stale or foreign"):
        submit(actor, leases[0], requests[0])
    result = wait_result(actor, fresh, submit(actor, fresh, request(fresh, 0)))
    ack(actor, fresh, result)


@pytest.mark.parametrize("mode", ["off", "required"])
def test_real_client_B2_lost_delivery_ACK_and_strict_mode_handshake(mode, make_dual):
    obj, actor = make_dual(mode)
    clients, leases = [], []
    for i in range(2):
        client = TransactionPolicyClient(
            "http://cpu",
            timeout_s=5,
            batch_profile=True,
            max_batch_size=2,
            local_memory_mode=mode,
            transport=DirectTransport(obj, ("/v2/inference/ack",)),
        )
        leases.append(client.register(registration(f"slot{i}"), "cpu-owner"))
        clients.append(client)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda pair: pair[0].predict(no_ttt_payload(pair[1]) if mode == "off" else payload(pair[1])),
                zip(clients, leases),
            )
        )
    assert len(obj.native_calls) == 1 and {r["native_batch"]["size"] for r in results} == {2}
    wrong = TransactionPolicyClient("http://cpu", local_memory_mode="required" if mode == "off" else "off")
    with pytest.raises(ValueError, match="mode/profile"):
        wrong.bind_assignment(actor._registration_reply(actor._leases["slot0"]).json())
    for client in clients:
        client.end()


@pytest.mark.parametrize("mode", ["off", "required"])
@pytest.mark.parametrize("fault", ["timeout", "end", "generation"])
def test_sealed_failure_never_publishes_response_and_End_waits_for_quiescence(mode, fault, make_dual):
    obj, actor = make_dual(mode)
    entered, release = threading.Event(), threading.Event()
    original = actor.generate_many
    now = [time.monotonic()]
    actor.clock = lambda: now[0]

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        if fault == "generation":
            raise RuntimeError("native generation failure")
        return original(*args)

    actor.generate_many = blocked
    leases = [register(actor, f"slot{i}") for i in range(2)]
    pending = [submit(actor, lease, no_ttt_payload(lease) if mode == "off" else payload(lease)) for lease in leases]
    try:
        assert entered.wait(5)
        assert actor._responses == {}
        if fault == "timeout":
            now[0] += 6
        if fault == "end":
            assert actor.end(leases[0]).json()["acknowledged"] is False
        assert actor.info()["busy"]
    finally:
        release.set()
    for lease, p in zip(leases, pending, strict=True):
        wait_result(actor, lease, p, expected=409)
    assert actor._responses == {} and actor.info()["tickets"] == 0
    if mode == "required":
        assert not obj.local_memory_adapter.snapshot.fast
    for lease in leases:
        assert actor.end(lease).json()["acknowledged"]


def test_no_ttt_native_32_step_loop_uses_no_Local_client_and_Ends_common_Session(tmp_path, monkeypatch, make_dual):
    from cosmos_framework.simulation.robocasa import eval_v2_adapter as adapter
    from cosmos_framework.simulation.robocasa.eval_v2_transactions import PolicyTransport
    from cosmos_framework.simulation.robocasa.eval_v2_transactions_test import FakeEnv, settings

    obj, actor = make_dual("off", 1)
    lease = register(actor)
    transport = PolicyTransport(
        dict(settings(), local_memory_mode="off"),
        actor._registration_reply(actor._leases["slot0"]).json(),
        time.monotonic() + 10,
        transport=DirectTransport(obj),
        media_transport=DirectTransport(obj),
    )
    assert transport.local is None
    monkeypatch.setattr(
        "cosmos_framework.simulation.robocasa.eval_v2_transactions.PolicyTransport", lambda *args: transport
    )
    monkeypatch.setattr(adapter, "RoboCasaLocalMemoryClient", forbidden)
    monkeypatch.setattr(adapter.native, "build_state_token", lambda _: [0.0] * 15)
    engine = adapter.NativeEpisode(FakeEnv(), dict(settings(), local_memory_mode="off"))
    assert engine.initialize()["initial_success"] is False
    attempt = dict(
        task="CloseFridge", episode=0, attempt_id="a" * 32, stream_id="b" * 32, seed=0, manifest_digest="a" * 64
    )
    result = engine.execute(
        attempt,
        reset_index=0,
        directory=tmp_path / "episode",
        transaction_assignment=actor._registration_reply(actor._leases["slot0"]).json(),
        episode_deadline=time.monotonic() + 10,
    )
    assert result["observation"]["outcome"] == "success" and result["replans"] == 2
    assert result["local_session"]["session_id"] == lease["session_id"]
    assert result["local_session"]["reset_ack"] and actor.info()["sessions"] == 0
    assert len(obj.native_calls) == 2 and transport._ended


@pytest.mark.parametrize("mode", ["off", "required"])
def test_real_small_DCP_inventory_and_config_must_match_selected_mode(tmp_path, mode):
    from torch.distributed.checkpoint import save

    model = nn.Module()
    model.net = nn.Module()
    model.net.config = SimpleNamespace(local_memory_enabled=mode == "required")
    model.net.weight = nn.Parameter(torch.zeros(1))
    if mode == "required":
        model.net.local_memory_runtime = nn.Linear(2, 3)
    checkpoint = tmp_path / "checkpoint"
    save(model.state_dict(), checkpoint_id=checkpoint / "model")
    cfg = {"model": {"config": {"local_memory_enabled": mode == "required"}}}
    validate_model_mode(model, cfg, checkpoint, mode)
    with pytest.raises(ValueError, match="config"):
        validate_model_mode(model, cfg, checkpoint, "off" if mode == "required" else "required")
    # A declared config and loaded capability cannot conceal opposite DCP bytes.
    other = nn.Module()
    other.net = nn.Module()
    other.net.config = SimpleNamespace(local_memory_enabled=mode != "required")
    if mode == "off":
        other.net.local_memory_runtime = nn.Linear(2, 3)
    with pytest.raises(ValueError, match="inventory"):
        validate_model_mode(
            other,
            {"model": {"config": {"local_memory_enabled": mode != "required"}}},
            checkpoint,
            "required" if mode == "off" else "off",
        )


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("fault", [None, "missing", "shape"])
def test_selected_DCP_tree_accepts_regular_and_EMA_without_weakening_inventory(tmp_path, ema, fault):
    from torch.distributed.checkpoint import save

    model = nn.Module()
    model.net = nn.Module()
    model.net.config = SimpleNamespace(local_memory_enabled=True)
    model.net.local_memory_runtime = nn.Linear(2, 3)
    model.net_ema = nn.Module()
    model.net_ema.local_memory_runtime = nn.Linear(2, 3)
    state = model.state_dict()
    key = ("net_ema." if ema else "net.") + "local_memory_runtime.weight"
    if fault == "missing":
        del state[key]
    elif fault == "shape":
        state[key] = torch.zeros(1)
    save(state, checkpoint_id=tmp_path / "model")
    config = {"model": {"config": {"local_memory_enabled": True}}}
    if fault:
        with pytest.raises(ValueError, match="inventory"):
            validate_model_mode(model, config, tmp_path, "required", use_ema_weights=ema)
    else:
        validate_model_mode(model, config, tmp_path, "required", use_ema_weights=ema)


@pytest.mark.parametrize("mode", ["off", "required"])
def test_response_allocation_failure_before_commit_leaves_both_sessions_unpublished(mode, make_dual, monkeypatch):
    obj, actor = make_dual(mode)
    leases = [register(actor, f"slot{i}") for i in range(2)]

    from cosmos_framework.inference import robocasa_session_transactions as module

    original, allocated = module._CommittedResponse, []

    def allocation_failure(*args, **kwargs):
        allocated.append(True)
        if len(allocated) == 2:
            raise MemoryError("injected transaction replacement allocation")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_CommittedResponse", allocation_failure)
    pending = [submit(actor, lease, no_ttt_payload(lease) if mode == "off" else payload(lease)) for lease in leases]
    for lease, p in zip(leases, pending, strict=True):
        wait_result(actor, lease, p, expected=409)
    assert actor._responses == {} and len(obj.native_calls) == 1
    if mode == "required":
        assert obj.local_memory_adapter.snapshot.fast == obj.local_memory_adapter.snapshot.responses == {}


@pytest.mark.parametrize("mode", ["off", "required"])
@pytest.mark.parametrize("mismatch", [False, True])
def test_public_server_constructor_wires_actual_mode_and_rejects_checkpoint_mismatch(
    tmp_path, monkeypatch, mode, mismatch
):
    from torch.distributed.checkpoint import save

    from cosmos_framework.inference.args import OmniSetupArgs
    from cosmos_framework.inference.common.args import ConfigFileType
    from cosmos_framework.inference.robocasa_session_transactions_test import RealLocalModel

    model = RealLocalModel() if mode == "required" else nn.Module()
    if mode == "off":
        model.net = nn.Module()
        model.net.weight = nn.Parameter(torch.zeros(1))
    model.net.config = SimpleNamespace(local_memory_enabled=mode == "required")
    model.config = SimpleNamespace(max_action_dim=64)
    checkpoint = tmp_path / "checkpoint"
    state = model.state_dict()
    if mismatch:
        if mode == "off":
            state["net.local_memory_runtime.weight"] = torch.zeros(1)
        else:
            state = {key: value for key, value in state.items() if "local_memory_runtime" not in key}
    save(state, checkpoint_id=checkpoint / "model")
    config = {
        "model": {"config": {"local_memory_enabled": mode == "required"}},
        "dataloader_train": {"dataset": {"use_state": True}},
    }
    setup = OmniSetupArgs.model_construct(
        checkpoint_path=str(checkpoint),
        use_ema_weights=True,
        experiment="CPU fixture",
        output_dir=tmp_path / "loader-output",
        config_file="CPU fixture config",
        config_file_type=ConfigFileType.YAML,
    )
    monkeypatch.setattr(
        server.ActionServerArgs, "build_setup_overrides", lambda _: SimpleNamespace(build_setup=lambda: setup)
    )
    monkeypatch.setattr(server, "init_output_dir", lambda _: None)
    monkeypatch.setattr(server, "disable_runtime_ema_for_frozen_config", lambda value: value)
    monkeypatch.setattr(server, "maybe_init_distributed", lambda: None)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(server.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(server.OmniInference, "create", lambda _: SimpleNamespace(model=model, setup_args=setup))
    monkeypatch.setattr(server, "_load_introspection_config_dict", lambda _: config)
    token = tmp_path / "capability"
    token.write_text("cpu-owner")
    token.chmod(0o600)
    args = server.ActionServerArgs.model_construct(
        local_memory_mode=mode,
        raw_action_dim=15,
        action_chunk_size=16,
        transaction_profile="batch",
        transaction_slot_ids="slot0,slot1",
        local_memory_max_sessions=2,
        transaction_max_batch_size=2,
        transaction_owner_token_file=token,
        transaction_run_lease="run",
        transaction_manifest_digest="a" * 64,
    )
    if mismatch:
        with pytest.raises(ValueError, match="inventory"):
            server.ActionModelService(args)
    else:
        obj = server.ActionModelService(args)
        try:
            assert obj.transactions.info()["local_memory_mode"] == mode
            assert (obj.local_memory_adapter.memory is not None) == (mode == "required")
            assert obj.transactions.max_batch_size == 2
        finally:
            assert obj.transactions.close()


def test_required_Local_commit_validation_failure_cannot_publish_common_responses(make_dual, monkeypatch):
    obj, actor = make_dual("required")
    before = obj.local_memory_adapter.snapshot
    original = obj.local_memory_adapter.memory.validate_commit
    validated = []

    def fail_second(update):
        original(update)
        validated.append(update)
        if len(validated) == 2:
            raise RuntimeError("injected second Local validation failure")

    monkeypatch.setattr(obj.local_memory_adapter.memory, "validate_commit", fail_second)
    leases = [register(actor, f"slot{i}") for i in range(2)]
    pending = [submit(actor, lease, payload(lease)) for lease in leases]
    for lease, p in zip(leases, pending, strict=True):
        wait_result(actor, lease, p, expected=409)
    assert actor._responses == {} and obj.local_memory_adapter.snapshot is before
    assert not obj.local_memory_adapter._visual_pending and not obj.local_memory_adapter.memory._pending


@pytest.mark.parametrize("damage", ["foreign_session", "foreign_episode", "gap", "boolean", "Local_payload"])
def test_NoTTT_public_admission_rejects_foreign_or_fake_Local_frontier(damage, make_dual):
    obj, actor = make_dual("off", 1)
    lease = register(actor)
    request = no_ttt_payload(lease)
    if damage == "foreign_session":
        request["inference"]["session_id"] = "foreign"
    elif damage == "foreign_episode":
        request["inference"]["episode_id"] = "foreign"
    elif damage == "gap":
        request["inference"]["request_index"] = 4
    elif damage == "boolean":
        request["inference"]["request_index"] = True
    else:
        request["local_memory"] = payload(lease)["local_memory"]
    with pytest.raises(ValueError):
        submit(actor, lease, request)
    assert actor._responses == {} and actor.info()["tickets"] == 0 and obj.native_calls == []

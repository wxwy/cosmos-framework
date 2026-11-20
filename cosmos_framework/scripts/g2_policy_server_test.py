"""Real service/Session/Local CPU integration; only spatial/model kernels doubled."""

import json
import sys
import threading
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.utils import static_task_text
from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskText, StaticTaskTextRecipe
from cosmos_framework.inference.g2_model_hooks_test import actual_tiny_model
from cosmos_framework.inference.g2_model_hooks_test import hooks as actual_hooks
from cosmos_framework.inference.g2_native_policy import NativeG2R1Authority
from cosmos_framework.inference.g2_policy_conditions import G2ConditionRuntime
from cosmos_framework.inference.g2_policy_conditions_test import request as condition_request
from cosmos_framework.inference.g2_policy_conditions_test import runtime
from cosmos_framework.inference.robocasa_batch_transactions import BatchedSessionTransactions
from cosmos_framework.inference.robocasa_session_transactions import SessionTransactions
from cosmos_framework.inference.robocasa_session_transactions_test import (
    ack,
    clock_kernel,  # noqa: F401 - shared explicit kernel fixture, no real host/GPU claim
    make_adapter,
    payload,
    register,
    submit,
    wait_result,
)
from cosmos_framework.scripts.robocasa_transactions_b1_test import service


def g2_payload(lease, frame=0, start=0, *, subtask="subtask", scene="scene", value=1):
    result = payload(lease, frame, start, value)
    extra = condition_request(
        session=lease["session_id"],
        episode=lease["episode_id"],
        scene=scene,
        subtask=subtask,
        frame=frame,
    )
    result.update(prompt=extra["prompt"], g2=extra["g2"])
    return result


def setup(monkeypatch, *, batch=False, cache=True, profile="g2_r2", slots=2):
    obj = service(monkeypatch)
    obj.local_memory_adapter, obj.model = make_adapter(slots)
    obj.model.eval()
    obj.model.tokenize_text = lambda text, **_kwargs: [ord(c) for c in text]
    obj.model.llm_special_tokens = {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3}
    rt, hooks, provider = runtime(profile=profile, cache=cache)
    obj.g2_conditions = rt
    obj.calls = []
    obj.before_sample = lambda *_: None

    def generate(data, **kwargs):
        obj.before_sample(data, kwargs)
        obj.calls.append((data, kwargs))
        output = []
        for i, item in enumerate(kwargs["_g2_fixture"]):
            prefix = kwargs["_local_memory_prefixes"][i]
            local = 0.0 if prefix is None else float(prefix.square().sum()) * 0.001
            global_value = 0.0 if item.global_tokens is None else float(item.global_tokens.mean()) * 1e-5
            state = data["action"][i][0][0, 0]
            output.append(torch.full((1, 17, 64), local + global_value + state))
        return {"action": output, "vision": [torch.empty(0) for _ in output]}

    obj.model.generate_samples_from_batch = generate
    kwargs = dict(
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=tuple(f"slot{i}" for i in range(slots)),
        model_lock=obj._lock,
        timeout_s=10,
        conditions=rt,
    )
    if batch:
        actor = BatchedSessionTransactions(
            obj.local_memory_adapter,
            obj._transaction_batch,
            lambda: None,
            max_batch_size=slots,
            batch_wait_ms=80,
            **kwargs,
        )
    else:
        actor = SessionTransactions(obj.local_memory_adapter, obj._transaction_action, lambda: None, **kwargs)
    obj.transactions = actor
    return obj, actor, rt, hooks, provider


@pytest.mark.parametrize("profile", ["g2_r1", "g2_r2"])
def test_real_service_calls_C_shared_prompt_and_keeps_state_fast(monkeypatch, profile):
    obj, actor, rt, _, _ = setup(monkeypatch, profile=profile)
    called = []
    serialize = static_task_text.serialize_static_task_prompt

    def observe(task, recipe):
        called.append((task, recipe))
        return serialize(task, recipe)

    monkeypatch.setattr(static_task_text, "serialize_static_task_prompt", observe)
    lease = register(actor)
    try:
        req = g2_payload(lease)
        first = obj._prep_policy_item(req)
        req["state"][0] = 0.8
        req["local_memory"]["consumer_step"] = req["g2"]["identity"]["decision_frame"] = 16
        req["g2"]["identity"]["logical_state_frame"] = 16
        second = obj._prep_policy_item(req)
        assert len(called) == 2
        assert first["augmented_prompt"] == second["augmented_prompt"]
        assert "idle" not in json.dumps(json.loads(first["augmented_prompt"]))
        assert first["state_token"][0] == 0 and second["state_token"][0] == pytest.approx(0.8)
        task = StaticTaskText(**req["g2"]["static_text"])
        h, w = map(int, second["padded_image_size"][:2].tolist())
        assert (h, w) == second["video_padded"].shape[-2:]
        recipe = StaticTaskTextRecipe(17, 20, h, w, rt.hooks.tokenizer_identity)
        assert first["augmented_prompt"] == serialize(task, recipe)
        assert first["g2_text_recipe_id"] == recipe.identity
        assert asdict(called[0][0]) == req["g2"]["static_text"]
        effective = first["g2_effective_text"]
        expected_ids = (1, *obj.model.tokenize_text(first["augmented_prompt"]), 2, 3)
        assert effective.token_ids == (expected_ids,)
        assert effective.position_ids == ((tuple(range(len(expected_ids))),) * 3,)
        assert effective.dropped == (False,)
        assert rt.hooks.calls["tokenize"] == 0
    finally:
        assert actor.close()


def test_b1_retry_inflight_and_post_ack_does_not_duplicate_Local_or_extend_deadline(monkeypatch):
    obj, actor, rt, hooks, _ = setup(monkeypatch)
    lease = register(actor)
    try:
        cold = wait_result(actor, lease, submit(actor, lease, g2_payload(lease)))
        ack(actor, lease, cold)
        entered, release = threading.Event(), threading.Event()

        def hold(*_):
            entered.set()
            assert release.wait(3)

        obj.before_sample = hold
        req = g2_payload(lease, 2, 0)
        pending = submit(actor, lease, req)
        assert entered.wait(3)
        key = pending.json()["request_key"]
        original = actor._tickets[key].deadline
        duplicate = submit(actor, lease, deepcopy(req))
        assert duplicate.json()["request_key"] == key
        assert actor._tickets[key].deadline == original
        assert obj.model.scan_calls == 1
        release.set()
        result = wait_result(actor, lease, pending)
        ack(actor, lease, result)
        replay = submit(actor, lease, req)
        assert replay.status == 200 and replay.body == result.body
        assert obj.model.scan_calls == 1 and len(obj.calls) == 2
        assert hooks.calls["global"] == 1
        assert rt.info()["host_ns"]["queue"] and len(rt.info()["host_ns"]["backbone_sampling"]) == 2
        with pytest.raises(ValueError, match="stale"):
            submit(actor, lease, g2_payload(lease, 1, 0))
    finally:
        if "release" in locals():
            release.set()
        assert actor.close()
    assert rt.info()["cache"]["entries"] == rt.info()["task_bindings"] == 0


@pytest.mark.parametrize("fault", ["exception", "expired", "end-during-sampling"])
def test_abort_expired_and_End_preserve_Local_and_release_conditions(monkeypatch, fault):
    obj, actor, rt, _, _ = setup(monkeypatch)
    lease = register(actor)
    try:
        cold = wait_result(actor, lease, submit(actor, lease, g2_payload(lease)))
        ack(actor, lease, cold)
        before = obj.local_memory_adapter.snapshot

        def break_sampling(*_):
            if fault == "exception":
                raise RuntimeError("synthetic model exception")
            if fault == "expired":
                ticket = next(iter(actor._tickets.values()))
                actor.clock = lambda: ticket.deadline + 1
            else:
                assert not actor.end(lease).json()["acknowledged"]

        obj.before_sample = break_sampling
        pending = submit(actor, lease, g2_payload(lease, 2, 0))
        result = wait_result(actor, lease, pending, expected=409)
        assert result.status == 409
        assert obj.local_memory_adapter.snapshot is before
        assert obj.model.scan_calls == 1
        assert rt.info()["cache"]["active_requests"] == 0
        assert actor.end(lease).json()["acknowledged"]
        assert rt.info()["task_bindings"] == 0
    finally:
        assert actor.close()


def test_subtask_switch_rebuilds_Global_without_resetting_Local_and_End_clears_both(monkeypatch):
    obj, actor, rt, hooks, _ = setup(monkeypatch)
    lease = register(actor)
    try:
        first = wait_result(actor, lease, submit(actor, lease, g2_payload(lease)))
        ack(actor, lease, first)
        second = wait_result(actor, lease, submit(actor, lease, g2_payload(lease, 2, 0)))
        ack(actor, lease, second)
        before = obj.local_memory_adapter.snapshot.fast[lease["session_id"]]
        next_task = wait_result(actor, lease, submit(actor, lease, g2_payload(lease, 4, 2, subtask="next")))
        after = obj.local_memory_adapter.snapshot.fast[lease["session_id"]]
        assert before.state is not None and after.consumer_step == 4 and obj.model.scan_calls == 2
        assert obj.calls[-1][1]["_local_memory_prefixes"][0] is not None
        assert hooks.calls["global"] == 2 and hooks.calls["task_queries"] == 2
        ack(actor, lease, next_task)
        assert actor.end(lease).json()["acknowledged"]
        assert lease["session_id"] not in obj.local_memory_adapter.snapshot.fast
        assert rt.info()["cache"]["entries"] == 0
        fresh = register(actor, generation=2)
        result = wait_result(actor, fresh, submit(actor, fresh, g2_payload(fresh)))
        assert result.json()["local_memory"]["prefix_present"] is False
        assert hooks.calls["global"] == 3
    finally:
        assert actor.close()


def test_native_mixed_Session_batch_routes_conditions_and_Local_by_original_index(monkeypatch):
    obj, actor, rt, hooks, provider = setup(monkeypatch, batch=True)
    provider.scene_offsets = {"different-scene": 3.0}
    leases = [register(actor, f"slot{i}") for i in range(2)]
    try:
        cold = wait_result(actor, leases[0], submit(actor, leases[0], g2_payload(leases[0])))
        ack(actor, leases[0], cold)
        reqs = [g2_payload(leases[0], 2, 0), g2_payload(leases[1], scene="different-scene")]
        reqs[0]["state"][0] = reqs[1]["state"][0] = 0.125
        pending = [submit(actor, lease, req) for lease, req in zip(leases, reqs, strict=True)]
        results = [wait_result(actor, lease, p) for lease, p in zip(leases, pending, strict=True)]
        assert [r.json()["assignment"]["session_id"] for r in results] == [l["session_id"] for l in leases]
        assert [r.json()["native_batch"]["index"] for r in results] == [0, 1]
        batch, kwargs = obj.calls[-1]
        assert len(batch["prompt"]) == len(kwargs["_g2_fixture"]) == 2
        assert kwargs["_local_memory_prefixes"][0] is not None
        assert kwargs["_local_memory_prefixes"][1] is None
        assert hooks.calls["global"] == 2  # Separate fill even though both instructions match.
        assert results[0].json()["action"] != results[1].json()["action"]
        assert [s.identity.scene_id for s in rt.last_request_sources] == ["scene", "different-scene"]
    finally:
        assert actor.close()


def _native_r1_cpu_service(monkeypatch, tmp_path, slots, batch_size):
    obj = service(monkeypatch)
    obj.local_memory_adapter, _ = make_adapter(slots)
    obj.model, _ = actual_tiny_model(monkeypatch, global_enabled=False)
    obj.model.vlm_processor = SimpleNamespace(tokenize_text=obj.model.tokenize_text)
    hooks = actual_hooks(obj.model)
    obj.g2_native_authority = NativeG2R1Authority(
        model_path=tmp_path / "model",
        config_path=tmp_path / "config.yaml",
        text_recipe=StaticTaskTextRecipe(17, 20, 192, 320, hooks.tokenizer_identity),
        state15_recipe_id="state15-v1",
        alignment_version="online-align-v1",
        data_identity="online-v1",
        initial_language_recipe_id="template-v1",
        top_variant="with_robot",
        versions={},
        files={},
    )
    rt = obj.g2_conditions = G2ConditionRuntime(hooks, profile="g2_r1")
    observed = {}

    def generate(batch, *, _g2_condition, _local_memory_prefixes, seed, **kwargs):
        # Native text preparation, packed UND/GEN KV, and Local fast weights
        # are real tiny CPU computations. Only the diffusion sampler is doubled.
        condition = _g2_condition
        count = len(batch["action"])
        assert seed == [obj.cfg.seed] * count
        assert condition.views == condition.global_tokens == (None,) * count
        assert condition.global_kv is None
        assert len(condition.text.lengths) == len(_local_memory_prefixes) == count
        features = condition.text_features
        offset = 0
        actions = []
        for i, (length, prefix) in enumerate(zip(condition.text.lengths, _local_memory_prefixes, strict=True)):
            state = batch["action"][i][0][0, :15]
            state_id = float(state[0])
            assert state_id not in observed
            assert int(features.valid_mask[i].sum()) == length
            layers = (*features.layers, *condition.gen_text_kv)
            observed[state_id] = (
                batch["prompt"][i],
                condition.text.token_ids[i],
                condition.text.position_ids[i],
                state.clone(),
                features.hidden[i, :length].clone(),
                tuple(t[offset : offset + length].clone() for layer in layers for t in (layer.key, layer.value)),
                None if prefix is None else prefix.clone(),
            )
            offset += length
            token_value = sum(condition.text.token_ids[i]) / 100_000
            local_value = 0 if prefix is None else float(prefix.square().sum()) / 10_000
            action = torch.zeros(1, 17, 64)
            action[:, 1:, :15] = state + token_value + local_value
            actions.append(action)
        return {"action": actions}

    obj.model.generate_samples_from_batch = generate
    actor = obj.transactions = BatchedSessionTransactions(
        obj.local_memory_adapter,
        obj._transaction_batch,
        lambda: None,
        owner_token="cpu-owner",
        run_lease="run",
        manifest_digest="a" * 64,
        slot_ids=tuple(f"slot{i}" for i in range(slots)),
        capacity_profile="streaming32",
        max_batch_size=batch_size,
        batch_wait_ms=100,
        timeout_s=30,
        model_lock=obj._lock,
        conditions=rt,
    )
    return obj, actor, rt, observed


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_native_r1_batch_matches_each_session_serially_with_dense_local_history(monkeypatch, tmp_path, batch_size):
    slots = 16 if batch_size == 8 else batch_size
    runs = [_native_r1_cpu_service(monkeypatch, tmp_path, slots, size) for size in (batch_size, 1)]

    def evaluate(run, limit):
        obj, actor, rt, observed = run
        leases = [
            register(actor, f"slot{i}", episode_index=i, task="OpenDrawer" if i % 2 else "CloseFridge")
            for i in range(slots)
        ]
        results, snapshots = {}, {}
        for frame in (0, 16):
            # Warm requests deliberately permute batch members and peers.
            order = list(range(slots)) if frame == 0 else list(reversed(range(slots)))
            for start in range(0, slots, limit):
                group = order[start : start + limit]
                pending = []
                for i in group:
                    req = g2_payload(leases[i], frame, scene=f"scene-{i}", subtask=f"subtask-{i}", value=i + 1)
                    req["state"] = [i / 32 + frame / 32 + j / 1024 for j in range(15)]
                    req["prompt"] = "Act on the " + "wooden " * i + f"fixture {i}."
                    req["g2"]["static_text"].update(enhanced_instruction=req["prompt"], source_identity=f"sg-{i}")
                    req["g2"]["identity"].update(task_binding_id=f"binding-{i}", logical_state_frame=0)
                    pending.append(submit(actor, leases[i], req))
                rows = [wait_result(actor, leases[i], p) for i, p in zip(group, pending, strict=True)]
                native = [r.json()["native_batch"] for r in rows]
                assert len({v["id"] for v in native}) == 1
                assert [v["size"] for v in native] == [len(group)] * len(group)
                assert [v["index"] for v in native] == list(range(len(group)))
                if len(group) > 1:
                    with pytest.raises(ValueError):
                        ack(actor, leases[group[1]], rows[0])
                for i, r in zip(group, rows, strict=True):
                    value = r.json()
                    assert value["assignment"]["session_id"] == leases[i]["session_id"]
                    local = value["local_memory"]
                    assert local["prefix_present"] == bool(frame)
                    assert local["encoded_steps"] == frame
                    if frame:
                        assert local["adapted_steps"] == frame
                    action = torch.tensor(value["action"])
                    assert action.shape == (16, 15) and torch.isfinite(action).all()
                    results[frame, i] = action
                    state = obj.local_memory_adapter.snapshot.fast[leases[i]["session_id"]]
                    assert state.consumer_step == frame
                    snapshots[frame, i] = state.state
                    assert ack(actor, leases[i], r).json()["acknowledged"]
        assert actor.info()["batch_histogram"] == {limit: 2 * slots // limit}
        assert rt.info()["task_bindings"] == slots
        for i, lease in enumerate(leases):
            assert rt._tasks[lease["session_id"]][:4] == (
                lease["episode_id"],
                f"scene-{i}",
                f"subtask-{i}",
                f"binding-{i}",
            )
        assert actor.end(leases[0]).json()["acknowledged"]
        assert rt.info()["task_bindings"] == slots - 1
        assert set(obj.local_memory_adapter.snapshot.fast) == {l["session_id"] for l in leases[1:]}
        return results, snapshots, observed

    try:
        packed, serial = (evaluate(run, limit) for run, limit in zip(runs, (batch_size, 1), strict=True))
        for key, action in serial[0].items():
            torch.testing.assert_close(packed[0][key], action, rtol=0, atol=0)
            reference = serial[1][key]
            if reference is None:
                assert packed[1][key] is None
            else:
                torch.testing.assert_close(packed[1][key], reference, rtol=0, atol=0)
        for key, reference in serial[2].items():
            actual = packed[2][key]
            assert actual[:3] == reference[:3]
            torch.testing.assert_close(actual[3:-1], reference[3:-1], rtol=0, atol=0)
            if reference[-1] is None:
                assert actual[-1] is None
            else:
                torch.testing.assert_close(actual[-1], reference[-1], rtol=0, atol=0)
    finally:
        for _, actor, _, _ in runs:
            assert actor.close()


def test_service_cache_on_off_exact_actions_and_real_Local_state_parity(monkeypatch):
    on = setup(monkeypatch, cache=True, slots=1)
    off = setup(monkeypatch, cache=False, slots=1)
    try:
        leases = [register(item[1]) for item in (on, off)]
        for frame, start in ((0, 0), (2, 0), (4, 2)):
            results = [
                wait_result(item[1], lease, submit(item[1], lease, g2_payload(lease, frame, start)))
                for item, lease in zip((on, off), leases, strict=True)
            ]
            assert results[0].json()["action"] == results[1].json()["action"]
            a = on[0].local_memory_adapter.snapshot.fast[leases[0]["session_id"]]
            b = off[0].local_memory_adapter.snapshot.fast[leases[1]["session_id"]]
            if a.state is not None:
                assert all(torch.equal(x, y) for x, y in zip(a.state, b.state, strict=True))
            for item, lease, result in zip((on, off), leases, results, strict=True):
                ack(item[1], lease, result)
        assert on[3].calls["global"] == 1 and off[3].calls["global"] == 3
    finally:
        assert on[1].close() and off[1].close()


@pytest.mark.parametrize(
    "fault,message",
    [
        ("transactions", "owned Session"),
        ("guidance", "guidance=1"),
        ("hooks", "shared model hooks"),
        ("provider", "current-rollout provider"),
    ],
)
def test_explicit_profile_rejects_missing_interfaces_before_model_or_GPU_load(monkeypatch, fault, message):
    obj = service(monkeypatch)
    cls = type(obj)
    options = None if fault == "transactions" else {}
    monkeypatch.setitem(cls.__init__.__globals__, "_transaction_options", lambda _: options)
    args = SimpleNamespace(
        g2_profile="g2_r2",
        guidance=2.0 if fault == "guidance" else 1.0,
        g2_hooks_factory=None if fault == "hooks" else "B.fixed:model_hooks",
        g2_provider_factory=None if fault == "provider" else "online.fixed:spatial",
    )
    with pytest.raises(ValueError, match=message):
        cls(args)


def test_R1_factory_does_not_import_or_construct_spatial_provider(monkeypatch):
    obj = service(monkeypatch)
    _, hooks, _ = runtime(profile="g2_r1")
    module = SimpleNamespace(model_hooks=lambda _service: hooks)
    monkeypatch.setitem(sys.modules, "g2_test_factory", module)
    args = SimpleNamespace(
        g2_hooks_factory="g2_test_factory:model_hooks",
        g2_provider_factory="never_import_this:provider",
        g2_profile="g2_r1",
        g2_cache_enabled=True,
        g2_cache_max_sessions=1,
        g2_cache_max_entries=16,
        g2_cache_cpu_bytes=1024**2,
        g2_cache_gpu_bytes=1024**2,
        g2_cache_handles=64,
    )
    obj._configure_g2_conditions(args)
    assert obj.g2_conditions.profile == "g2_r1" and obj.g2_conditions.provider is None

"""CPU source/GA/frontier contracts. Synthetic native loss is not a GPU model Gate."""

from __future__ import annotations

import pickle
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.local_memory_grouped_window import GroupedLocalMemoryWindow
from cosmos_framework.model.generator.mot.local_memory_grouped_window_test import _Net
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowRankPlanner
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _setup
from cosmos_framework.trainer import matched_no_ttt as module
from cosmos_framework.trainer.local_memory_grouped_test import _Hooks


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(local_memory_enabled=False)
        self.net = nn.Module()
        self.net.moe_gen = nn.Parameter(torch.tensor(2.0))
        self.seen = []
        self.before_zero = None

    def training_step(self, batch, iteration):
        # Deliberately no **kwargs: even all-None prefix injection would fail.
        self.seen.append(tuple((p["episode_index"], p["start_frame"]) for p in batch["payloads"]))
        value = torch.stack([p["action_raw"][1].mean() for p in batch["payloads"]]).mean()
        loss = self.net.moe_gen * value
        return {"flow_matching_loss_action": loss, "aux_loss_gen": loss * 0.1}, loss

    def on_after_backward(self):
        pass

    def on_before_optimizer_step(self, *args, **kwargs):
        pass

    def on_before_zero_grad(self, *args, **kwargs):
        self.before_zero = self.net.moe_gen.grad.detach().clone()


def trainer(count=33, *, t=16, b=8, ga=2):
    _, _, catalog, producer = _setup((count,) * max(16, b * ga), t=t)
    instance = object.__new__(module.MatchedNoTTTTrainer)
    instance.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=ga, distributed_parallelism="fsdp"))
    instance.callbacks = _Hooks()
    instance.training_timer = lambda name: nullcontext()
    instance.bind_grouped_stream(
        ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=b, active_ga=ga),
        producer,
        config_digest="config",
    )
    return instance


@pytest.fixture(autouse=True)
def cpu_batch(monkeypatch):
    monkeypatch.setattr(module, "collate_grouped_native_batch", lambda payloads: {"payloads": payloads})
    monkeypatch.setattr(module.misc, "to", lambda value, **kwargs: value)


def optim(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)


def update(instance, model, state, iteration=0):
    outputs = []
    for member in range(instance._grouped_planner.active_ga):
        outputs.append(instance.training_step(model, *state, {}, iteration, member))
    return outputs


@pytest.mark.parametrize("count,expected", [(33, (128, 128)), (17, (128, 8)), (1, (8, 8))])
def test_full_and_terminal_partial_match_g1_order_weights_and_boundary(count, expected):
    instance, model = trainer(count), Model()
    planner, producer = instance._grouped_planner, instance._grouped_producer
    reference_model = nn.Module()
    reference_model.net = _Net(16)
    reference = GroupedLocalMemoryWindow(reference_model, planner)
    plan = reference.begin()
    assert plan.member_counts == expected
    source_order, weighted = [], []
    for member in range(2):
        segments = tuple(producer.produce(request) for request in plan.members[member])

        def native(payloads, prefixes, positions):
            source_order.append(tuple((p["episode_index"], p["start_frame"]) for p in payloads))
            return reference_model.net.moe_gen * torch.tensor([p["action_raw"][1].mean() for p in payloads]).mean()

        weighted.append(reference.run_member(segments, native, lambda loss, retain: loss.backward()))
    reference.finish(lambda: True)
    state = optim(model)
    before = planner.initial_frontier()
    out, loss, accum = instance.training_step(model, *state, {}, 0, 0)
    assert accum == 1 and instance._grouped_window.live.frontier == before
    with pytest.raises(RuntimeError, match="committed"):
        instance.callbacks._callbacks[0].state_dict()
    _, last_loss, accum = instance.training_step(model, *state, {}, 0, 1)
    assert accum == 0 and model.seen == source_order
    torch.testing.assert_close(loss + last_loss, sum(weighted))
    torch.testing.assert_close(model.before_zero, reference_model.net.moe_gen.grad)
    assert instance._grouped_window.live.frontier == reference.live.frontier
    assert state[1].last_epoch == 1 and model.net.moe_gen.grad is None
    assert instance._grouped_completed_iteration == 1
    assert not hasattr(instance._grouped_window.live, "sidecar")
    assert not hasattr(instance._grouped_window.live, "scheduler")
    assert not any("local_memory" in name for name, _ in model.named_parameters())


def test_snapshot_roundtrip_next_source_loss_optimizer_and_scheduler():
    source, model = trainer(80), Model()
    state = optim(model)
    update(source, model, state)
    saved = pickle.loads(pickle.dumps(source.callbacks._callbacks[0].state_dict()))
    assert set(saved) == {
        "format",
        "iteration",
        "frontier",
        "config_digest",
        "profile",
        "cache_manifest_sha256",
        "cache_corpus_digest",
        "source_binding_digest",
    }
    resumed, other = trainer(80), Model()
    other.load_state_dict(model.state_dict())
    other_state = optim(other)
    other_state[0].load_state_dict(deepcopy(state[0].state_dict()))
    other_state[1].load_state_dict(deepcopy(state[1].state_dict()))
    resumed._resume_required = True
    resumed.callbacks._callbacks[0].load_state_dict(saved)
    update(source, model, state, iteration=1)
    update(resumed, other, other_state, iteration=1)
    assert other.seen == model.seen[-2:]
    assert source._grouped_window.live == resumed._grouped_window.live
    torch.testing.assert_close(model.net.moe_gen, other.net.moe_gen)
    assert state[1].state_dict() == other_state[1].state_dict()


@pytest.mark.parametrize(
    "key",
    [
        "format",
        "iteration",
        "config_digest",
        "profile",
        "cache_manifest_sha256",
        "cache_corpus_digest",
        "source_binding_digest",
        "sidecar",
        "frontier_counter",
    ],
)
def test_corrupt_or_local_checkpoint_rejected_before_native_forward(key):
    source, model = trainer(80), Model()
    update(source, model, optim(model))
    saved = deepcopy(source.callbacks._callbacks[0].state_dict())
    if key == "iteration":
        saved[key] = True
    elif key == "profile":
        saved[key] = (1, *saved[key][1:])
    elif key == "frontier_counter":
        frontier = saved["frontier"]
        saved["frontier"] = replace(
            frontier, slots=(replace(frontier.slots[0], next_segment_id=99), *frontier.slots[1:])
        )
    else:
        saved[key] = "wrong"
    target, other = trainer(80), Model()
    target._resume_required = True
    target.callbacks._callbacks[0].load_state_dict(saved)
    with pytest.raises(ValueError, match="No-TTT"):
        target.training_step(other, *optim(other), {}, 1, 0)
    assert other.seen == [] and target._no_ttt_failed
    assert target._grouped_window.live.frontier == target._grouped_planner.initial_frontier()


@pytest.mark.parametrize("required,iteration", [(True, 0), (True, 1), (False, 1)])
def test_missing_frontier_never_silently_restarts(required, iteration):
    target, model = trainer(), Model()
    target._resume_required = required
    with pytest.raises(RuntimeError, match="requires its rank frontier"):
        target.training_step(model, *optim(model), {}, iteration, 0)
    assert not model.seen


@pytest.mark.parametrize("failure", ["nan_loss", "nan_gradient", "hook_inf", "step_skip", "native_error"])
def test_failure_does_not_publish_frontier_or_allow_checkpoint(monkeypatch, failure):
    instance, model = trainer(), Model()
    state = optim(model)
    if failure == "nan_loss":
        monkeypatch.setattr(model, "training_step", lambda *args: ({}, model.net.moe_gen * float("nan")))
    elif failure == "nan_gradient":
        model.net.moe_gen.register_hook(lambda grad: grad * float("nan"))
    elif failure == "hook_inf":
        monkeypatch.setattr(
            model, "on_before_optimizer_step", lambda *a, **k: model.net.moe_gen.grad.fill_(float("inf"))
        )
    elif failure == "step_skip":
        monkeypatch.setattr(instance, "_optimizer_step_success", lambda *a: False)
    else:

        def explode(*args):
            raise OSError("synthetic native failure")

        monkeypatch.setattr(model, "training_step", explode)
    with pytest.raises((ValueError, FloatingPointError, RuntimeError, OSError)):
        update(instance, model, state)
    assert instance._grouped_window.live.frontier == instance._grouped_planner.initial_frontier()
    assert instance._grouped_completed_iteration == 0 and state[1].last_epoch == 0
    with pytest.raises(RuntimeError, match="committed"):
        instance.callbacks._callbacks[0].state_dict()
    with pytest.raises(RuntimeError, match="failed instance"):
        update(instance, model, state)


@pytest.mark.parametrize("failure", ["enabled", "runtime", "parameter"])
def test_local_module_forbidden(failure):
    instance, model = trainer(), Model()
    if failure == "enabled":
        model.config.local_memory_enabled = True
    elif failure == "runtime":
        model.net.local_memory_runtime = nn.Identity()
    else:
        model.net.local_memory_modality_embed = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="Local|local_memory_enabled"):
        update(instance, model, optim(model))
    assert model.seen == []


def test_no_ttt_is_independent_native_trainer_and_refuses_background_data():
    from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer

    assert not issubclass(module.MatchedNoTTTTrainer, GroupedLocalMemoryTrainer)
    target = trainer()
    with pytest.raises(ValueError, match="synchronous"):
        object.__new__(module.MatchedNoTTTTrainer).bind_grouped_stream(
            target._grouped_planner, target._grouped_producer, config_digest="config", num_workers=1
        )


def test_native_network_constructor_does_not_instantiate_local(monkeypatch):
    from transformers import PretrainedConfig

    from cosmos_framework.model.generator.mot import cosmos3_vfm_network as network

    language = PretrainedConfig(
        hidden_size=8, num_attention_heads=2, num_key_value_heads=2, head_dim=4, num_hidden_layers=1
    )
    config = network.Cosmos3VFMNetworkConfig(
        vlm_config=language,
        local_memory_enabled=False,
        vision_gen=True,
        action_gen=True,
        action_dim=64,
        num_embodiment_domains=32,
        joint_attn_implementation="two_way",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("No-TTT native constructor attempted to create LocalMemoryRuntime")

    monkeypatch.setattr(network, "LocalMemoryRuntime", forbidden)
    net = network.Cosmos3VFMNetwork(nn.Identity(), config)
    model = nn.Module()
    model.config, model.net = config, net
    module.require_no_local_model(model)
    assert hasattr(net, "action2llm") and hasattr(net, "llm2action") and hasattr(net, "vae2llm")


def test_producer_transform_rng_and_order_are_unchanged():
    import random

    instance = trainer(33)
    planner, producer = instance._grouped_planner, instance._grouped_producer
    transform = planner.catalog.wrapped_sft._transform

    def stochastic(item, resolution):
        result = transform(item, resolution)
        result["rng_witness"] = (random.random(), float(torch.rand(())))
        return result

    planner.catalog.wrapped_sft._transform = stochastic
    window = module.MatchedNoTTTWindow(Model(), planner)
    window.begin()
    requests = window.plan.members[0]
    random.seed(37)
    torch.manual_seed(37)
    reference = tuple(producer.produce(request) for request in requests)
    reference_payloads = tuple(
        payload for segment in reference for payload in segment.consumer_payload[0] if payload is not None
    )
    random.seed(37)
    torch.manual_seed(37)
    actual, _ = window.pack(tuple(producer.produce(request) for request in requests))
    assert [(p["start_frame"], p["rng_witness"]) for p in actual] == [
        (p["start_frame"], p["rng_witness"]) for p in reference_payloads
    ]


def test_no_ttt_passive_telemetry_records_consumer_source_without_fake_fast_state(monkeypatch):
    from examples.psm_wma_robocasa_matched_no_ttt import NoTTTPlanObserver

    monkeypatch.setenv("PSM_V3_STATE_AUDIT_EVERY", "1")
    instance, model = trainer(), Model()
    rows = []
    observer = NoTTTPlanObserver(rank=0, parameter_group=lambda _: "generation", emit=rows.append)
    instance.grouped_observer = observer
    observer.start_iteration(0)
    update(instance, model, optim(model))
    records = [json_row for json_row in rows if "actual_consumer_identity_count" in str(json_row)]
    assert len(records) == 1
    import json

    record = (
        records[0] if isinstance(records[0], dict) else json.loads(records[0].removeprefix("[CorrectedV3][train] "))
    )
    assert record["actual_consumer_identity_count"] == 256
    assert record["consumer_audit_status"] == "OBSERVED"
    assert record["fast_state_status"] == "DISABLED_NOT_INSTANTIATED"
    assert record["state_fingerprint_sampled"] is False
    assert "committed_fast_state_sha256" not in record
    assert not any("local_read" in error for error in observer._errors)

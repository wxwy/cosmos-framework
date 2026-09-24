"""A2 regression: source order, gradients, repeated-slot dependencies and resume."""

from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .active_local_memory_driver import ActiveLocalMemoryWindowDriver
from .active_local_memory_driver_test import _FakeProducer, _FakeStream, _model, _registry, _trainer
from .grouped_active_driver import GroupedActiveLocalMemoryWindowDriver
from .grouped_active_runtime import GroupedActiveWiringRegistry, GroupedSegmentRuntimeOwner
from .local_memory_segment import RankLocalSegmentScheduler, SegmentIdentity
from .production_segment_bridge import NativeBatchResult
from .production_segment_wiring import CanonicalSegmentWiring


class Producer(_FakeProducer):
    def produce(self, stream, *, cursor):
        segment = super().produce(stream, cursor=cursor)
        rng = torch.Generator().manual_seed(stream.slot_id * 10007 + stream.episode_index * 101 + cursor)
        return replace(
            segment,
            evidence_visual_summary_prev=torch.randn(1, self.ttt_tbptt_steps, 96, generator=rng),
            evidence_executed_action_prev=torch.randn(1, self.ttt_tbptt_steps, 10, generator=rng),
        )


def make_driver(*, grouped=True, group_size=2, ga=2, widths=(4, 4), template=None):
    producer = Producer()
    streams = tuple(_FakeStream(slot, slot + 10, "suite") for slot in range(len(widths)))
    for stream, width in zip(streams, widths, strict=True):
        producer.blocks[(stream.slot_id, stream.episode_index)] = width
    registry = _registry({"suite": 1.0})
    adapter = registry.owner.adapter
    if template is not None:
        adapter.encoder.load_state_dict(template.owner.adapter.encoder.state_dict())
        adapter.core.load_state_dict(template.owner.adapter.core.state_dict())
    parameters = tuple(adapter.encoder.parameters()) + tuple(adapter.core.parameters())
    registry.owner.wiring.local_slow_parameters = parameters
    if grouped:
        wiring = CanonicalSegmentWiring(adapter, parameters)
        registry = GroupedActiveWiringRegistry(
            GroupedSegmentRuntimeOwner(RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0}), wiring)
        )
    kwargs = dict(
        registry=registry,
        producer=producer,
        streams=streams,
        window_members=ga if grouped else ga * group_size,
        queue_seed=19,
    )
    if grouped:
        driver = GroupedActiveLocalMemoryWindowDriver(
            group_size=group_size, manifest_digest="manifest", config_digest="config", **kwargs
        )
    else:
        driver = ActiveLocalMemoryWindowDriver(**kwargs)
    trainer = _trainer(registry, grad_accum_iter=ga if grouped else ga * group_size)
    model = _model()
    driver.attach(trainer, model)
    return driver, trainer, model


def run_window(driver, trainer, model):
    collected, identities = [], []
    registry = driver.registry
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    for index in range(trainer.config.trainer.grad_accum_iter):
        driver.arm_next_member(trainer, model)
        prepared = trainer._psm_active_armed_prepared
        registry.consume_prepared_for_model(prepared)
        local = prepared.forward.result.local_tokens
        collected.append(local.detach().flatten(0, 1))
        identities.extend(prepared.inputs.identities)
        primary = local.square().sum() / prepared.actual_n_valid
        active = registry.publish_active_forward(prepared, NativeBatchResult(primary, primary * 0))
        trainer._run_active_local_memory_backward(model, {"psm_local_memory_active_forward": active}, scaler, index)
        trainer._psm_active_armed_prepared = None
    registry.owner.resolve_local_memory_slow_window(trainer._psm_active_completed_window, scaler_skipped=False)
    registry.retire_resolved_window(registry.owner)
    trainer._psm_active_completed_window = trainer._psm_active_registry = None
    return torch.cat(collected), tuple(identities)


def assert_state_equal(left, right):
    assert left["window_index"] == right["window_index"]
    for key in ("slot_epoch", "stream_index", "active_stream", "active_cursor"):
        assert left[key] == right[key]
    assert left["runtime"].scheduler == right["runtime"].scheduler
    a, b = left["runtime"].committed, right["runtime"].committed
    assert len(a) == len(b)
    for (i, state), (j, other) in zip(a, b, strict=True):
        assert i == j
        for value, expected in zip(state, other, strict=True):
            torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)


def test_group_plan_is_stable_slot_synchronized():
    driver, trainer, _ = make_driver(group_size=2, ga=3, widths=(10, 10))
    freeze = driver.freeze_window()
    plan = driver._plan_groups(freeze)
    assert plan.ga_effective == trainer.config.trainer.grad_accum_iter == 3
    for group_index, group in enumerate(plan.members):
        assert tuple(identity.slot_id for identity in group.row_identities) == (0, 1)
        assert tuple(identity.cursor for identity in group.row_identities) == (group_index, group_index)


def test_synchronized_groups_keep_one_wave_and_reuse_short_slot_only_after_terminal():
    driver, trainer, model = make_driver(group_size=2, ga=2, widths=(7, 3))
    # Window 1: both slots advance 0 -> 1.
    run_window(driver, trainer, model)
    first = driver._group_plan
    assert [tuple(i.cursor for i in group.row_identities) for group in first.members] == [(0, 0), (1, 1)]
    assert driver.registry.owner.last_dependency_wave_count == 1

    # Window 2: slot1 reaches terminal at cursor2, then only at the next
    # microbatch boundary reuses its catalog from cursor0. Slot0 continues.
    run_window(driver, trainer, model)
    second = driver._group_plan
    assert [tuple(i.slot_id for i in group.row_identities) for group in second.members] == [(0, 1), (0, 1)]
    assert [tuple(i.cursor for i in group.row_identities) for group in second.members] == [(2, 2), (3, 0)]
    assert driver._slot_epoch[1] == 1
    assert driver.registry.owner.last_dependency_wave_count == 1

    # The synchronized route remains executable after epoch reuse.
    run_window(driver, trainer, model)
    assert driver.registry.owner.last_dependency_wave_count == 1


def test_group_prepare_and_failure_do_not_publish_partial_rows():
    driver, trainer, model = make_driver()
    before = driver.registry.owner.snapshot()
    driver.arm_next_member(trainer, model)
    prepared = trainer._psm_active_armed_prepared
    owner = driver.registry.owner
    assert owner.scheduler.snapshot() == before.scheduler
    assert owner.adapter.sidecar._records == {}
    for parameter in owner.wiring.local_slow_parameters:
        parameter.grad = torch.ones_like(parameter)
    owner.abort_terminal(prepared.transaction, prepared.forward, "injected")
    assert owner.scheduler.snapshot() == before.scheduler
    assert owner.adapter.sidecar._records == {}
    assert all(p.grad is None for p in owner.wiring.local_slow_parameters)
    with pytest.raises(RuntimeError):
        owner.commit(prepared.transaction, prepared.forward)


def test_group_resume_matches_uninterrupted_next_window():
    original = make_driver(widths=(7, 3))
    run_window(*original)
    run_window(*original)
    saved = copy.deepcopy(original[0].state_dict())
    resumed = make_driver(widths=(7, 3), template=original[0].registry)
    resumed[0].load_state_dict(saved)
    a, ids_a = run_window(*original)
    b, ids_b = run_window(*resumed)
    assert ids_a == ids_b
    torch.testing.assert_close(a, b)
    assert_state_equal(original[0].state_dict(), resumed[0].state_dict())


@pytest.mark.parametrize("key,value", [("group_size", 99), ("native_members", 99), ("member_layout", "single")])
def test_invalid_resume_geometry_rejects_without_mutation(key, value):
    original = make_driver()
    run_window(*original)
    saved = original[0].state_dict()
    saved[key] = value
    other = make_driver()
    before = other[0].state_dict()
    with pytest.raises(RuntimeError, match="geometry"):
        other[0].load_state_dict(saved)
    assert_state_equal(before, other[0].state_dict())


def test_new_layout_cannot_load_in_legacy_driver():
    original = make_driver()
    run_window(*original)
    legacy = make_driver(grouped=False)
    with pytest.raises(RuntimeError, match="layout"):
        legacy[0].load_state_dict(original[0].state_dict())


def test_a2_default_geometry_is_16_forwards_2048_consumers():
    driver, trainer, _ = make_driver(group_size=8, ga=16, widths=(20,) * 8)
    driver.producer.ttt_tbptt_steps = 16
    freeze = driver.freeze_window()
    plan = driver._plan_groups(freeze)
    assert plan.ga_effective == trainer.config.trainer.grad_accum_iter == 16
    assert plan.planned_n_valid == (128,) * 16
    assert plan.n_window == 2048
    assert tuple(i for group in plan.members for i in group.row_identities) == freeze.identities


def _original_row_update(core, evidence, state, valid, *, prewrite=False):
    """Independent pre-A2 algorithm: per-row F.linear and per-row autograd.grad."""
    key, query_base, value = core.project_evidence(evidence)
    queries = core.project_queries(query_base)
    outputs, states = [], []
    from .local_evidence import ContinualTTTFastState

    for row in range(len(valid)):
        original = ContinualTTTFastState(*(v[row] for v in state))
        if not valid[row]:
            outputs.append(torch.zeros(core.k_local, core.local_dim))
            states.append(original)
            continue
        work = ContinualTTTFastState(*(v if v.requires_grad else v.detach().requires_grad_(True) for v in original))
        prediction = core._fast_mlp(key[row], work)
        inner = (prediction - value[row]).square().mean()
        gradients = torch.autograd.grad(inner, work, create_graph=True)
        updated = ContinualTTTFastState(*(v - core.inner_lr * g for v, g in zip(work, gradients, strict=True)))
        outputs.append(core._fast_mlp(queries[row], original if prewrite else updated))
        states.append(updated)
    return torch.stack(outputs), ContinualTTTFastState(*(torch.stack([s[i] for s in states]) for i in range(4)))


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("prewrite", [False, True])
def test_vectorized_inner_update_matches_original_per_row_algorithm(batch, prewrite):
    from .local_evidence import ContinualTTTLocalMemoryCore

    torch.manual_seed(47)
    reference = ContinualTTTLocalMemoryCore(evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10, k_local=2)
    vectorized = copy.deepcopy(reference)
    evidence = torch.randn(batch, 8)
    valid = torch.ones(batch, dtype=torch.bool)
    if batch > 1:
        valid[0] = False
    expected, expected_state = _original_row_update(
        reference, evidence, reference.initial_state(batch), valid, prewrite=prewrite
    )
    key, query_base, value = vectorized.project_evidence(evidence)
    actual, actual_state, _ = vectorized.step_projected_many(
        key_t=key,
        query_base_t=query_base,
        value_t=value,
        state_in=vectorized.initial_state(batch),
        valid=valid,
        create_graph=True,
        emit_prewrite_tokens=prewrite,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for a, b in zip(actual_state, expected_state, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(vectorized.parameters(), reference.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("changed", [False, True])
def test_first_group_retry_is_content_bound_and_commits_nothing_before_backward(changed):
    driver, trainer, model = make_driver()
    driver.arm_next_member(trainer, model)
    first = trainer._psm_active_armed_prepared
    registry = driver.registry
    expected = first.forward.result.local_tokens.detach().clone()
    before = registry.owner.scheduler.snapshot()
    plan = registry.abort_source_transient(first)
    assert registry.owner.scheduler.snapshot() == before
    assert registry.owner.adapter.sidecar._records == {}
    if changed:
        first.segment.evidence_visual_summary_prev[0, 1, 0] += 1
        with pytest.raises(RuntimeError, match="SOURCE_CHANGED"):
            registry.prepare_retry(first.segment, plan, trainer_grad_accum_iter=0)
        assert registry.owner.scheduler.snapshot() == before
        assert registry.owner.adapter.sidecar._records == {}
    else:
        retry = registry.prepare_retry(first.segment, plan, trainer_grad_accum_iter=0)
        torch.testing.assert_close(retry.forward.result.local_tokens, expected)
        assert registry.owner.scheduler.snapshot() == before
        with pytest.raises(RuntimeError, match="EXHAUSTED"):
            registry.abort_source_transient(retry)
        assert registry.owner.adapter.sidecar._records == {}


def test_resume_rejects_changed_episode_catalog_before_mutation():
    original = make_driver(widths=(4, 4))
    run_window(*original)
    changed = make_driver(widths=(5, 4))
    before = changed[0].state_dict()
    with pytest.raises(RuntimeError, match="geometry"):
        changed[0].load_state_dict(original[0].state_dict())
    assert_state_equal(before, changed[0].state_dict())


def test_live_delivery_metrics_records_actual_geometry(tmp_path):
    import json

    from .grouped_active_metrics import ActiveDeliveryMetrics

    driver, trainer, model = make_driver()
    run_window(driver, trainer, model)
    trainer.last_optimizer_step_timing = {"step_wall_s": 1.0}
    metrics = ActiveDeliveryMetrics(trainer=trainer, driver=driver, path=str(tmp_path / "metrics.jsonl"))
    metrics._window_started = 0.0
    metrics._starting_state_sha = metrics._fast_state_sha()
    metrics._losses = [1.0, 2.0]
    metrics._backward_objectives = [0.5, 1.0]
    metrics._waves = [1, 1]
    metrics._identities = [(0, "test", i) for i in range(driver._group_plan.n_window)]
    model._psm_native_forward_calls = 2
    metrics.on_training_step_end(model, {}, {}, torch.tensor(1.0), iteration=1)
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record["valid_consumers"] == 8 and record["native_forwards"] == 2
    assert record["window_index"] == 1 and len(record["actual_consumer_identities"]) == 8
    assert record["fast_state_sha256"] == metrics._fast_state_sha()
    assert record["fast_state_before_first_group_sha256"] is not None
    assert record["raw_native_loss_mean"] == 1.5
    assert record["backward_objective_sum"] == 1.5


def test_active_group_explicitly_rejects_partial_tbptt_row():
    driver, _, _ = make_driver()
    freeze = driver.freeze_window()
    plan = driver._plan_groups(freeze)
    group = plan.members[0]
    with pytest.raises(ValueError, match="whole TBPTT"):
        replace(group, row_planned_n_valid=(1, 2))


def test_speculative_rebind_does_not_modify_live_scheduler():
    driver, trainer, model = make_driver(widths=(2, 2))
    run_window(driver, trainer, model)
    before = driver.registry.owner.scheduler.snapshot()
    terminal_slots = dict(driver.registry.owner.scheduler.terminal_slots)
    assert terminal_slots
    for slot in terminal_slots:
        driver._rebind_terminal(slot)
    assert driver.registry.owner.scheduler.snapshot() == before


def test_prepare_error_preserves_original_exception_and_no_row_commit():
    driver, trainer, model = make_driver()
    with patch.object(driver.registry.owner, "_scan_group", side_effect=TypeError("original payload failure")):
        with pytest.raises(TypeError, match="original payload failure"):
            driver.arm_next_member(trainer, model)
    assert driver.registry.owner.adapter.sidecar._records == {}
    assert driver.registry.owner.scheduler.committed_identities == []


@pytest.mark.parametrize("slots", [4, 8, 12])
def test_configurable_slot_count_preserves_consumer_accounting(slots):
    driver, trainer, _ = make_driver(group_size=slots, ga=3, widths=(20,) * slots)
    plan = driver._plan_groups(driver.freeze_window())
    assert plan.ga_effective == trainer.config.trainer.grad_accum_iter == 3
    assert plan.n_window == slots * driver.producer.ttt_tbptt_steps * 3
    assert all(len(member.row_identities) == slots for member in plan.members)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires authorized CUDA")
def test_cuda_vectorized_fast_state_and_outer_gradients_match_scalar():
    from .local_evidence import ContinualTTTLocalMemoryCore

    torch.manual_seed(59)
    reference = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10, k_local=2
    ).cuda()
    vectorized = copy.deepcopy(reference)
    evidence = torch.randn(8, 8, device="cuda")
    valid = torch.ones(8, dtype=torch.bool, device="cuda")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected, state = _original_row_update(
            reference, evidence, reference.initial_state(8, device="cuda"), valid
        )
        key, query, value = vectorized.project_evidence(evidence)
        actual, got, _ = vectorized.step_projected_many(
            key_t=key,
            query_base_t=query,
            value_t=value,
            state_in=vectorized.initial_state(8, device="cuda"),
            valid=valid,
            create_graph=True,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for a, b in zip(got, state, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for a, b in zip(vectorized.parameters(), reference.parameters(), strict=True):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def test_a2_and_scalar_window_objective_have_identical_consumer_mean_scale():
    from .grouped_active_contract import GroupedGAWindowPlan, GroupedPlanMember
    from .local_memory_segment import GAWindowPlan, SegmentIdentity

    # One window has the same 2048 consumers in both layouts.  Let each scalar
    # 16-consumer segment have an arbitrary native mean.  Each A2 128-consumer
    # group is the equal-size mean of eight consecutive scalar segments.
    scalar_means = [torch.tensor(1.0 + index / 100.0, dtype=torch.float64) for index in range(128)]
    scalar_plan = GAWindowPlan(
        members=tuple((index % 8, f"e{index % 8}", index // 8) for index in range(128)),
        planned_n_valid=(16,) * 128,
    )
    scalar_total = sum(
        scalar_plan.objective(index, mean, torch.tensor(0.0, dtype=torch.float64), 16)
        for index, mean in enumerate(scalar_means)
    )

    identities = tuple(
        SegmentIdentity(index % 8, f"e{index % 8}", "suite", index // 8, index, "source")
        for index in range(128)
    )
    groups = tuple(
        GroupedPlanMember(
            identities[start : start + 8], (16,) * 8, 16, "manifest", "config", "source"
        )
        for start in range(0, 128, 8)
    )
    grouped_plan = GroupedGAWindowPlan(
        members=groups, planned_n_valid=(128,) * 16, plan_chain_id="scale-parity"
    )
    grouped_means = [torch.stack(scalar_means[start : start + 8]).mean() for start in range(0, 128, 8)]
    grouped_total = sum(
        grouped_plan.objective(index, mean, torch.tensor(0.0, dtype=torch.float64), 128)
        for index, mean in enumerate(grouped_means)
    )
    torch.testing.assert_close(grouped_total, scalar_total, rtol=0, atol=1e-12)
    torch.testing.assert_close(grouped_total, torch.stack(scalar_means).mean(), rtol=0, atol=1e-12)


def test_resume_compressed_audit_does_not_collide_on_reused_episode_cursor() -> None:
    """A resumed slot may retain only an old terminal cursor, not its old cursor0.

    Rebinding at fresh cursor0 must clear the whole slot audit so reaching the
    same terminal cursor again in a later catalog epoch cannot be rejected as a
    duplicate committed identity.
    """
    driver, _, _ = make_driver(group_size=1, ga=1, widths=(10,))
    owner = driver.registry.owner
    scheduler = owner.scheduler
    source = driver.producer.source_digest

    old_terminal = SegmentIdentity(
        slot_id=0,
        episode_id="10",
        category="suite",
        cursor=9,
        segment_id=9,
        source_digest=source,
        training_stream_end=True,
    )
    # Model the checkpoint-compressed runtime frontier: only the latest audit
    # identity for the slot survives snapshot/rebuild.
    scheduler.admission_order = [old_terminal]
    scheduler.committed_identities = [old_terminal]
    scheduler.stable_slots = {0: old_terminal}
    scheduler.terminal_slots = {0: old_terminal}

    for cursor in range(10):
        identity = SegmentIdentity(
            slot_id=0,
            episode_id="10",
            category="suite",
            cursor=cursor,
            segment_id=cursor,
            source_digest=source,
            training_stream_end=cursor == 9,
        )
        group = GroupedPlanMember(
            (identity,),
            (4,),
            4,
            "manifest",
            "config",
            source,
        )
        owner.scheduler = owner._stage_scheduler(group)

    assert owner.scheduler.committed_identities[-1].cursor == 9
    assert owner.scheduler.committed_identities[-1].episode_id == "10"
    assert len(owner.scheduler.committed_identities) == 10

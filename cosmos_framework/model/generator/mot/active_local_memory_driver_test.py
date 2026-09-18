# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only contract tests for the active Local-Memory window driver."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.active_local_memory_driver import (
    ActiveLocalMemoryWindowDriver,
)
from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import queue_permutation
from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    RankLocalSegmentScheduler,
    SegmentBatch,
    SegmentProvenance,
)
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import (
    CanonicalLocalMemorySegmentAdapter,
    LocalMemorySegmentSidecar,
)
from cosmos_framework.model.generator.mot.production_active_wiring import ProductionActiveWiringRegistry
from cosmos_framework.model.generator.mot.production_segment_bridge import NativeBatchResult
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.callback import Callback, _missing_callback_hooks


class _FakeStream:
    """Minimal stand-in for ``CanonicalSegmentStream``'s driver-facing surface."""

    def __init__(self, slot_id: int, episode_index: int, category: str) -> None:
        self.slot_id = slot_id
        self.episode_index = episode_index
        self.episode_position = episode_index
        self.category = category


class _FakeProducer:
    """Deterministic whole-block geometry without touching the latent cache."""

    def __init__(self, *, ttt_tbptt_steps: int = 2, source_digest: str = "source") -> None:
        self.ttt_tbptt_steps = ttt_tbptt_steps
        self.source_digest = source_digest
        self.blocks: dict[tuple[int, int], int] = {}
        self.produced: list[tuple[int, int, int]] = []

    def block_count(self, stream: _FakeStream) -> int:
        return self.blocks.get((stream.slot_id, stream.episode_index), 0)

    def produce(self, stream: _FakeStream, *, cursor: int) -> SegmentBatch:
        self.produced.append((stream.slot_id, stream.episode_index, cursor))
        return _segment(stream, cursor, self.ttt_tbptt_steps)


def _segment(stream: _FakeStream, cursor: int, width: int) -> SegmentBatch:
    steps = torch.arange(cursor * width, (cursor + 1) * width, dtype=torch.long).unsqueeze(0)
    has_evidence = steps > 0
    return SegmentBatch(
        consumer_visual_summary=torch.zeros(1, width, 96),
        consumer_payload=(tuple({"step": int(step)} for step in steps[0]),),
        consumer_valid=torch.ones(1, width, dtype=torch.bool),
        consumer_step=steps,
        evidence_visual_summary_prev=torch.zeros(1, width, 96),
        evidence_executed_action_prev=torch.zeros(1, width, 10),
        evidence_valid=has_evidence,
        evidence_source_step=torch.where(has_evidence, steps - 1, torch.full_like(steps, -1)),
        slot_id=torch.tensor([stream.slot_id]),
        episode_id=(str(stream.episode_index),),
        category=(stream.category,),
        segment_provenance=SegmentProvenance("manifest", "config", "source", cursor),
    )


def _registry(target_distribution: dict[str, float]) -> ProductionActiveWiringRegistry:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution=target_distribution)
    adapter = CanonicalLocalMemorySegmentAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(),
        LocalMemorySegmentSidecar(),
    )
    wiring = CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),))
    return ProductionActiveWiringRegistry(CanonicalSegmentRuntimeOwner(scheduler, wiring))


def _trainer(registry: ProductionActiveWiringRegistry, *, grad_accum_iter: int) -> ImaginaireTrainer:
    trainer = object.__new__(ImaginaireTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=grad_accum_iter))
    trainer.callbacks = SimpleNamespace(_callbacks=[])
    return trainer


def _model() -> torch.nn.Module:
    model = object.__new__(torch.nn.Module)
    torch.nn.Module.__init__(model)
    return model


def _driver(
    registry: ProductionActiveWiringRegistry,
    producer: _FakeProducer,
    streams: tuple[_FakeStream, ...],
    *,
    window_members: int,
    queue_seed: int = 0,
    prefetch_depth: int = 0,
) -> ActiveLocalMemoryWindowDriver:
    return ActiveLocalMemoryWindowDriver(
        registry=registry,
        producer=producer,
        streams=streams,
        window_members=window_members,
        queue_seed=queue_seed,
        prefetch_depth=prefetch_depth,
    )


def test_freeze_covers_exactly_one_optimizer_window_of_whole_blocks() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 7)] = 4
    stream = _FakeStream(0, 7, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)

    freeze = driver.freeze_window()

    assert freeze.plan.members == ((0, "7", 0), (0, "7", 1))
    assert freeze.plan.planned_n_valid == (2, 2)
    assert freeze.plan.ga_effective == 2
    assert [identity.cursor for identity in freeze.identities] == [0, 1]
    assert [member.cursor for member in freeze.members] == [0, 1]
    assert producer.produced == []


def test_slot_rotation_follows_the_scheduler_deficit_rule() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    producer.blocks[(1, 2)] = 4
    driver = _driver(
        _registry({"a": 0.5, "b": 0.5}),
        producer,
        (_FakeStream(0, 1, "a"), _FakeStream(1, 2, "b")),
        window_members=2,
    )

    freeze = driver.freeze_window()

    # The lagging category is served first, so the second member returns to it.
    assert [member.stream.category for member in freeze.members] == ["b", "a"]
    assert [identity.slot_id for identity in freeze.identities] == [1, 0]


def test_slot_rotation_covers_every_slot_sharing_one_category() -> None:
    """Two slots of the same category must alternate inside one window.

    Production always looks like this: ``canonical_segment_streams`` assigns a
    suite's episodes round-robin to the two slots where
    ``slot % len(categories) == index``.  Two such slots have an identical
    deficit — it is computed per category — so the tie-break alone decides, and
    a tie-break that prefers the larger slot_id starves the smaller one for the
    whole run.
    """
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    producer.blocks[(4, 2)] = 4
    driver = _driver(
        _registry({"suite": 1.0}),
        producer,
        (_FakeStream(0, 1, "suite"), _FakeStream(4, 2, "suite")),
        window_members=4,
    )

    freeze = driver.freeze_window()

    slots = [identity.slot_id for identity in freeze.identities]
    assert set(slots) == {0, 4}  # was [4, 4, 4, 4]: slot 0 never selected
    assert slots == [0, 4, 0, 4]
    assert [identity.category for identity in freeze.identities] == ["suite"] * 4


def test_no_slot_starves_at_the_production_stream_geometry() -> None:
    """A ``b_stream=8`` / 4-category window must visit all eight slots.

    Each category owns exactly two slots, so the window decomposes as
    ``8 slots x 16 members = 128 = b_stream * ga`` — the shape the frozen (B)
    ruling calls numerically equivalent to one eight-row batch.
    """
    categories = ("c0", "c1", "c2", "c3")
    producer = _FakeProducer()
    streams = []
    for index, category in enumerate(categories):
        for slot_id in (index, index + 4):
            producer.blocks[(slot_id, slot_id)] = 64
            streams.append(_FakeStream(slot_id, slot_id, category))
    driver = _driver(
        _registry({category: 0.25 for category in categories}),
        producer,
        tuple(streams),
        window_members=128,
    )

    freeze = driver.freeze_window()

    per_slot = Counter(identity.slot_id for identity in freeze.identities)
    assert sorted(per_slot) == list(range(8))  # was [4, 5, 6, 7]
    assert set(per_slot.values()) == {16}
    per_category = Counter(identity.category for identity in freeze.identities)
    assert set(per_category.values()) == {32}  # quota stays a quarter each


def test_last_block_of_an_episode_is_terminal_and_rebinds_its_successor() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 2
    producer.blocks[(0, 2)] = 3
    driver = _driver(
        _registry({"suite": 1.0}),
        producer,
        (_FakeStream(0, 1, "suite"), _FakeStream(0, 2, "suite")),
        window_members=3,
    )

    freeze = driver.freeze_window()

    assert [identity.episode_id for identity in freeze.identities] == ["1", "1", "2"]
    assert [identity.cursor for identity in freeze.identities] == [0, 1, 0]
    assert [identity.training_stream_end for identity in freeze.identities] == [False, True, False]
    assert [member.rebind_before_admit for member in freeze.members] == [False, False, True]


def test_driver_raises_when_every_stream_is_exhausted() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 1
    driver = _driver(_registry({"suite": 1.0}), producer, (_FakeStream(0, 1, "suite"),), window_members=2)

    with pytest.raises(RuntimeError, match="exhausted every segment stream"):
        driver.freeze_window()


def test_driver_skips_an_episode_without_a_whole_block() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 0  # fewer frames than one TBPTT block
    producer.blocks[(0, 2)] = 2
    driver = _driver(
        _registry({"suite": 1.0}),
        producer,
        (_FakeStream(0, 1, "suite"), _FakeStream(0, 2, "suite")),
        window_members=1,
    )

    freeze = driver.freeze_window()

    assert freeze.identities[0].episode_id == "2"
    assert freeze.identities[0].cursor == 0


def test_attach_binds_the_registry_and_registers_one_batch_start_hook() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 2
    driver = _driver(registry, producer, (_FakeStream(0, 1, "suite"),), window_members=1)
    trainer, model = _trainer(registry, grad_accum_iter=1), _model()

    driver.attach(trainer, model)
    driver.attach(trainer, model)

    assert trainer._psm_active_wiring_registry is registry
    assert model._psm_active_wiring_registry is registry
    assert trainer.callbacks._callbacks == [driver]
    assert trainer._psm_active_window_driver is driver


def test_driver_rejects_a_member_count_off_the_accumulation_boundary() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    driver = _driver(registry, producer, (_FakeStream(0, 1, "suite"),), window_members=2)
    trainer, model = _trainer(registry, grad_accum_iter=1), _model()
    driver.attach(trainer, model)

    with pytest.raises(RuntimeError, match="differs from the accumulation boundary"):
        driver.arm_next_member(trainer, model)
    assert registry.owner.phase.name == "IDLE" and registry.owner.adapter.pending() is None


def test_driver_arms_initial_then_continuation_at_the_trainer_counter() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    driver = _driver(registry, producer, (_FakeStream(0, 1, "suite"),), window_members=2)
    trainer, model = _trainer(registry, grad_accum_iter=2), _model()
    driver.attach(trainer, model)

    driver.arm_next_member(trainer, model)
    first = trainer._psm_active_armed_prepared
    assert first.member_index == 0 and first.identity.cursor == 0
    assert registry.owner.phase.name == "PREPARED"

    registry.consume_prepared_for_model(first)
    published = registry.publish_active_forward(first, NativeBatchResult(torch.ones(()), torch.zeros(())))
    registry.consume_active_forward(published)
    transaction = registry.owner.transaction
    transaction.successful_backward(0, first.identity, first.actual_n_valid)
    registry.owner.commit(transaction, first.forward)
    trainer._psm_active_armed_prepared = None

    driver.arm_next_member(trainer, model)
    second = trainer._psm_active_armed_prepared
    assert second.member_index == 1 and second.identity.cursor == 1
    assert second.transaction is transaction


def test_driver_refuses_an_unarmable_owner_phase() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    driver = _driver(registry, producer, (_FakeStream(0, 1, "suite"),), window_members=2)
    trainer, model = _trainer(registry, grad_accum_iter=2), _model()
    driver.attach(trainer, model)
    driver.arm_next_member(trainer, model)
    trainer._psm_active_armed_prepared = None  # already consumed by the trainer forward

    with pytest.raises(RuntimeError, match="unarmable owner phase: PREPARED"):
        driver.arm_next_member(trainer, model)


def test_driver_satisfies_every_callback_group_hook() -> None:
    """``CallBackGroup.__getattr__`` asserts each member implements every hook."""
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 2
    driver = _driver(registry, producer, (_FakeStream(0, 1, "suite"),), window_members=1)

    assert isinstance(driver, Callback)
    assert _missing_callback_hooks(driver) == []


def test_continuation_rebinds_a_finished_episode_before_admitting_its_successor() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 1
    producer.blocks[(0, 2)] = 2
    driver = _driver(
        registry,
        producer,
        (_FakeStream(0, 1, "suite"), _FakeStream(0, 2, "suite")),
        window_members=2,
    )
    trainer, model = _trainer(registry, grad_accum_iter=2), _model()
    driver.attach(trainer, model)

    driver.arm_next_member(trainer, model)
    first = trainer._psm_active_armed_prepared
    registry.consume_prepared_for_model(first)
    published = registry.publish_active_forward(first, NativeBatchResult(torch.ones(()), torch.zeros(())))
    registry.consume_active_forward(published)
    transaction = registry.owner.transaction
    transaction.successful_backward(0, first.identity, first.actual_n_valid)
    registry.owner.commit(transaction, first.forward)
    trainer._psm_active_armed_prepared = None
    assert registry.owner.scheduler.terminal_slots[0] is first.identity

    driver.arm_next_member(trainer, model)

    second = trainer._psm_active_armed_prepared
    assert second.identity.episode_id == "2" and second.identity.cursor == 0
    assert 0 not in registry.owner.scheduler.terminal_slots
    assert producer.produced == [(0, 1, 0), (0, 2, 0)]


def test_load_state_dict_rejects_source_digest_mismatch() -> None:
    producer = _FakeProducer(source_digest="source-a")
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()

    state["source_digest"] = "source-b"
    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    with pytest.raises(RuntimeError):
        other.load_state_dict(state)


def test_state_dict_round_trips_cursor_state() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    other.load_state_dict(state)

    assert other._stream_index == driver._stream_index
    assert other._active_cursor == driver._active_cursor
    assert other._window_index == driver._window_index
    assert [
        (s.slot_id, s.episode_index, s.category) for s in other._active_stream.values()
    ] == [(s.slot_id, s.episode_index, s.category) for s in driver._active_stream.values()]


def test_load_state_dict_rejects_unrebuildable_slot_binding() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()
    slot = next(iter(state["active_stream"]))
    slot_id, _, episode_position, category = state["active_stream"][slot]
    state["active_stream"][slot] = (slot_id, 999, episode_position, category)

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    with pytest.raises(RuntimeError, match="ambiguous"):
        other.load_state_dict(state)


def test_load_state_dict_rejects_ambiguous_slot_binding() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    twin = (_FakeStream(0, 1, "suite"), _FakeStream(0, 1, "suite"))
    driver = _driver(_registry({"suite": 1.0}), producer, twin, window_members=2)
    driver.freeze_window()
    state = driver.state_dict()

    other = _driver(_registry({"suite": 1.0}), producer, twin, window_members=2)
    with pytest.raises(RuntimeError, match="ambiguous"):
        other.load_state_dict(state)


def test_runtime_round_trip_preserves_committed_identity_objects() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(registry, producer, (stream,), window_members=2)
    trainer, model = _trainer(registry, grad_accum_iter=2), _model()
    driver.attach(trainer, model)

    transaction = None
    for index in range(2):
        driver.arm_next_member(trainer, model)
        prepared = trainer._psm_active_armed_prepared
        registry.consume_prepared_for_model(prepared)
        published = registry.publish_active_forward(prepared, NativeBatchResult(torch.ones(()), torch.zeros(())))
        registry.consume_active_forward(published)
        transaction = registry.owner.transaction
        transaction.successful_backward(index, prepared.identity, prepared.actual_n_valid)
        registry.owner.commit(transaction, prepared.forward)
        trainer._psm_active_armed_prepared = None
    registry.owner.resolve_local_memory_slow_window(registry.owner.finish_window(transaction), scaler_skipped=False)
    assert registry.owner.phase.name == "IDLE"

    state = driver.state_dict()

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    other.load_state_dict(state)

    owner = other.registry.owner
    committed_by_slot = {i.slot_id: i for i in owner.scheduler.committed_identities}
    assert committed_by_slot
    for slot, identity in committed_by_slot.items():
        assert identity is owner.scheduler.stable_slots[slot]
    other.state_dict()  # re-snapshot must not raise (is-checks hold)


def _live_identity(driver: ActiveLocalMemoryWindowDriver) -> dict[str, object]:
    """Capture the live objects whose identity must survive a rejected load."""
    owner = driver.registry.owner
    return {
        "scheduler": owner.scheduler,
        "stream_index": dict(driver._stream_index),
        "active_stream": dict(driver._active_stream),
        "active_cursor": dict(driver._active_cursor),
        "window_index": driver._window_index,
        "sidecar_keys": set(owner.adapter.sidecar._records.keys()),
        "phase": owner.phase,
    }


def _reject_without_live_mutation(state: dict, *, match: str) -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    before = _live_identity(other)
    with pytest.raises(RuntimeError, match=match):
        other.load_state_dict(state)
    assert _live_identity(other) == before


def test_load_state_dict_rejects_misaligned_stream_index() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()
    state["stream_index"][0] = 999
    _reject_without_live_mutation(state, match="stream_index")


def test_load_state_dict_rejects_out_of_range_cursor() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()
    state["active_cursor"][0] = 99
    _reject_without_live_mutation(state, match="cursor")


def test_load_state_dict_rejects_bad_active_stream_key_set() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    driver.freeze_window()
    state = driver.state_dict()
    state["active_stream"][7] = (7, 1, 1, "suite")  # key not in the catalog
    _reject_without_live_mutation(state, match="key")


def test_load_state_dict_rejects_runtime_without_scheduler_counterpart() -> None:
    from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalRuntimeSnapshot

    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _driver(registry, producer, (stream,), window_members=2)
    trainer, model = _trainer(registry, grad_accum_iter=2), _model()
    driver.attach(trainer, model)

    transaction = None
    for index in range(2):
        driver.arm_next_member(trainer, model)
        prepared = trainer._psm_active_armed_prepared
        registry.consume_prepared_for_model(prepared)
        published = registry.publish_active_forward(prepared, NativeBatchResult(torch.ones(()), torch.zeros(())))
        registry.consume_active_forward(published)
        transaction = registry.owner.transaction
        transaction.successful_backward(index, prepared.identity, prepared.actual_n_valid)
        registry.owner.commit(transaction, prepared.forward)
        trainer._psm_active_armed_prepared = None
    registry.owner.resolve_local_memory_slow_window(registry.owner.finish_window(transaction), scaler_skipped=False)

    state = driver.state_dict()
    runtime = state["runtime"]
    stripped = dict(runtime.scheduler)
    stripped["admission_order"] = ()
    stripped["committed_identities"] = ()
    stripped["stable_slots"] = {}
    state["runtime"] = CanonicalRuntimeSnapshot(runtime.generation, stripped, runtime.committed)

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    before = _live_identity(other)
    with pytest.raises(RuntimeError, match="counterpart"):
        other.load_state_dict(state)
    assert _live_identity(other) == before


def _run_full_window(registry, producer, stream, *, window_members: int = 2, prefetch_depth: int = 0):
    driver = _driver(registry, producer, (stream,), window_members=window_members, prefetch_depth=prefetch_depth)
    trainer, model = _trainer(registry, grad_accum_iter=window_members), _model()
    driver.attach(trainer, model)
    transaction = None
    for index in range(window_members):
        driver.arm_next_member(trainer, model)
        prepared = trainer._psm_active_armed_prepared
        registry.consume_prepared_for_model(prepared)
        published = registry.publish_active_forward(prepared, NativeBatchResult(torch.ones(()), torch.zeros(())))
        registry.consume_active_forward(published)
        transaction = registry.owner.transaction
        transaction.successful_backward(index, prepared.identity, prepared.actual_n_valid)
        registry.owner.commit(transaction, prepared.forward)
        trainer._psm_active_armed_prepared = None
    registry.owner.resolve_local_memory_slow_window(registry.owner.finish_window(transaction), scaler_skipped=False)
    return driver


def test_load_state_dict_accepts_terminal_frontier_without_sidecar_carry() -> None:
    """A slot may terminalize (sidecar popped) while still in committed_identities."""
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 2  # cursor 1 == blocks-1 is terminal
    stream = _FakeStream(0, 1, "suite")
    driver = _run_full_window(registry, producer, stream)

    assert registry.owner.scheduler.terminal_slots
    assert registry.owner.adapter.sidecar._records == {}
    state = driver.state_dict()

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    other.load_state_dict(state)  # must be accepted, not mis-rejected
    other.state_dict()  # re-snapshot must succeed


def test_load_state_dict_rejects_sidecar_identity_value_mismatch() -> None:
    from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalRuntimeSnapshot
    from cosmos_framework.model.generator.mot.local_memory_segment import SegmentIdentity

    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")
    driver = _run_full_window(registry, producer, stream)

    state = driver.state_dict()
    runtime = state["runtime"]
    identity, fast_state = runtime.committed[0]
    bogus = SegmentIdentity(identity.slot_id, "bogus", identity.category, 99, 99, identity.source_digest)
    state["runtime"] = CanonicalRuntimeSnapshot(runtime.generation, runtime.scheduler, ((bogus, fast_state),))

    other = _driver(_registry({"suite": 1.0}), producer, (stream,), window_members=2)
    before = _live_identity(other)
    with pytest.raises(RuntimeError, match="differs"):
        other.load_state_dict(state)
    assert _live_identity(other) == before


# ---- epoch reuse (catalog epoch-reuse design v0.8) ---------------------------


def _run_windows(
    registry: ProductionActiveWiringRegistry,
    producer: _FakeProducer,
    streams: tuple[_FakeStream, ...],
    *,
    window_members: int,
    num_windows: int,
) -> ActiveLocalMemoryWindowDriver:
    """Run ``num_windows`` complete windows through the trainer seam.

    Every window boundary re-enters ``_arm_initial`` (owner phase back to IDLE),
    which is exactly where the epoch-reuse boundary probe runs.
    """
    driver = _driver(registry, producer, streams, window_members=window_members)
    trainer, model = _trainer(registry, grad_accum_iter=window_members), _model()
    driver.attach(trainer, model)
    for _ in range(num_windows):
        transaction = None
        for index in range(window_members):
            driver.arm_next_member(trainer, model)
            prepared = trainer._psm_active_armed_prepared
            registry.consume_prepared_for_model(prepared)
            published = registry.publish_active_forward(prepared, NativeBatchResult(torch.ones(()), torch.zeros(())))
            registry.consume_active_forward(published)
            transaction = registry.owner.transaction
            transaction.successful_backward(index, prepared.identity, prepared.actual_n_valid)
            registry.owner.commit(transaction, prepared.forward)
            trainer._psm_active_armed_prepared = None
        registry.owner.resolve_local_memory_slow_window(registry.owner.finish_window(transaction), scaler_skipped=False)
    return driver


def test_remaining_blocks_probe_is_pure() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    driver = _driver(_registry({"suite": 1.0}), producer, (_FakeStream(0, 1, "suite"),), window_members=2)
    driver.freeze_window()

    before = (dict(driver._stream_index), dict(driver._active_stream), dict(driver._active_cursor))
    assert driver._remaining_blocks(0) == 2
    after = (dict(driver._stream_index), dict(driver._active_stream), dict(driver._active_cursor))
    assert before == after


def test_rollover_triggers_only_below_window_members() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    driver = _driver(_registry({"suite": 1.0}), producer, (_FakeStream(0, 1, "suite"),), window_members=2)

    driver._maybe_rollover()
    assert driver._slot_epoch == {0: 0}

    driver.freeze_window()  # cursor 0,1 -> remaining 2, still fillable
    driver._maybe_rollover()
    assert driver._slot_epoch == {0: 0}

    driver.freeze_window()  # cursor 2,3 -> terminal, remaining 0 < 2
    driver._maybe_rollover()
    assert driver._slot_epoch == {0: 1}
    assert driver._active_stream == {} and driver._active_cursor == {}


def test_reordered_by_slot_matches_category_permutation() -> None:
    producer = _FakeProducer()
    for episode in (1, 2, 10):
        producer.blocks[(0, episode)] = 2
    streams = (_FakeStream(0, 1, "suite"), _FakeStream(0, 2, "suite"), _FakeStream(0, 10, "suite"))
    driver = _driver(_registry({"suite": 1.0}), producer, streams, window_members=2)

    reference = sorted(streams, key=lambda s: (producer.source_digest, str(s.episode_index)))
    # string comparison: "1" < "10" < "2"
    assert [s.episode_index for s in reference] == [1, 10, 2]
    permutation = queue_permutation(queue_seed=0, epoch=1, category="suite", catalog_size=len(reference))
    assert driver._reordered_by_slot(0, 1) == tuple(reference[i] for i in permutation)


def test_non_terminal_slot_is_not_reset_by_rollover() -> None:
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    producer.blocks[(1, 2)] = 4
    streams = (_FakeStream(0, 1, "suite"), _FakeStream(1, 2, "suite"))
    driver = _driver(_registry({"suite": 1.0}), producer, streams, window_members=2)

    driver._active_stream[0] = streams[0]
    driver._active_cursor[0] = 1  # blocks - 1 == 3, so non-terminal
    driver._stream_index[0] = 0
    driver._slot_epoch[0] = 2

    driver._rollover_slot(0)

    assert driver._active_stream[0] is streams[0]
    assert driver._active_cursor[0] == 1
    assert driver._slot_epoch[0] == 2


def test_terminal_slot_reuse_prunes_guards_and_replays_fresh_episode() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4  # cursor 0..3, terminal at 3
    producer.blocks[(1, 2)] = 1  # single block, terminal at 0
    streams = (_FakeStream(0, 1, "suite"), _FakeStream(1, 2, "suite"))
    driver = _run_windows(registry, producer, streams, window_members=2, num_windows=3)

    # slot 0 stayed non-terminal across the first boundary (cursor 2 < 3); slot 1
    # terminal-reused once.  The three windows only succeed if the reused slot's
    # stale guard entries were pruned (criterion 9).
    assert driver._slot_epoch[1] == 1
    slot1_admissions = [i for i in registry.owner.scheduler.admission_order if i.slot_id == 1]
    assert len(slot1_admissions) == 1
    assert slot1_admissions[0].episode_id == "2" and slot1_admissions[0].cursor == 0


def test_slot_epoch_round_trips_through_state_dict() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    producer.blocks[(1, 2)] = 1
    streams = (_FakeStream(0, 1, "suite"), _FakeStream(1, 2, "suite"))
    driver = _run_windows(registry, producer, streams, window_members=2, num_windows=3)
    assert driver._slot_epoch[1] == 1

    state = driver.state_dict()
    other = _driver(_registry({"suite": 1.0}), producer, streams, window_members=2)
    other.load_state_dict(state)

    assert other._slot_epoch == driver._slot_epoch
    assert other._by_slot == driver._by_slot
    assert other._stream_index == driver._stream_index
    assert other._active_cursor == driver._active_cursor


def test_rollover_discards_sidecar_carry_only_for_terminal_slot() -> None:
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    producer.blocks[(1, 2)] = 4
    streams = (_FakeStream(0, 1, "suite"), _FakeStream(1, 2, "suite"))
    driver = _driver(registry, producer, streams, window_members=2)

    # slot 0 non-terminal (cursor 1 < blocks-1 == 3), slot 1 terminal (cursor 3).
    driver._active_stream[0] = streams[0]
    driver._active_cursor[0] = 1
    driver._stream_index[0] = 0
    driver._active_stream[1] = streams[1]
    driver._active_cursor[1] = 3
    driver._stream_index[1] = 0

    sidecar = registry.owner.adapter.sidecar
    sidecar._records[0] = ("slot0", "carry0")
    sidecar._records[1] = ("slot1", "carry1")

    driver._rollover_slot(0)  # non-terminal: keep the detached carry
    driver._rollover_slot(1)  # terminal: drop the detached carry

    assert sidecar._records[0] == ("slot0", "carry0")
    assert 1 not in sidecar._records



def test_prefetch_builds_the_same_window_members_as_sequential() -> None:
    """A window armed with prefetch workers commits exactly the frozen members.

    Per-sample data determinism is covered separately; this pins that the
    background workers neither reorder nor drop a member.
    """
    registry = _registry({"suite": 1.0})
    producer = _FakeProducer()
    producer.blocks[(0, 1)] = 4
    stream = _FakeStream(0, 1, "suite")

    driver = _run_full_window(registry, producer, stream, window_members=2, prefetch_depth=2)

    assert registry.owner.phase.name == "IDLE"
    assert [identity.cursor for identity in registry.owner.scheduler.committed_identities] == [0, 1]
    assert driver._prefetch_futures == {}

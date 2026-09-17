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
) -> ActiveLocalMemoryWindowDriver:
    return ActiveLocalMemoryWindowDriver(
        registry=registry,
        producer=producer,
        streams=streams,
        window_members=window_members,
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

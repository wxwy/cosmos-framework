# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only contract tests for the active Local-Memory launch assembly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.active_local_memory_launch import (
    ActiveLocalMemoryLaunchCallback,
    SuiteRoutedSegmentProducer,
    canonical_runtime_module,
    canonical_segment_adapter_from_model,
    canonical_segment_streams,
    canonical_slow_parameters_from_model,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LEGACY_EVIDENCE_FEATURE_CONFIG,
    LocalEvidenceEncoder,
)

_TBPTT = 16


class _FakeDataset:
    """Stand-in for ``ActionSFTDataset`` over a latent-cached frame source."""

    def __init__(self, episodes: dict[int, int]) -> None:
        self._dataset = SimpleNamespace(_ep_vals=tuple(sorted(episodes)))
        self.episodes = episodes


class _FakeProducer:
    """Deterministic whole-block geometry without touching the latent cache."""

    def __init__(self, dataset: _FakeDataset, *, category: str, **_: object) -> None:
        self.frame_source = dataset._dataset
        self.episodes = dataset.episodes
        self.category = category
        self.ttt_tbptt_steps = _TBPTT
        self.produced: list[tuple[str, int, int]] = []

    def block_count(self, stream) -> int:
        return self.episodes[stream.episode_index] // self.ttt_tbptt_steps

    def produce(self, stream, *, cursor: int):
        self.produced.append((self.category, stream.episode_index, cursor))
        return SimpleNamespace(consumer_valid=torch.ones(1, _TBPTT, dtype=torch.bool))


def _model(*, feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG, projector_dim: int = 32):
    encoder = LocalEvidenceEncoder(feature_config=feature_config)
    core = ContinualTTTLocalMemoryCore()
    runtime = torch.nn.Module()
    runtime.evidence_encoder = encoder
    runtime.ttt_core = core
    net = torch.nn.Module()
    net.local_memory_runtime = runtime
    net.local_memory2llm = torch.nn.Linear(projector_dim, 2048)
    net.local_memory_modality_embed = torch.nn.Parameter(torch.zeros(2048))
    model = torch.nn.Module()
    model.net = net
    return model, runtime, encoder, core


def _trainer(*, grad_accum_iter: int):
    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=grad_accum_iter)),
        callbacks=SimpleNamespace(_callbacks=[]),
    )
    trainer.bind_active_local_memory_registry = lambda model, registry: (
        setattr(trainer, "_psm_active_wiring_registry", registry),
        setattr(model, "_psm_active_wiring_registry", registry),
    )
    return trainer


def test_runtime_resolution_fails_closed_without_the_registered_owner() -> None:
    with pytest.raises(RuntimeError, match="net.local_memory_runtime owner"):
        canonical_runtime_module(torch.nn.Module())


def test_adapter_binds_to_the_registered_modules_and_is_cached() -> None:
    model, _, encoder, core = _model()

    adapter = canonical_segment_adapter_from_model(model)

    assert adapter.encoder is encoder and adapter.core is core
    assert canonical_segment_adapter_from_model(model) is adapter


def test_adapter_rejects_a_non_canonical_feature_config() -> None:
    model, _, _, _ = _model(feature_config=LEGACY_EVIDENCE_FEATURE_CONFIG)
    with pytest.raises(RuntimeError, match="non-canonical feature config"):
        canonical_segment_adapter_from_model(model)


def test_adapter_rejects_a_swapped_registered_module() -> None:
    model, runtime, _, _ = _model()
    canonical_segment_adapter_from_model(model)
    runtime.ttt_core = ContinualTTTLocalMemoryCore()  # a second copy the optimizer never sees

    with pytest.raises(RuntimeError, match="not bound to the registered Local-Memory owner"):
        canonical_segment_adapter_from_model(model)


def test_slow_parameters_cover_the_whole_trained_selector_groups() -> None:
    model, runtime, _, _ = _model()

    parameters = canonical_slow_parameters_from_model(model)

    expected = {id(value) for value in runtime.parameters()}
    expected |= {id(value) for value in model.net.local_memory2llm.parameters()}
    expected.add(id(model.net.local_memory_modality_embed))
    assert {id(parameter) for parameter in parameters} == expected


def test_router_dispatches_by_category_and_rejects_unknown_suites() -> None:
    spatial = _FakeProducer(_FakeDataset({7: 32}), category="libero_spatial")
    goal = _FakeProducer(_FakeDataset({3: 32}), category="libero_goal")
    router = SuiteRoutedSegmentProducer(
        {"libero_spatial": spatial, "libero_goal": goal}, source_digest="latent-cache-root"
    )

    assert router.ttt_tbptt_steps == _TBPTT and router.source_digest == "latent-cache-root"
    stream = SimpleNamespace(category="libero_goal", episode_index=3, cursor=1)
    assert router.block_count(stream) == 2
    router.produce(stream, cursor=1)
    assert goal.produced == [("libero_goal", 3, 1)] and spatial.produced == []
    with pytest.raises(ValueError, match="no producer for category"):
        router.block_count(SimpleNamespace(category="libero_object", episode_index=0))


def test_router_rejects_producers_that_disagree_on_the_tbptt_width() -> None:
    spatial = _FakeProducer(_FakeDataset({7: 32}), category="libero_spatial")
    goal = _FakeProducer(_FakeDataset({3: 32}), category="libero_goal")
    goal.ttt_tbptt_steps = 8

    with pytest.raises(ValueError, match="disagree on the TBPTT width"):
        SuiteRoutedSegmentProducer({"libero_spatial": spatial, "libero_goal": goal}, source_digest="digest")


def test_streams_spread_episodes_over_slots_and_skip_short_ones() -> None:
    # Categories sort alphabetically, so libero_goal takes slots {0,2,4,6} and
    # libero_spatial {1,3,5,7}; episode 2 cannot fill one whole block.
    spatial = _FakeProducer(_FakeDataset({0: 32, 1: 32, 2: 4}), category="libero_spatial")
    goal = _FakeProducer(_FakeDataset({5: 32, 6: 32}), category="libero_goal")

    streams = canonical_segment_streams({"libero_spatial": spatial, "libero_goal": goal}, b_stream=8)

    assert [(stream.slot_id, stream.episode_index) for stream in streams] == [
        (0, 5),
        (1, 0),
        (2, 6),
        (3, 1),
    ]
    assert all(
        stream.category == ("libero_goal" if stream.slot_id % 2 == 0 else "libero_spatial") for stream in streams
    )


def test_launch_assembles_the_driver_at_the_accumulation_boundary(monkeypatch) -> None:
    from cosmos_framework.data.generator.action.datasets import canonical_local_memory_producer as producer_module

    monkeypatch.setattr(producer_module, "CanonicalLocalMemorySegmentProducer", _FakeProducer)
    model, _, _, _ = _model()
    trainer = _trainer(grad_accum_iter=24)
    callback = ActiveLocalMemoryLaunchCallback(
        {"libero_spatial": _FakeDataset({0: 32, 1: 32})},
        ttt_tbptt_steps=_TBPTT,
        manifest_digest="manifest",
        config_digest="config",
        source_digest="latent-cache-root",
    )
    callback.trainer = trainer

    callback.on_train_start(model, iteration=0)

    driver = callback.driver
    assert driver is not None and driver.window_members == 24
    assert trainer.callbacks._callbacks == [driver]
    assert trainer._psm_active_wiring_registry is driver.registry
    assert driver.registry.owner.phase.name == "IDLE"
    with pytest.raises(RuntimeError, match="ran twice"):
        callback.on_train_start(model, iteration=0)


def test_launch_requires_a_bound_trainer() -> None:
    model, _, _, _ = _model()
    callback = ActiveLocalMemoryLaunchCallback(
        {"libero_spatial": _FakeDataset({0: 32})},
        manifest_digest="manifest",
        config_digest="config",
        source_digest="latent-cache-root",
    )
    with pytest.raises(RuntimeError, match="requires a bound trainer"):
        callback.on_train_start(model, iteration=0)

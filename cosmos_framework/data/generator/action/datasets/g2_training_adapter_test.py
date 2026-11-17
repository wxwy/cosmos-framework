# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit DATA_C fixtures, not real source/state15 alignment acceptance."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.g2_training_adapter import (
    G2DataRecipe,
    G2TrainingDataAdapter,
    move_g2_views,
)
from cosmos_framework.data.generator.action.utils.static_task_text_test import task_fixture
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_view
from cosmos_framework.model.generator.mot.g2_contracts import G2SampleIdentity, GlobalMemoryConfig


class ProviderFixture:
    identity = "fixture-data"
    evidence_scope = "fixture"

    def __init__(self):
        self.views_read = 0
        self.task = task_fixture()
        self.state = torch.arange(15, dtype=torch.float32)
        self.views = (replace(synthetic_view(), top_variant="with_robot"),)

    def read_common(self, *, task_class, episode_index, decision_frame):
        return dict(
            identity=G2SampleIdentity(
                episode_id=self.task.episode_id,
                scene_id="fixture-scene",
                subtask_id=self.task.subtask_id,
                decision_frame=decision_frame,
                logical_state_frame=decision_frame,
                data_version=self.identity,
                alignment_version="fixture-alignment",
                state15_recipe_id="fixture-data-b",
                task_binding_id="fixture-binding",
            ),
            static_task_text=self.task,
            state15=self.state,
            task_class=task_class,
            episode_index=episode_index,
            source_binding_digest="fixture-source",
            state15_space="data_b_model_input",
        )

    def read_selected_views(self, identity, *, top_view_variant):
        self.views_read += 1
        return tuple(replace(view, top_variant=top_view_variant) for view in self.views)


def setup_fixture(enabled=True):
    provider = ProviderFixture()
    recipe = G2DataRecipe(
        provider.identity,
        "fixture-alignment",
        "fixture-data-b",
        provider.views[0].identity.recipe_id if enabled else None,
        GlobalMemoryConfig(enabled=enabled),
    )
    raw = torch.randn(17, 15)
    raw[0, 5:] = provider.state[5:]
    sample = dict(
        task_class="fixture-task",
        episode_index=0,
        start_frame=3,
        action=raw,
        ai_caption=provider.task.original_instruction,
        source_binding_digest="fixture-source",
    )
    return provider, G2TrainingDataAdapter(provider, recipe), sample


def test_r1_does_not_even_resolve_global_method_or_payload():
    provider, adapter, sample = setup_fixture(False)

    class NoGlobal:
        identity, evidence_scope = provider.identity, provider.evidence_scope
        read_common = provider.read_common

        def __getattribute__(self, name):
            if name in {"read_selected_views", "views"}:
                raise AssertionError("R1 touched Global provider")
            return object.__getattribute__(self, name)

    adapter.provider = NoGlobal()
    result = adapter(sample)
    assert result["g2_views"] == () and result["g2_provider_scope"] == "fixture"
    assert torch.equal(result["action"][1:], sample["action"][1:])
    assert torch.equal(result["action"][0], provider.state)
    assert result["action"].data_ptr() != sample["action"].data_ptr()


def test_r2_reads_selected_variable_views_with_geometry_versions_and_causal_identity():
    provider, adapter, sample = setup_fixture()
    result = adapter(sample)
    assert provider.views_read == 1 and len(result["g2_views"]) == 1
    view = result["g2_views"][0]
    assert view.projection == provider.views[0].projection and view.identity == provider.views[0].identity
    assert torch.equal(result["action"][1:], sample["action"][1:])
    batch = move_g2_views({"g2_views": [result["g2_views"]]}, torch.device("cpu"))
    assert batch["g2_views"][0][0].identity == view.identity


def test_missing_direction_is_normal_but_declared_asset_damage_propagates():
    provider, adapter, sample = setup_fixture()
    assert len(adapter(sample)["g2_views"]) == 1  # No placeholder directions.

    def broken(*args, **kwargs):
        raise FileNotFoundError("declared selected latent is damaged")

    provider.read_selected_views = broken
    with pytest.raises(FileNotFoundError, match="damaged"):
        adapter(sample)


@pytest.mark.parametrize("kind", ["future", "recipe", "eef", "text", "source"])
def test_bad_data_is_not_silently_repaired(kind):
    provider, adapter, sample = setup_fixture()
    if kind == "future":
        view = provider.views[0]
        provider.views = (replace(view, identity=replace(view.identity, image_owner_frame=4)),)
    elif kind == "recipe":
        view = provider.views[0]
        provider.views = (replace(view, identity=replace(view.identity, recipe_id="other")),)
    elif kind == "eef":
        provider.state[5] += 1
    elif kind == "text":
        adapter(sample)
        provider.task = replace(provider.task, enhanced_instruction="New current location, forbidden.")
    else:
        sample["source_binding_digest"] = "other"
    with pytest.raises(ValueError):
        adapter(sample)


def test_common_inputs_r1_r2_are_identical():
    provider, r2, sample = setup_fixture()
    r1 = G2TrainingDataAdapter(
        provider, replace(r2.recipe, global_config=GlobalMemoryConfig(enabled=False), view_recipe_id=None)
    )
    left, right = r1(sample), r2(sample)
    assert torch.equal(left["action"], right["action"])
    assert left["static_task_text"] == right["static_task_text"]
    assert left["g2_sample_identity"] == right["g2_sample_identity"]

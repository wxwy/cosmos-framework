# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Thin consumer of the offline DATA_B/C provider, on the existing G1 dataset.

No provider/producer implementation, directory guessing, VAE, or Pose5 conversion
is imported here. The entry point injects the verified provider. Until DATA_C
exists, tests inject an explicitly synthetic provider at exactly this boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

import torch
from torch.profiler import record_function

from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskText
from cosmos_framework.model.generator.mot.g2_contracts import G2SampleIdentity, GlobalMemoryConfig, SpatialView


class AlignedG2Provider(Protocol):
    identity: str
    evidence_scope: str  # "fixture" or provider-owned verified DATA_C receipt
    latent_channels: int  # small recipe metadata; only queried when Global is enabled

    def read_common(self, *, task_class: str, episode_index: int, decision_frame: int) -> Mapping[str, Any]: ...

    def read_selected_views(self, identity: G2SampleIdentity, *, top_view_variant: str) -> tuple[SpatialView, ...]: ...


@dataclass(frozen=True)
class G2DataRecipe:
    data_identity: str
    alignment_version: str
    state15_recipe_id: str
    view_recipe_id: str | None
    global_config: GlobalMemoryConfig

    def __post_init__(self):
        if not all(
            isinstance(value, str) and value
            for value in (
                self.data_identity,
                self.alignment_version,
                self.state15_recipe_id,
            )
        ):
            raise ValueError("DATA_C/data/state15 recipe identities must be explicit")
        if self.global_config.enabled and not self.view_recipe_id:
            raise ValueError("R2 requires the selected views' actual VAE recipe identity")


class G2TrainingDataAdapter:
    """Attach to RoboCasaExactWindowCachedDataset; no second Dataset/Reader."""

    def __init__(self, provider: AlignedG2Provider, recipe: G2DataRecipe):
        if provider.identity != recipe.data_identity or not provider.evidence_scope:
            raise ValueError("offline provider/data identity mismatch")
        self.provider, self.recipe = provider, recipe
        self._static_ids: OrderedDict[str, str] = OrderedDict()

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        task, episode, frame = (sample[key] for key in ("task_class", "episode_index", "start_frame"))
        with record_function("psm.g2.data_c_common"):
            common = self.provider.read_common(task_class=task, episode_index=episode, decision_frame=frame)
        identity, text, state15 = (common[key] for key in ("identity", "static_task_text", "state15"))
        if isinstance(text, dict):
            text = StaticTaskText(**text)
        if (
            not isinstance(identity, G2SampleIdentity)
            or not isinstance(text, StaticTaskText)
            or common["task_class"] != task
            or common["episode_index"] != episode
            or identity.decision_frame != frame
            or type(identity.decision_frame) is not int
            or type(identity.logical_state_frame) is not int
            or identity.data_version != self.recipe.data_identity
            or identity.alignment_version != self.recipe.alignment_version
            or identity.state15_recipe_id != self.recipe.state15_recipe_id
            or common["source_binding_digest"] != sample["source_binding_digest"]
            or (text.episode_id, text.subtask_id) != (identity.episode_id, identity.subtask_id)
            or text.original_instruction != sample["ai_caption"]
            or common["state15_space"] != "data_b_model_input"
        ):
            raise ValueError("G2 exact-window/common/static text/state15 provenance mismatch")
        if (
            not isinstance(state15, torch.Tensor)
            or state15.shape != (15,)
            or state15.dtype != torch.float32
            or state15.device.type != "cpu"
            or not bool(torch.isfinite(state15).all())
        ):
            raise ValueError("DATA_B must supply a finite CPU fp32 model-input state15")
        original_action = sample["action"]
        if original_action.shape != (17, 15) or not torch.equal(state15[5:], original_action[0, 5:]):
            raise ValueError("DATA_B must preserve the existing EEF10 handling")
        asset_key = identity.episode_id  # Atomic training episode is one subtask.
        previous = self._static_ids.get(asset_key)
        if previous is not None and previous != text.identity:
            raise ValueError("static initial instruction changed within one subtask")
        self._static_ids[asset_key] = text.identity
        self._static_ids.move_to_end(asset_key)
        if len(self._static_ids) > 256:
            self._static_ids.popitem(last=False)
        result = dict(sample)
        result["action"] = original_action.clone()
        result["action"][0] = state15
        result["static_task_text"] = asdict(text)
        result["g2_sample_identity"] = identity
        result["g2_provider_scope"] = self.provider.evidence_scope
        result["g2_views"] = ()
        # R1 NEVER looks up the view method, opens a latent, or creates Reader.
        if self.recipe.global_config.enabled:
            with record_function("psm.g2.data_c_selected_views"):
                views = self.provider.read_selected_views(
                    identity, top_view_variant=self.recipe.global_config.top_view_variant
                )
            if not isinstance(views, tuple) or not views or not all(isinstance(view, SpatialView) for view in views):
                raise ValueError("R2 provider must return selected actual SpatialView entries")
            ids = [view.identity.view_id for view in views]
            if len(set(ids)) != len(ids):
                raise ValueError("selected view identity is duplicated")
            tops = [view for view in views if view.top_variant is not None]
            if len(tops) != 1 or tops[0].top_variant != self.recipe.global_config.top_view_variant:
                raise ValueError("selected top variant is missing or ambiguous")
            for view in views:
                if (
                    view.identity.image_owner_frame > identity.logical_state_frame
                    or view.identity.recipe_id != self.recipe.view_recipe_id
                    or view.latent.device.type != "cpu"
                    or not bool(torch.isfinite(view.latent).all())
                ):
                    raise ValueError("future/corrupt/wrong-recipe selected view payload")
            result["g2_views"] = tuple(sorted(views, key=lambda view: view.identity.view_id))
        return result

    def summary(self):
        return {
            "data_identity": self.recipe.data_identity,
            "alignment_version": self.recipe.alignment_version,
            "state15_recipe_id": self.recipe.state15_recipe_id,
            "view_recipe_id": self.recipe.view_recipe_id,
            "global_memory_enabled": self.recipe.global_config.enabled,
            "top_view_variant": self.recipe.global_config.top_view_variant,
            "provider_scope": self.provider.evidence_scope,
        }


def move_g2_views(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """C0 dataclasses preserve metadata; only selected tensor payloads move."""
    if "g2_views" in batch:
        batch["g2_views"] = [
            tuple(replace(view, latent=view.latent.to(device), valid_mask=view.valid_mask.to(device)) for view in views)
            for views in batch["g2_views"]
        ]
    return batch

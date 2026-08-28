# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contract tests for the R08 causal LIBERO Local-evidence fields.

Set ``PSM_R08_LIBERO_ROOT`` and ``PSM_R08_LATENT_CACHE_ROOT`` to run these
integration tests against the pinned local LIBERO dataset and exact-window cache.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

from cosmos_framework.data.generator.action.action_normalization import normalize_action
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline


def _required_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run R08 causal-history integration tests")
    if not Path(value).is_dir():
        pytest.skip(f"{name} does not exist: {value}")
    return value


@pytest.fixture(scope="module")
def dataset() -> LIBEROLeRobotDataset:
    return LIBEROLeRobotDataset(
        root=_required_path("PSM_R08_LIBERO_ROOT"),
        latent_cache_root=_required_path("PSM_R08_LATENT_CACHE_ROOT"),
        split="full",
        local_history_horizon=16,
    )


def _item(dataset: LIBEROLeRobotDataset, local_start_frame: int) -> dict[str, torch.Tensor]:
    # The first selected episode starts at dataset index zero in split="full".
    return dataset._build_item(local_start_frame)


@pytest.mark.L0
def test_local_history_padding_alignment_and_normalization(dataset: LIBEROLeRobotDataset) -> None:
    start = _item(dataset, 0)
    assert not start["history_mask"].any()
    assert torch.equal(start["history_frame_indices"], torch.full((16,), -1, dtype=torch.long))
    assert torch.equal(start["history_global_row_indices"], torch.full((16,), -1, dtype=torch.long))
    assert torch.count_nonzero(start["local_history_action"]) == 0

    partial = _item(dataset, 3)
    assert partial["history_mask"].tolist() == [False] * 13 + [True] * 3
    assert partial["history_frame_indices"].tolist() == [-1] * 13 + [0, 1, 2]
    assert partial["history_age_steps"].tolist() == [0] * 13 + [3, 2, 1]
    assert partial["history_dt_s"][13:].gt(0).all()

    full = _item(dataset, 16)
    assert full["history_mask"].all()
    assert full["history_frame_indices"].tolist() == list(range(16))
    assert full["history_age_steps"].tolist() == list(range(16, 0, -1))
    source_rows = full["history_global_row_indices"].numpy()
    expected_raw = dataset._build_frame_wise_action(dataset._row_action[source_rows])
    expected_normalized = normalize_action(expected_raw, dataset.action_normalization, dataset._load_norm_stats())
    torch.testing.assert_close(full["local_history_action_raw"], expected_raw)
    torch.testing.assert_close(full["local_history_action"], expected_normalized)
    torch.testing.assert_close(full["history_state_raw"], torch.from_numpy(dataset._row_state[source_rows]).float())
    expected_visual = torch.stack(
        [
            torch.nn.functional.adaptive_avg_pool2d(
                dataset._load_cached_latent(int(dataset._ep_vals[0]), frame)[0].unsqueeze(0), output_size=(1, 2)
            ).flatten()
            for frame in full["history_frame_indices"].tolist()
        ]
    )
    torch.testing.assert_close(full["history_visual_summary"], expected_visual)
    assert int(full["history_global_row_indices"][-1]) + 1 == int(dataset._ep_starts[0]) + 16
    assert int(full["history_frame_indices"][-1]) == 15


@pytest.mark.L0
def test_local_history_h1_is_immediately_prior_frame() -> None:
    dataset = LIBEROLeRobotDataset(
        root=_required_path("PSM_R08_LIBERO_ROOT"),
        latent_cache_root=_required_path("PSM_R08_LATENT_CACHE_ROOT"),
        split="full",
        local_history_horizon=1,
    )
    item = dataset._build_item(3)
    assert item["history_mask"].tolist() == [True]
    assert item["history_frame_indices"].tolist() == [2]
    assert item["history_age_steps"].tolist() == [1]


@pytest.mark.L0
def test_local_history_default_off_has_no_evidence_fields() -> None:
    dataset = LIBEROLeRobotDataset(
        root=_required_path("PSM_R08_LIBERO_ROOT"),
        latent_cache_root=_required_path("PSM_R08_LATENT_CACHE_ROOT"),
        split="full",
    )
    item = dataset._build_item(0)
    assert not any(key.startswith("local_history_") or key.startswith("history_") for key in item)


@pytest.mark.L0
def test_local_history_does_not_enter_native_action_prefix(dataset: LIBEROLeRobotDataset) -> None:
    raw = _item(dataset, 16)
    expected_action = raw["action"].clone()
    expected_length = expected_action.shape[0]
    result = ActionTransformPipeline(tokenizer_config=None, max_action_dim=64)(copy.deepcopy(raw), resolution="256")

    assert "history_action" not in result
    assert "local_history_action" in result
    assert result["action_raw"].shape[0] == expected_length
    torch.testing.assert_close(result["action_raw"], expected_action)
    assert result["sequence_plan"].condition_frame_indexes_action == []

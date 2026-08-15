# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic Edge LIBERO tiny-overfit experiment for G0-R05."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_warmstart import (
    action_policy_libero_edge_warmstart,
)


action_policy_libero_edge_tiny_overfit = copy.deepcopy(action_policy_libero_edge_warmstart)
action_policy_libero_edge_tiny_overfit["job"].update(name="action_policy_libero_edge_tiny_overfit")

_libero_dataset = action_policy_libero_edge_tiny_overfit["dataloader_train"]["dataloader"]["datasets"]["libero"][
    "dataset"
]
_libero_dataset.update(
    split="full",
    iterable_shuffle=False,
    cfg_dropout_rate=0.0,
    tiny_overfit_num_samples=4,
    tiny_overfit_start_index=0,
)
action_policy_libero_edge_tiny_overfit["dataloader_val"] = copy.deepcopy(
    action_policy_libero_edge_tiny_overfit["dataloader_train"]
)
_libero_val_dataset = action_policy_libero_edge_tiny_overfit["dataloader_val"]["dataloader"]["datasets"]["libero"][
    "dataset"
]
_libero_val_dataset.update(
    split="full",
    iterable_shuffle=False,
    cfg_dropout_rate=0.0,
    tiny_overfit_num_samples=1,
    tiny_overfit_start_index=4,
)
action_policy_libero_edge_tiny_overfit["trainer"].update(
    run_validation=True,
    run_validation_on_start=False,
    validation_iter=100,
    max_val_iter=1,
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_tiny_overfit",
    node=action_policy_libero_edge_tiny_overfit,
)

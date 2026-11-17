# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict G1 model-only initialization, including every existing Local tensor.

The only allowed destination-only names are the exact new Global inventory
provided by the model owner. Same-job G2 resume keeps SafeLocal DCP behavior.
No G1 optimizer/scheduler/loader/RNG/episode state is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import SafeLocalDistributedCheckpointer


def validate_g1_model_inventory(
    destination_shapes: Mapping[str, tuple[int, ...]],
    source_shapes: Mapping[str, tuple[int, ...]],
    *,
    new_global_names: frozenset[str],
) -> dict[str, Any]:
    destination, source = set(destination_shapes), set(source_shapes)
    global_prefixes = ("net.global_memory_reader.", "net.global_memory_projector.")
    if not new_global_names.issubset(destination) or any(
        not name.startswith(global_prefixes) for name in new_global_names
    ):
        raise ValueError("new Global inventory must list exact destination net state names")
    if not any(name.startswith("net.local_memory_runtime.") for name in destination):
        raise ValueError("G1 warm-start requires the existing Local runtime")
    required = destination - new_global_names
    missing = required - source
    # EMA is explicitly excluded by the inherited Phase5 model-only contract.
    unexpected = {name for name in source - required if not name.startswith("net_ema.")}
    mismatches = {
        name: (tuple(source_shapes[name]), tuple(destination_shapes[name]))
        for name in required & source
        if tuple(source_shapes[name]) != tuple(destination_shapes[name])
    }
    if missing or unexpected or mismatches:
        raise ValueError(
            f"G1 model inventory mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, shape_mismatches={mismatches}"
        )
    return {
        "loaded_names": tuple(sorted(required)),
        "initialized_global_names": tuple(sorted(new_global_names)),
        "local_loaded_names": tuple(sorted(name for name in required if name.startswith("net.local_memory"))),
        "model_only": True,
    }


def load_g1_model_only(
    model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    new_global_names: frozenset[str],
) -> dict[str, Any]:
    """CPU/small-model reference; strict load after validating *all* keys/shapes."""
    destination = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in (*destination.values(), *source_state.values())):
        raise TypeError("G1 model-only loader expects tensor state dictionaries")
    report = validate_g1_model_inventory(
        {name: tuple(value.shape) for name, value in destination.items()},
        {name: tuple(value.shape) for name, value in source_state.items()},
        new_global_names=new_global_names,
    )
    merged = {name: destination[name] if name in new_global_names else source_state[name] for name in destination}
    model.load_state_dict(merged, strict=True)
    return report


class G2ModelOnlyCheckpointer(SafeLocalDistributedCheckpointer):
    def load(self, model, optimizer=None, scheduler=None, grad_scaler=None):
        _, source = self.keys_to_resume_during_load()
        if source is not None and source.warm_start:
            if self.load_training_state:
                raise ValueError("G1 warm-start must not load training state")
            names = getattr(model, "_g2_new_global_state_names", None)
            if not isinstance(names, frozenset):
                raise ValueError("model owner must declare exact new Global state inventory")

            def validate():
                reader = self.get_storage_reader(source.path.rstrip("/") + "/model", source)
                metadata = reader.read_metadata().state_dict_metadata
                target = get_model_state_dict(model)
                report = validate_g1_model_inventory(
                    {name: tuple(value.shape) for name, value in target.items()},
                    {name: tuple(value.size) for name, value in metadata.items() if hasattr(value, "size")},
                    new_global_names=names,
                )
                # The inherited DCP planner uses substring skips. Prove the
                # exact list cannot accidentally skip any existing tensor.
                skips = ["net_ema.", *sorted(names)]
                skipped = {name for name in target if any(part in name for part in skips)}
                expected = names | {name for name in target if name.startswith("net_ema.")}
                if skipped != expected:
                    raise ValueError("Global skip names overlap existing model state")
                return report, skips

            report, skips = self._checked("G2 validate G1 model-only inventory", validate)
            self.config_checkpoint.keys_to_skip_loading = skips
            self.keys_to_skip_loading = skips
            self.g2_warm_start_report = report
        return super().load(model, optimizer, scheduler, grad_scaler)

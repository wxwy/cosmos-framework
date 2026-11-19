# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real single-process CPU DCP metadata/planner; no FSDP/GPU/real checkpoint."""

from types import SimpleNamespace

import pytest
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, SafeLocalDistributedCheckpointer
from cosmos_framework.checkpoint.g2_warm_start import G2ModelOnlyCheckpointer
from cosmos_framework.trainer.g2_training_test import TinyPolicy, equal_tree
from cosmos_framework.utils.config import CheckpointConfig


@pytest.mark.parametrize("damage", [None, "missing_local", "shape"])
def test_actual_cpu_dcp_model_only_global_allowlist_and_local_inheritance(tmp_path, monkeypatch, damage):
    source_model, destination = TinyPolicy(False), TinyPolicy(True)
    source = dict(source_model.state_dict())
    if damage == "missing_local":
        source.pop("net.local_memory_runtime.core.w0_fast_in_weight")
    if damage == "shape":
        source["net.moe_gen.weight"] = source["net.moe_gen.weight"][:1]
    dcp.save(source, checkpoint_id=tmp_path / "model", no_dist=True)
    destination._g2_new_global_state_names = frozenset(
        name
        for name in destination.state_dict()
        if name.startswith(("net.global_memory_reader.", "net.global_memory_projector."))
    )
    checkpointer = object.__new__(G2ModelOnlyCheckpointer)
    checkpointer.load_training_state = False
    original_config = CheckpointConfig(keys_to_skip_loading=["net_ema.", "local_memory"])
    original_config.freeze()  # Trainer freezes configs before actual Phase5 load.
    checkpointer.config_checkpoint = original_config
    checkpointer.keys_to_resume_during_load = lambda: ({"model"}, SimpleNamespace(path=str(tmp_path), warm_start=True))
    checkpointer.get_storage_reader = lambda path, source: dcp.FileSystemReader(path)
    checkpointer._checked = lambda label, operation: operation()
    called = []

    def cpu_native_load(self, model, optimizer, scheduler, grad_scaler):
        called.append(1)
        assert optimizer is scheduler is grad_scaler is None
        target = get_model_state_dict(model)
        dcp.load(
            target,
            checkpoint_id=tmp_path / "model",
            planner=CustomLoadPlanner(
                allow_partial_load=False, keys_to_skip_loading=self.config_checkpoint.keys_to_skip_loading
            ),
            no_dist=True,
        )
        set_model_state_dict(model, target, options=StateDictOptions(strict=True))
        return 0

    # Only the distributed orchestration shell is isolated. Actual CPU DCP
    # write/read metadata, exact key validation, strict planner and load run.
    monkeypatch.setattr(SafeLocalDistributedCheckpointer, "load", cpu_native_load)
    if damage:
        with pytest.raises(ValueError, match="inventory mismatch"):
            checkpointer.load(destination)
        assert not called
        assert checkpointer.config_checkpoint is original_config
    else:
        assert checkpointer.load(destination) == 0
        assert checkpointer.config_checkpoint is not original_config
        assert checkpointer.config_checkpoint._is_frozen is True
        with pytest.raises(AttributeError, match="Cannot modify frozen"):
            checkpointer.config_checkpoint.keys_to_skip_loading = ["unsafe"]
        for name, value in source.items():
            equal_tree(value, destination.state_dict()[name])
        assert "local_memory" not in checkpointer.keys_to_skip_loading
        assert checkpointer.g2_warm_start_report["local_loaded_names"]
    assert original_config.keys_to_skip_loading == ["net_ema.", "local_memory"]
    assert original_config._is_frozen is True

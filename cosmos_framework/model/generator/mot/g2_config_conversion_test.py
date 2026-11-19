# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression for native Phase5 OmegaConf-to-GlobalMemoryConfig admission."""

import pytest
from omegaconf import OmegaConf

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetworkConfig,
)
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig


@pytest.mark.parametrize("enabled", [False, True])
def test_native_phase5_dictconfig_has_only_public_memory_fields(enabled):
    # Preflight produces a typed OmniMoT DictConfig. During model setup it is
    # passed as-is to Cosmos3VFMNetworkConfig, not converted to a plain dict.
    given = OmegaConf.create(
        {
            "enabled": enabled,
            "k_global": 64,
            "hidden_dim": 512,
            "num_attention_heads": 8,
            "ffn_hidden_dim": 2048,
            "num_language_blocks": 1,
            "num_visual_blocks": 2,
            "top_view_variant": "without_robot",
        }
    )
    assert "_metadata" in vars(given)  # Original vars(DictConfig) bug is real.
    config = Cosmos3VFMNetworkConfig(
        global_memory_enabled=enabled, global_memory_config=given
    )
    assert config.global_memory_config == OmegaConf.to_container(given, resolve=True)
    assert config.global_memory_config["enabled"] is enabled
    assert "_metadata" not in config.global_memory_config


def test_plain_dict_dataclass_and_none_contracts_are_preserved():
    expected = vars(GlobalMemoryConfig(enabled=True, top_view_variant="with_robot"))
    for value in (
        dict(expected),
        GlobalMemoryConfig(enabled=True, top_view_variant="with_robot"),
    ):
        config = Cosmos3VFMNetworkConfig(
            global_memory_enabled=True, global_memory_config=value
        )
        assert config.global_memory_config == expected
    default = Cosmos3VFMNetworkConfig(global_memory_enabled=False)
    assert default.global_memory_config == vars(GlobalMemoryConfig())


def test_phase5_global_flag_conflict_and_unknown_fields_fail_closed():
    with pytest.raises(ValueError, match="disagree"):
        Cosmos3VFMNetworkConfig(
            global_memory_enabled=False,
            global_memory_config=OmegaConf.create({"enabled": True}),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        Cosmos3VFMNetworkConfig(
            global_memory_enabled=True,
            global_memory_config=OmegaConf.create({"enabled": True, "_metadata": {}}),
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        Cosmos3VFMNetworkConfig(
            global_memory_enabled=True,
            global_memory_config=OmegaConf.create(["bad"]),
        )

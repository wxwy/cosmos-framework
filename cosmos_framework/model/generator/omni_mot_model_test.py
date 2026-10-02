# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
        lidar_tokenizer=None,
        radar_tokenizer=None,
        sound_gen=False,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )
    instantiated = []

    def _instantiate(candidate):
        instantiated.append(candidate)
        return vlm_processor

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert instantiated == [vlm_config.tokenizer]
    assert model.tokenizer_vision_gen is None
    assert model.tokenizer_sound_gen is None


def test_default_setup_loads_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vision_tokenizer = SimpleNamespace(latent_ch=48, reset_dtype=Mock())
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=True,
        lidar_tokenizer=None,
        radar_tokenizer=None,
        sound_gen=False,
        state_ch=48,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )

    def _instantiate(candidate):
        if candidate == vlm_config.tokenizer:
            return vlm_processor
        assert candidate is vision_config
        return vision_tokenizer

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert model.tokenizer_vision_gen is vision_tokenizer
    vision_tokenizer.reset_dtype.assert_called_once_with()


def test_velocity_repack_preserves_local_memory_tokens() -> None:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    local = torch.randn(4, 32)
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 1, 1)],
        x0_tokens_local_memory=[local],
    )
    captured = {}

    def stop_at_pack(sequence_plans, text_tokens, repacked, *args, **kwargs):
        captured["local"] = repacked.x0_tokens_local_memory
        raise RuntimeError("captured-repack")

    holder = SimpleNamespace(
        config=SimpleNamespace(action_gen=False, sound_gen=False),
        _pack_input_sequence=stop_at_pack,
        _derive_include_end_of_generation_token=lambda: False,
    )
    plan = SimpleNamespace(has_action=False)
    with pytest.raises(RuntimeError, match="captured-repack"):
        OmniMoTModel._get_velocity(
            holder,
            noise_x=[torch.zeros(1)],
            timestep=torch.tensor([[0.5]]),
            text_tokens=[[1]],
            sequence_plans=[plan],
            gen_data_clean=clean,
            has_noisy_actions=False,
        )
    assert captured["local"] is clean.x0_tokens_local_memory
    torch.testing.assert_close(captured["local"][0], local)

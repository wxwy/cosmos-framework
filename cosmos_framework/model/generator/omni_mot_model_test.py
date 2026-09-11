# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
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


def test_canonical_marker_has_disable_first_precedence_without_legacy_lifecycle() -> None:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    lifecycle = SimpleNamespace(process_sample=Mock())
    enabled = SimpleNamespace(
        config=SimpleNamespace(local_ttt_enabled=True),
        _canonical_local_memory_segment_forward=Mock(),
        _ttt_lifecycle=lifecycle,
    )
    with pytest.raises(ValueError, match="exact canonical-production mode"):
        OmniMoTModel.training_step(enabled, {"canonical_local_memory_segment": True}, 0)
    enabled._canonical_local_memory_segment_forward.assert_not_called()
    lifecycle.process_sample.assert_not_called()

    disabled = SimpleNamespace(
        config=SimpleNamespace(local_ttt_enabled=False),
        _canonical_local_memory_segment_forward=Mock(side_effect=AssertionError("canonical marker must be disabled")),
        _get_training_inputs=Mock(),
    )
    with pytest.raises(ValueError, match="canonical or legacy Local-Memory markers"):
        OmniMoTModel.training_step(disabled, {"canonical_local_memory_segment": True}, 0)
    disabled._canonical_local_memory_segment_forward.assert_not_called()
    disabled._get_training_inputs.assert_not_called()


def test_canonical_adapter_binds_only_the_registered_ttt_owner() -> None:
    import torch

    from cosmos_framework.model.generator.mot.local_evidence import (
        CANONICAL_EVIDENCE_FEATURE_CONFIG,
        ContinualTTTLocalMemoryCore,
        LocalEvidenceEncoder,
    )
    from cosmos_framework.model.generator.omni_mot_model import _canonical_production_adapter_from_model

    runtime = torch.nn.Module()
    runtime.evidence_encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
    runtime.ttt_core = ContinualTTTLocalMemoryCore()
    model = SimpleNamespace(net=SimpleNamespace(local_memory_runtime=runtime))

    adapter = _canonical_production_adapter_from_model(model)
    assert adapter.encoder is runtime.evidence_encoder
    assert adapter.core is runtime.ttt_core
    assert _canonical_production_adapter_from_model(model) is adapter
    model.net.local_history_runtime = SimpleNamespace(encoder=runtime.evidence_encoder, recurrent_backend=runtime.ttt_core)
    assert _canonical_production_adapter_from_model(model) is adapter

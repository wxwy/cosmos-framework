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


def test_build_net_registers_only_active_ttt_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from cosmos_framework.model.generator import omni_mot_model

    class _Network(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def to_empty(self, **_: object) -> "_Network":
            return self

    config = SimpleNamespace(
        lora_enabled=False,
        lbl=SimpleNamespace(method="none", coeff_und=None, coeff_gen=None),
        rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
        diffusion_expert_config=SimpleNamespace(
            patch_spatial=2, max_vae_latent_side_after_patchify=20, enable_fps_modulation=False,
            enable_vision_modality_embeddings=False, enable_media_modality_embedding=False,
            base_fps=24, timestep_range=1.0,
        ),
        latent_downsample_factor=8,
        state_ch=16,
        state_t=4,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        joint_attn_implementation="eager",
        flex_attention=SimpleNamespace(enabled=False, backend="", mask=SimpleNamespace(noisy_attention_scope="all")),
        max_action_dim=32,
        num_embodiment_domains=1,
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        natten_parameter_list=(),
        video_temporal_causal=False,
        sound_dim=None,
        sound_latent_fps=25,
        local_memory_enabled=True,
        local_memory_dim=32,
        enable_input_bias=True,
        local_history_enabled=True,
        local_history_backend="ttt_fast_weight",
        local_ttt_enabled=True,
        local_history_evidence_dim=8,
        ttt_inner_lr=0.1,
        ttt_tbptt_steps=16,
        k_local=1,
        compile=SimpleNamespace(use_cuda_graphs=False),
        quantization=SimpleNamespace(modelopt_fp8_checkpoint_path=None),
        activation_checkpointing=SimpleNamespace(),
    )
    model = SimpleNamespace(
        config=config,
        vlm_config=SimpleNamespace(model_instance=object()),
        tokenizer_vision_gen=None,
        parallel_dims=object(),
        install_attention_dispatch=lambda _: None,
    )
    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", lambda *_: SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr(omni_mot_model, "Cosmos3VFMNetworkConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(omni_mot_model, "Cosmos3VFMNetwork", lambda **kwargs: _Network(**kwargs))
    monkeypatch.setattr(omni_mot_model, "parallelize_vfm_network", lambda net, **_: net)
    monkeypatch.setattr(omni_mot_model, "DEVICE", omni_mot_model.Device.CPU)

    net = omni_mot_model.OmniMoTModel.build_net(model, torch.float32)

    assert tuple(net.local_memory_runtime._modules) == ("evidence_encoder", "ttt_core")
    assert not hasattr(net, "local_history_runtime")
    assert not hasattr(net, "readout")

"""H1 actual OmegaConf containers at the loaded Wan identity boundary (CPU only)."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from cosmos_framework.inference import robocasa_vae_daemon as daemon
from cosmos_framework.inference.robocasa_vae_protocol import canonical_json, strict_json


def _tokenizer(*, omega=False, durations=(17, 61, 73)):
    frames = {"256": 68, "480": 24, "720": 8}
    durations = None if durations is None else list(durations)
    module = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    with torch.no_grad():
        module.weight.fill_(0.25)
    module.temporal_window = OmegaConf.create(frames) if omega else deepcopy(frames)
    module._encode_exact_durations = set(durations or [])
    return SimpleNamespace(
        chunk_duration=93,
        encode_chunk_frames=OmegaConf.create(frames) if omega else deepcopy(frames),
        encode_exact_durations=OmegaConf.create(durations) if omega and durations is not None else durations,
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        is_causal=True,
        use_streaming_encode=False,
        _keep_encoder_cache=False,
        model=SimpleNamespace(
            model=module,
            dtype=torch.bfloat16,
            scale=(torch.ones(48, dtype=torch.bfloat16), torch.ones(48, dtype=torch.bfloat16)),
        ),
    )


def test_real_omegaconf_constructor_equals_native_without_mutating_order_or_values():
    tokenizer = _tokenizer(omega=True, durations=(73, 17, 61, 17))
    assert isinstance(tokenizer.encode_exact_durations, ListConfig)
    assert isinstance(tokenizer.encode_chunk_frames, DictConfig)
    before = OmegaConf.to_container(tokenizer.encode_exact_durations)
    actual = daemon.worker_constructor_config(tokenizer)
    expected = daemon.worker_constructor_config(_tokenizer(durations=before))
    assert actual == expected and canonical_json(actual) == canonical_json(expected)
    assert type(actual["encode_exact_durations"]) is list
    assert type(actual["encode_chunk_frames"]) is dict
    assert actual["encode_exact_durations"] == [73, 17, 61, 17]
    actual["encode_exact_durations"][0] = 99
    actual["encode_chunk_frames"]["256"] = 100
    assert OmegaConf.to_container(tokenizer.encode_exact_durations) == before
    assert tokenizer.encode_chunk_frames["256"] == 68


def test_real_omegaconf_loaded_identity_matches_native_and_preserves_weight_authority():
    tokenizer = _tokenizer(omega=True)
    expected = daemon.loaded_vae_identity(_tokenizer())
    actual = daemon.loaded_vae_identity(tokenizer)
    assert actual == expected
    constructor = daemon.worker_constructor_config(tokenizer, vae_path="/cpu/not-a-checkpoint")
    assert daemon.loaded_vae_identity(tokenizer, constructor) == expected
    with torch.no_grad():
        tokenizer.model.model.weight[0, 0].add_(1)
    changed = daemon.loaded_vae_identity(tokenizer)
    assert changed["vae_weights_sha256"] != expected["vae_weights_sha256"]
    assert changed["vae_config_sha256"] == expected["vae_config_sha256"]


@pytest.mark.parametrize("durations", [None, (), (17, 61, 73)])
def test_optional_and_empty_durations_keep_explicit_semantics(durations):
    native, omega = _tokenizer(durations=durations), _tokenizer(omega=True, durations=durations)
    assert daemon.worker_constructor_config(omega) == daemon.worker_constructor_config(native)
    assert daemon.loaded_vae_identity(omega) == daemon.loaded_vae_identity(native)
    value = daemon.worker_constructor_config(omega)["encode_exact_durations"]
    assert value == (None if durations is None else list(durations))


def test_interpolation_uses_original_effective_omegaconf_parent_context():
    config = OmegaConf.create({"duration": 17, "values": ["${duration}", 61, 73]})
    tokenizer = _tokenizer()
    tokenizer.encode_exact_durations = config["values"]
    assert daemon.worker_constructor_config(tokenizer)["encode_exact_durations"] == [17, 61, 73]


@pytest.mark.parametrize("omega", [False, True])
@pytest.mark.parametrize("value", [[True], [17.0], ["17"], [0], [-1], [[17]], {"0": 17}])
def test_ambiguous_or_invalid_duration_settings_fail_closed(value, omega):
    tokenizer = _tokenizer()
    tokenizer.encode_exact_durations = OmegaConf.create(value) if omega else value
    with pytest.raises(ValueError):
        daemon.worker_constructor_config(tokenizer)


@pytest.mark.parametrize("value", [object(), {17}, [object()]])
def test_unsupported_duration_objects_are_not_stringified(value):
    tokenizer = _tokenizer()
    tokenizer.encode_exact_durations = value
    with pytest.raises(ValueError):
        daemon.worker_constructor_config(tokenizer)


@pytest.mark.parametrize("omega", [False, True])
@pytest.mark.parametrize(
    "value", [{"256": True}, {"256": 68.0}, {"256": "68"}, {"256": 0}, {256: 68}, {"": 68}, {"256": 67}]
)
def test_invalid_chunk_mapping_settings_fail_closed(value, omega):
    tokenizer = _tokenizer()
    tokenizer.encode_chunk_frames = OmegaConf.create(value) if omega else value
    with pytest.raises(ValueError):
        daemon.worker_constructor_config(tokenizer)


def test_missing_omegaconf_values_and_oversized_configuration_fail_closed():
    tokenizer = _tokenizer()
    tokenizer.encode_exact_durations = OmegaConf.create(["???"])
    with pytest.raises(ValueError):
        daemon.worker_constructor_config(tokenizer)
    tokenizer.encode_exact_durations = [17] * 32768
    with pytest.raises(ValueError):
        daemon.worker_constructor_config(tokenizer)


def test_changed_config_is_hashed_and_resolved_mismatch_is_rejected():
    tokenizer = _tokenizer(omega=True)
    original = daemon.loaded_vae_identity(tokenizer)
    supplied = daemon.worker_constructor_config(tokenizer)
    tokenizer.encode_exact_durations[0] = 21
    changed = daemon.loaded_vae_identity(tokenizer)
    assert changed["vae_weights_sha256"] == original["vae_weights_sha256"]
    assert changed["vae_config_sha256"] != original["vae_config_sha256"]
    with pytest.raises(ValueError, match="differs"):
        daemon.loaded_vae_identity(tokenizer, supplied)


def test_worker_loader_matches_real_policy_constructor_and_rejects_changed_identity(tmp_path, monkeypatch):
    policy = _tokenizer(omega=True)
    identity = daemon.loaded_vae_identity(policy)
    constructor = daemon.worker_constructor_config(policy)
    received = []

    def cpu_wan(**kwargs):
        received.append(kwargs)
        worker = _tokenizer(durations=kwargs["encode_exact_durations"])
        worker.encode_chunk_frames = kwargs["encode_chunk_frames"]
        worker.model.model.temporal_window = kwargs["encode_chunk_frames"]
        return worker

    target = "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16"
    monkeypatch.setitem(sys.modules, target, SimpleNamespace(Wan2pt2VAEInterface=cpu_wan))
    from cosmos_framework.inference import robocasa_vae_encoder

    monkeypatch.setattr(robocasa_vae_encoder, "LoadedWanEncode1", lambda tokenizer, identity: (tokenizer, identity))
    weights = tmp_path / "synthetic-vae.bin"
    weights.write_bytes(b"cpu-constructor-fixture-only")
    config = strict_json(
        canonical_json(
            {
                "vae_weights": str(weights),
                "vae_file_sha256": daemon.file_sha256(weights),
                "identity": identity,
                "constructor_config": constructor,
            }
        )
    )
    (_, worker_identity), proof = daemon.load_real_encoder(config)
    assert worker_identity == identity == proof["identity"]
    assert received[0]["encode_exact_durations"] == [17, 61, 73]
    assert type(received[0]["encode_exact_durations"]) is list
    config["constructor_config"]["encode_exact_durations"][0] = 21
    with pytest.raises(ValueError, match="does not match"):
        daemon.load_real_encoder(config)

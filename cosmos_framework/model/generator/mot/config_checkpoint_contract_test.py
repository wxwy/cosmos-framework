import pytest
import torch
from torch import nn

from .config_checkpoint_contract import LocalMemoryConfig, canonical_slow_inventory, slow_checkpoint_payload, strict_restore, strict_restore_into, validate_exact_optimizer_membership, validate_optimizer_membership, validate_slow_inventory


class _Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.readout = nn.Linear(2, 2)
        self.recurrent_backend = nn.Linear(2, 2)


def test_config_defaults_and_validation() -> None:
    config = LocalMemoryConfig(); config.validate()
    assert config.ttt_tbptt_steps == 16 and config.inner_lr == 0.1 and config.k_local == 1
    LocalMemoryConfig(ttt_tbptt_steps=1, k_local=8).validate()


@pytest.mark.parametrize("kwargs", [{"inner_lr": 0}, {"inner_lr": float("nan")}, {"ttt_tbptt_steps": 0}, {"runtime_evidence_steps": 2}, {"ttt_tbptt_steps": True}, {"k_local": 2}, {"inner_lr": True}, {"runtime_evidence_steps": 1.0}])
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LocalMemoryConfig(**kwargs).validate()


def test_inventory_requires_identity_and_excludes_dormant_readout() -> None:
    owner = _Owner()
    owner.readout = nn.Identity(); owner.recurrent_backend = nn.Identity()
    names = validate_slow_inventory(owner, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend)
    assert names and all(not name.startswith("readout.") for name in names)


def test_inventory_rejects_duplicate_runtime_objects() -> None:
    owner = _Owner()
    with pytest.raises(ValueError, match="registered"):
        validate_slow_inventory(owner, runtime_encoder=nn.Linear(2, 2), runtime_backend=owner.recurrent_backend)


def test_checkpoint_is_slow_only_and_cloned() -> None:
    config = LocalMemoryConfig(k_local=4)
    value = torch.ones(2)
    payload = slow_checkpoint_payload({"recurrent_backend.q.weight": value}, config)
    assert payload["config"] == config and payload["parameters"]["recurrent_backend.q.weight"] is not value
    with pytest.raises(ValueError, match="fast"):
        slow_checkpoint_payload({"fast_state": value}, config)
    expected = {"recurrent_backend.q.weight": value}
    assert torch.equal(strict_restore(payload, expected, config)["recurrent_backend.q.weight"], value)
    owner = _Owner()
    owner.readout = nn.Identity(); owner.recurrent_backend = nn.Identity()
    own_expected = {f"local_history_runtime.{name}": value.detach().clone() for name, value in owner.named_parameters()}
    own_payload = slow_checkpoint_payload(own_expected, config)
    strict_restore_into(owner, own_payload, own_expected, config, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend)
    for bad in ({}, {**payload["parameters"], "extra": value}, {"recurrent_backend.q.weight": torch.ones(3)}, {"recurrent_backend.q.weight": value.to(torch.float64)}):
        with pytest.raises(ValueError):
            strict_restore({"config": config, "parameters": bad}, expected, config)


def test_strict_restore_round_trips_all_four_slow_groups_into_same_objects() -> None:
    config = LocalMemoryConfig(k_local=4)
    owner = _Owner()
    owner.readout = nn.Identity()
    projector = nn.Linear(2, 2)
    modality = nn.Parameter(torch.ones(2))
    expected = canonical_slow_inventory(owner, projector, modality)
    assert set(expected) == {"local_history_runtime.encoder.weight", "local_history_runtime.encoder.bias", "local_history_runtime.recurrent_backend.weight", "local_history_runtime.recurrent_backend.bias", "local_memory2llm.weight", "local_memory2llm.bias", "local_memory_modality_embed"}
    payload = slow_checkpoint_payload(expected, config)
    snapshot = {name: value.detach().clone() for name, value in expected.items()}
    encoder_ref, backend_ref = owner.encoder, owner.recurrent_backend
    with torch.no_grad():
        for value in expected.values():
            value.add_(torch.full_like(value, 3.0))
    assert any(not torch.equal(value, snapshot[name]) for name, value in expected.items())
    strict_restore_into(owner, payload, expected, config, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend, local_memory2llm=projector, modality=modality)
    for name, value in expected.items():
        assert torch.equal(value, snapshot[name]), name
    assert owner.encoder is encoder_ref and owner.recurrent_backend is backend_ref
    assert canonical_slow_inventory(owner, projector, modality)["local_memory_modality_embed"] is modality
    with pytest.raises(ValueError, match="local_memory2llm"):
        strict_restore_into(owner, payload, expected, config, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend)
    with pytest.raises(ValueError, match="local_memory_modality_embed"):
        strict_restore_into(owner, payload, expected, config, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend, local_memory2llm=projector)


def test_optimizer_membership_is_exact() -> None:
    values = {"local_history_runtime.encoder.weight": torch.ones(1), "local_history_runtime.recurrent_backend.weight": torch.ones(1)}
    external = {"local_memory2llm.weight": torch.ones(1), "local_memory_modality_embed": torch.ones(1)}
    assert len(validate_optimizer_membership(values, external)) == 4
    with pytest.raises(ValueError):
        validate_optimizer_membership(values, {})
    with pytest.raises(ValueError):
        validate_optimizer_membership({**values, "other.weight": torch.ones(1)}, external)
    owner = _Owner(); owner.readout = nn.Identity(); owner.recurrent_backend = nn.Identity()
    expected = canonical_slow_inventory(owner, nn.Linear(2, 2), nn.Parameter(torch.ones(2)))
    validate_exact_optimizer_membership(expected, expected)
    with pytest.raises(ValueError):
        validate_exact_optimizer_membership({key: value for key, value in expected.items() if key != next(iter(expected))}, expected)

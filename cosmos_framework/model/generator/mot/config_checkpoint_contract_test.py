import pytest
import torch
from torch import nn

from .config_checkpoint_contract import LocalMemoryConfig, slow_checkpoint_payload, validate_slow_inventory


class _Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.readout = nn.Linear(2, 2)
        self.recurrent_backend = nn.Linear(2, 2)


def test_config_defaults_and_validation() -> None:
    config = LocalMemoryConfig(); config.validate()
    assert config.ttt_tbptt_steps == 16 and config.inner_lr == 0.1 and config.k_local == 1


@pytest.mark.parametrize("kwargs", [{"inner_lr": 0}, {"inner_lr": float("nan")}, {"ttt_tbptt_steps": 0}, {"runtime_evidence_steps": 2}])
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LocalMemoryConfig(**kwargs).validate()


def test_inventory_requires_identity_and_excludes_dormant_readout() -> None:
    owner = _Owner()
    with pytest.raises(ValueError, match="dormant"):
        validate_slow_inventory(owner, runtime_encoder=owner.encoder, runtime_backend=owner.recurrent_backend)
    owner.readout = nn.Identity()
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

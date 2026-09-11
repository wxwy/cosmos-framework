from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import CanonicalProductionAdapter
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)

from .config_checkpoint_contract import (
    LocalMemoryConfig,
    canonical_slow_inventory,
    slow_checkpoint_payload,
    strict_restore,
    strict_restore_into,
    validate_exact_optimizer_membership,
    validate_optimizer_membership,
    validate_runtime_admission,
    validate_slow_inventory,
)


class _RuntimeRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evidence_encoder = LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
        self.ttt_core = ContinualTTTLocalMemoryCore()


def _fixture() -> tuple[_RuntimeRoot, nn.Linear, nn.Parameter, CanonicalProductionAdapter, CanonicalBatchScheduler]:
    root = _RuntimeRoot()
    projector, modality = nn.Linear(32, 2048), nn.Parameter(torch.zeros(2048))
    adapter = CanonicalProductionAdapter(root.evidence_encoder, root.ttt_core)
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    return root, projector, modality, adapter, scheduler


def _restore(root: _RuntimeRoot, projector: nn.Linear, modality: nn.Parameter, adapter: CanonicalProductionAdapter, scheduler: CanonicalBatchScheduler, payload: dict[str, object], expected: dict[str, nn.Parameter], **kwargs: object) -> None:
    strict_restore_into(root, payload, expected, LocalMemoryConfig(), runtime_encoder=root.evidence_encoder, runtime_core=root.ttt_core, local_memory2llm=projector, modality=modality, adapter=adapter, scheduler=scheduler, **kwargs)


def test_config_identity_is_versioned_deterministic_and_fail_closed() -> None:
    config = LocalMemoryConfig()
    config.validate()
    mapping = config.to_mapping()
    assert LocalMemoryConfig.from_mapping(mapping) == config
    for kwargs in ({"ttt_inner_lr": 0}, {"ttt_inner_lr": True}, {"ttt_tbptt_steps": True}, {"k_local": 4}, {"local_fast_state_dtype": "bf16"}, {"local_runtime_resume_mode": "resume"}):
        with pytest.raises(ValueError):
            LocalMemoryConfig(**kwargs).validate()
    for bad in ({key: value for key, value in mapping.items() if key != "k_local"}, {**mapping, "runtime_evidence_steps": 1}):
        with pytest.raises(ValueError):
            LocalMemoryConfig.from_mapping(bad)


def test_active_owner_inventory_selectors_and_adapter_are_exact() -> None:
    root, projector, modality, adapter, _ = _fixture()
    names = validate_slow_inventory(root, runtime_encoder=root.evidence_encoder, runtime_core=root.ttt_core, adapter=adapter)
    expected = canonical_slow_inventory(root, projector, modality)
    assert set(names) < set(expected)
    assert "local_memory_runtime.ttt_core.w0_fast_in_weight" in expected
    assert "local_memory_runtime.ttt_core.query_proj.weight" in expected
    validate_optimizer_membership(expected, expected)
    with pytest.raises(ValueError, match="exact"):
        validate_exact_optimizer_membership({**expected, "local_memory2llm.alias": expected["local_memory2llm.weight"]}, expected)
    root.readout = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="exactly"):
        validate_slow_inventory(root, runtime_encoder=root.evidence_encoder, runtime_core=root.ttt_core, adapter=adapter)


def test_slow_payload_rejects_runtime_keys_and_stages_without_mutation() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = slow_checkpoint_payload(expected, LocalMemoryConfig())
    assert strict_restore(payload, expected, LocalMemoryConfig())
    with pytest.raises(ValueError, match="runtime"):
        slow_checkpoint_payload({"local_memory_runtime.frontier": next(iter(expected.values()))}, LocalMemoryConfig())
    before = {name: value.detach().clone() for name, value in expected.items()}
    damaged = {**payload, "parameters": {**payload["parameters"], "local_memory2llm.weight": torch.zeros(1)}}
    with pytest.raises(ValueError, match="tensor"):
        _restore(root, projector, modality, adapter, scheduler, damaged, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_preflight_is_atomic_and_real_authorities_reject_live_state() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = slow_checkpoint_payload(expected, LocalMemoryConfig())
    snapshot = {name: value.detach().clone() for name, value in expected.items()}
    with torch.no_grad():
        for value in expected.values():
            value.add_(1)
    _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, snapshot[name]) for name, value in expected.items())
    adapter._scan_requests.add(1)
    before = {name: value.detach().clone() for name, value in expected.items()}
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())
    adapter._scan_requests.clear()
    scheduler._frozen_transitions.append(SimpleNamespace())
    with pytest.raises(ValueError, match="frozen"):
        validate_runtime_admission(adapter=adapter, scheduler=scheduler)
    scheduler._frozen_transitions.clear()
    validate_runtime_admission(adapter=adapter, scheduler=scheduler)

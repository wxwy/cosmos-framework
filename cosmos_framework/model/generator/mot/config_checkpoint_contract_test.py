import copy

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    CatalogRow,
    ChronologyCountRecord,
    MicrobatchPlanMember,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalProductionAdapter,
    CanonicalProductionSegmentRequest,
    build_canonical_native_loss_split,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_integration_test import (
    _bound_request_and_carrier,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import (
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)

from .config_checkpoint_contract import (
    FeatureConfigIdentity,
    LocalMemoryConfig,
    build_base_identity,
    canonical_slow_inventory,
    slow_checkpoint_payload,
    strict_restore,
    strict_restore_into,
    validate_exact_optimizer_membership,
    validate_optimizer_membership,
    validate_pristine_progress,
    validate_runtime_admission,
    validate_slow_inventory,
)


class _RuntimeRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evidence_encoder = LocalEvidenceEncoder(evidence_dim=8, feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG)
        self.ttt_core = ContinualTTTLocalMemoryCore(evidence_dim=8, ttt_dim=4, fast_hidden_dim=8)


def _fixture() -> tuple[_RuntimeRoot, nn.Linear, nn.Parameter, CanonicalProductionAdapter, CanonicalBatchScheduler]:
    root = _RuntimeRoot()
    projector, modality = nn.Linear(32, 2048), nn.Parameter(torch.zeros(2048))
    adapter = CanonicalProductionAdapter(root.evidence_encoder, root.ttt_core)
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    return root, projector, modality, adapter, scheduler


def _base_identity() -> dict[str, object]:
    return build_base_identity(
        child_git_revision="f" * 40,
        feature_config=FeatureConfigIdentity(),
        checkpoint_source_fingerprint="synthetic-source",
        manifest_sha256="m" * 64,
        source_sha256="s" * 64,
    )


def _feature_config(config: LocalMemoryConfig = LocalMemoryConfig()) -> FeatureConfigIdentity:
    return FeatureConfigIdentity(
        ttt_tbptt_steps=config.ttt_tbptt_steps,
        ttt_inner_lr=config.ttt_inner_lr,
        k_local=config.k_local,
        local_evidence_feature_version=config.local_evidence_feature_version,
        local_fast_state_dtype=config.local_fast_state_dtype,
        local_runtime_resume_mode=config.local_runtime_resume_mode,
    )


def _payload(expected: dict[str, nn.Parameter], config: LocalMemoryConfig = LocalMemoryConfig(), **kwargs: object) -> dict[str, object]:
    return slow_checkpoint_payload(expected, config, feature_config=_feature_config(config), base_identity=_base_identity(), **kwargs)


def _restore(root: _RuntimeRoot, projector: nn.Linear, modality: nn.Parameter, adapter: CanonicalProductionAdapter, scheduler: CanonicalBatchScheduler, payload: dict[str, object], expected: dict[str, nn.Parameter], **kwargs: object) -> None:
    config = LocalMemoryConfig()
    strict_restore_into(root, payload, expected, config, runtime_encoder=root.evidence_encoder, runtime_core=root.ttt_core, local_memory2llm=projector, modality=modality, adapter=adapter, scheduler=scheduler, feature_config=_feature_config(config), base_identity=_base_identity(), **kwargs)


def _optimizer_and_scheduler(expected: dict[str, nn.Parameter]) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ExponentialLR]:
    optimizer = torch.optim.AdamW(tuple(expected.values()), lr=0.1)
    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def _assert_state_equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and set(actual) == set(expected)
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual)) and len(actual) == len(expected)
        for item, other in zip(actual, expected, strict=True):
            _assert_state_equal(item, other)
    else:
        assert actual == expected


def _real_request() -> tuple[CanonicalBatchScheduler, CanonicalBatchWindowTransaction, CanonicalProductionSegmentRequest, SegmentBatch]:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identity = SegmentIdentity(0, "episode", "category", 0, 0, "source")
    chronology = ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest")
    row = CatalogRow(identity, chronology, provenance)
    snapshot = QueueEpochSnapshot(1, 0, "catalog", (), (("category", (0,)),))
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(snapshot, (), target_distribution=(("category", 1.0),), catalog=(row,))
    )
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="restore-authority")
    batch = SegmentBatch(
        torch.zeros(1, 2, 96),
        (("s0", "s1"),),
        torch.tensor([[True, True]]),
        torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96),
        torch.zeros(1, 2, 10),
        torch.tensor([[False, True]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([0]),
        ("episode",),
        ("category",),
        provenance,
    )
    transaction = CanonicalBatchWindowTransaction(plan)
    request = CanonicalProductionSegmentRequest(scheduler, plan, transaction, plan.members[0], 0, batch)
    return scheduler, transaction, request, batch


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


def test_feature_config_and_base_identity_are_exact_and_versioned() -> None:
    feature = FeatureConfigIdentity()
    mapping = feature.to_mapping()
    assert len(mapping) == 15
    assert FeatureConfigIdentity.from_mapping(mapping) == feature
    for bad in (
        {key: value for key, value in mapping.items() if key != "enable_input_bias"},
        {**mapping, "foreign": 1},
        {**mapping, "local_memory_dim": 31},
        {**mapping, "local_history_state_enabled": True},
    ):
        with pytest.raises(ValueError):
            FeatureConfigIdentity.from_mapping(bad)
    identity = build_base_identity(
        child_git_revision="f" * 40,
        feature_config=feature,
        checkpoint_source_fingerprint="source-fingerprint",
        manifest_sha256="m" * 64,
        source_sha256="s" * 64,
    )
    assert set(identity) == {
        "schema",
        "child_git_revision",
        "canonical_model_config_sha256",
        "checkpoint_source_fingerprint",
        "manifest_sha256",
        "source_sha256",
    }
    with pytest.raises(ValueError):
        build_base_identity(
            child_git_revision="",
            feature_config=feature,
            checkpoint_source_fingerprint="source-fingerprint",
            manifest_sha256="m" * 64,
            source_sha256="s" * 64,
        )


def test_pristine_progress_predicate_is_exact_and_fail_closed() -> None:
    pristine = {"last_epoch": 0, "_step_count": 1}
    validate_pristine_progress(
        iteration=0,
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=pristine,
        pristine_scheduler_state=pristine,
        optimizer_present=True,
        scheduler_present=True,
    )
    for kwargs in (
        {"iteration": 1, "optimizer_state": {"state": {}, "param_groups": []}, "scheduler_state": pristine, "pristine_scheduler_state": pristine, "optimizer_present": True, "scheduler_present": True},
        {"iteration": 0, "optimizer_state": {"state": {0: {"step": 1}}, "param_groups": []}, "scheduler_state": pristine, "pristine_scheduler_state": pristine, "optimizer_present": True, "scheduler_present": True},
        {"iteration": 0, "optimizer_state": {"state": {}, "param_groups": []}, "scheduler_state": {"last_epoch": 1, "_step_count": 1}, "pristine_scheduler_state": pristine, "optimizer_present": True, "scheduler_present": True},
        {"iteration": 0, "optimizer_state": None, "scheduler_state": {"last_epoch": 0}, "pristine_scheduler_state": None, "optimizer_present": False, "scheduler_present": False},
    ):
        with pytest.raises(ValueError):
            validate_pristine_progress(**kwargs)


def test_active_ttt_projection_abi_is_fail_closed() -> None:
    OmniMoTModelConfig(
        local_ttt_enabled=True,
        local_history_enabled=True,
        local_history_backend="ttt_fast_weight",
        local_memory_enabled=True,
        local_memory_dim=32,
    )
    with pytest.raises(ValueError, match="local_memory_dim=32"):
        OmniMoTModelConfig(
            local_ttt_enabled=True,
            local_history_enabled=True,
            local_history_backend="ttt_fast_weight",
            local_memory_enabled=True,
            local_memory_dim=31,
        )
    root, _, modality, _, _ = _fixture()
    with pytest.raises(ValueError, match="32 -> 2048"):
        canonical_slow_inventory(root, nn.Linear(31, 2048), modality)
    with pytest.raises(ValueError, match="32 -> 2048"):
        canonical_slow_inventory(root, nn.Linear(32, 2048), nn.Parameter(torch.zeros(32)))


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
    payload = _payload(expected)
    assert strict_restore(payload, expected, LocalMemoryConfig(), feature_config=_feature_config(), base_identity=_base_identity())
    with pytest.raises(ValueError, match="runtime"):
        _payload({"local_memory_runtime.frontier": next(iter(expected.values()))})
    before = {name: value.detach().clone() for name, value in expected.items()}
    damaged = {**payload, "parameters": {**payload["parameters"], "local_memory2llm.weight": torch.zeros(1)}}
    with pytest.raises(ValueError, match="tensor"):
        _restore(root, projector, modality, adapter, scheduler, damaged, expected)
    feature_drift = copy.deepcopy(payload)
    feature_drift["feature_config"]["enable_input_bias"] = False
    with pytest.raises(ValueError, match="identity"):
        _restore(root, projector, modality, adapter, scheduler, feature_drift, expected)
    base_drift = copy.deepcopy(payload)
    base_drift["base_identity"]["source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="identity"):
        _restore(root, projector, modality, adapter, scheduler, base_drift, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_preflight_is_atomic() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = _payload(expected)
    snapshot = {name: value.detach().clone() for name, value in expected.items()}
    with torch.no_grad():
        for value in expected.values():
            value.add_(1)
    _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, snapshot[name]) for name, value in expected.items())
    validate_runtime_admission(adapter=adapter, scheduler=scheduler)


def test_restore_round_trip_preflights_optimizer_scheduler_and_object_membership() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    optimizer, state_scheduler = _optimizer_and_scheduler(expected)
    payload = _payload(expected, optimizer=optimizer, scheduler=state_scheduler, iteration=0)
    slow_snapshot = {name: value.detach().clone() for name, value in expected.items()}
    optimizer_snapshot = copy.deepcopy(payload["optimizer"])
    scheduler_snapshot = copy.deepcopy(payload["scheduler"])
    with torch.no_grad():
        for value in expected.values():
            value.add_(1)
    _restore(
        root,
        projector,
        modality,
        adapter,
        scheduler,
        payload,
        expected,
        optimizer=optimizer,
        state_scheduler=state_scheduler,
        iteration=0,
    )
    assert all(torch.equal(value, slow_snapshot[name]) for name, value in expected.items())
    _assert_state_equal(optimizer.state_dict(), optimizer_snapshot)
    _assert_state_equal(state_scheduler.state_dict(), scheduler_snapshot)

    foreign = torch.optim.AdamW(tuple(nn.Parameter(torch.zeros_like(value)) for value in expected.values()), lr=0.1)
    before = {name: value.detach().clone() for name, value in expected.items()}
    with pytest.raises(ValueError, match="optimizer group"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected, optimizer=foreign, state_scheduler=state_scheduler, iteration=0)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_rejects_late_optimizer_or_scheduler_defect_before_mutation() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    optimizer, state_scheduler = _optimizer_and_scheduler(expected)
    payload = _payload(expected, optimizer=optimizer, scheduler=state_scheduler, iteration=0)
    before_slow = {name: value.detach().clone() for name, value in expected.items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(state_scheduler.state_dict())
    damaged_optimizer = copy.deepcopy(payload)
    damaged_optimizer["optimizer"].pop("state")
    with pytest.raises(ValueError, match="optimizer"):
        _restore(root, projector, modality, adapter, scheduler, damaged_optimizer, expected, optimizer=optimizer, state_scheduler=state_scheduler, iteration=0)
    damaged_scheduler = copy.deepcopy(payload)
    damaged_scheduler["scheduler"]["foreign"] = 1
    with pytest.raises(ValueError, match="scheduler"):
        _restore(root, projector, modality, adapter, scheduler, damaged_scheduler, expected, optimizer=optimizer, state_scheduler=state_scheduler, iteration=0)
    damaged_optimizer_identity = copy.deepcopy(payload)
    damaged_optimizer_identity["optimizer_identity"]["groups"][0]["hyperparameters"]["lr"] = 0.2
    with pytest.raises(ValueError, match="optimizer identity"):
        _restore(root, projector, modality, adapter, scheduler, damaged_optimizer_identity, expected, optimizer=optimizer, state_scheduler=state_scheduler, iteration=0)
    damaged_scheduler_identity = copy.deepcopy(payload)
    damaged_scheduler_identity["scheduler_identity"]["class"] = "foreign.Scheduler"
    with pytest.raises(ValueError, match="scheduler identity"):
        _restore(root, projector, modality, adapter, scheduler, damaged_scheduler_identity, expected, optimizer=optimizer, state_scheduler=state_scheduler, iteration=0)
    assert all(torch.equal(value, before_slow[name]) for name, value in expected.items())
    _assert_state_equal(optimizer.state_dict(), before_optimizer)
    _assert_state_equal(state_scheduler.state_dict(), before_scheduler)


def test_restore_rejects_reordered_duplicate_and_missing_optimizer_membership_before_mutation() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    optimizer, state_scheduler = _optimizer_and_scheduler(expected)
    payload = _payload(expected, optimizer=optimizer, scheduler=state_scheduler)
    before = {name: value.detach().clone() for name, value in expected.items()}
    candidates = (
        torch.optim.AdamW(tuple(reversed(tuple(expected.values()))), lr=0.1),
        torch.optim.AdamW(tuple(expected.values())[:-1], lr=0.1),
    )
    for candidate in candidates:
        with pytest.raises(ValueError, match="optimizer group"):
            _restore(root, projector, modality, adapter, scheduler, payload, expected, optimizer=candidate, state_scheduler=state_scheduler)
        assert all(torch.equal(value, before[name]) for name, value in expected.items())
    duplicate = torch.optim.AdamW((tuple(expected.values())[0], *tuple(expected.values())), lr=0.1)
    with pytest.raises(ValueError, match="optimizer group"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected, optimizer=duplicate, state_scheduler=state_scheduler)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_rejects_real_native_forward_commit_and_retry_authorities_before_mutation() -> None:
    root, projector, modality, adapter, _ = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = _payload(expected)
    before = {name: value.detach().clone() for name, value in expected.items()}
    request, carrier = _bound_request_and_carrier()
    result = adapter.scan(request)
    prepared = adapter.prepare_native_inputs(request, result, carrier, input_image_key="images", input_video_key="video")
    prepared = adapter.attach_native_preparation(
        prepared, input_text_indexes=[[], []], sequence_plans=[SequencePlan(has_text=False), SequencePlan(has_text=False, has_local_memory=True)],
        gen_data_clean=object(), memory_info={}, data_resolutions=None, vae_pixel_shapes=[]
    )
    anchor = torch.ones((), requires_grad=True)
    native = adapter.bind_native_forward(
        prepared,
        build_canonical_native_loss_split(
            consumer_identities=prepared.traversal.identities,
            modalities={},
            sample_level_scale=torch.ones(()),
            auxiliary_loss=anchor * 0.0,
            graph_anchor=anchor,
        ),
    )
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, request.scheduler, payload, expected)
    adapter.abort_native_forward(native)
    scheduler, transaction, retry_request, _ = _real_request()
    retry = adapter.retry_first_member_pre_backward(retry_request)
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    adapter.consume_retry(retry)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_rejects_real_suffix_recovery_and_receipt_authorities_before_mutation() -> None:
    root, projector, modality, adapter, scheduler = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = _payload(expected)
    provenance = SegmentProvenance("manifest", "config", "source", 0)

    def member(index: int, count: int) -> MicrobatchPlanMember:
        identity = SegmentIdentity(0, "episode", "category", index, index, "source")
        record = ChronologyCountRecord(0, "episode", "category", "source", 0, count, False, "manifest")
        return MicrobatchPlanMember(
            index, (identity,), (provenance,), (record,), (count,), count,
            QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)), (),
        )

    def batch(count: int) -> SegmentBatch:
        return SegmentBatch(
            torch.zeros(1, count, 96), (tuple(f"s{step}" for step in range(count)),),
            torch.ones(1, count, dtype=torch.bool), torch.arange(count).reshape(1, count),
            torch.zeros(1, count, 96), torch.zeros(1, count, 10),
            torch.tensor([[False, *([True] * (count - 1))]]), torch.tensor([[-1, *range(count - 1)]]),
            torch.tensor([0]), ("episode",), ("category",), provenance,
        )

    members = (member(0, 2), member(1, 5), member(2, 3))
    plan = CanonicalGAWindowPlan(members, 10, 3, "restore-suffix")
    transaction = CanonicalBatchWindowTransaction(plan)
    transaction.mark_backward_started(0)
    transaction.mark_reconciled(0)
    request = CanonicalProductionSegmentRequest(scheduler, plan, transaction, members[1], 1, batch(5))
    before = {name: value.detach().clone() for name, value in expected.items()}
    suffix = adapter.derive_suffix_recovery(request, source_transient=adapter.declare_retryable_source_transient(request))
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    recovered = adapter.consume_suffix_recovery(suffix, segment_batches=(batch(5), batch(3)))
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    fresh_adapter = CanonicalProductionAdapter(root.evidence_encoder, root.ttt_core)
    fresh_scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    with pytest.raises(ValueError, match="open canonical transaction"):
        _restore(root, projector, modality, fresh_adapter, fresh_scheduler, payload, expected, transaction=suffix.recovery)
    assert recovered and all(torch.equal(value, before[name]) for name, value in expected.items())


def test_restore_rejects_real_pending_and_committed_runtime_authorities_before_mutation() -> None:
    root, projector, modality, adapter, _ = _fixture()
    expected = canonical_slow_inventory(root, projector, modality)
    payload = _payload(expected)
    scheduler, transaction, request, _ = _real_request()
    before = {name: value.detach().clone() for name, value in expected.items()}
    result = adapter.scan(request)
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())
    adapter.abort_scan(request, result)
    with pytest.raises(ValueError, match="frozen"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    result = adapter.scan(request)
    transaction.mark_backward_started(0)
    capability = adapter.prepare_commit(request, result)
    with pytest.raises(ValueError, match="pending"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())
    adapter.commit_success(capability)
    with pytest.raises(ValueError, match="fast-state frontier"):
        _restore(root, projector, modality, adapter, scheduler, payload, expected)
    assert all(torch.equal(value, before[name]) for name, value in expected.items())
    fresh_adapter = CanonicalProductionAdapter(root.evidence_encoder, root.ttt_core)
    fresh_scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    with pytest.raises(ValueError, match="open canonical transaction"):
        _restore(root, projector, modality, fresh_adapter, fresh_scheduler, payload, expected, transaction=transaction)

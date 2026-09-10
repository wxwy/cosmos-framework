from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_adapter_scheduler import (
    CanonicalBatchScheduler,
    CanonicalBatchWindowTransaction,
    CanonicalGAWindowPlan,
    CanonicalSegmentContractError,
    CatalogRow,
    ChronologyCountRecord,
    MicrobatchPlanMember,
    ProjectedSchedulerState,
    QueueEpochSnapshot,
)
from cosmos_framework.model.generator.mot.canonical_segment_production_adapter import (
    CanonicalProductionAdapter,
    CanonicalProductionFastStateFrontier,
    CanonicalProductionSegmentRequest,
    CanonicalRawRowCarrier,
    build_prepared_canonical_native_loss_split,
)
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance


def test_adapter_scan_derives_stream_major_gather_and_fp32_state() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    batch = SegmentBatch(
        torch.zeros(1, 2, 96), (("s0", "s1"),), torch.tensor([[True, True]]), torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96), torch.zeros(1, 2, 10), torch.tensor([[False, True]]), torch.tensor([[-1, 0]]),
        torch.tensor([0]), ("episode",), ("category",), provenance,
    )
    member = MicrobatchPlanMember(
        0, (SegmentIdentity(0, "episode", "category", 0, 0, "source"),), (provenance,),
        (ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest"),), (2,), 2,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)), (),
    )
    plan = CanonicalGAWindowPlan((member,), 2, 1, "chain")
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    request = CanonicalProductionSegmentRequest(scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, batch)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core)
    result = adapter.scan(request)
    assert result.gathered.item_count == 2
    assert result.gathered.local_prefixes[0] is None and result.gathered.local_prefixes[1] is not None
    assert all(value.dtype is torch.float32 for value in result.candidate_state_out)
    assert result.slot_chain == ((0, "episode", "source", 0),)
    with pytest.raises(Exception, match="already has a scan capability"):
        adapter.scan(request)
    frontier_before = dict(adapter.frontier._states)
    adapter.abort_scan(request, result)
    assert adapter.frontier._states == frontier_before
    with pytest.raises(CanonicalSegmentContractError, match="exact pending"):
        adapter.abort_scan(request, result)
    adapter.abort_scan(request, adapter.scan(request))


def test_nested_carrier_derives_expected_traversal_and_rejects_foreign_identity() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    batch = SegmentBatch(
        torch.zeros(2, 3, 96), (("s00", "s01", None), ("s10", "s11", "s12")),
        torch.tensor([[True, True, False], [True, True, True]]), torch.tensor([[0, 1, -1], [0, 1, 2]]),
        torch.zeros(2, 3, 96), torch.zeros(2, 3, 10), torch.tensor([[False, True, False], [False, True, True]]),
        torch.tensor([[-1, 0, -1], [-1, 0, 1]]), torch.tensor([0, 1]), ("episode-0", "episode-1"),
        ("category", "category"), provenance,
    )
    member = MicrobatchPlanMember(
        0,
        (SegmentIdentity(0, "episode-0", "category", 0, 0, "source"), SegmentIdentity(1, "episode-1", "category", 0, 0, "source")),
        (provenance, provenance),
        (ChronologyCountRecord(0, "episode-0", "category", "source", 0, 2, False, "manifest"), ChronologyCountRecord(1, "episode-1", "category", "source", 0, 3, False, "manifest")),
        (2, 3), 5,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)), (),
    )
    plan = CanonicalGAWindowPlan((member,), 5, 1, "carrier")
    scheduler = CanonicalBatchScheduler(ProjectedSchedulerState(QueueEpochSnapshot(1, 0, "catalog", ()), ()))
    request = CanonicalProductionSegmentRequest(scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, batch)
    samples = tuple(
        tuple({"canonical_identity": (row, f"episode-{row}", step)} if step >= 0 else None for step in steps)
        for row, steps in enumerate(((0, 1, -1), (0, 1, 2)))
    )
    carrier = CanonicalRawRowCarrier(
        request, member, batch, member.row_identities, member.row_chronology, samples, samples, {"sequence_plan": ()}
    )
    assert carrier.expected_for(request).logical_indexes == ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))
    foreign = CanonicalRawRowCarrier(
        request, member, batch, member.row_identities, member.row_chronology,
        (({"canonical_identity": (1, "episode-0", 0)}, samples[0][1], None), samples[1]), samples, {},
    )
    with pytest.raises(CanonicalSegmentContractError, match="raw identity is foreign"):
        foreign.expected_for(request)
    model_samples = tuple(
        tuple(
            None
            if sample is None
            else {
                "canonical_identity": sample["canonical_identity"],
                "text_token_ids": sample,
                "images": sample,
            }
            for sample in row
        )
        for row in samples
    )
    source_identities = tuple(
        tuple(
            None if raw is None else (row, f"episode-{row}", "source", raw["canonical_identity"][2])
            for raw in raw_row
        )
        for row, raw_row in enumerate(samples)
    )
    bound = CanonicalRawRowCarrier(
        request, member, batch, member.row_identities, member.row_chronology, samples, model_samples,
        {
            "text_token_ids": [model_samples[row][index]["text_token_ids"] for row, index in carrier.expected_for(request).logical_indexes],
            "images": [model_samples[row][index]["images"] for row, index in carrier.expected_for(request).logical_indexes],
        },
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    expected = bound.expected_for(request)
    bound.validate_model_data_batch(expected, input_image_key="images", input_video_key="video")
    owner_maps = bound.native_owner_maps(expected, input_image_key="images", input_video_key="video")
    assert owner_maps.vision_owner_indexes == (0, 1, 2, 3, 4)
    assert owner_maps.action_owner_indexes == ()
    assert owner_maps.sound_owner_indexes == ()
    foreign_batch = CanonicalRawRowCarrier(
        request, member, batch, member.row_identities, member.row_chronology, samples, model_samples,
        {
            "text_token_ids": [dict(value) for value in bound.model_data_batch["text_token_ids"]],
            "images": bound.model_data_batch["images"],
        },
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    with pytest.raises(CanonicalSegmentContractError, match="model batch source is foreign"):
        foreign_batch.validate_model_data_batch(expected, input_image_key="images", input_video_key="video")
    stacked_sources = tuple(
        torch.tensor([position], dtype=torch.int64) for position in range(len(expected.logical_indexes))
    )
    for source, (row, index) in zip(stacked_sources, expected.logical_indexes, strict=True):
        model_samples[row][index]["image_size"] = source
    stacked_batch = CanonicalRawRowCarrier(
        request,
        member,
        batch,
        member.row_identities,
        member.row_chronology,
        samples,
        model_samples,
        {
            "images": bound.model_data_batch["images"],
            "image_size": torch.stack(stacked_sources),
        },
        {"image_size": stacked_sources},
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    stacked_batch.validate_model_data_batch(expected, input_image_key="images", input_video_key="video")
    foreign_stacked_batch = CanonicalRawRowCarrier(
        request,
        member,
        batch,
        member.row_identities,
        member.row_chronology,
        samples,
        model_samples,
        {
            "images": bound.model_data_batch["images"],
            "image_size": torch.stack(tuple(reversed(stacked_sources))),
        },
        {"image_size": stacked_sources},
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    with pytest.raises(CanonicalSegmentContractError, match="stacked model batch is foreign"):
        foreign_stacked_batch.validate_model_data_batch(
            expected, input_image_key="images", input_video_key="video"
        )
    conflicting_vision_batch = CanonicalRawRowCarrier(
        request,
        member,
        batch,
        member.row_identities,
        member.row_chronology,
        samples,
        model_samples,
        {"images": bound.model_data_batch["images"], "video": bound.model_data_batch["images"]},
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    with pytest.raises(CanonicalSegmentContractError, match="exactly one vision input key"):
        conflicting_vision_batch.validate_model_data_batch(
            expected, input_image_key="images", input_video_key="video"
        )

    flat_samples = tuple(model_samples[row][index] for row, index in expected.logical_indexes)
    vision_counts = (2, 1, 1, 1, 3)
    for sample, count in zip(flat_samples, vision_counts, strict=True):
        sample["num_vision_items_per_sample"] = count
    flat_samples[1]["action"] = torch.tensor([1.0])
    flat_samples[4]["action"] = torch.tensor([2.0])
    flat_samples[3]["sound"] = torch.tensor([3.0])
    dense_batch = CanonicalRawRowCarrier(
        request,
        member,
        batch,
        member.row_identities,
        member.row_chronology,
        samples,
        model_samples,
        {
            "text_token_ids": [sample["text_token_ids"] for sample in flat_samples],
            "images": [sample["images"] for sample in flat_samples],
            "num_vision_items_per_sample": list(vision_counts),
            "action": [flat_samples[1]["action"], flat_samples[4]["action"]],
            "sound": [flat_samples[3]["sound"]],
        },
        raw_row_source_identities=source_identities,
        row_model_source_rows=samples,
    )
    dense_batch.validate_model_data_batch(expected, input_image_key="images", input_video_key="video")
    dense_maps = dense_batch.native_owner_maps(expected, input_image_key="images", input_video_key="video")
    assert dense_maps.vision_owner_indexes == (0, 0, 1, 2, 3, 4, 4, 4)
    assert dense_maps.action_owner_indexes == (1, 4)
    assert dense_maps.sound_owner_indexes == (3,)
    dense_adapter = CanonicalProductionAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(evidence_dim=256),
    )
    dense_result = dense_adapter.scan(request)
    dense_prepared = dense_adapter.prepare_native_inputs(
        request, dense_result, dense_batch, input_image_key="images", input_video_key="video"
    )
    anchor = torch.tensor(1.0, requires_grad=True)
    split = build_prepared_canonical_native_loss_split(
        prepared=dense_prepared,
        vision_weighted_terms=torch.arange(1.0, 9.0),
        action_weighted_terms=torch.tensor([2.0, 4.0]),
        sound_weighted_terms=torch.tensor([6.0]),
        vision_weight=1.0,
        action_weight=0.5,
        sound_weight=0.25,
        sample_level_scale=torch.tensor(0.5),
        auxiliary_loss=anchor * 7.0,
        graph_anchor=anchor,
    )
    torch.testing.assert_close(split.consumer_loss, torch.tensor(3.75))
    torch.testing.assert_close(split.auxiliary_loss, anchor * 7.0)
    dense_adapter.abort_scan(request, dense_result)


def test_adapter_commit_is_exact_once_and_preflights_before_frontier_mutation() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identity = SegmentIdentity(0, "episode", "category", 0, 0, "source")
    record = ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest")
    state = ProjectedSchedulerState(
        QueueEpochSnapshot(
            1,
            0,
            "catalog",
            (("category", 0),),
            (("category", (0,)),),
        ),
        (("category", 0),),
        target_distribution=(("category", 1.0),),
        catalog=(CatalogRow(identity, record, provenance),),
    )
    scheduler = CanonicalBatchScheduler(state)
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="commit")
    member = plan.members[0]
    batch = SegmentBatch(
        torch.zeros(1, 2, 96), (("s0", "s1"),), torch.tensor([[True, True]]), torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96), torch.zeros(1, 2, 10), torch.tensor([[False, True]]), torch.tensor([[-1, 0]]),
        torch.tensor([0]), ("episode",), ("category",), provenance,
    )
    transaction = CanonicalBatchWindowTransaction(plan)
    request = CanonicalProductionSegmentRequest(scheduler, plan, transaction, member, 0, batch)
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    adapter = CanonicalProductionAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), core)
    result = adapter.scan(request)
    capability = adapter.prepare_commit(request, result)
    scheduler_before = scheduler.snapshot
    with pytest.raises(CanonicalSegmentContractError, match="current backward member"):
        adapter.commit_success(capability)
    assert scheduler.snapshot == scheduler_before
    continuation = MicrobatchPlanMember(
        0,
        (SegmentIdentity(0, "episode", "category", 1, 1, "source"),),
        (provenance,),
        (ChronologyCountRecord(0, "episode", "category", "source", 2, 3, False, "manifest"),),
        (1,),
        1,
        member.queue_snapshot,
        member.projected_exposure_before,
    )
    with pytest.raises(CanonicalSegmentContractError, match="lacks exact committed"):
        adapter.frontier.state_for(continuation)
    transaction.mark_backward_started(0)
    adapter.commit_success(capability)
    assert transaction.snapshot().completed_members == (0,)
    with pytest.raises(CanonicalSegmentContractError, match="foreign or already consumed"):
        adapter.commit_success(capability)


def test_adapter_abort_commit_consumes_exact_capability_without_reconcile(monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identity = SegmentIdentity(0, "episode", "category", 0, 0, "source")
    record = ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest")
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(
            QueueEpochSnapshot(1, 0, "catalog", (("category", 0),), (("category", (0,)),)),
            (("category", 0),),
            target_distribution=(("category", 1.0),),
            catalog=(CatalogRow(identity, record, provenance),),
        )
    )
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="abort-commit")
    member = plan.members[0]
    batch = SegmentBatch(
        torch.zeros(1, 2, 96), (("s0", "s1"),), torch.tensor([[True, True]]), torch.tensor([[0, 1]]),
        torch.zeros(1, 2, 96), torch.zeros(1, 2, 10), torch.tensor([[False, True]]), torch.tensor([[-1, 0]]),
        torch.tensor([0]), ("episode",), ("category",), provenance,
    )
    request = CanonicalProductionSegmentRequest(
        scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, batch
    )
    adapter = CanonicalProductionAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(evidence_dim=256),
    )
    result = adapter.scan(request)
    request.transaction.mark_backward_started(0)
    capability = adapter.prepare_commit(request, result)
    scheduler_before = scheduler.snapshot
    with pytest.raises(CanonicalSegmentContractError, match="exact pending capability"):
        adapter.abort_commit(replace(capability))
    assert adapter._commit_capabilities == {id(capability)}
    assert adapter._scan_requests == {id(request)}
    monkeypatch.setattr(
        adapter.frontier,
        "commit",
        lambda member, state: (_ for _ in ()).throw(RuntimeError("injected pre-mutation commit failure")),
    )
    with pytest.raises(RuntimeError, match="injected pre-mutation commit failure"):
        adapter.commit_success(capability)
    assert adapter._commit_capabilities == {id(capability)}
    assert adapter._scan_requests == {id(request)}
    adapter.abort_commit(capability)
    assert adapter._commit_capabilities == set()
    assert adapter._scan_requests == set()
    assert adapter._scan_results == {}
    assert adapter.frontier._states == {}
    assert scheduler.snapshot == scheduler_before
    assert request.transaction.snapshot().completed_members == ()
    request.transaction.terminalize(0, "CANONICAL_NATIVE_COMMIT_FAILURE")
    assert request.transaction.snapshot().terminal_failure_code == "CANONICAL_NATIVE_COMMIT_FAILURE"
    with pytest.raises(CanonicalSegmentContractError, match="exact pending capability"):
        adapter.abort_commit(capability)


def test_adapter_retry_preserves_original_frozen_transition_exactly_once() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identity = SegmentIdentity(0, "episode", "category", 0, 0, "source")
    record = ChronologyCountRecord(0, "episode", "category", "source", 0, 2, False, "manifest")
    scheduler = CanonicalBatchScheduler(
        ProjectedSchedulerState(
            QueueEpochSnapshot(1, 0, "catalog", (("category", 0),), (("category", (0,)),)),
            (("category", 0),),
            target_distribution=(("category", 1.0),),
            catalog=(CatalogRow(identity, record, provenance),),
        )
    )
    plan = scheduler.freeze_plan(slot_groups=((0,),), plan_chain_id="retry")
    member = plan.members[0]
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
    request = CanonicalProductionSegmentRequest(
        scheduler, plan, CanonicalBatchWindowTransaction(plan), member, 0, batch
    )
    adapter = CanonicalProductionAdapter(
        LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ContinualTTTLocalMemoryCore(evidence_dim=256),
    )

    with pytest.raises(CanonicalSegmentContractError, match="exact unscanned"):
        adapter.retry_first_member_pre_backward(replace(request, member_index=1))
    capability = adapter.retry_first_member_pre_backward(request)
    assert capability.original_plan is plan
    assert capability.retry_plan.attempt == 1
    assert capability.retry_plan.members == plan.members
    assert capability.retry_plan.members[0] is member
    assert capability.retry_plan.original_n_valid_window == plan.original_n_valid_window
    assert capability.retry_plan.original_ga_effective == plan.original_ga_effective
    assert capability.retry_plan.plan_chain_id == plan.plan_chain_id
    assert len(scheduler._frozen_transitions) == 1
    with pytest.raises(CanonicalSegmentContractError, match="unstarted batch window"):
        adapter.retry_first_member_pre_backward(request)
    with pytest.raises(CanonicalSegmentContractError, match="foreign or already consumed"):
        adapter.consume_retry(replace(capability, retry_request=request))

    retry_request = adapter.consume_retry(capability)
    with pytest.raises(CanonicalSegmentContractError, match="foreign or already consumed"):
        adapter.consume_retry(capability)
    result = adapter.scan(retry_request)
    prepared = adapter.prepare_commit(retry_request, result)
    retry_request.transaction.mark_backward_started(0)
    adapter.commit_success(prepared)
    assert scheduler._frozen_transitions == []
    assert retry_request.transaction.snapshot().completed_members == (0,)


def test_fast_state_frontier_preserves_w0_gradients_and_slot_isolation() -> None:
    provenance = SegmentProvenance("manifest", "config", "source", 0)
    identities = (
        SegmentIdentity(0, "episode-0", "category", 0, 0, "source-0"),
        SegmentIdentity(1, "episode-1", "category", 0, 0, "source-1"),
    )
    member = MicrobatchPlanMember(
        0,
        identities,
        (provenance, provenance),
        (
            ChronologyCountRecord(0, "episode-0", "category", "source-0", 0, 1, False, "manifest"),
            ChronologyCountRecord(1, "episode-1", "category", "source-1", 0, 1, False, "manifest"),
        ),
        (1, 1),
        2,
        QueueEpochSnapshot(1, 0, "catalog", (("category", 0),)),
        (),
    )
    core = ContinualTTTLocalMemoryCore(evidence_dim=256)
    frontier = CanonicalProductionFastStateFrontier(core)
    fresh = frontier.state_for(member)
    assert all(value.dtype is torch.float32 for value in fresh)
    sum(value.sum() for value in fresh).backward()
    assert all(parameter.grad is not None for parameter in core._w0)
    # Give each slot a distinct numeric state without introducing a shared storage alias.
    committed = type(fresh)(*(value + torch.arange(2, dtype=torch.float32).reshape(2, *([1] * (value.ndim - 1))) for value in fresh))
    frontier.commit(member, committed)
    continued = MicrobatchPlanMember(
        0,
        (
            SegmentIdentity(0, "episode-0", "category", 1, 1, "source-0"),
            SegmentIdentity(1, "episode-1", "category", 1, 1, "source-1"),
        ),
        (provenance, provenance),
        (
            ChronologyCountRecord(0, "episode-0", "category", "source-0", 1, 2, False, "manifest"),
            ChronologyCountRecord(1, "episode-1", "category", "source-1", 1, 2, False, "manifest"),
        ),
        (1, 1),
        2,
        member.queue_snapshot,
        (),
    )
    state = frontier.state_for(continued)
    for value, expected in zip(state, committed, strict=True):
        torch.testing.assert_close(value, expected)
        assert value[0].data_ptr() != value[1].data_ptr()

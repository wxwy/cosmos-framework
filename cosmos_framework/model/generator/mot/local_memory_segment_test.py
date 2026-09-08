# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.local_memory_segment import (
    GAWindowPlan,
    RankLocalSegmentScheduler,
    SegmentBatch,
    SegmentIdentity,
    SegmentProvenance,
)


def _batch() -> SegmentBatch:
    return SegmentBatch(
        consumer_visual_summary=torch.randn(1, 3, 96),
        consumer_payload=(("s0", "s1", None),),
        consumer_valid=torch.tensor([[True, True, False]]),
        consumer_step=torch.tensor([[0, 1, -1]]),
        evidence_visual_summary_prev=torch.randn(1, 3, 96),
        evidence_executed_action_prev=torch.randn(1, 3, 10),
        evidence_valid=torch.tensor([[False, True, False]]),
        evidence_source_step=torch.tensor([[-1, 0, -1]]),
        slot_id=torch.tensor([4]),
        episode_id=("episode",),
        category=("category",),
        segment_provenance=SegmentProvenance("m", "c", "s", 0),
    )


def test_segment_batch_shifted_validation_and_common_gather() -> None:
    batch = _batch()
    batch.validate(16)
    tokens = torch.randn(1, 3, 1, 32)
    payloads, local, identities = batch.gather_consumers(tokens, torch.tensor([[False, True, False]]))
    assert payloads == ["s0", "s1"]
    assert local[0] is None and torch.equal(local[1], tokens[0, 1])
    assert identities == [(4, "episode", 0), (4, "episode", 1)]


def test_segment_batch_rejects_step0_evidence_and_plan_counts() -> None:
    batch = _batch()
    invalid = SegmentBatch(
        **{
            **batch.__dict__,
            "evidence_valid": torch.tensor([[True, True, False]]),
            "evidence_source_step": torch.tensor([[0, 0, -1]]),
        }
    )
    with pytest.raises(ValueError, match="step0"):
        invalid.validate(16)
    plan = GAWindowPlan(((4, "episode", 0), (4, "episode", 1)), (1, 3))
    assert torch.equal(plan.objective(1, torch.tensor(4.0), torch.tensor(2.0), 3), torch.tensor(4.0))
    with pytest.raises(ValueError, match="actual"):
        plan.objective(0, torch.tensor(1.0), torch.tensor(1.0), 2)


def test_rank_local_scheduler_is_deterministic_and_commits_exposure() -> None:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0, "b": 1.0})
    candidates = (
        SegmentIdentity(0, "a0", "a", 0, 0, "x"),
        SegmentIdentity(1, "b0", "b", 0, 0, "x"),
    )
    first = scheduler.admit(candidates)
    assert first.category == "b"
    scheduler.commit(first, 3)
    second = scheduler.admit(candidates)
    assert second.category == "a"
    scheduler.commit(second, 1)
    snapshot = scheduler.snapshot()
    assert snapshot["cumulative_valid_consumer_exposure"] == {"a": 1, "b": 3}
    with pytest.raises(ValueError, match="num_workers"):
        RankLocalSegmentScheduler(rank=0, target_distribution={"a": 1.0}, num_workers=1)

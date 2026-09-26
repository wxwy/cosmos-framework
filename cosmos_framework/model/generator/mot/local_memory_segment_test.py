"""B0 的时序与窗口加权合同，全部使用合成 CPU tensors。"""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, SegmentBatch, SegmentProvenance


def make_segment(cursor=0, slot=0, episode="episode", action_dim=15, steps=16):
    consumer_step = torch.arange(cursor * 16, cursor * 16 + steps)[None]
    valid = torch.ones(1, steps, dtype=torch.bool)
    evidence = consumer_step > 0
    return SegmentBatch(
        consumer_visual_summary=torch.zeros(1, steps, 96),
        consumer_payload=(tuple({"step": int(step)} for step in consumer_step[0]),),
        consumer_valid=valid,
        consumer_step=consumer_step,
        evidence_visual_summary_prev=torch.randn(1, steps, 96),
        evidence_executed_action_prev=torch.randn(1, steps, action_dim),
        evidence_valid=evidence,
        evidence_source_step=torch.where(evidence, consumer_step - 1, -1),
        slot_id=torch.tensor([slot]),
        episode_id=(episode,),
        category=("robocasa",),
        segment_provenance=SegmentProvenance("manifest", "config", "source", cursor),
    )


def test_s0_absent_and_continuation_previous_evidence():
    fresh, continuation = make_segment(), make_segment(cursor=1)
    fresh.validate(16)
    continuation.validate(16)
    assert not fresh.evidence_valid[0, 0]
    assert fresh.evidence_source_step[0, 0] == -1
    assert continuation.evidence_source_step[0, 0] == 15
    assert torch.equal(continuation.evidence_source_step, continuation.consumer_step - 1)


@pytest.mark.parametrize("bad_source", [0, 2, 9])
def test_non_s0_rejects_other_source_step(bad_source):
    segment = make_segment()
    source = segment.evidence_source_step.clone()
    source[0, 2] = bad_source
    with pytest.raises(ValueError):
        replace(segment, evidence_source_step=source).validate(16)


def test_s0_evidence_and_missing_non_s0_rejected():
    segment = make_segment()
    evidence = segment.evidence_valid.clone()
    evidence[0, 0] = True
    with pytest.raises(ValueError):
        replace(segment, evidence_valid=evidence).validate(16)
    evidence = segment.evidence_valid.clone()
    evidence[0, 1] = False
    source = segment.evidence_source_step.clone()
    source[0, 1] = -1
    with pytest.raises(ValueError):
        replace(segment, evidence_valid=evidence, evidence_source_step=source).validate(16)


@pytest.mark.parametrize("action_dim", [1, 10, 15, 64])
def test_segment_action_dimension_is_generic_positive(action_dim):
    make_segment(action_dim=action_dim).validate(16)


def test_zero_action_dim_and_discontinuous_consumer_rejected():
    with pytest.raises(ValueError):
        make_segment(action_dim=0).validate(16)
    segment = make_segment()
    steps = segment.consumer_step.clone()
    steps[0, 4] += 1
    source = torch.where(segment.evidence_valid, steps - 1, -1)
    with pytest.raises(ValueError):
        replace(segment, consumer_step=steps, evidence_source_step=source).validate(16)


def test_ga_fixture_128_plus_128_and_no_second_division():
    b_stream, tbptt, active_ga = 8, 16, 2
    # 两个抽象 group member；本层不接 grouped driver 或 trainer。
    plan = GAWindowPlan(((0, "group", 0), (0, "group", 1)), (b_stream * tbptt,) * active_ga)
    assert plan.n_window == 256
    assert plan.ga_effective == 2
    losses = torch.tensor([2.0, 6.0], requires_grad=True)
    auxiliary = torch.tensor([1.0, 3.0], requires_grad=True)
    objective = sum(plan.objective(index, losses[index], auxiliary[index], 128) for index in range(2))
    assert objective.item() == 6.0
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(auxiliary.grad, torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError):
        plan.objective(0, losses[0], auxiliary[0], 127)


def test_unequal_valid_counts_use_consumer_weight():
    plan = GAWindowPlan(((0, "a", 0), (1, "b", 0)), (16, 8))
    objective = plan.objective(0, torch.tensor(3.0), torch.tensor(0.0), 16)
    objective += plan.objective(1, torch.tensor(9.0), torch.tensor(0.0), 8)
    assert objective.item() == 5.0

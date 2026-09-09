from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.canonical_segment_runtime import CanonicalSegmentRuntimeOwner, RuntimePhase
from cosmos_framework.model.generator.mot.local_evidence import CANONICAL_EVIDENCE_FEATURE_CONFIG, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from cosmos_framework.model.generator.mot.production_segment_bridge import NativeBatchOutcome, NativeBatchResult, OpenMemberCapability, RetryMemberCapability, TerminalMemberResult, run_disabled, run_member
from cosmos_framework.model.generator.mot.production_segment_wiring import CanonicalSegmentWiring


def _owner() -> CanonicalSegmentRuntimeOwner:
    scheduler = RankLocalSegmentScheduler(rank=0, target_distribution={"suite": 1.0})
    adapter = CanonicalLocalMemorySegmentAdapter(LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG), ContinualTTTLocalMemoryCore(), LocalMemorySegmentSidecar())
    return CanonicalSegmentRuntimeOwner(scheduler, CanonicalSegmentWiring(adapter, (torch.nn.Parameter(torch.ones(())),)))


def _identity(cursor: int) -> SegmentIdentity:
    return SegmentIdentity(0, "episode", "suite", cursor, cursor, "source")


def _segment(identity: SegmentIdentity) -> SegmentBatch:
    return SegmentBatch(torch.zeros(1, 1, 96), ((object(),),), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long), torch.full((1, 1, 96), float("nan")), torch.full((1, 1, 10), float("nan")), torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1), -1, dtype=torch.long), torch.tensor([0]), ("episode",), ("suite",), SegmentProvenance("manifest", "config", identity.source_digest, identity.segment_id))


def _success(payloads, locals) -> NativeBatchOutcome:
    assert len(payloads) == len(locals)
    return NativeBatchOutcome(result=NativeBatchResult(torch.ones((), requires_grad=True), torch.zeros((), requires_grad=True)))


def _assert_terminal(
    owner: CanonicalSegmentRuntimeOwner, transaction, result: TerminalMemberResult, code: str
) -> None:
    assert result.code == code
    snapshot = transaction.snapshot()
    assert owner.phase is RuntimePhase.ABORTED
    assert owner.adapter.pending() is None
    assert snapshot.terminal_failure_code == code
    assert snapshot.remaining_members_suppressed and snapshot.slow_grads_cleared
    assert owner.wiring.local_slow_parameters[0].grad is None


def test_initial_continuation_finishes_inside_bridge_and_requires_resolution() -> None:
    owner, first, second = _owner(), _identity(0), _identity(1)
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 1))
    prior = run_member(owner, first, _segment(first), _success, initial_plan=plan)
    assert isinstance(prior, OpenMemberCapability)
    completed = run_member(owner, second, _segment(second), _success, prior=prior)
    assert owner.phase is RuntimePhase.SLOW_RESOLUTION_PENDING
    owner.resolve_local_memory_slow_window(completed, scaler_skipped=False)
    assert owner.phase is RuntimePhase.IDLE


def test_attempt_zero_source_failure_retries_exact_suffix_once() -> None:
    owner, identity = _owner(), _identity(0)
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    retry = run_member(owner, identity, _segment(identity), lambda *_: NativeBatchOutcome(source_failure="LOAD_DECODE_TRANSIENT"), initial_plan=plan)
    assert isinstance(retry, RetryMemberCapability)
    completed = run_member(owner, identity, _segment(identity), _success, retry=retry)
    owner.resolve_local_memory_slow_window(completed, scaler_skipped=True)
    assert owner.phase is RuntimePhase.IDLE


def test_gather_count_mismatch_terminalizes_first_member_before_callback() -> None:
    owner, identity = _owner(), _identity(0)
    calls = []
    parameter = owner.wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    result = run_member(owner, identity, _segment(identity), lambda *_: calls.append(True), initial_plan=GAWindowPlan(((0, "episode", 0),), (2,)))
    assert isinstance(result, TerminalMemberResult)
    _assert_terminal(owner, owner.transaction, result, "LOCAL_MEM_IDENTITY_CONTRACT_FAILURE")
    assert calls == [] and owner.scheduler.committed_identities == [] and owner.adapter.committed_snapshot() == ()


def test_gather_count_mismatch_terminalizes_later_member_and_preserves_fast_frontier() -> None:
    owner, first, second = _owner(), _identity(0), _identity(1)
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1)), (1, 2))
    prior = run_member(owner, first, _segment(first), _success, initial_plan=plan)
    assert isinstance(prior, OpenMemberCapability)
    parameter = owner.wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    result = run_member(owner, second, _segment(second), lambda *_: pytest.fail("native callback must not run"), prior=prior)
    assert isinstance(result, TerminalMemberResult)
    _assert_terminal(owner, owner.transaction, result, "LOCAL_MEM_IDENTITY_CONTRACT_FAILURE")
    assert tuple(item[0] for item in owner.adapter.committed_snapshot()) == (first,)
    assert owner.scheduler.committed_identities == [first]


def test_attempt_one_source_failure_terminalizes_and_clears_partial_grad_once() -> None:
    owner, identity = _owner(), _identity(0)
    plan = GAWindowPlan(((0, "episode", 0),), (1,))
    retry = run_member(owner, identity, _segment(identity), lambda *_: NativeBatchOutcome(source_failure="LOAD_DECODE_TRANSIENT"), initial_plan=plan)
    assert isinstance(retry, RetryMemberCapability)
    parameter = owner.wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    result = run_member(owner, identity, _segment(identity), lambda *_: NativeBatchOutcome(source_failure="LOAD_DECODE_TRANSIENT"), retry=retry)
    assert isinstance(result, TerminalMemberResult)
    _assert_terminal(owner, owner.transaction, result, "LOCAL_MEM_RETRY_EXHAUSTED")
    assert owner.adapter.committed_snapshot() == ()


@pytest.mark.parametrize(
    ("native_batch", "code"),
    (
        (lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic callback failure")), "LOCAL_MEM_OUTER_FAILURE"),
        (lambda *_: object(), "LOCAL_MEM_OUTER_FAILURE"),
        (lambda *_: NativeBatchOutcome(result=NativeBatchResult(torch.tensor(float("nan"), requires_grad=True), torch.zeros((), requires_grad=True))), "LOCAL_MEM_NUMERICAL_FAILURE"),
        (lambda *_: NativeBatchOutcome(result=NativeBatchResult(torch.ones(()), torch.zeros(()))), "LOCAL_MEM_OUTER_FAILURE"),
    ),
)
def test_public_terminal_failures_clear_pending_grad_and_sidecar(native_batch, code: str) -> None:
    owner, identity = _owner(), _identity(0)
    parameter = owner.wiring.local_slow_parameters[0]
    parameter.grad = torch.ones_like(parameter)
    result = run_member(owner, identity, _segment(identity), native_batch, initial_plan=GAWindowPlan(((0, "episode", 0),), (1,)))
    assert isinstance(result, TerminalMemberResult)
    _assert_terminal(owner, owner.transaction, result, code)
    assert owner.adapter.committed_snapshot() == ()


def test_stale_open_capability_is_rejected_without_new_prepare() -> None:
    owner, first, second, third = _owner(), _identity(0), _identity(1), _identity(2)
    plan = GAWindowPlan(((0, "episode", 0), (0, "episode", 1), (0, "episode", 2)), (1, 1, 1))
    stale = run_member(owner, first, _segment(first), _success, initial_plan=plan)
    assert isinstance(stale, OpenMemberCapability)
    current = run_member(owner, second, _segment(second), _success, prior=stale)
    assert isinstance(current, OpenMemberCapability)
    with pytest.raises(RuntimeError, match="exact open capability"):
        run_member(owner, third, _segment(third), _success, prior=stale)
    completed = run_member(owner, third, _segment(third), _success, prior=current)
    owner.resolve_local_memory_slow_window(completed, scaler_skipped=False)


def test_disabled_path_preserves_payload_callback_loss_and_slow_grad_parity() -> None:
    payloads, calls = (object(), object()), []
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def no_local_batch(received):
        calls.append(received)
        return parameter * 3

    disabled = run_disabled(payloads, no_local_batch)
    disabled.backward()
    disabled_grad = parameter.grad.detach().clone()
    parameter.grad = None
    baseline = no_local_batch(payloads)
    baseline.backward()
    assert calls == [payloads, payloads]
    assert disabled.item() == baseline.item() == 6.0
    assert torch.equal(disabled_grad, parameter.grad)

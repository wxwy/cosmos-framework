"""CPU/static-only bridge for canonical Local-Memory segment wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .local_memory_segment import LocalMemoryTransaction, SegmentBatch, SegmentIdentity
from .local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, SegmentScanResult


@dataclass(frozen=True)
class CanonicalSegmentForward:
    """Exact graph-bearing scan result and gathered consumer ABI."""

    wiring: "CanonicalSegmentWiring"
    result: SegmentScanResult
    payloads: tuple[Any, ...]
    locals: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]


class CanonicalSegmentWiring:
    """Own the one in-memory adapter capability for a synthetic segment."""

    def __init__(
        self,
        adapter: CanonicalLocalMemorySegmentAdapter,
        local_slow_parameters: tuple[torch.nn.Parameter, ...],
    ) -> None:
        if len({id(parameter) for parameter in local_slow_parameters}) != len(local_slow_parameters):
            raise ValueError("local slow parameters must be unique.")
        if any(not parameter.is_leaf for parameter in local_slow_parameters):
            raise ValueError("local slow parameters must be leaf parameters.")
        self.adapter = adapter
        self.local_slow_parameters = local_slow_parameters

    def prepare(
        self,
        segment: SegmentBatch,
        identity: SegmentIdentity,
        transaction: LocalMemoryTransaction,
    ) -> CanonicalSegmentForward:
        result = self.adapter.scan(segment, identity=identity, transaction=transaction)
        return CanonicalSegmentForward(self, result, result.payloads, result.locals, result.identities)

    def clear_local_slow_grads(self) -> None:
        for parameter in self.local_slow_parameters:
            parameter.grad = None


def run_native_forward_for_test(
    payloads: tuple[Any, ...],
    locals: tuple[torch.Tensor | None, ...],
    all_local_tokens: torch.Tensor,
    local_slow_parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor spy; never invokes a model, data loader, or filesystem."""
    if len(payloads) != len(locals):
        raise ValueError("payload/local cardinality must match.")
    if not local_slow_parameters:
        raise ValueError("canonical test spy requires a Local slow-parameter graph anchor.")
    present_tokens = tuple(token for token in locals if token is not None)
    # A valid S0 consumer has no Local payload by contract.  Its zero-valued
    # scan/slow-owner anchors preserve a graph without changing a visible
    # consumer's exactly-once Local contribution.
    local_sum = sum((token.sum() for token in present_tokens), all_local_tokens.sum() * 0)
    local_sum = local_sum + sum((parameter.sum() * 0 for parameter in local_slow_parameters))
    return local_sum, torch.zeros_like(local_sum)


def build_segment_batch_for_test(segment: SegmentBatch) -> SegmentBatch:
    """Explicit test-only marker constructor without data-loader dependencies."""
    return segment

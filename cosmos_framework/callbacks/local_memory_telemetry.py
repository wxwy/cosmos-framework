"""Pure CPU/static Local Memory telemetry reduction for R09-B TTT O2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalTelemetrySnapshot:
    local_tokens: Tensor | None
    fast_state: Tensor | None
    fast_update: Tensor | None
    consumer_valid_count: int
    local_present_count: int
    admitted_segments: int
    committed_segments: int
    pad_rows: int
    terminal_remainders: int
    initialized_fraction: float | None
    segment_progress_mean: float | None
    txn_backward_success: int
    txn_commit: int
    txn_transient_failure: int
    txn_suffix_retry_begin: int
    txn_scaler_skip: int
    txn_slow_optimizer_step: int
    txn_retry_exhausted: int
    txn_identity_failure: int
    txn_numerical_failure: int
    txn_outer_failure: int


_COUNT_FIELDS = (
    "consumer_valid_count",
    "local_present_count",
    "admitted_segments",
    "committed_segments",
    "pad_rows",
    "terminal_remainders",
    "txn_backward_success",
    "txn_commit",
    "txn_transient_failure",
    "txn_suffix_retry_begin",
    "txn_scaler_skip",
    "txn_slow_optimizer_step",
    "txn_retry_exhausted",
    "txn_identity_failure",
    "txn_numerical_failure",
    "txn_outer_failure",
)


class LocalMemoryTelemetryProducer:
    """Validate a frozen snapshot and reduce it without side effects."""

    def record(self, snapshot: LocalTelemetrySnapshot) -> Mapping[str, float | int]:
        self._validate_counts(snapshot)
        local_tokens = self._validate_local_tokens(snapshot)
        fast_state = self._validate_fast_observation(snapshot.fast_state, "fast_state")
        fast_update = self._validate_fast_observation(snapshot.fast_update, "fast_update")
        initialized = self._validate_fraction(snapshot.initialized_fraction, "initialized_fraction")
        progress = self._validate_fraction(snapshot.segment_progress_mean, "segment_progress_mean")

        metrics: dict[str, float | int] = {
            "local/token/present_fraction": self._present_fraction(snapshot),
            "local/exposure/admitted_segments": snapshot.admitted_segments,
            "local/exposure/valid_consumers": snapshot.consumer_valid_count,
            "local/exposure/pad_rows": snapshot.pad_rows,
            "local/exposure/segments_committed": snapshot.committed_segments,
            "local/exposure/terminal_remainders": snapshot.terminal_remainders,
            "local/txn/backward_success": snapshot.txn_backward_success,
            "local/txn/commit": snapshot.txn_commit,
            "local/txn/transient_failure": snapshot.txn_transient_failure,
            "local/txn/suffix_retry_begin": snapshot.txn_suffix_retry_begin,
            "local/txn/scaler_skip": snapshot.txn_scaler_skip,
            "local/txn/slow_optimizer_step": snapshot.txn_slow_optimizer_step,
            "local/txn/retry_exhausted": snapshot.txn_retry_exhausted,
            "local/txn/identity_failure": snapshot.txn_identity_failure,
            "local/txn/numerical_failure": snapshot.txn_numerical_failure,
            "local/txn/outer_failure": snapshot.txn_outer_failure,
        }
        if local_tokens is not None:
            token_rows = local_tokens.reshape(-1, local_tokens.shape[-1])
            norms = torch.linalg.vector_norm(token_rows, dim=-1)
            metrics.update(
                {
                    "local/token/l2_mean": float(norms.mean().item()),
                    "local/token/l2_max": float(norms.max().item()),
                    "local/token/abs_max": float(local_tokens.abs().max().item()),
                }
            )
        self._add_fast_metrics(metrics, fast_state, fast_update)
        if initialized is not None:
            metrics["local/fast/initialized_fraction"] = initialized
        if progress is not None:
            metrics["local/fast/segment_progress_mean"] = progress
        return MappingProxyType(dict(metrics))

    @staticmethod
    def _validate_counts(snapshot: LocalTelemetrySnapshot) -> None:
        for name in _COUNT_FIELDS:
            value = getattr(snapshot, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative Python int")
        if snapshot.committed_segments > snapshot.admitted_segments:
            raise ValueError("committed_segments must not exceed admitted_segments")
        if snapshot.terminal_remainders > snapshot.admitted_segments:
            raise ValueError("terminal_remainders must not exceed admitted_segments")
        if snapshot.local_present_count > snapshot.consumer_valid_count:
            raise ValueError("local_present_count must not exceed consumer_valid_count")

    @staticmethod
    def _validate_local_tokens(snapshot: LocalTelemetrySnapshot) -> Tensor | None:
        tokens = snapshot.local_tokens
        if tokens is None:
            if snapshot.local_present_count != 0:
                raise ValueError("local_tokens must be present when local_present_count is non-zero")
            return None
        if snapshot.local_present_count == 0:
            raise ValueError("local_tokens must be absent when local_present_count is zero")
        detached = tokens.detach()
        if detached.device.type != "cpu" or detached.dtype != torch.float32:
            raise ValueError("local_tokens must be a CPU float32 tensor")
        if detached.ndim not in (2, 3) or detached.shape[0] != snapshot.local_present_count:
            raise ValueError("local_tokens must have shape [N,D] or [N,K,D] matching local_present_count")
        if any(size < 1 for size in detached.shape[1:]) or not torch.isfinite(detached).all().item():
            raise ValueError("local_tokens must have non-empty finite trailing dimensions")
        return detached

    @staticmethod
    def _validate_fast_observation(value: Tensor | None, name: str) -> Tensor | None:
        if value is None:
            return None
        detached = value.detach()
        if (
            type(detached).__name__ == "DTensor"
            or
            detached.device.type != "cpu"
            or detached.dtype != torch.float32
            or detached.ndim != 2
            or not detached.is_contiguous()
            or any(size < 1 for size in detached.shape)
            or not torch.isfinite(detached).all().item()
        ):
            raise ValueError(f"{name} must be a contiguous finite CPU float32 [N,D] tensor")
        return detached

    @staticmethod
    def _validate_fraction(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite float in [0, 1]")
        return value

    @staticmethod
    def _present_fraction(snapshot: LocalTelemetrySnapshot) -> float:
        if snapshot.consumer_valid_count == 0:
            return 0.0
        return float(snapshot.local_present_count / snapshot.consumer_valid_count)

    @staticmethod
    def _add_fast_metrics(
        metrics: dict[str, float | int], fast_state: Tensor | None, fast_update: Tensor | None
    ) -> None:
        observations = (("state", fast_state), ("update", fast_update))
        present = [value for _, value in observations if value is not None]
        for name, value in observations:
            if value is not None:
                norms = torch.linalg.vector_norm(value, dim=-1)
                metrics[f"local/fast/{name}_l2_mean"] = float(norms.mean().item())
                metrics[f"local/fast/{name}_l2_max"] = float(norms.max().item())
        if present:
            metrics["local/fast/finite_fraction"] = 1.0

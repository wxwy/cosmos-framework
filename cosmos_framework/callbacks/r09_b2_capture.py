"""Pure, CPU-testable non-mutating Local capture helpers for R09-B2 P2."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import torch

from cosmos_framework.utils.callback import Callback


TTT_STATE_KEYS = ("W", "pending_evidence", "last_evidence", "initialized", "segment_progress")


@contextmanager
def isolated_rng():
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def clone_local_payload(payload: list[torch.Tensor], mode: str) -> list[torch.Tensor]:
    if mode not in {"normal", "zero", "shuffle"}:
        raise ValueError(f"unsupported capture mode: {mode}")
    if not all(isinstance(tensor, torch.Tensor) for tensor in payload):
        raise TypeError("local payload must contain tensors only")
    cloned = [tensor.detach().clone() for tensor in payload]
    if mode == "zero":
        return [torch.zeros_like(tensor) for tensor in cloned]
    if mode == "shuffle":
        return list(reversed(cloned))
    return cloned


def snapshot_hash(value: object) -> str:
    """Stable CPU hash for nested immutable capture evidence."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        raw = tensor.view(torch.uint16).numpy().tobytes() if tensor.dtype is torch.bfloat16 else tensor.numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    if isinstance(value, dict):
        return hashlib.sha256(json.dumps({str(key): snapshot_hash(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}, sort_keys=True).encode()).hexdigest()
    if isinstance(value, (list, tuple)):
        return hashlib.sha256(json.dumps([snapshot_hash(item) for item in value]).encode()).hexdigest()
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _validate_ttt_state(ttt_state: object) -> None:
    if not isinstance(ttt_state, dict) or tuple(sorted(ttt_state)) != tuple(sorted(TTT_STATE_KEYS)):
        raise ValueError(f"TTT state must contain exactly {TTT_STATE_KEYS}")


@dataclass(frozen=True)
class IsolationSnapshot:
    parameters: str
    buffers: str
    optimizer: str
    scheduler: str
    cpu_rng: str
    cuda_rng: str
    batch_metadata: str
    recurrent_state: str
    ttt_state: str


def take_isolation_snapshot(*, parameters: object, buffers: object, optimizer: object, scheduler: object, batch_metadata: object, recurrent_state: object, ttt_state: object) -> IsolationSnapshot:
    _validate_ttt_state(ttt_state)
    return IsolationSnapshot(
        parameters=snapshot_hash(parameters), buffers=snapshot_hash(buffers), optimizer=snapshot_hash(optimizer),
        scheduler=snapshot_hash(scheduler), cpu_rng=snapshot_hash(torch.get_rng_state()),
        cuda_rng=snapshot_hash(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        batch_metadata=snapshot_hash(batch_metadata), recurrent_state=snapshot_hash(recurrent_state), ttt_state=snapshot_hash(ttt_state),
    )


def require_unchanged(before: IsolationSnapshot, after: IsolationSnapshot) -> None:
    if before != after:
        changed = [name for name in before.__dataclass_fields__ if getattr(before, name) != getattr(after, name)]
        raise RuntimeError(f"non-mutating capture changed protected state: {changed}")


def capture_with_isolation(
    payload: list[torch.Tensor], mode: str, snapshot_provider: Callable[[], IsolationSnapshot]
) -> list[torch.Tensor]:
    """The sole P2 capture entrypoint; always verifies supplied protected snapshots."""
    before = snapshot_provider()
    try:
        with isolated_rng():
            return clone_local_payload(payload, mode)
    finally:
        require_unchanged(before, snapshot_provider())


class R09B2NonMutatingCaptureCallback(Callback):
    """Capture detached Local payload only; never invokes the training model."""

    def __init__(self, mode: str, snapshot_provider: Callable[[], IsolationSnapshot]) -> None:
        self.mode = mode
        self._snapshot_provider = snapshot_provider

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, output_batch, loss, iteration
        required_metadata = ("b2_stream_ordinal", "b2_stream_epoch", "b2_stream_microbatch")
        missing = [key for key in required_metadata if key not in data_batch]
        if missing:
            raise ValueError(f"B2 capture batch is missing immutable metadata: {missing}")
        payload = data_batch.get("local_memory", [])
        self.last_capture = capture_with_isolation(payload, self.mode, self._snapshot_provider)
        self.last_ordinals = tuple(int(data_batch[key].item()) for key in required_metadata)

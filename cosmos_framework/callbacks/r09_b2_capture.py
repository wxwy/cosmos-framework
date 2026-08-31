"""Pure, CPU-testable non-mutating Local capture helpers for R09-B2 P2."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from dataclasses import dataclass

import torch

from cosmos_framework.utils.callback import Callback


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


@dataclass(frozen=True)
class IsolationSnapshot:
    parameters: str
    buffers: str
    optimizer: str
    scheduler: str
    rng: str
    batch_metadata: str
    recurrent_state: str
    ttt_state: str


def take_isolation_snapshot(*, parameters: object, buffers: object, optimizer: object, scheduler: object, batch_metadata: object, recurrent_state: object, ttt_state: object) -> IsolationSnapshot:
    return IsolationSnapshot(
        parameters=snapshot_hash(parameters), buffers=snapshot_hash(buffers), optimizer=snapshot_hash(optimizer),
        scheduler=snapshot_hash(scheduler), rng=snapshot_hash(torch.get_rng_state()),
        batch_metadata=snapshot_hash(batch_metadata), recurrent_state=snapshot_hash(recurrent_state), ttt_state=snapshot_hash(ttt_state),
    )


def require_unchanged(before: IsolationSnapshot, after: IsolationSnapshot) -> None:
    if before != after:
        changed = [name for name in before.__dataclass_fields__ if getattr(before, name) != getattr(after, name)]
        raise RuntimeError(f"non-mutating capture changed protected state: {changed}")


class R09B2NonMutatingCaptureCallback(Callback):
    """Capture detached Local payload only; never invokes the training model."""

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, output_batch, loss, iteration
        payload = data_batch.get("local_memory", [])
        with isolated_rng():
            self.last_capture = clone_local_payload(payload, self.mode)
        self.last_ordinals = tuple(
            int(data_batch[key].item()) for key in ("b2_stream_ordinal", "b2_stream_epoch") if key in data_batch
        )

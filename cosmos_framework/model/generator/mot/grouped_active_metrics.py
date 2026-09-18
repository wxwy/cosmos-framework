"""Opt-in run evidence from the live A2 trainer, never an authorization gate."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


class ActiveDeliveryMetrics(Callback):
    def __init__(self, *, trainer, driver, path: str) -> None:
        super().__init__()
        self.trainer, self.driver = trainer, driver
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = None
        self.last_native_calls = 0
        self.member_counts = []
        self.wave_counts = []
        self.gradients = {}
        self.losses = []
        self.starting_state_sha = None

    def on_training_step_batch_start(self, model, data, iteration=0):
        if self.started is None:
            self.started = time.monotonic()
            self.starting_state_sha = self._fast_state_sha()
        prepared = getattr(self.trainer, "_psm_active_armed_prepared", None)
        if prepared is not None:
            self.member_counts.append(prepared.actual_n_valid)
            self.wave_counts.append(getattr(self.driver.registry.owner, "last_dependency_wave_count", 1))

    def on_before_backward(self, model, loss, iteration=0):
        self.losses.append(float(loss.detach()))

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        groups = {
            "encoder": "local_memory_runtime.evidence_encoder.",
            "ttt_core": "local_memory_runtime.ttt_core.",
            "projector": "local_memory2llm.",
            "modality": "local_memory_modality_embed",
        }
        self.gradients = {}
        for label, prefix in groups.items():
            grads = [p.grad for name, p in model.net.named_parameters() if prefix in name and p.grad is not None]
            self.gradients[label] = {
                "tensors": len(grads),
                "finite": all(bool(torch.isfinite(g).all()) for g in grads),
                "max_abs": max((float(g.detach().abs().max()) for g in grads), default=0.0),
            }

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        owner = self.driver.registry.owner
        if owner.phase.name != "IDLE":
            raise RuntimeError("delivery metrics requires resolved window")
        native = getattr(model, "_psm_native_forward_calls", 0)
        optimizers = getattr(optimizer, "optimizers", (optimizer,))
        param_groups = [group for opt in optimizers for group in opt.param_groups]
        record = {
            "iteration_callback": int(iteration),
            "window_index": self.driver._window_index,
            "member_layout": getattr(self.driver, "member_layout", "single"),
            "native_forward_calls": native - self.last_native_calls,
            "member_valid_counts": self.member_counts,
            "consumers": sum(self.member_counts),
            "dependency_waves": self.wave_counts,
            "slot_epoch": dict(self.driver._slot_epoch),
            "stream_index": dict(self.driver._stream_index),
            "active_cursor": dict(self.driver._active_cursor),
            "exposure": dict(owner.scheduler.cumulative_valid_consumer_exposure),
            "identities": [
                (i.slot_id, i.episode_id, i.cursor, i.training_stream_end) for i in self.driver._window.identities
            ],
            "fast_state_sha256": self._fast_state_sha(),
            "fast_state_before_first_group_sha256": self.starting_state_sha,
            "fast_state_records": len(owner.adapter.sidecar._records),
            "local_gradients": self.gradients,
            "losses": self.losses,
            "wall_seconds": time.monotonic() - self.started,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "optimizer_group_sizes": [len(group["params"]) for group in param_groups],
            "optimizer_lrs": [group["lr"] for group in param_groups],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"[A2-EVIDENCE] window={record['window_index']} native={record['native_forward_calls']} "
            f"consumers={record['consumers']} peak_GiB={record['cuda_peak_allocated'] / 2**30:.2f}",
            flush=True,
        )
        self.last_native_calls = native
        self.member_counts, self.wave_counts, self.losses = [], [], []
        self.started = None

    def _fast_state_sha(self):
        digest = hashlib.sha256()
        for slot, (identity, state) in sorted(self.driver.registry.owner.adapter.sidecar._records.items()):
            digest.update(repr((slot, identity)).encode())
            for tensor in state:
                digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

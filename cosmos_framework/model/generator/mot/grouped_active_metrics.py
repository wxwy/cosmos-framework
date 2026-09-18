"""Opt-in delivery telemetry: actual forwards, gradients, state, and GPU budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _group(name):
    for key in (
        "local_memory_runtime.evidence_encoder",
        "local_memory_runtime.ttt_core",
        "local_memory2llm",
        "local_memory_modality_embed",
        "moe_gen",
        "time_embedder",
        "vae2llm",
        "llm2vae",
        "action2llm",
        "llm2action",
        "action_modality_embed",
    ):
        if key in name:
            return key
    return "OTHER"


class ActiveDeliveryMetrics(Callback):
    def __init__(self, *, trainer, driver, path):
        super().__init__(trainer=trainer, config=trainer.config)
        self.trainer, self.config = trainer, trainer.config
        self.driver, self.path = driver, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._previous_calls = 0
        self._losses, self._waves, self._identities = [], [], []
        self._gradients, self._before = {}, {}

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self._losses.append(float(loss.detach()))
        self._waves.append(getattr(self.driver.registry.owner, "last_dependency_wave_count", 1))
        prepared = output_batch["psm_local_memory_active_forward"].prepared
        self._identities.extend(prepared.inputs.identities)

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        self._gradients, self._before = {}, {}
        for name, parameter in model.net.named_parameters():
            if not parameter.requires_grad:
                continue
            group = self._gradients.setdefault(
                _group(name), {"tensors": 0, "with_grad": 0, "nonzero_grad": 0, "finite": True, "changed_samples": 0}
            )
            group["tensors"] += 1
            self._before[name] = _local(parameter).detach().flatten()[:8].float().cpu().clone()
            if parameter.grad is not None:
                grad = _local(parameter.grad).detach()
                group["with_grad"] += 1
                group["finite"] = group["finite"] and bool(torch.isfinite(grad).all())
                group["nonzero_grad"] += int(bool(torch.count_nonzero(grad)))

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        for name, parameter in model.net.named_parameters():
            before = self._before.get(name)
            if before is not None:
                after = _local(parameter).detach().flatten()[:8].float().cpu()
                self._gradients[_group(name)]["changed_samples"] += int(not torch.equal(before, after))
        self._before.clear()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        count = getattr(model, "_psm_native_forward_calls", 0)
        calls, self._previous_calls = count - self._previous_calls, count
        plan = getattr(self.driver, "_group_plan", None) or self.driver._window.plan
        expected = int(self.trainer.config.trainer.grad_accum_iter)
        if calls != expected or len(self._losses) != expected:
            raise RuntimeError(f"native forward accounting mismatch: {calls}, {len(self._losses)}, {expected}")
        if len(self._identities) != plan.n_window:
            raise RuntimeError("actual consumer identity count differs from the completed plan")
        identity_bytes = json.dumps(self._identities, separators=(",", ":")).encode()
        snapshot = self.driver.state_dict()
        record = {
            "iteration": int(iteration),
            "window_index": snapshot["window_index"],
            "layout": getattr(self.driver, "member_layout", "single"),
            "native_forwards": calls,
            "valid_consumers": plan.n_window,
            "group_counts": list(plan.planned_n_valid),
            "actual_consumer_identities": list(self._identities),
            "actual_consumer_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "objective_version": "native_weighted_total_separate_aux_v1",
            "loss_mean": sum(self._losses) / len(self._losses),
            "loss_min": min(self._losses),
            "loss_max": max(self._losses),
            "dependency_waves": list(self._waves),
            "gradients": self._gradients,
            "timing": self.trainer.last_optimizer_step_timing,
            "slot_epoch": snapshot["slot_epoch"],
            "stream_index": snapshot["stream_index"],
            "active_stream": snapshot["active_stream"],
            "active_cursor": snapshot["active_cursor"],
            "exposure": snapshot["runtime"].scheduler["cumulative_valid_consumer_exposure"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"[A2-EVIDENCE] iter={iteration} native={calls} consumers={plan.n_window} "
            f"loss_mean={record['loss_mean']:.6f} peak_GiB={record['peak_allocated_bytes'] / 2**30:.2f}",
            flush=True,
        )
        self._losses.clear()
        self._waves.clear()
        self._identities.clear()

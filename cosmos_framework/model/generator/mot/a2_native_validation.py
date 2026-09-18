"""Opt-in, bounded validation on the loaded real model after training.

The parity probe fixes sigma=0.5 and epsilon=0 and disables dropout: it tests
batch regrouping under identical numerical inputs, not RNG-stream equivalence.
It never steps an optimizer, changes committed stream state, or trains weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.utils import misc
from cosmos_framework.utils.callback import Callback

from .grouped_active_contract import stack_segments
from .local_memory_online import OnlineLocalMemorySession, OnlineTransition, generate_with_local_memory
from .production_active_wiring import ActiveNativeBatchInputs


def _local_native_inputs(model, segment):
    runtime = model.net.local_memory_runtime
    device = next(runtime.parameters()).device
    local, _, present = runtime.ttt_core.scan_segment_masked_encoded_many(
        runtime.evidence_encoder,
        segment.evidence_visual_summary_prev.to(device),
        segment.evidence_executed_action_prev.to(device),
        segment.evidence_valid.to(device),
        create_graph=True,
    )
    payloads, tokens, identities = segment.gather_consumers(local, present)
    return ActiveNativeBatchInputs(tuple(payloads), tuple(tokens), tuple(identities))


def _host(value):
    value = value.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.cpu().clone()


def native_group_parity(model, segments):
    parameters = {n: p for n, p in model.net.named_parameters() if p.requires_grad}
    if not parameters or any(p.grad is not None for p in parameters.values()):
        raise RuntimeError("native parity requires a completed, zero-grad training boundary")
    old_add_noise = model._add_noise_to_input
    old_calls = getattr(model, "_psm_native_forward_calls", 0)

    def schedule(*, batch_size, **kwargs):
        sigma = torch.full((batch_size, 1), 0.5, **model.tensor_kwargs_fp32)
        return sigma * model.rectified_flow_video.noise_scheduler.config.num_train_timesteps, sigma

    def zero_noise(*args, **kwargs):
        kwargs.pop("generator", None)
        return torch.zeros(*args, **kwargs)

    def add_noise(*args, **kwargs):
        with patch("torch.randn", zero_noise):
            return old_add_noise(*args, **kwargs)

    training = model.training
    started = time.monotonic()
    try:
        model.eval()
        with (
            torch.random.fork_rng(devices=[torch.cuda.current_device()]),
            torch.enable_grad(),
            patch.object(model, "_get_train_noise_level_vision", schedule),
            patch.object(model, "_get_train_noise_level_action", schedule),
            patch.object(model, "_add_noise_to_input", add_noise),
        ):
            grouped = stack_segments(tuple(segments), segment_id=0)
            inputs = _local_native_inputs(model, grouped)
            result = model._run_active_local_memory_native_forward(inputs, 0)
            objective = result.primary_consumer_mean + result.auxiliary_loss
            grouped_loss = float(objective.detach())
            objective.backward()
            grouped_grad = {name: _host(p.grad) for name, p in parameters.items() if p.grad is not None}
            del inputs, result, objective, grouped
            model.net.zero_grad(set_to_none=True)
            scalar_loss = 0.0
            for segment in segments:
                inputs = _local_native_inputs(model, segment)
                result = model._run_active_local_memory_native_forward(inputs, 0)
                objective = (result.primary_consumer_mean + result.auxiliary_loss) / len(segments)
                scalar_loss += float(objective.detach())
                objective.backward()
                del inputs, result, objective
            scalar_names = {name for name, p in parameters.items() if p.grad is not None}
            if scalar_names != set(grouped_grad):
                raise RuntimeError("native regrouping changed gradient membership")
            diff2 = ref2 = local_diff2 = local_ref2 = 0.0
            for name, expected in grouped_grad.items():
                actual = _host(parameters[name].grad).float()
                expected = expected.float()
                delta = float((actual - expected).square().sum())
                norm = float(expected.square().sum())
                diff2 += delta
                ref2 += norm
                if name.startswith(("local_memory_runtime.", "local_memory2llm", "local_memory_modality_embed")):
                    local_diff2 += delta
                    local_ref2 += norm
            error = abs(scalar_loss - grouped_loss) / max(abs(grouped_loss), 1e-12)
            grad_error = (diff2 / max(ref2, 1e-30)) ** 0.5
            local_error = (local_diff2 / max(local_ref2, 1e-30)) ** 0.5
            report = {
                "grouped_loss": grouped_loss,
                "scalar_loss": scalar_loss,
                "loss_relative_error": error,
                "gradient_relative_l2": grad_error,
                "local_gradient_relative_l2": local_error,
                "gradient_tensors": len(grouped_grad),
                "grouped_forwards": 1,
                "scalar_forwards": len(segments),
                "consumers": sum(int(s.consumer_valid.sum()) for s in segments),
                "fixture": "real model/data, eval dropout, sigma=0.5, epsilon=0; no optimizer step",
                "tolerances": {"loss_relative": 0.01, "gradient_relative_l2": 0.05},
                "wall_seconds": time.monotonic() - started,
            }
            report["result"] = "PASS" if error <= 0.01 and grad_error <= 0.05 and local_error <= 0.05 else "FAIL"
            return report
    finally:
        model.net.zero_grad(set_to_none=True)
        model.train(training)
        model._psm_native_forward_calls = old_calls



def rgb_visual_summary_parity(model, producer, stream):
    """Real raw-RGB witness for the causal visual96 evidence used online.

    The exact-window cache is [T_latent,C_latent,H,W].  We decode the same
    17-frame dataset item, encode it through the loaded model VAE, and also
    encode only its first causal frame (the online-server route).  Causality
    requires that the first latent / pooled visual96 agree.
    """
    source = producer.frame_source
    old_ratio = source._latent_cache_verify_ratio
    try:
        source._latent_cache_verify_ratio = 1.0
        item = source._build_item(producer._flat_index(stream, 0))
    finally:
        source._latent_cache_verify_ratio = old_ratio
    raw = item.get("video")
    cached = item.get("video_latent")
    if not isinstance(raw, torch.Tensor) or raw.dtype is not torch.uint8:
        raise RuntimeError("RGB parity requires the real uint8 dataset window")
    if not isinstance(cached, torch.Tensor) or cached.ndim != 4:
        raise RuntimeError("RGB parity requires the exact-window cached latent")
    device = next(model.net.parameters()).device
    with torch.inference_mode():
        full = model._encode_uint8_vision_item(raw.unsqueeze(0).to(device)).float()
        first = model._encode_uint8_vision_item(raw[:, :1].unsqueeze(0).to(device)).float()
    expected = cached.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float32)
    if tuple(full.shape) != tuple(expected.shape) or first.shape[2] != 1:
        raise RuntimeError(
            f"RGB parity latent shape mismatch: full={tuple(full.shape)} cache={tuple(expected.shape)} first={tuple(first.shape)}"
        )
    full_max = float((full - expected).abs().max())
    first_max = float((first[:, :, 0] - expected[:, :, 0]).abs().max())
    cached96 = F.adaptive_avg_pool2d(expected[0, :, 0].unsqueeze(0), (1, 2)).flatten()
    online96 = F.adaptive_avg_pool2d(first[0, :, 0].unsqueeze(0), (1, 2)).flatten()
    visual96_max = float((online96 - cached96).abs().max())
    # Existing exact-window parity is normally near numerical roundoff.  Keep a
    # strict but non-bitwise tolerance because VAE kernels may use different
    # legal batch/temporal launch geometry.
    tolerance = 2e-4
    return {
        "result": "PASS" if full_max <= tolerance and first_max <= tolerance and visual96_max <= tolerance else "FAIL",
        "episode_id": str(stream.episode_index),
        "raw_shape": list(raw.shape),
        "cache_shape": list(cached.shape),
        "full_latent_max_abs": full_max,
        "first_causal_latent_max_abs": first_max,
        "visual96_max_abs": visual96_max,
        "tolerance": tolerance,
        "scope": "same real raw 17-frame LIBERO window: cache vs full VAE and first-frame causal online visual96",
    }

def native_online_probe(model, segment):
    """Recorded observations exercise real generation, not an environment rollout."""
    runtime = model.net.local_memory_runtime
    before = {name: _host(value) for name, value in runtime.state_dict().items()}
    session = OnlineLocalMemorySession(runtime)
    device = next(model.net.parameters()).device
    was_training = model.training
    try:
        model.eval()
        outputs = []
        for step in (0, 1):
            batch = custom_collate_fn([segment.consumer_payload[0][step]])
            batch = misc.to(batch, device=device)
            transition = OnlineTransition(
                int(segment.slot_id[0]),
                segment.episode_id[0],
                step,
                None if step == 0 else segment.evidence_visual_summary_prev[0, step],
                None if step == 0 else segment.evidence_executed_action_prev[0, step],
            )
            result = generate_with_local_memory(
                model, batch, [transition], session, guidance=1.0, seed=[0], num_steps=2, has_negative_prompt=False
            )
            outputs.append(result["action"][0].detach().cpu())
        without = generate_with_local_memory(
            model,
            batch,
            [transition],
            session,
            use_local_tokens=False,
            guidance=1.0,
            seed=[0],
            num_steps=2,
            has_negative_prompt=False,
        )
        max_difference = float((outputs[-1] - without["action"][0].detach().cpu()).abs().max())
        token = session._records[int(segment.slot_id[0])][4]
        unchanged = all(torch.equal(_host(value), before[name]) for name, value in runtime.state_dict().items())
        no_slow_grads = all(p.grad is None for p in model.net.parameters())
        return {
            "result": "PASS" if unchanged and no_slow_grads and token is not None else "FAIL",
            "scope": "real cached-observation policy generation; not closed-loop success evaluation",
            "consumer_steps": [0, 1],
            "denoise_steps": 2,
            "action_shapes": [list(value.shape) for value in outputs],
            "actions_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
            "local_token_max_abs": float(token.abs().max()),
            "local_on_off_action_max_difference": max_difference,
            "slow_weights_unchanged": unchanged,
            "no_slow_parameter_grads": no_slow_grads,
        }
    finally:
        model.train(was_training)


class A2NativeValidation(Callback):
    def __init__(self, *, driver, path: str) -> None:
        super().__init__()
        self.driver, self.path = driver, Path(path)

    def on_train_end(self, model, iteration=0):
        if self.driver.registry.owner.phase.name != "IDLE":
            raise RuntimeError("native delivery probe requires a completed training boundary")
        streams = tuple(rows[0] for _, rows in sorted(self.driver._by_slot.items()))[: self.driver.group_size]
        segments = tuple(self.driver.producer.produce(stream, cursor=0) for stream in streams)
        report = {"iteration": int(iteration), "result": "BLOCKED", "independent_review": False}
        try:
            report["native_group_parity"] = native_group_parity(model, segments)
            report["online_generation"] = native_online_probe(model, segments[0])
            producer = self.driver.producer._route(streams[0])
            report["rgb_visual96_parity"] = rgb_visual_summary_parity(model, producer, streams[0])
            report["result"] = (
                "PASS"
                if all(
                    report[key]["result"] == "PASS"
                    for key in ("native_group_parity", "online_generation", "rgb_visual96_parity")
                )
                else "FAIL"
            )
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            report["result"] = "FAIL"
            raise
        finally:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if report["result"] != "PASS":
            raise RuntimeError("A2 native validation failed; inspect its report")
        print(f"[A2-NATIVE-VALIDATION] PASS: {self.path}", flush=True)

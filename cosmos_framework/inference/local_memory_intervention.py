# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired Local-prefix intervention using the unchanged public generation method.

Caller supplies an already-prepared native batch and committed Local prefix.
This function does not execute an environment step, prepare/commit a Local
session, load a checkpoint, decode an env command, or touch optimizer state.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch


def run_matched_local_intervention(
    model, batch, prefixes, *, seed: int, num_steps: int, guidance: float, raw_action_dim: int = 15
) -> dict:
    if getattr(model, "training", True):
        raise ValueError("matched intervention requires model.eval() before invocation")
    if not isinstance(prefixes, (tuple, list)) or len(prefixes) != 1 or prefixes[0] is None:
        raise ValueError("matched intervention requires one nonempty committed prefix")
    if not isinstance(prefixes[0], torch.Tensor) or not torch.isfinite(prefixes[0]).all():
        raise ValueError("Local prefix must be a finite tensor")
    if type(seed) is not int or type(num_steps) is not int or num_steps <= 0 or raw_action_dim != 15:
        raise ValueError("expected explicit fixed seed, positive steps, raw15")
    runtime = getattr(getattr(model, "net", None), "local_memory_runtime", None)
    if runtime is None:
        raise ValueError("required Local runtime missing")
    before = {k: v.detach().cpu().clone() for k, v in runtime.state_dict().items()}
    params_before = [(id(p), p._version) for p in model.parameters()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    outputs = []
    try:
        with torch.random.fork_rng(devices=devices):
            cpu_state = torch.get_rng_state()
            cuda_states = {device: torch.cuda.get_rng_state(device) for device in devices}
            for selected in (tuple(prefixes), None):
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(cpu_state)
                for device, state in cuda_states.items():
                    torch.cuda.set_rng_state(state, device)
                with torch.inference_mode():
                    result = model.generate_samples_from_batch(
                        copy.deepcopy(batch),
                        guidance=guidance,
                        seed=[seed],
                        num_steps=num_steps,
                        has_negative_prompt=False,
                        _local_memory_prefixes=selected,
                    )
                action = result["action"][0].detach().float().cpu()
                if action.ndim == 3 and action.shape[0] == 1:
                    action = action[0]
                if action.ndim != 2 or action.shape[-1] < raw_action_dim or not torch.isfinite(action).all():
                    raise ValueError("native generation returned invalid action")
                outputs.append(action[:, :raw_action_dim].clone())
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    if outputs[0].shape != outputs[1].shape:
        raise ValueError("Local intervention changed action shape")
    after = runtime.state_dict()
    local_unchanged = before.keys() == after.keys() and all(
        torch.equal(before[k], after[k].detach().cpu()) for k in before
    )
    versions_unchanged = params_before == [(id(p), p._version) for p in model.parameters()]
    if not local_unchanged or not versions_unchanged:
        raise RuntimeError("generation mutated Local slow state or model parameter versions")
    difference = outputs[0] - outputs[1]
    return {
        "scope": "same_batch_seed_native_generation_only_not_task_success",
        "seed": seed,
        "num_steps": num_steps,
        "guidance": guidance,
        "action_shape": list(outputs[0].shape),
        "raw_action_dim": raw_action_dim,
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "l2_difference": float(torch.linalg.vector_norm(difference)),
        "exact_equal": torch.equal(outputs[0], outputs[1]),
        "local_slow_state_bitwise_unchanged": local_unchanged,
        "host_parameter_versions_unchanged": versions_unchanged,
        "local_on_raw15": outputs[0].tolist(),
        "local_off_raw15": outputs[1].tolist(),
    }

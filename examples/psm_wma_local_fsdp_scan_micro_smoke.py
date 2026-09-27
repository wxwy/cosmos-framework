"""V3 B2-C R1-B tiny Local FSDP2 lifecycle smoke；GPU 执行须另行审核。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import DTensor

from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTFastState
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh

LOCAL_PARAMS = 165_312
STEPS = 16
K_LOCAL = 4
LOCAL_DIM = 32
ROOT_ENV = "PSM_WMA_V3_ROOT"


class TinyLocalOwner(nn.Module):
    """仅持有正式 Local slow 参数；不实例化 Cosmos host。"""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(local_memory_enabled=True)
        self.local_memory_runtime = LocalMemoryRuntime()
        self.local_memory2llm = nn.Linear(LOCAL_DIM, 2048)
        self.local_memory_modality_embed = nn.Parameter(torch.empty(2048))
        nn.init.normal_(self.local_memory_modality_embed, std=2048**-0.5)

    def scan_local_memory(
        self,
        visual_summary: torch.Tensor,
        executed_action: torch.Tensor,
        valid: torch.Tensor,
        state_in: ContinualTTTFastState | None,
    ) -> tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]:
        if not self.config.local_memory_enabled:
            raise RuntimeError("Local Memory is disabled")
        runtime = self.local_memory_runtime
        return runtime.core.scan_segment_masked_encoded_many(
            runtime.encoder, visual_summary, executed_action, valid, state_in
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preflight", action="store_true", help="仅检查 tiny owner/合同，不接触 CUDA")
    result.add_argument("--output", type=Path, required=True, help="必须是尚不存在的唯一目录")
    return result


def make_parallel_dims() -> ParallelDims:
    return ParallelDims(
        world_size=1,
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        cfgp=1,
        enable_inference_mode=False,
    )


def wrap_local_owner(owner: TinyLocalOwner, dims: ParallelDims) -> TinyLocalOwner:
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    owner = fully_shard(owner, mesh=fsdp_mesh(dims), mp_policy=policy)
    register_fsdp_forward_method(owner, "scan_local_memory")
    owner._local_memory_scan_fsdp_registered = True
    return owner


def validate_world_size() -> None:
    for key, expected in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)):
        value = int(os.environ.get(key, str(expected)))
        if value != expected:
            raise ValueError(f"R1-B requires {key}={expected}, got {value}")


def lock_implementation_pair() -> dict[str, str]:
    root_text = os.environ.get(ROOT_ENV)
    if not root_text:
        raise ValueError(f"GPU run requires {ROOT_ENV} pointing to the V3 root worktree")
    root = Path(root_text).expanduser().resolve()
    child = Path(__file__).resolve().parents[1]

    def git(path: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    root_sha = git(root, "rev-parse", "HEAD")
    child_sha = git(child, "rev-parse", "HEAD")
    gitlink = git(root, "ls-tree", "HEAD", "cosmos-framework").split()[2]
    if child_sha != gitlink or git(child, "status", "--porcelain"):
        raise ValueError("R1-B root Gitlink、child HEAD 或 child 工作树不匹配")
    return {"root": root_sha, "child": child_sha, "gitlink": gitlink}


def local_inventory(owner: TinyLocalOwner) -> int:
    selected = [(name, parameter) for name, parameter in owner.named_parameters() if "local_memory" in name]
    count = sum(parameter.numel() for _, parameter in selected)
    if count != LOCAL_PARAMS or len(selected) != len(list(owner.parameters())):
        raise ValueError(f"tiny Local inventory mismatch: {count}")
    return count


def cuda_memory(trace: list[dict[str, Any]], phase: str) -> None:
    torch.cuda.synchronize()
    trace.append(
        {
            "phase": phase,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )


def inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = torch.linspace(-1, 1, STEPS * 96, device=device, dtype=torch.float32).reshape(1, STEPS, 96)
    action = torch.linspace(-1, 1, STEPS * 15, device=device, dtype=torch.float32).reshape(1, STEPS, 15)
    valid = torch.ones((1, STEPS), device=device, dtype=torch.bool)
    valid[:, 0] = False
    return visual, action, valid


def _ordinary_finite(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if isinstance(value, DTensor) or value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
        raise ValueError(f"Local scan output must be finite ordinary {dtype} tensor with shape {shape}")


def _gradient_norm(parameter: nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise ValueError(f"missing Local gradient: {name}")
    gradient = parameter.grad.detach()
    if isinstance(gradient, DTensor):
        gradient = gradient.full_tensor()
    if not torch.isfinite(gradient).all():
        raise ValueError(f"nonfinite Local gradient: {name}")
    norm = float(gradient.float().norm())
    if norm <= 0:
        raise ValueError(f"zero Local gradient: {name}")
    return norm


def run_gpu(output: Path, result: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    validate_world_size()  # 不符合单 rank 合同时，必须在任何 CUDA 调用之前退出。
    result["phase"] = "pair_lock"
    result["pair"] = lock_implementation_pair()
    if dist.is_initialized():
        raise RuntimeError("R1-B requires an isolated process group")
    if not torch.cuda.is_available():
        raise RuntimeError("R1-B GPU run requires CUDA")
    result["phase"] = "cuda_init"
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    cuda_memory(trace, "cuda_init")
    result["device"] = {
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    result["phase"] = "process_group"
    dist.init_process_group(backend="nccl", init_method=(output / "process_group.store").as_uri(), rank=0, world_size=1)
    result["process_group"] = {"backend": dist.get_backend(), "world_size": dist.get_world_size()}
    result["phase"] = "parallel_dims"
    dims = make_parallel_dims()
    dims.build_meshes("cuda")
    if not dims.dp_enabled or dims.enable_inference_mode:
        raise RuntimeError("R1-B requires training-mode root FSDP")
    result["parallel_dims"] = {
        "world_size": dims.world_size,
        "dp_shard": dims.dp_shard,
        "dp_replicate": dims.dp_replicate,
        "cp": dims.cp,
        "cfgp": dims.cfgp,
        "enable_inference_mode": dims.enable_inference_mode,
    }
    result["mixed_precision"] = {
        "master": "float32",
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "cast_forward_inputs": False,
    }
    result["phase"] = "owner_materialized"
    torch.manual_seed(27)
    owner = TinyLocalOwner().to(device=device, dtype=torch.float32).train()
    result["local_parameter_count"] = local_inventory(owner)
    cuda_memory(trace, "owner_materialized")
    result["phase"] = "root_fsdp_wrapped"
    owner = wrap_local_owner(owner, dims)
    cuda_memory(trace, "root_fsdp_wrapped")
    result["local_parameter_types_before_scan"] = {
        name: type(parameter).__name__ for name, parameter in owner.local_memory_runtime.named_parameters()
    }
    if not any(isinstance(parameter, DTensor) for parameter in owner.local_memory_runtime.parameters()):
        raise RuntimeError("R1-B requires DTensor-owned Local parameters before scan")
    result["phase"] = "before_scan"
    visual, action, valid = inputs(device)
    if any(isinstance(value, DTensor) for value in (visual, action, valid)):
        raise RuntimeError("B0 evidence must be ordinary CUDA tensors")
    result["inputs"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype), "type": type(value).__name__}
        for name, value in (("visual", visual), ("action", action), ("valid", valid))
    }
    cuda_memory(trace, "before_scan")
    result["phase"] = "scan"
    tokens, candidate, present = owner.scan_local_memory(visual, action, valid, None)
    cuda_memory(trace, "after_scan")
    _ordinary_finite(tokens, (1, STEPS, K_LOCAL, LOCAL_DIM), torch.float32)
    if isinstance(present, DTensor) or present.shape != (1, STEPS) or present.dtype != torch.bool:
        raise ValueError("Local present must be ordinary [1,16] bool")
    if not torch.equal(present, valid):
        raise ValueError("S0/present mask mismatch")
    owner.local_memory_runtime.core.validate_state(candidate, 1)
    if any(isinstance(value, DTensor) for value in candidate):
        raise ValueError("candidate fast state must not be DTensor")
    result["outputs"] = {
        "tokens_shape": list(tokens.shape),
        "tokens_dtype": str(tokens.dtype),
        "present": present.tolist(),
        "candidate_shapes": [list(value.shape) for value in candidate],
        "candidate_dtypes": [str(value.dtype) for value in candidate],
        "finite": True,
    }
    scalar = tokens[present].square().mean()
    if not torch.isfinite(scalar):
        raise ValueError("Local scalar is nonfinite")
    result["phase"] = "backward"
    scalar.backward()
    cuda_memory(trace, "after_backward")
    runtime = owner.local_memory_runtime
    result["gradient_norms"] = {
        "encoder.visual_proj.weight": _gradient_norm(runtime.encoder.visual_proj.weight, "encoder.visual_proj.weight"),
        "core.slot_queries": _gradient_norm(runtime.core.slot_queries, "core.slot_queries"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    requested = args.output.expanduser().resolve()
    result: dict[str, Any] = {"status": "FAIL", "preflight": args.preflight, "phase": "arguments"}
    trace: list[dict[str, Any]] = []
    output: Path | None = None
    initial_group = dist.is_initialized()
    try:
        requested.mkdir(parents=True, exist_ok=False)
        output = requested
        result["output"] = str(output)
        result["command"] = sys.argv if argv is None else [sys.argv[0], *argv]
        result["environment"] = {
            key: os.environ.get(key) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES", ROOT_ENV)
        }
        result["phase"] = "cpu_preflight" if args.preflight else "gpu_lifecycle"
        if args.preflight:
            owner = TinyLocalOwner()
            result["local_parameter_count"] = local_inventory(owner)
            dims = make_parallel_dims()
            result["parallel_dims"] = {
                "world_size": dims.world_size,
                "dp_shard": dims.dp_shard,
                "dp_replicate": dims.dp_replicate,
                "cp": dims.cp,
                "cfgp": dims.cfgp,
                "enable_inference_mode": dims.enable_inference_mode,
                "dp_enabled": dims.dp_enabled,
            }
            result["model"] = "TinyLocalOwner"
        else:
            run_gpu(output, result, trace)
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "phase": result["phase"]}
        result["traceback"] = traceback.format_exc()
        if not args.preflight and torch.cuda.is_initialized():
            try:
                cuda_memory(trace, "failure")
            except Exception as memory_error:
                result["memory_trace_error"] = str(memory_error).splitlines()[0]
        if output is None:
            requested.parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=f"{requested.name}.rejected_", dir=requested.parent))
            result["output"] = str(output)
    finally:
        if dist.is_initialized() and not initial_group:
            try:
                dist.destroy_process_group()
                result["process_group_destroyed"] = True
            except Exception as teardown_error:
                result["status"] = "FAIL"
                result["process_group_destroyed"] = False
                result["process_group_destroy_error"] = str(teardown_error)
        else:
            result["process_group_destroyed"] = False
        if output is not None:
            (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            (output / "cuda_memory_trace.json").write_text(json.dumps(trace, indent=2))
    print(
        json.dumps(
            {"status": result["status"], "output": result.get("output"), "error": result.get("error")},
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

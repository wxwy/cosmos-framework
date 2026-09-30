"""V3 H3-F：RoboCasa target-atomic 8×H100 generation+Local-TTT 30k formal training launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.model.generator.mot.robocasa_grouped_segment import (
    RankLocalGroupedPlanner,
    StageARoboCasaEpisodeBinder,
)
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer
from cosmos_framework.utils import distributed
from cosmos_framework.utils.context_managers import model_init
from cosmos_framework.utils.lazy_config import instantiate
from examples.psm_wma_robocasa_h100 import (
    H3F_FORMAL_CHECKPOINT_ITERS,
    H3F_FORMAL_MAX_ITER,
    LOCAL_PARAMS,
    MANIFEST_DIGEST,
    GroupedTriggerLoader,
    _local,
    _optimizer_parameter_ids,
    _paths,
    build_stage_a_action_transform,
    load_stage_a_config,
    lock_pair,
    make_catalog,
    preflight_native_batch,
    read_stage_a_contract,
    runtime_paths,
    validate_h100_asset_authority,
)
from examples.psm_wma_robocasa_h100 import (
    config_digest as h3e_config_digest,
)

H3F_SAVE_ITER = 500
H3F_WARMUP_STEPS = 500
H3F_GROUP = "h3f_edge_local_h100"
H3F_READINESS_GROUP = "h3f_edge_local_h100_readiness"
H3F_READINESS_MAX_STEPS = 100
H3F_GENERATION_KEYS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
H3F_LOCAL_KEYS = (
    "local_memory_runtime.encoder.",
    "local_memory_runtime.core.",
    "local_memory2llm.",
    "local_memory_modality_embed",
)
H3F_ACTION_KEYS = ("action2llm", "action_modality_embed", "llm2action")
H3F_OPTIMIZER_KEYS = (*H3F_GENERATION_KEYS, *H3F_LOCAL_KEYS)
H3F_TRAINABLE_PROFILE = "v2_semantic_generation+local_v3_raw15"
H3F_MESH_PROFILE = "dp_shard8_generation_and_local"
H3F_DATA_PROFILE = "official_v30_raw15"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("fresh", "resume"), required=True)
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--job-name", default="edge_local_target_atomic_30k")
    result.add_argument("--attempt", type=int, default=1)
    result.add_argument("--readiness-steps", type=int)
    result.add_argument("--expected-root", required=True)
    result.add_argument("--expected-child", required=True)
    return result


def overlay_h3f_config(
    config: Any,
    *,
    phase: str,
    job_name: str,
    readiness_steps: int | None = None,
    save_iter: int = H3F_SAVE_ITER,
    runtime=None,
    resume_checkpoint: Path | None = None,
) -> None:
    from examples.psm_wma_robocasa_h100 import overlay_h100_config

    runtime = runtime or runtime_paths()
    overlay_h100_config(config, phase=phase, job_name=job_name, runtime=runtime)
    if phase == "fresh":
        config.checkpoint.load_path = str(runtime.base_checkpoint)
        config.checkpoint.load_training_state = False
        config.checkpoint.keys_to_skip_loading = ["net_ema.", "local_memory"]
    else:
        if resume_checkpoint is None:
            raise ValueError("H3-F resume 必须显式提供 same-job checkpoint")
        resume_checkpoint = resume_checkpoint.expanduser().resolve()
        config.checkpoint.load_path = str(resume_checkpoint)
        config.checkpoint.load_training_state = True
        config.checkpoint.keys_to_skip_loading = ["net_ema."]
    target_iteration = readiness_steps if readiness_steps is not None else H3F_FORMAL_MAX_ITER
    config.optimizer.keys_to_select = list(H3F_OPTIMIZER_KEYS)
    config.optimizer.lr_multipliers = {key: 5.0 for key in H3F_ACTION_KEYS}
    config.trainer.max_iter = target_iteration
    config.trainer.logging_iter = 50
    config.scheduler.cycle_lengths = [H3F_FORMAL_MAX_ITER]
    config.scheduler.warm_up_steps = [H3F_WARMUP_STEPS]
    config.checkpoint.save_iter = target_iteration if readiness_steps is not None else save_iter
    config.job.group = H3F_READINESS_GROUP if readiness_steps is not None else H3F_GROUP
    config.job.name = job_name
    if (
        config.optimizer.keys_to_select != list(H3F_OPTIMIZER_KEYS)
        or config.optimizer.lr_multipliers != {key: 5.0 for key in H3F_ACTION_KEYS}
        or float(config.optimizer.lr) != 5e-5
        or float(config.optimizer.weight_decay) != 0.05
        or config.model.config.parallelism.data_parallel_shard_degree != 8
        or config.model.config.parallelism.data_parallel_replicate_degree != 1
        or config.model.config.local_memory_action_dim != 15
        or config.trainer.grad_accum_iter != 2
        or config.trainer.max_iter != target_iteration
        or config.scheduler.cycle_lengths != [30_000]
        or config.scheduler.warm_up_steps != [500]
        or config.checkpoint.save_iter != (target_iteration if readiness_steps is not None else save_iter)
        or config.checkpoint.load_path
        != str(runtime.base_checkpoint if phase == "fresh" else resume_checkpoint)
        or config.checkpoint.load_training_state != (phase == "resume")
        or save_iter <= 0
        or any(step % save_iter for step in H3F_FORMAL_CHECKPOINT_ITERS)
    ):
        raise ValueError("H3-F 30k/scheduler/checkpoint 合同不匹配")


def config_digest(save_iter: int = H3F_SAVE_ITER, runtime=None) -> str:
    runtime = runtime or runtime_paths()
    authority = {
        "h3e_runtime": h3e_config_digest(runtime),
        "trainable_profile": H3F_TRAINABLE_PROFILE,
        "optimizer_keys": list(H3F_OPTIMIZER_KEYS),
        "lr_multipliers": {key: 5.0 for key in H3F_ACTION_KEYS},
        "mesh_profile": H3F_MESH_PROFILE,
        "data_profile": H3F_DATA_PROFILE,
        "local_memory_action_dim": 15,
        "max_iter": H3F_FORMAL_MAX_ITER,
        "save_iter": save_iter,
        "warmup_steps": H3F_WARMUP_STEPS,
        "cycle_lengths": [H3F_FORMAL_MAX_ITER],
        "primary_eval_iters": list(H3F_FORMAL_CHECKPOINT_ITERS),
    }
    return hashlib.sha256(json.dumps(authority, sort_keys=True).encode()).hexdigest()


def readiness_config_digest(steps: int, save_iter: int = H3F_SAVE_ITER, runtime=None) -> str:
    runtime = runtime or runtime_paths()
    authority = {
        "formal_config_digest": config_digest(save_iter, runtime),
        "readiness_steps": steps,
        "scheduler_cycle": [H3F_FORMAL_MAX_ITER],
        "scheduler_warmup": [H3F_WARMUP_STEPS],
    }
    return hashlib.sha256(json.dumps(authority, sort_keys=True).encode()).hexdigest()


def _job(output_root: Path, job_name: str, *, readiness_steps: int | None = None) -> Path:
    group = H3F_READINESS_GROUP if readiness_steps is not None else H3F_GROUP
    return output_root / "psm_wma_v3" / group / job_name


def _resume_iteration(job: Path) -> int:
    latest = job / "checkpoints/latest_checkpoint.txt"
    if not latest.is_file():
        raise FileNotFoundError("H3-F resume 缺少 same-job latest_checkpoint.txt")
    value = latest.read_text().strip()
    prefix = "iter_"
    if not value.startswith(prefix) or not value[len(prefix) :].isdigit():
        raise ValueError("H3-F latest_checkpoint.txt 格式非法")
    iteration = int(value[len(prefix) :])
    if not 0 < iteration < H3F_FORMAL_MAX_ITER:
        raise ValueError("H3-F resume iteration 必须位于 1..29999")
    return iteration


def _evidence_dir(job: Path, *, phase: str, attempt: int, start_iteration: int) -> Path:
    return job / "h3f_evidence" / f"attempt_{attempt:04d}_{phase}_from_{start_iteration:09d}"


def _authority_preflight_rank() -> bool:
    rank = os.environ.get("RANK")
    return rank is None or int(rank) == 0


def _validate_output_root(output_root: Path, runtime) -> None:
    if output_root == runtime.root_worktree:
        raise ValueError("H3-F 产物不能直接写入 root worktree")
    if runtime.root_worktree in output_root.parents:
        allowed = runtime.root_worktree / "outputs"
        if output_root != allowed and allowed not in output_root.parents:
            raise ValueError("H3-F worktree 内只允许写入已忽略的 outputs/ 子树")


def _validate_contract_env(output_root: Path, runtime) -> dict[str, Any]:
    exact_ints = {
        "TTT_TBPTT_STEPS": 16,
        "TTT_DIM": 64,
        "TTT_FAST_HIDDEN_DIM": 256,
        "TTT_K_LOCAL": 4,
        "TTT_B_STREAM": 8,
        "TTT_ACTIVE_GA": 2,
    }
    observed: dict[str, Any] = {}

    save_value = os.environ.get("SAVE_ITER", str(H3F_SAVE_ITER))
    try:
        save_iter = int(save_value)
    except ValueError as error:
        raise ValueError("H3-F SAVE_ITER 必须为正整数") from error
    if save_iter <= 0 or any(step % save_iter for step in H3F_FORMAL_CHECKPOINT_ITERS):
        raise ValueError("H3-F SAVE_ITER 必须为正整数且整除全部 primary eval milestones")
    observed["SAVE_ITER"] = save_iter

    for name, expected in exact_ints.items():
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"H3-F {name} 必须为整数") from error
        if parsed != expected:
            raise ValueError(f"H3-F {name}={parsed} 与冻结值 {expected} 不匹配")
        observed[name] = parsed

    inner_lr = os.environ.get("TTT_INNER_LR")
    if inner_lr is not None:
        try:
            parsed_lr = float(inner_lr)
        except ValueError as error:
            raise ValueError("H3-F TTT_INNER_LR 必须为浮点数") from error
        if parsed_lr != 0.1:
            raise ValueError(f"H3-F TTT_INNER_LR={parsed_lr} 与冻结值 0.1 不匹配")
        observed["TTT_INNER_LR"] = parsed_lr

    for name, expected in (
        ("PSM_WMA_ROOT", runtime.root_worktree),
        ("STAGE_A_CHECKPOINT_PATH", runtime.stage_a_checkpoint),
        ("STAGE_A_CONFIG_PATH", runtime.stage_a_config),
        ("ROBOCASA_ROOT", runtime.dataset_root),
        ("ROBOCASA_LATENT_CACHE_ROOT", runtime.cache_root),
        ("ROBOCASA_LATENT_CACHE_PROBE", runtime.cache_probe),
        ("EDGE_POLICY_CHECKPOINT", runtime.edge),
        ("WAN_VAE_PATH", runtime.vae),
        ("BASE_CHECKPOINT_PATH", runtime.base_checkpoint),
    ):
        value = os.environ.get(name)
        if value is None:
            raise ValueError(f"H3-F 缺少必需运行参数 {name}")
        resolved = Path(value).expanduser().resolve()
        if resolved != expected:
            raise ValueError(f"H3-F {name} 与解析后的运行参数不匹配")
        observed[name] = str(resolved)

    if not (runtime.base_checkpoint / "model/.metadata").is_file():
        raise FileNotFoundError(
            f"H3-F BASE_CHECKPOINT_PATH 缺少 model/.metadata：{runtime.base_checkpoint}"
        )
    observed["DIRECT_BASE_CHECKPOINT"] = str(runtime.base_checkpoint)

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None and cuda != "0,1,2,3,4,5,6,7":
        raise ValueError("H3-F CUDA_VISIBLE_DEVICES 必须为 0,1,2,3,4,5,6,7")
    if cuda is not None:
        observed["CUDA_VISIBLE_DEVICES"] = cuda

    output_env = os.environ.get("OUTPUT_ROOT")
    if output_env is not None and Path(output_env).expanduser().resolve() != output_root:
        raise ValueError("H3-F OUTPUT_ROOT 与 --output-root 不一致")
    if output_env is not None:
        observed["OUTPUT_ROOT"] = str(output_root)

    workers = os.environ.get("ROBOCASA_NUM_WORKERS")
    if workers is not None:
        try:
            parsed_workers = int(workers)
        except ValueError as error:
            raise ValueError("H3-F ROBOCASA_NUM_WORKERS 必须为正整数") from error
        if parsed_workers <= 0:
            raise ValueError("H3-F ROBOCASA_NUM_WORKERS 必须为正整数")
        observed["ROBOCASA_NUM_WORKERS"] = parsed_workers
        observed["ROBOCASA_NUM_WORKERS_EFFECT"] = (
            "contract-only; grouped planner/binder 同步物化，不驱动 DataLoader workers"
        )

    return observed


def _disk_free_bytes(output_root: Path) -> int:
    probe = output_root.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    runtime = runtime_paths()
    pair = lock_pair(args.expected_root, args.expected_child, runtime)
    output_root = args.output_root.expanduser().resolve()
    _validate_output_root(output_root, runtime)
    if not args.job_name or "/" in args.job_name or args.job_name in (".", ".."):
        raise ValueError("H3-F job-name 不合法")
    if type(args.attempt) is not int or args.attempt <= 0:
        raise ValueError("H3-F attempt 必须为正整数")

    readiness_steps = args.readiness_steps
    if readiness_steps is not None:
        if args.phase != "fresh" or not 2 <= readiness_steps <= H3F_READINESS_MAX_STEPS:
            raise ValueError("H3-F readiness-steps 仅允许 fresh 且范围 2..100")
        if args.attempt != 1:
            raise ValueError("H3-F readiness smoke 必须使用 attempt=1")
    job = _job(output_root, args.job_name, readiness_steps=readiness_steps)
    authority_rank = _authority_preflight_rank()
    if args.phase == "fresh":
        if args.attempt != 1:
            raise ValueError("H3-F fresh 必须使用 attempt=1")
        if job.exists() and authority_rank:
            raise FileExistsError("H3-F fresh job 已存在；禁止覆盖")
        start_iteration = 0
        resume_checkpoint = None
    else:
        if args.attempt < 2:
            raise ValueError("H3-F resume 必须使用 attempt>=2")
        start_iteration = _resume_iteration(job)
        resume_checkpoint = job / "checkpoints" / f"iter_{start_iteration:09d}"
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(f"H3-F same-job resume checkpoint 不存在：{resume_checkpoint}")

    evidence = _evidence_dir(job, phase=args.phase, attempt=args.attempt, start_iteration=start_iteration)
    if evidence.exists() and authority_rank:
        raise FileExistsError("H3-F evidence attempt 已存在；禁止覆盖")

    contract_env = _validate_contract_env(output_root, runtime)
    save_iter = int(contract_env["SAVE_ITER"])
    paths = _paths(output_root, runtime)
    asset_authority = validate_h100_asset_authority(runtime)
    contract = read_stage_a_contract(paths)
    config = load_stage_a_config(runtime)
    overlay_h3f_config(
        config,
        phase=args.phase,
        job_name=args.job_name,
        readiness_steps=readiness_steps,
        save_iter=save_iter,
        runtime=runtime,
        resume_checkpoint=resume_checkpoint,
    )
    dataset, catalog = make_catalog(runtime)
    if catalog.manifest_digest != MANIFEST_DIGEST:
        raise ValueError("H3-F manifest authority 漂移")
    digest = (
        readiness_config_digest(readiness_steps, save_iter, runtime)
        if readiness_steps is not None
        else config_digest(save_iter, runtime)
    )
    native = preflight_native_batch(dataset, catalog, paths, digest=digest, runtime=runtime)
    return {
        "pair": pair,
        "phase": args.phase,
        "attempt": args.attempt,
        "start_iteration": start_iteration,
        "job": str(job),
        "evidence_dir": str(evidence),
        "catalog_episodes": len(catalog.episodes),
        "manifest_digest": catalog.manifest_digest,
        "config_digest": digest,
        "stage_a_dcp_keys": contract["dcp_keys"],
        "asset_authority": asset_authority,
        "contract_env": contract_env,
        "checkpoint_load_path": config.checkpoint.load_path,
        "checkpoint_load_training_state": config.checkpoint.load_training_state,
        "initialization_source": "direct BASE_CHECKPOINT_PATH" if args.phase == "fresh" else "same-job checkpoint",
        **native,
        "geometry": [8, 8, 2, 16, 4],
        "selector": list(H3F_OPTIMIZER_KEYS),
        "trainable_profile": H3F_TRAINABLE_PROFILE,
        "mesh_profile": H3F_MESH_PROFILE,
        "data_profile": H3F_DATA_PROFILE,
        "local_memory_action_dim": 15,
        "formal_max_iter": H3F_FORMAL_MAX_ITER,
        "readiness_steps": readiness_steps,
        "target_iteration": readiness_steps if readiness_steps is not None else H3F_FORMAL_MAX_ITER,
        "save_iter": readiness_steps if readiness_steps is not None else save_iter,
        "scheduler_cycle": [H3F_FORMAL_MAX_ITER],
        "scheduler_warmup": [H3F_WARMUP_STEPS],
        "primary_eval_iters": list(H3F_FORMAL_CHECKPOINT_ITERS),
        "disk_free_bytes": _disk_free_bytes(output_root),
    }


class FormalObserver:
    """Aggregate one global training record per committed optimizer iteration."""

    def __init__(self, path: Path, model: torch.nn.Module) -> None:
        self.path = path
        self.model = model
        self.iteration: int | None = None
        self.forward = 0
        self.backward = 0
        self.pre_optimizer = 0
        self.loss_sum: torch.Tensor | None = None
        self.loss_min: torch.Tensor | None = None
        self.loss_max: torch.Tensor | None = None
        self.metric_sums: dict[str, torch.Tensor] = {}
        self.metric_counts: dict[str, int] = {}
        self.completed = 0
        self.last_record: dict[str, Any] | None = None
        self.last_commit_time = time.perf_counter()
        self.step_wall_samples: list[float] = []

    def _bind_iteration(self, iteration: int) -> None:
        if self.iteration is None:
            self.iteration = iteration
        elif self.iteration != iteration:
            raise RuntimeError("H3-F observer 跨 iteration 状态未提交")

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _reduce_scalar(self, value: torch.Tensor | float | int, op: Any = None) -> float:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=self._device, dtype=torch.float64)
        else:
            tensor = torch.tensor(float(value), device=self._device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.SUM if op is None else op,
            )
        return float(tensor.cpu())

    def _global_mean(self, total: torch.Tensor | float, count: int | float) -> float:
        global_total = self._reduce_scalar(total)
        global_count = self._reduce_scalar(count)
        if global_count <= 0:
            raise RuntimeError("H3-F global metric count 必须为正")
        return global_total / global_count

    def _global_max(self, value: torch.Tensor | float | int) -> float:
        return self._reduce_scalar(value, torch.distributed.ReduceOp.MAX)

    def _global_min(self, value: torch.Tensor | float | int) -> float:
        return self._reduce_scalar(value, torch.distributed.ReduceOp.MIN)

    def _accumulate_metrics(self, metrics: dict[str, Any] | None) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0 or not bool(torch.isfinite(value)):
                    continue
                scalar = value.detach().float()
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalar = torch.tensor(float(value), device=self._device, dtype=torch.float32)
            else:
                continue
            self.metric_sums[key] = scalar if key not in self.metric_sums else self.metric_sums[key] + scalar
            self.metric_counts[key] = self.metric_counts.get(key, 0) + 1

    def _collect_grad_stats(self) -> dict[str, Any]:
        categories = {
            "local": {"sq": None, "nonempty": 0, "nonzero": 0},
            "generation": {"sq": None, "nonempty": 0, "nonzero": 0},
            "action": {"sq": None, "nonempty": 0, "nonzero": 0},
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not (_is_h3f_generation_parameter(name) or _is_h3f_local_parameter(name)):
                continue
            gradient = _local(parameter.grad)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("H3-F Generation/Local gradient 非有限")
            if _is_h3f_local_parameter(name):
                category = "local"
            elif any(key in name for key in H3F_ACTION_KEYS):
                category = "action"
            else:
                category = "generation"
            stats = categories[category]
            if gradient.numel() > 0:
                sq = gradient.detach().float().square().sum()
                stats["sq"] = sq if stats["sq"] is None else stats["sq"] + sq
                stats["nonempty"] += 1
                stats["nonzero"] += int(bool(sq > 0))
        for category, stats in categories.items():
            if stats["sq"] is None or stats["nonzero"] <= 0:
                raise FloatingPointError(f"H3-F 缺少非零 {category} gradient witness")
        return categories

    def _global_grad_record(self, categories: dict[str, Any]) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        total_sq = 0.0
        for category, stats in categories.items():
            global_sq = self._reduce_scalar(stats["sq"])
            total_sq += global_sq
            result[f"{category}_grad_norm"] = math.sqrt(max(global_sq, 0.0))
            result[f"{category}_grad_nonempty_shards"] = int(round(self._reduce_scalar(stats["nonempty"])))
            result[f"{category}_grad_nonzero_shards"] = int(round(self._reduce_scalar(stats["nonzero"])))
        result["grad_norm"] = math.sqrt(max(total_sq, 0.0))
        return result

    def _drain_ttt_telemetry(self) -> dict[str, float]:
        runtime = getattr(getattr(self.model, "net", None), "local_memory_runtime", None)
        core = getattr(runtime, "core", None)
        drain = getattr(core, "drain_telemetry", None)
        if drain is None:
            return {}
        telemetry = drain()
        result: dict[str, float] = {}
        for stem in ("inner_loss", "fast_state_norm", "fast_update_norm"):
            sum_key = f"ttt_{stem}_sum"
            count_key = f"ttt_{stem}_count"
            max_key = f"ttt_{stem}_max"
            if sum_key not in telemetry:
                continue
            result[f"ttt_{stem}_mean"] = self._global_mean(telemetry[sum_key], telemetry[count_key])
            result[f"ttt_{stem}_max"] = self._global_max(telemetry[max_key])
        return result

    def _format_console(self, record: dict[str, Any], target: int) -> str:
        parts = [
            f"[H3-F][train] iter={record['iteration']:06d}/{target:06d}",
            f"outer={record['loss_mean']:.4f}",
        ]
        for key, label in (
            ("flow_matching_loss_action", "action"),
            ("flow_matching_loss_vision", "vision"),
            ("ttt_inner_loss_mean", "inner"),
        ):
            if key in record:
                parts.append(f"{label}={record[key]:.4f}")
        parts.extend(
            [
                f"gnorm={record['grad_norm']:.3f}",
                f"gen={record['generation_grad_norm']:.3f}",
                f"act={record['action_grad_norm']:.3f}",
                f"local={record['local_grad_norm']:.3f}",
            ]
        )
        if "ttt_fast_state_norm_mean" in record:
            parts.append(f"fast={record['ttt_fast_state_norm_mean']:.3f}")
        if "ttt_fast_update_norm_mean" in record:
            parts.append(f"dFast={record['ttt_fast_update_norm_mean']:.3f}")
        if "lr_min" in record:
            if abs(record["lr_max"] - record["lr_min"]) < 1e-16:
                parts.append(f"lr={record['lr_min']:.2e}")
            else:
                parts.append(f"lr={record['lr_min']:.2e}..{record['lr_max']:.2e}")
        parts.extend(
            [
                f"step={record['step_wall_seconds']:.1f}s",
                f"peak={record['peak_allocated_bytes'] / (1024**3):.1f}GB",
                f"epoch={record['frontier_epoch']}",
            ]
        )
        return " ".join(parts)

    def __call__(
        self,
        *,
        phase: str,
        iteration: int,
        member: int,
        index: int | None,
        loss: torch.Tensor | None,
        metrics: dict[str, Any] | None = None,
        trainer: Any,
    ) -> None:
        del member, index
        self._bind_iteration(iteration)
        if phase == "native_forward":
            if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("H3-F native loss 非有限或非标量")
            value = loss.detach().float()
            self.forward += 1
            self.loss_sum = value if self.loss_sum is None else self.loss_sum + value
            self.loss_min = value if self.loss_min is None else torch.minimum(self.loss_min, value)
            self.loss_max = value if self.loss_max is None else torch.maximum(self.loss_max, value)
            self._accumulate_metrics(metrics)
            return
        if phase == "native_backward":
            self.backward += 1
            return
        if phase == "pre_optimizer":
            self.pre_optimizer += 1
            self.grad_categories = self._collect_grad_stats()
            self._accumulate_metrics(metrics)
            return
        if phase != "post_commit":
            raise ValueError(f"H3-F 未知 grouped observer phase: {phase}")
        if (
            self.forward != 32
            or self.backward != 32
            or self.pre_optimizer != 1
            or self.loss_sum is None
            or self.loss_min is None
            or self.loss_max is None
        ):
            raise RuntimeError(
                f"H3-F iteration event count 错误: fwd={self.forward}, bwd={self.backward}, pre={self.pre_optimizer}"
            )

        completed = trainer._grouped_completed_iteration + 1
        now = time.perf_counter()
        local_step_wall = now - self.last_commit_time
        self.last_commit_time = now
        step_wall_seconds = self._global_max(local_step_wall)
        self.step_wall_samples.append(step_wall_seconds)

        record: dict[str, Any] = {
            "iteration": completed,
            "native_forward": int(round(self._global_max(self.forward))),
            "native_backward": int(round(self._global_max(self.backward))),
            "pre_optimizer": int(round(self._global_max(self.pre_optimizer))),
            "post_commit": 1,
            "loss_mean": self._global_mean(self.loss_sum, self.forward),
            "loss_min": self._global_min(self.loss_min),
            "loss_max": self._global_max(self.loss_max),
            "frontier_epoch": trainer._grouped_window.live.frontier.epoch,
            "step_wall_seconds": step_wall_seconds,
            "allocated_bytes": int(self._global_max(torch.cuda.memory_allocated())),
            "reserved_bytes": int(self._global_max(torch.cuda.memory_reserved())),
            "peak_allocated_bytes": int(self._global_max(torch.cuda.max_memory_allocated())),
        }
        for key, total in self.metric_sums.items():
            record[key] = self._global_mean(total, self.metric_counts[key])
        record.update(self._global_grad_record(self.grad_categories))
        record.update(self._drain_ttt_telemetry())

        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        if rank == 0:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            trainer_config = getattr(getattr(trainer, "config", None), "trainer", None)
            target = int(getattr(trainer_config, "max_iter", H3F_FORMAL_MAX_ITER))
            print(self._format_console(record, target), flush=True)

        self.completed += 1
        self.last_record = record
        self.iteration = None
        self.forward = self.backward = self.pre_optimizer = 0
        self.loss_sum = self.loss_min = self.loss_max = None
        self.metric_sums.clear()
        self.metric_counts.clear()
        torch.cuda.reset_peak_memory_stats()

    def timing_summary(self) -> dict[str, float | int | None]:
        if not self.step_wall_samples:
            return {"samples": 0, "min": None, "median": None, "max": None, "mean": None}
        values = sorted(self.step_wall_samples)
        midpoint = len(values) // 2
        median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2
        return {
            "samples": len(values),
            "min": values[0],
            "median": median,
            "max": values[-1],
            "mean": sum(values) / len(values),
        }


def _is_h3f_generation_parameter(name: str) -> bool:
    return any(key in name for key in H3F_GENERATION_KEYS)


def _is_h3f_local_parameter(name: str) -> bool:
    return any(key in name for key in H3F_LOCAL_KEYS)


def _install_optimizer_inventory_check(model: torch.nn.Module, result: dict[str, Any]) -> None:
    original = model.init_optimizer_scheduler

    def checked_optimizer(optimizer_config, scheduler_config):
        named = dict(model.named_parameters())
        expected_names = {name for name in named if _is_h3f_generation_parameter(name) or _is_h3f_local_parameter(name)}
        generation_names = {name for name in expected_names if _is_h3f_generation_parameter(name)}
        local_names = {name for name in expected_names if _is_h3f_local_parameter(name)}
        if not generation_names:
            raise ValueError("H3-F generation inventory 为空")
        if sum(named[name].numel() for name in local_names) != LOCAL_PARAMS:
            raise ValueError("H3-F raw15 Local-TTT inventory 必须精确为 165312")

        optimizer, scheduler = original(optimizer_config, scheduler_config)
        selected = _optimizer_parameter_ids(optimizer)
        selected_names = {name for name, parameter in named.items() if id(parameter) in selected}
        reasoner_names = {
            name for name in selected_names if name.startswith("net.language_model.") and "_moe_gen" not in name
        }
        if (
            selected_names != expected_names
            or reasoner_names
            or any(parameter.requires_grad != (name in expected_names) for name, parameter in named.items())
        ):
            raise ValueError("H3-F optimizer 必须精确为 V2 语义 generation + V3 raw15 Local-TTT；reasoner 必须冻结")

        result["selected_names"] = sorted(selected_names)
        result["selected_generation_tensors"] = len(generation_names)
        result["selected_generation_params"] = sum(named[name].numel() for name in generation_names)
        result["selected_local_tensors"] = len(local_names)
        result["selected_local_params"] = LOCAL_PARAMS
        result["selected_reasoner_params"] = 0
        result["trainable_profile"] = H3F_TRAINABLE_PROFILE
        result["mesh_profile"] = H3F_MESH_PROFILE
        result["data_profile"] = H3F_DATA_PROFILE
        return optimizer, scheduler

    model.init_optimizer_scheduler = checked_optimizer


def execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or not torch.cuda.is_available():
        raise RuntimeError("H3-F 必须由 8-rank CUDA torchrun 启动")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < 8 or "H100" not in torch.cuda.get_device_name(local_rank):
        raise RuntimeError("H3-F 每个 local rank 必须绑定 H100")

    runtime = runtime_paths()
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root.expanduser().resolve())
    config = load_stage_a_config(runtime)
    resume_checkpoint = Path(report["checkpoint_load_path"]) if args.phase == "resume" else None
    overlay_h3f_config(
        config,
        phase=args.phase,
        job_name=args.job_name,
        readiness_steps=report["readiness_steps"],
        save_iter=int(report["contract_env"]["SAVE_ITER"]),
        runtime=runtime,
        resume_checkpoint=resume_checkpoint,
    )

    job = Path(report["job"])
    evidence = Path(report["evidence_dir"])
    result_path = evidence / f"rank_{rank}.json"
    progress_path = evidence / f"rank_{rank}_progress.jsonl"
    result: dict[str, Any] = dict(report, rank=rank, result="FAIL")

    try:
        distributed.init()
        if rank == 0:
            evidence.mkdir(parents=True, exist_ok=False)
        torch.distributed.barrier()
        if not evidence.is_dir():
            raise FileNotFoundError("H3-F rank0 未发布 evidence attempt 目录")
        config.validate()
        config.freeze()
        trainer = GroupedLocalMemoryTrainer(config)
        with model_init():
            model = instantiate(config.model)
        _install_optimizer_inventory_check(model, result)

        dataset, catalog = make_catalog(runtime)
        transform, resolution = build_stage_a_action_transform(_paths(args.output_root, runtime))
        binder = StageARoboCasaEpisodeBinder(
            dataset,
            catalog,
            source_root=runtime.dataset_root,
            cache_root=runtime.cache_root,
            transform=transform,
            resolution=resolution,
            config_digest=report["config_digest"],
        )
        planner = RankLocalGroupedPlanner(catalog, rank=rank, world_size=8, seed=0)
        trainer.bind_grouped_stream(planner, binder.producer_for, config_digest=report["config_digest"])
        observer = FormalObserver(progress_path, model)
        trainer.grouped_observer = observer
        trainer.train(model, GroupedTriggerLoader(config.trainer.max_iter), None)

        target_iteration = report["target_iteration"]
        if trainer._grouped_completed_iteration != target_iteration:
            raise RuntimeError("H3-F 未完成目标 optimizer iterations")
        if observer.completed != target_iteration - report["start_iteration"]:
            raise RuntimeError("H3-F observer 完成 iteration 数与 resume 起点不匹配")

        final = job / "checkpoints" / f"iter_{target_iteration:09d}"
        if (
            (job / "checkpoints/latest_checkpoint.txt").read_text().strip() != final.name
            or any(not (final / key / ".metadata").is_file() for key in ("model", "optim", "scheduler", "trainer"))
            or not (final / "dataloader" / f"rank_{rank}.pkl").is_file()
        ):
            raise RuntimeError("H3-F final DCP 不完整")

        result.update(
            result="PASS",
            resume_required=trainer._resume_required,
            completed_iteration=trainer._grouped_completed_iteration,
            progress_records=observer.completed,
            final_record=observer.last_record,
            timing=observer.timing_summary(),
            final_checkpoint=str(final),
        )
    except Exception:
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if evidence.is_dir():
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    report = preflight(args)
    if args.preflight:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        execute(args, report)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"H3-F FAIL: {error}", file=sys.stderr)
        raise

"""V3 H3-F：RoboCasa target-atomic 8×H100 generation+Local-TTT 30k formal training launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EDGE,
    DEFAULT_VAE,
    H3F_FORMAL_CHECKPOINT_ITERS,
    H3F_FORMAL_MAX_ITER,
    LOCAL_PARAMS,
    MANIFEST_DIGEST,
    ROOT_WORKTREE,
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
H3F_BASE_CHECKPOINT_LINEAGE = Path("/mnt/data/shenzhen/szrobot/logs/.tmp_backup/models/Cosmos3-Edge-Policy-DROID-dcp")
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
) -> None:
    from examples.psm_wma_robocasa_h100 import overlay_h100_config

    overlay_h100_config(config, phase=phase, job_name=job_name)
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
        or save_iter <= 0
        or any(step % save_iter for step in H3F_FORMAL_CHECKPOINT_ITERS)
    ):
        raise ValueError("H3-F 30k/scheduler/checkpoint 合同不匹配")


def config_digest(save_iter: int = H3F_SAVE_ITER) -> str:
    authority = {
        "h3e_runtime": h3e_config_digest(),
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


def readiness_config_digest(steps: int, save_iter: int = H3F_SAVE_ITER) -> str:
    authority = {
        "formal_config_digest": config_digest(save_iter),
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


def _validate_output_root(output_root: Path) -> None:
    if output_root == ROOT_WORKTREE:
        raise ValueError("H3-F 产物不能直接写入 root worktree")
    if ROOT_WORKTREE in output_root.parents:
        allowed = ROOT_WORKTREE / "outputs"
        if output_root != allowed and allowed not in output_root.parents:
            raise ValueError("H3-F worktree 内只允许写入已忽略的 outputs/ 子树")


def _validate_contract_env(output_root: Path) -> dict[str, Any]:
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
        ("ROBOCASA_ROOT", DEFAULT_DATASET_ROOT),
        ("ROBOCASA_LATENT_CACHE_ROOT", CACHE_ROOT),
        ("EDGE_POLICY_CHECKPOINT", DEFAULT_EDGE),
        ("WAN_VAE_PATH", DEFAULT_VAE),
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        resolved = Path(value).expanduser().resolve()
        if resolved != expected.resolve():
            raise ValueError(f"H3-F {name} 与冻结 authority 不匹配")
        observed[name] = str(resolved)

    base_env = os.environ.get("BASE_CHECKPOINT_PATH")
    if base_env is not None:
        base = Path(base_env).expanduser().resolve()
        if base != H3F_BASE_CHECKPOINT_LINEAGE.resolve():
            raise ValueError("H3-F BASE_CHECKPOINT_PATH 与冻结 Edge-DROID lineage 不匹配")
        if not (base / "model/.metadata").is_file():
            raise FileNotFoundError(f"H3-F BASE_CHECKPOINT_PATH 缺少 model/.metadata：{base}")
        observed["BASE_CHECKPOINT_PATH"] = str(base)
        observed["EFFECTIVE_WARMSTART"] = "frozen Stage-A H100 iter1 authority"

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
    pair = lock_pair(args.expected_root, args.expected_child)
    output_root = args.output_root.expanduser().resolve()
    _validate_output_root(output_root)
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
    else:
        if args.attempt < 2:
            raise ValueError("H3-F resume 必须使用 attempt>=2")
        start_iteration = _resume_iteration(job)

    evidence = _evidence_dir(job, phase=args.phase, attempt=args.attempt, start_iteration=start_iteration)
    if evidence.exists() and authority_rank:
        raise FileExistsError("H3-F evidence attempt 已存在；禁止覆盖")

    contract_env = _validate_contract_env(output_root)
    save_iter = int(contract_env["SAVE_ITER"])
    paths = _paths(output_root)
    asset_authority = validate_h100_asset_authority()
    contract = read_stage_a_contract(paths)
    config = load_stage_a_config()
    overlay_h3f_config(
        config,
        phase=args.phase,
        job_name=args.job_name,
        readiness_steps=readiness_steps,
        save_iter=save_iter,
    )
    dataset, catalog = make_catalog()
    if catalog.manifest_digest != MANIFEST_DIGEST:
        raise ValueError("H3-F manifest authority 漂移")
    digest = (
        readiness_config_digest(readiness_steps, save_iter) if readiness_steps is not None else config_digest(save_iter)
    )
    native = preflight_native_batch(dataset, catalog, paths, digest=digest)
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
    """Aggregate 64 native events into one durable JSONL record per optimizer iteration."""

    def __init__(self, path: Path, model: torch.nn.Module) -> None:
        self.path = path
        self.model = model
        self.iteration: int | None = None
        self.forward = 0
        self.backward = 0
        self.pre_optimizer = 0
        self.loss_sum = 0.0
        self.loss_min = float("inf")
        self.loss_max = float("-inf")
        self.completed = 0
        self.last_record: dict[str, Any] | None = None
        self.last_commit_time: float | None = None
        self.step_wall_samples: list[float] = []

    def _bind_iteration(self, iteration: int) -> None:
        if self.iteration is None:
            self.iteration = iteration
        elif self.iteration != iteration:
            raise RuntimeError("H3-F observer 跨 iteration 状态未提交")

    def __call__(
        self, *, phase: str, iteration: int, member: int, index: int | None, loss: torch.Tensor | None, trainer: Any
    ) -> None:
        del member, index
        self._bind_iteration(iteration)
        if phase == "native_forward":
            if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("H3-F native loss 非有限或非标量")
            value = float(loss.detach())
            self.forward += 1
            self.loss_sum += value
            self.loss_min = min(self.loss_min, value)
            self.loss_max = max(self.loss_max, value)
            return
        if phase == "native_backward":
            self.backward += 1
            return
        if phase == "pre_optimizer":
            self.pre_optimizer += 1
            selected_local = []
            selected_generation = []
            selected_action = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                if not (_is_h3f_generation_parameter(name) or _is_h3f_local_parameter(name)):
                    continue
                gradient = _local(parameter.grad)
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("H3-F Generation/Local gradient 非有限")
                witness = (name, int(gradient.numel()), float(gradient.float().norm()))
                if _is_h3f_local_parameter(name):
                    selected_local.append(witness)
                elif any(key in name for key in H3F_ACTION_KEYS):
                    selected_action.append(witness)
                else:
                    selected_generation.append(witness)
            if not any(numel > 0 and norm > 0 for _, numel, norm in selected_local):
                raise FloatingPointError("H3-F 缺少非零 Local gradient witness")
            if not any(numel > 0 and norm > 0 for _, numel, norm in selected_generation):
                raise FloatingPointError("H3-F 缺少非零 generation-core gradient witness")
            if not any(numel > 0 and norm > 0 for _, numel, norm in selected_action):
                raise FloatingPointError("H3-F 缺少非零 action-head gradient witness")
            self.local_grad_witness = selected_local
            self.generation_grad_witness = selected_generation
            self.action_grad_witness = selected_action
            return
        if phase != "post_commit":
            raise ValueError(f"H3-F 未知 grouped observer phase: {phase}")
        if self.forward != 32 or self.backward != 32 or self.pre_optimizer != 1:
            raise RuntimeError(
                f"H3-F iteration event count 错误: fwd={self.forward}, bwd={self.backward}, pre={self.pre_optimizer}"
            )
        completed = trainer._grouped_completed_iteration + 1
        now = time.perf_counter()
        step_wall_seconds = None if self.last_commit_time is None else now - self.last_commit_time
        self.last_commit_time = now
        if step_wall_seconds is not None:
            self.step_wall_samples.append(step_wall_seconds)
        record = {
            "iteration": completed,
            "native_forward": self.forward,
            "native_backward": self.backward,
            "pre_optimizer": self.pre_optimizer,
            "post_commit": 1,
            "loss_mean": self.loss_sum / self.forward,
            "loss_min": self.loss_min,
            "loss_max": self.loss_max,
            "frontier_epoch": trainer._grouped_window.live.frontier.epoch,
            "step_wall_seconds": step_wall_seconds,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "local_grad_nonempty_shards": sum(numel > 0 for _, numel, _ in self.local_grad_witness),
            "local_grad_nonzero_shards": sum(numel > 0 and norm > 0 for _, numel, norm in self.local_grad_witness),
            "generation_grad_nonempty_shards": sum(numel > 0 for _, numel, _ in self.generation_grad_witness),
            "generation_grad_nonzero_shards": sum(
                numel > 0 and norm > 0 for _, numel, norm in self.generation_grad_witness
            ),
            "action_grad_nonempty_shards": sum(numel > 0 for _, numel, _ in self.action_grad_witness),
            "action_grad_nonzero_shards": sum(numel > 0 and norm > 0 for _, numel, norm in self.action_grad_witness),
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.completed += 1
        self.last_record = record
        self.iteration = None
        self.forward = self.backward = self.pre_optimizer = 0
        self.loss_sum = 0.0
        self.loss_min = float("inf")
        self.loss_max = float("-inf")
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

    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root.expanduser().resolve())
    os.environ["EDGE_POLICY_CHECKPOINT"] = str(DEFAULT_EDGE)
    os.environ["WAN_VAE_PATH"] = str(DEFAULT_VAE)
    config = load_stage_a_config()
    overlay_h3f_config(
        config,
        phase=args.phase,
        job_name=args.job_name,
        readiness_steps=report["readiness_steps"],
        save_iter=int(report["contract_env"]["SAVE_ITER"]),
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

        dataset, catalog = make_catalog()
        transform, resolution = build_stage_a_action_transform(_paths(args.output_root))
        binder = StageARoboCasaEpisodeBinder(
            dataset,
            catalog,
            source_root=DEFAULT_DATASET_ROOT,
            cache_root=CACHE_ROOT,
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

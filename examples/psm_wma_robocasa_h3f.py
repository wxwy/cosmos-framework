"""V3 H3-F：RoboCasa target-atomic 8×H100 Local-TTT 30k formal training launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
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
    HOST_KEYS,
    LOCAL_PARAMS,
    MANIFEST_DIGEST,
    ROOT_WORKTREE,
    SELECTED_KEYS,
    GroupedTriggerLoader,
    _local,
    _optimizer_parameter_ids,
    _paths,
    build_stage_a_action_transform,
    config_digest as h3e_config_digest,
    load_stage_a_config,
    lock_pair,
    make_catalog,
    preflight_native_batch,
    read_stage_a_contract,
    validate_h100_asset_authority,
)

H3F_SAVE_ITER = 1_000
H3F_WARMUP_STEPS = 500
H3F_GROUP = "h3f_edge_local_h100"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("fresh", "resume"), required=True)
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--job-name", default="edge_local_target_atomic_30k")
    result.add_argument("--attempt", type=int, default=1)
    result.add_argument("--expected-root", required=True)
    result.add_argument("--expected-child", required=True)
    return result


def overlay_h3f_config(config: Any, *, phase: str, job_name: str) -> None:
    from examples.psm_wma_robocasa_h100 import overlay_h100_config

    overlay_h100_config(config, phase=phase, job_name=job_name)
    config.trainer.max_iter = H3F_FORMAL_MAX_ITER
    config.trainer.logging_iter = 50
    config.scheduler.cycle_lengths = [H3F_FORMAL_MAX_ITER]
    config.scheduler.warm_up_steps = [H3F_WARMUP_STEPS]
    config.checkpoint.save_iter = H3F_SAVE_ITER
    config.job.group = H3F_GROUP
    config.job.name = job_name
    if (
        config.trainer.grad_accum_iter != 2
        or config.trainer.max_iter != 30_000
        or config.scheduler.cycle_lengths != [30_000]
        or config.scheduler.warm_up_steps != [500]
        or config.checkpoint.save_iter != 1_000
        or any(step % H3F_SAVE_ITER for step in H3F_FORMAL_CHECKPOINT_ITERS)
    ):
        raise ValueError("H3-F 30k/scheduler/checkpoint 合同不匹配")


def config_digest() -> str:
    authority = {
        "h3e_runtime": h3e_config_digest(),
        "max_iter": H3F_FORMAL_MAX_ITER,
        "save_iter": H3F_SAVE_ITER,
        "warmup_steps": H3F_WARMUP_STEPS,
        "cycle_lengths": [H3F_FORMAL_MAX_ITER],
        "primary_eval_iters": list(H3F_FORMAL_CHECKPOINT_ITERS),
    }
    return hashlib.sha256(json.dumps(authority, sort_keys=True).encode()).hexdigest()


def _job(output_root: Path, job_name: str) -> Path:
    return output_root / "psm_wma_v3" / H3F_GROUP / job_name


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


def _disk_free_bytes(output_root: Path) -> int:
    probe = output_root.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    pair = lock_pair(args.expected_root, args.expected_child)
    output_root = args.output_root.expanduser().resolve()
    if output_root == ROOT_WORKTREE or ROOT_WORKTREE in output_root.parents:
        raise ValueError("H3-F 产物必须位于 root worktree 之外")
    if not args.job_name or "/" in args.job_name or args.job_name in (".", ".."):
        raise ValueError("H3-F job-name 不合法")
    if type(args.attempt) is not int or args.attempt <= 0:
        raise ValueError("H3-F attempt 必须为正整数")

    job = _job(output_root, args.job_name)
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

    paths = _paths(output_root)
    asset_authority = validate_h100_asset_authority()
    contract = read_stage_a_contract(paths)
    config = load_stage_a_config()
    overlay_h3f_config(config, phase=args.phase, job_name=args.job_name)
    dataset, catalog = make_catalog()
    if catalog.manifest_digest != MANIFEST_DIGEST:
        raise ValueError("H3-F manifest authority 漂移")
    digest = config_digest()
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
        **native,
        "geometry": [8, 8, 2, 16, 4],
        "selector": list(SELECTED_KEYS),
        "max_iter": H3F_FORMAL_MAX_ITER,
        "save_iter": H3F_SAVE_ITER,
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
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad or "local_memory" not in name or parameter.grad is None:
                    continue
                gradient = _local(parameter.grad)
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("H3-F Local gradient 非有限")
                selected_local.append((name, int(gradient.numel()), float(gradient.float().norm())))
            if not selected_local:
                raise FloatingPointError("H3-F 缺少 Local gradient witness")
            self.local_grad_witness = selected_local
            return
        if phase != "post_commit":
            raise ValueError(f"H3-F 未知 grouped observer phase: {phase}")
        if self.forward != 32 or self.backward != 32 or self.pre_optimizer != 1:
            raise RuntimeError(
                f"H3-F iteration event count 错误: fwd={self.forward}, bwd={self.backward}, pre={self.pre_optimizer}"
            )
        completed = trainer._grouped_completed_iteration + 1
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
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "local_grad_nonempty_shards": sum(numel > 0 for _, numel, _ in self.local_grad_witness),
            "local_grad_nonzero_shards": sum(numel > 0 and norm > 0 for _, numel, norm in self.local_grad_witness),
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


def _install_optimizer_inventory_check(model: torch.nn.Module, result: dict[str, Any]) -> None:
    original = model.init_optimizer_scheduler

    def checked_optimizer(optimizer_config, scheduler_config):
        optimizer, scheduler = original(optimizer_config, scheduler_config)
        named = dict(model.named_parameters())
        selected = _optimizer_parameter_ids(optimizer)
        selected_names = {name for name, parameter in named.items() if id(parameter) in selected}
        local_names = {name for name in named if name.startswith("net.local_memory")}
        if (
            not local_names <= selected_names
            or not all(any(key in name for name in selected_names) for key in HOST_KEYS)
            or sum(named[name].numel() for name in local_names) != LOCAL_PARAMS
            or not all(any(key in name for key in SELECTED_KEYS) for name in selected_names)
            or any(parameter.requires_grad for name, parameter in named.items() if name not in selected_names)
        ):
            raise ValueError("H3-F optimizer Edge+Local inventory 不匹配")
        result["selected_names"] = sorted(selected_names)
        result["selected_local_params"] = LOCAL_PARAMS
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
    overlay_h3f_config(config, phase=args.phase, job_name=args.job_name)

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

        if trainer._grouped_completed_iteration != H3F_FORMAL_MAX_ITER:
            raise RuntimeError("H3-F 未完成 30000 optimizer iterations")
        if observer.completed != H3F_FORMAL_MAX_ITER - report["start_iteration"]:
            raise RuntimeError("H3-F observer 完成 iteration 数与 resume 起点不匹配")

        final = job / "checkpoints" / f"iter_{H3F_FORMAL_MAX_ITER:09d}"
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

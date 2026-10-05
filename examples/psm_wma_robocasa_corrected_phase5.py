"""Corrected exact-window RoboCasa Phase5 trainer 入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import RoboCasaExactWindowCacheCatalog
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    get_action_robocasa_exact_window_cached_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
from cosmos_framework.utils import distributed
from cosmos_framework.utils.context_managers import model_init
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.utils.lazy_config import instantiate
from examples.psm_wma_robocasa_native import RECIPE, check_droid_dcp, check_edge_checkpoint

GENERATION_KEYS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
LOCAL_KEYS = (
    "local_memory_runtime.encoder.",
    "local_memory_runtime.core.",
    "local_memory2llm.",
    "local_memory_modality_embed",
)
OPTIMIZER_KEYS = (*GENERATION_KEYS, *LOCAL_KEYS)
ACTION_KEYS = ("action2llm", "llm2action", "action_modality_embed")


class GroupedTriggerLoader:
    """每个 optimizer iteration 精确提供 active_ga 个空触发。"""

    def __init__(self, max_iter: int, active_ga: int) -> None:
        if any(type(value) is not int or value <= 0 for value in (max_iter, active_ga)):
            raise ValueError("max_iter/active_ga 必须是正整数")
        self.max_iter, self.active_ga, self.start = max_iter, active_ga, 0

    def set_start_iteration(self, fetched: int) -> None:
        if type(fetched) is not int or not 0 <= fetched <= self.max_iter * self.active_ga:
            raise ValueError("trigger resume 偏移越界")
        self.start = fetched

    def __iter__(self):
        for _ in range(self.start, self.max_iter * self.active_ga):
            yield {}

    def __len__(self) -> int:
        return self.max_iter * self.active_ga - self.start


class GroupedPlanObserver:
    """按本次实际 plan 的有效前缀校验原生 forward/backward 次数。"""

    def __init__(self) -> None:
        self.forward = 0
        self.backward = 0
        self.completed = 0

    def __call__(self, *, phase: str, trainer: GroupedLocalMemoryTrainer, **_: Any) -> None:
        if phase == "native_forward":
            self.forward += 1
        elif phase == "native_backward":
            self.backward += 1
        elif phase == "pre_optimizer":
            plan = trainer._grouped_window.plan
            if plan is None:
                raise RuntimeError("observer 缺少 pending grouped plan")
            expected = sum(max(request.valid_count for request in member) for member in plan.members)
            if (self.forward, self.backward) != (expected, expected):
                raise RuntimeError(
                    f"native 调用次数不匹配：forward={self.forward}, backward={self.backward}, expected={expected}"
                )
        elif phase == "post_commit":
            self.completed += 1
            self.forward = self.backward = 0


def _selected_name(name: str) -> bool:
    if not name.startswith("net."):
        return False
    if name.startswith("net.language_model."):
        return "_moe_gen" in name
    root = name.split(".", 2)[1]
    if root in GENERATION_KEYS:
        return True
    return (
        name.startswith("net.local_memory_runtime.encoder.")
        or name.startswith("net.local_memory_runtime.core.")
        or name.startswith("net.local_memory2llm.")
        or root == "local_memory_modality_embed"
    )


def validate_optimizer_inventory(model: torch.nn.Module, optimizer: Any) -> dict[str, int]:
    """用实际模型身份和实际 optimizer 参数集合核对精确 allowlist。"""
    named = dict(model.named_parameters())
    expected = {name for name in named if _selected_name(name)}
    generation = {name for name in expected if any(key in name for key in GENERATION_KEYS)}
    local = expected - generation
    if not generation or not local or any(not any(key in name for key in LOCAL_KEYS) for name in local):
        raise ValueError("generation/Local 参数集合缺失或重叠")
    if any(parameter.requires_grad != (name in expected) for name, parameter in named.items()):
        raise ValueError("非 generation/Local 参数必须冻结，选中参数必须可训练")
    inners = optimizer.optimizers if isinstance(optimizer, OptimizersContainer) else (optimizer,)
    selected = [parameter for inner in inners for group in inner.param_groups for parameter in group["params"]]
    by_id = {id(parameter): name for name, parameter in named.items()}
    if len(selected) != len({id(parameter) for parameter in selected}) or any(id(p) not in by_id for p in selected):
        raise ValueError("optimizer 参数重复或不属于模型")
    selected_names = {by_id[id(parameter)] for parameter in selected}
    if selected_names != expected:
        raise ValueError(
            f"optimizer inventory 不匹配：缺少={sorted(expected - selected_names)[:4]}，多出={sorted(selected_names - expected)[:4]}"
        )
    return {
        "generation_tensors": len(generation),
        "local_tensors": len(local),
        "generation_params": sum(named[name].numel() for name in generation),
        "local_params": sum(named[name].numel() for name in local),
    }


def install_optimizer_inventory_check(model: torch.nn.Module, report: dict[str, Any]) -> None:
    original = model.init_optimizer_scheduler

    def checked(optimizer_config, scheduler_config):
        optimizer, scheduler = original(optimizer_config, scheduler_config)
        report["optimizer_inventory"] = validate_optimizer_inventory(model, optimizer)
        return optimizer, scheduler

    model.init_optimizer_scheduler = checked


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_witnesses(edge: Path, base_checkpoint: Path) -> dict[str, str]:
    return {
        "edge_config_sha256": _file_sha256(edge / "config.json"),
        "base_model_metadata_sha256": _file_sha256(base_checkpoint / "model/.metadata"),
    }


def config_digest(
    catalog: ExactWindowLocalCatalog,
    config: Any,
    *,
    b_stream: int,
    active_ga: int,
    witnesses: dict[str, str],
) -> str:
    model = config.model.config
    local = {
        key: getattr(model, f"local_memory_{key}")
        for key in ("ttt_tbptt_steps", "k_local", "ttt_dim", "fast_hidden_dim", "dim", "evidence_dim", "inner_lr")
    }
    authority = {
        "cache_manifest_sha256": catalog.manifest_digest,
        "cache_corpus_digest": catalog.cache_corpus_digest,
        "source_binding_digest": catalog.source_binding_digest,
        "model": "Cosmos3-Edge-Policy-DROID/corrected-exact-window-v1",
        "model_witnesses": witnesses,
        "raw_action_dim": 15,
        "state_dim": 15,
        "h_pred": 16,
        "consumer_frames": 17,
        "max_action_dim": model.max_action_dim,
        "local": local,
        "b_stream": b_stream,
        "active_ga": active_ga,
        "optimizer": {
            "keys": list(config.optimizer.keys_to_select),
            "type": config.optimizer.optimizer_type,
            "lr": float(config.optimizer.lr),
            "action_lr_multipliers": dict(config.optimizer.lr_multipliers),
            "weight_decay": float(config.optimizer.weight_decay),
        },
        "mesh": {
            "world_size": model.parallelism.data_parallel_shard_degree
            * model.parallelism.data_parallel_replicate_degree,
            "shard": model.parallelism.data_parallel_shard_degree,
            "replicate": model.parallelism.data_parallel_replicate_degree,
            "context_parallel": getattr(getattr(config, "model_parallel", None), "context_parallel_size", 1),
        },
        "schedule": {
            "max_iter": config.trainer.max_iter,
            "cycle_lengths": list(config.scheduler.cycle_lengths),
            "warm_up_steps": list(config.scheduler.warm_up_steps),
            "save_iter": config.checkpoint.save_iter,
        },
    }
    return hashlib.sha256(json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def overlay_config(
    config: Any,
    args: argparse.Namespace,
    contract: CorrectedRoboCasaPolicyContract,
    *,
    resume_checkpoint: Path | None = None,
) -> None:
    from examples.psm_wma_robocasa_local_s1 import overlay_local_config

    overlay_local_config(config)
    model = config.model.config
    model.local_memory_ttt_tbptt_steps = args.t
    model.local_memory_k_local = args.k
    model.tokenizer = contract.resolve_tokenizer_config(model.tokenizer)
    contract.validate_tokenizer_config(model.tokenizer)
    model.parallelism.data_parallel_shard_degree = args.world_size
    model.parallelism.data_parallel_replicate_degree = 1
    config.trainer.type = GroupedLocalMemoryTrainer
    config.trainer.grad_accum_iter = args.ga
    config.trainer.max_iter = args.max_iter
    config.trainer.run_validation = False
    config.trainer.callbacks = {}
    config.trainer.save_zero_checkpoint = False
    config.optimizer.keys_to_select = list(OPTIMIZER_KEYS)
    config.optimizer.lr_multipliers = {key: 5.0 for key in ACTION_KEYS}
    config.scheduler.cycle_lengths = [args.max_iter]
    config.scheduler.warm_up_steps = [args.warmup]
    config.checkpoint.save_iter = args.save_iter
    config.checkpoint.strict_resume = True
    config.checkpoint.load_path = str(resume_checkpoint or args.base_checkpoint)
    config.checkpoint.load_training_state = resume_checkpoint is not None
    config.checkpoint.keys_to_skip_loading = ["net_ema."] if resume_checkpoint else ["net_ema.", "local_memory"]
    config.job.project = "psm_wma_v3"
    config.job.group = "corrected_phase5"
    config.job.name = args.job_name
    if model.max_action_dim != 64 or model.local_memory_action_dim != 15:
        raise ValueError("Edge raw15/64 合同不匹配")


def _load_config(args: argparse.Namespace):
    overrides = {
        "EDGE_POLICY_CHECKPOINT": str(args.edge),
        "WAN_VAE_PATH": str(args.vae),
        "ROBOCASA_ROOT": str(args.source_root),
        "BASE_CHECKPOINT_PATH": str(args.base_checkpoint),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        return load_experiment_from_toml(RECIPE)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.phase == "fresh":
        return None
    latest = args.output_root / "psm_wma_v3/corrected_phase5" / args.job_name / "checkpoints/latest_checkpoint.txt"
    name = latest.read_text().strip()
    if not name.startswith("iter_") or not name[5:].isdigit() or not 0 < int(name[5:]) < args.max_iter:
        raise ValueError("same-job latest checkpoint iteration 不合法")
    checkpoint = latest.parent / name
    for component in ("model", "optim", "scheduler", "trainer"):
        if not (checkpoint / component / ".metadata").is_file():
            raise FileNotFoundError(f"same-job DCP 缺少 {component}")
    for rank in range(args.world_size):
        if not (checkpoint / "dataloader" / f"rank_{rank}.pkl").is_file():
            raise FileNotFoundError(f"same-job DCP 缺少 rank_{rank} Local 状态")
    return checkpoint


def verify_root_child_lock(args: argparse.Namespace) -> dict[str, str]:
    child = Path(__file__).resolve().parents[1]

    def git(repo: Path, *command: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repo), *command), check=True, capture_output=True, text=True
        ).stdout.strip()

    root = args.root_worktree
    actual_root = git(root, "rev-parse", "HEAD")
    actual_child = git(child, "rev-parse", "HEAD")
    gitlink = git(root, "ls-tree", "HEAD", "cosmos-framework").split()[2]
    if (
        len(args.expected_root) != 40
        or len(args.expected_child) != 40
        or (actual_root, actual_child, gitlink) != (args.expected_root, args.expected_child, args.expected_child)
        or git(child, "status", "--porcelain")
        or git(root, "status", "--porcelain")
    ):
        raise ValueError("V3 root/child/Gitlink 或工作树不匹配")
    return {"root": actual_root, "child": actual_child, "gitlink": gitlink}


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, ExactWindowLocalCatalog, Any]:
    if any(
        type(value) is not int or value <= 0
        for value in (args.t, args.b, args.ga, args.k, args.world_size, args.max_iter, args.save_iter)
    ):
        raise ValueError("T/B/GA/K/world_size/max_iter/save_iter 必须是正整数")
    if args.warmup < 0 or args.warmup > args.max_iter:
        raise ValueError("warmup 越界")
    lock = verify_root_child_lock(args)
    os.environ["HF_HUB_OFFLINE"] = "1"
    check_edge_checkpoint(str(args.edge))
    check_droid_dcp(args.base_checkpoint)
    if not args.vae.is_file():
        raise FileNotFoundError("WAN VAE 文件缺失")
    resume = _resume_checkpoint(args)
    config = _load_config(args)
    cache = RoboCasaExactWindowCacheCatalog(args.cache_root)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(cache)
    overlay_config(config, args, contract, resume_checkpoint=resume)
    tokenizer = config.model.config.vlm_config.tokenizer
    dataset = get_action_robocasa_exact_window_cached_sft_dataset(
        cache_root=args.cache_root, source_root=args.source_root, tokenizer_config=tokenizer
    )
    catalog = ExactWindowLocalCatalog(dataset, ttt_tbptt_steps=args.t)
    if (
        catalog.manifest_digest != cache.manifest_sha256
        or catalog.raw.contract.vae_encode_contract != contract.vae_encode_contract
    ):
        raise ValueError("预检期间 cache VAE authority 漂移")
    contract.validate_tokenizer_config(config.model.config.tokenizer)
    witnesses = model_witnesses(args.edge, args.base_checkpoint)
    digest = config_digest(catalog, config, b_stream=args.b, active_ga=args.ga, witnesses=witnesses)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=args.world_size, b_stream=args.b, active_ga=args.ga)
    plan = planner.plan_window(planner.initial_frontier())
    if args.snapshot10:
        producer = ExactWindowSegmentProducer(catalog, config_digest=digest)
        checked = 0
        while checked < 10:
            for member in plan.members:
                for request in member:
                    if checked == 10:
                        break
                    segment = producer.produce(request)
                    if segment.consumer_payload[0][0]["cached_latent_required"] is not True:
                        raise ValueError("snapshot10 缺少 cached latent")
                    collate_grouped_native_batch((segment.consumer_payload[0][0],))
                    checked += 1
            if checked < 10:
                plan = planner.plan_window(plan.candidate_frontier)
    return (
        {
            "lock": lock,
            "config_digest": digest,
            "cache_manifest_sha256": catalog.manifest_digest,
            "cache_corpus_digest": catalog.cache_corpus_digest,
            "source_binding_digest": catalog.source_binding_digest,
            "model_witnesses": witnesses,
            "episodes": len(catalog.episodes),
            "checkpoint_load_path": str(resume or args.base_checkpoint),
        },
        config,
        catalog,
        dataset,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("fresh", "resume"), required=True)
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--snapshot10", action="store_true")
    for name in ("output-root", "source-root", "cache-root", "edge", "vae", "base-checkpoint"):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument("--root-worktree", required=True, type=Path)
    result.add_argument("--expected-root", required=True)
    result.add_argument("--expected-child", required=True)
    result.add_argument("--job-name", default="edge_local_exact_window")
    for name, default in (
        ("t", 16),
        ("b", 8),
        ("ga", 2),
        ("k", 4),
        ("world-size", 8),
        ("max-iter", 30000),
        ("save-iter", 100),
        ("warmup", 500),
    ):
        result.add_argument(f"--{name}", type=int, default=default)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    report, config, catalog, _ = preflight(args)
    if args.preflight:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if int(os.environ.get("WORLD_SIZE", "0")) != args.world_size or not torch.cuda.is_available():
        raise RuntimeError("训练要求匹配 world_size 的 CUDA torchrun")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root)
    distributed.init()
    config.validate()
    config.freeze()
    rank = int(os.environ["RANK"])
    trainer = GroupedLocalMemoryTrainer(config)
    with model_init():
        model = instantiate(config.model)
    install_optimizer_inventory_check(model, report)
    planner = ExactWindowRankPlanner(catalog, rank=rank, world_size=args.world_size, b_stream=args.b, active_ga=args.ga)
    producer = ExactWindowSegmentProducer(catalog, config_digest=report["config_digest"])
    trainer.bind_grouped_stream(planner, producer, config_digest=report["config_digest"])
    observer = GroupedPlanObserver()
    trainer.grouped_observer = observer
    trainer.train(model, GroupedTriggerLoader(args.max_iter, args.ga), None)
    start_iteration = int(Path(report["checkpoint_load_path"]).name[5:]) if args.phase == "resume" else 0
    if trainer._grouped_completed_iteration != args.max_iter or observer.completed != args.max_iter - start_iteration:
        raise RuntimeError("Phase5 完成的 optimizer iteration 不匹配")


if __name__ == "__main__":
    main()

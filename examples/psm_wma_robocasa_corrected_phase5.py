"""Corrected exact-window RoboCasa Phase5 trainer 入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

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
from examples.psm_wma_robocasa_corrected_telemetry import GroupedPlanObserver
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

    def __init__(
        self, max_iter: int, active_ga: int, *, on_iteration_start: Callable[[int], None] | None = None
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (max_iter, active_ga)):
            raise ValueError("max_iter/active_ga 必须是正整数")
        self.max_iter, self.active_ga, self.start = max_iter, active_ga, 0
        self.on_iteration_start = on_iteration_start

    def set_start_iteration(self, fetched: int) -> None:
        if type(fetched) is not int or not 0 <= fetched <= self.max_iter * self.active_ga:
            raise ValueError("trigger resume 偏移越界")
        self.start = fetched

    def __iter__(self):
        for fetched in range(self.start, self.max_iter * self.active_ga):
            if fetched % self.active_ga == 0 and self.on_iteration_start is not None:
                self.on_iteration_start(fetched // self.active_ga)
            yield {}

    def __len__(self) -> int:
        return self.max_iter * self.active_ga - self.start


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


def _telemetry_parameter_group(name: str) -> str | None:
    """The existing optimizer allowlist is the sole telemetry parameter authority."""
    if not _selected_name(name):
        return None
    if (
        name.startswith("net.local_memory_runtime.")
        or name.startswith("net.local_memory2llm.")
        or name.startswith("net.local_memory_modality_embed")
    ):
        return "local"
    if name.split(".", 2)[1] in ACTION_KEYS:
        return "action"
    return "generation"


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


def install_checkpoint_save_telemetry(checkpointer: Any, observer: GroupedPlanObserver) -> None:
    """Wrap DCP save for timing only; preserve its return value and exceptions."""
    original_save = checkpointer.save

    def observed_save(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_save(*args, **kwargs)
        iteration = kwargs.get("iteration")
        if iteration is not None:
            observer.log_checkpoint(int(iteration), (time.perf_counter() - started) * 1000.0)
        return result

    checkpointer.save = observed_save


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


def _validate_runtime_tokenizer(contract: CorrectedRoboCasaPolicyContract, tokenizer: Any) -> None:
    plain = OmegaConf.to_container(tokenizer, resolve=False) if isinstance(tokenizer, DictConfig) else tokenizer
    contract.validate_tokenizer_config(plain)


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
    resolved_tokenizer = contract.resolve_tokenizer_config(model.tokenizer)
    contract.validate_tokenizer_config(resolved_tokenizer)
    model.tokenizer = resolved_tokenizer
    _validate_runtime_tokenizer(contract, model.tokenizer)
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


def _audit_root_noncode_changes(porcelain_z: str) -> tuple[str, ...]:
    """Permit known MM notes/Evidence, never production-code or unknown root changes."""
    if not porcelain_z:
        return ()
    records = porcelain_z.split("\0")
    if records[-1] != "":
        raise ValueError("Root git status porcelain -z 格式不完整")
    permitted = []
    for record in records[:-1]:
        if len(record) < 4 or record[2] != " ":
            raise ValueError("Root git status porcelain -z 记录格式非法")
        status, path = record[:2], record[3:]
        notes = status in {" M", "M ", "MM"} and path in {"SESSION.md", "TODO.md"}
        evidence_json = status == "??" and path.startswith("artifacts/g0/") and path.endswith(".json")
        evidence_report = (
            status == "??"
            and path.startswith("docs/collab/chatgpt/DS_PRO_")
            and path.endswith(".md")
            and "/" not in path[len("docs/collab/chatgpt/"):]
        )
        if not (notes or evidence_json or evidence_report):
            raise ValueError(f"Root 工作树含非授权改动：{status} {path}")
        permitted.append(f"{status} {path}")
    return tuple(permitted)


def verify_root_child_lock(args: argparse.Namespace) -> dict[str, Any]:
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
    ):
        raise ValueError("V3 root/child/Gitlink 或 child 工作树不匹配")
    # Gitlink and child source remain strictly clean and SHA-pinned. The root
    # tree contains MM-owned notes and DS-only reports that must be preserved.
    # Read porcelain -z without .strip(), which would destroy its leading XY status.
    root_status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    noncode_changes = _audit_root_noncode_changes(root_status)
    return {
        "root": actual_root,
        "child": actual_child,
        "gitlink": gitlink,
        "allowed_root_noncode_changes": len(noncode_changes),
    }


def _execution_stop_iteration(args: argparse.Namespace) -> int:
    """Bound only fresh diagnostic execution, never the formal training config."""
    stop_after = getattr(args, "stop_after_iter", None)
    if stop_after is None:
        if args.phase == "resume" and args.job_name.startswith("bounded_"):
            raise ValueError("bounded diagnostic checkpoint must not be resumed as formal training")
        return args.max_iter
    if (
        args.phase != "fresh"
        or type(stop_after) is not int
        or not 1 <= stop_after < args.max_iter
        or (args.max_iter, args.warmup, args.save_iter) != (30000, 500, 100)
        or not args.job_name.startswith("bounded_")
    ):
        raise ValueError(
            "--stop-after-iter is fresh-only and requires formal 30000/500/100 schedule "
            "and a distinct bounded_ job-name"
        )
    namespace = args.output_root / "psm_wma_v3" / "corrected_phase5" / args.job_name
    if namespace.exists():
        raise FileExistsError(f"bounded diagnostic requires a new isolated job namespace: {namespace}")
    return stop_after


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, ExactWindowLocalCatalog, Any]:
    if any(
        type(value) is not int or value <= 0
        for value in (args.t, args.b, args.ga, args.k, args.world_size, args.max_iter, args.save_iter)
    ):
        raise ValueError("T/B/GA/K/world_size/max_iter/save_iter 必须是正整数")
    if args.warmup < 0 or args.warmup > args.max_iter:
        raise ValueError("warmup 越界")
    execution_stop = _execution_stop_iteration(args)
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
    _validate_runtime_tokenizer(contract, config.model.config.tokenizer)
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
            "execution_stop_after_iter": execution_stop,
            "execution_scope": "bounded_diagnostic" if execution_stop < args.max_iter else "full_formal",
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
    result.add_argument(
        "--stop-after-iter",
        type=int,
        default=None,
        help="Fresh-only bounded optimizer steps; preserves max_iter=30000/warmup=500/save_iter=100",
    )
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
    if getattr(args, "stop_after_iter", None) is not None:
        trainer._execution_max_iter = report["execution_stop_after_iter"]
    with model_init():
        model = instantiate(config.model)
    install_optimizer_inventory_check(model, report)
    planner = ExactWindowRankPlanner(catalog, rank=rank, world_size=args.world_size, b_stream=args.b, active_ga=args.ga)
    producer = ExactWindowSegmentProducer(catalog, config_digest=report["config_digest"])
    trainer.bind_grouped_stream(planner, producer, config_digest=report["config_digest"])
    observer = GroupedPlanObserver(rank=rank, parameter_group=_telemetry_parameter_group)
    trainer.grouped_observer = observer
    install_checkpoint_save_telemetry(trainer.checkpointer, observer)
    trainer.train(
        model,
        GroupedTriggerLoader(args.max_iter, args.ga, on_iteration_start=observer.start_iteration),
        None,
    )
    start_iteration = int(Path(report["checkpoint_load_path"]).name[5:]) if args.phase == "resume" else 0
    expected_iteration = report["execution_stop_after_iter"]
    if (
        trainer._grouped_completed_iteration != expected_iteration
        or observer.completed != expected_iteration - start_iteration
    ):
        raise RuntimeError("Phase5 完成的 optimizer iteration 不匹配")
    if rank == 0 and getattr(args, "stop_after_iter", None) is not None:
        print(
            "[CorrectedV3][bounded_stop] "
            + json.dumps(
                {
                    "completed_iteration": expected_iteration,
                    "configured_max_iter": args.max_iter,
                    "configured_warmup": args.warmup,
                    "config_digest": report["config_digest"],
                    "diagnostic_only_do_not_resume": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

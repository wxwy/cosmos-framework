"""V3 H3-E：8×H100 Edge+Local 两窗口训练与严格同 job 恢复。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    DEFAULT_ALL_ATOMIC_TASKS,
    RoboCasaLeRobotDataset,
)
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import (
    RankLocalGroupedPlanner,
    RoboCasaEpisodeCatalog,
    StageARoboCasaEpisodeBinder,
    materialize_member,
)
from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
from cosmos_framework.trainer.local_memory_grouped_resume import snapshot_grouped_local_state
from cosmos_framework.utils import distributed
from cosmos_framework.utils.context_managers import model_init
from cosmos_framework.utils.lazy_config import instantiate
from examples.psm_wma_robocasa_local_s1 import (
    LOCAL_PARAMS,
    SmokePaths,
    build_stage_a_action_transform,
    read_stage_a_contract,
)
from examples.psm_wma_robocasa_native import RECIPE

HOST_KEYS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
SELECTED_KEYS = (*HOST_KEYS, "local_memory")
MANIFEST_DIGEST = "a8cad3f053232b348ea155f15bf79c2c9cf807dedcf39b89e246b17e43f283df"

ROOT_WORKTREE = Path("/mnt/data/shenzhen/szrobot/logs/.tmp_backup/psm_wma_v3")
STAGE_A_ROOT = Path("/mnt/data1/data_v2_0617/psm_wma_v3_stage_a_h100/psm_wma_v3/edge_robocasa/smoke")
DEFAULT_CHECKPOINT = STAGE_A_ROOT / "checkpoints/iter_000000001"
DEFAULT_CONFIG = STAGE_A_ROOT / "config.yaml"
DEFAULT_DATASET_ROOT = Path("/mnt/data1/data_v2_0617/robocasa365_official_v30")
CACHE_ROOT = Path("/mnt/data1/data_v2_0617/robocasa365_official_v30_wan2.2vae_latent_b1")
DEFAULT_CACHE = CACHE_ROOT / "CloseFridge/20250816/lerobot/ep_000067.h5"
DEFAULT_EDGE = Path("/mnt/data/shenzhen/szrobot/logs/.tmp_backup/models/Cosmos3-Edge-Policy-DROID")
DEFAULT_VAE = Path("/mnt/data/shenzhen/szrobot/logs/.tmp_backup/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")

STAGE_A_CONFIG_SHA256 = "f64036c499f891979213469523a160ac08cb3d976a2add8d8fdf93750c5a5439"
STAGE_A_MODEL_METADATA_SHA256 = "53adef43a58e23f37d1132c4868ea8055be1b095cf76762b2e3e0b69ea287731"

H3F_FORMAL_MAX_ITER = 30_000
H3F_FORMAL_CHECKPOINT_ITERS = (1_000, 2_000, 4_000, 8_000, 12_000, 16_000, 20_000, 24_000, 30_000)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_h100_asset_authority() -> dict[str, str]:
    """Fail closed on any H100 source/cache/Stage-A authority drift."""
    required_dirs = (ROOT_WORKTREE, DEFAULT_CHECKPOINT, DEFAULT_DATASET_ROOT, CACHE_ROOT, DEFAULT_EDGE)
    required_files = (
        DEFAULT_CONFIG,
        DEFAULT_CHECKPOINT / "model/.metadata",
        DEFAULT_CHECKPOINT / "trainer/.metadata",
        DEFAULT_CACHE,
        DEFAULT_EDGE / "config.json",
        DEFAULT_VAE,
        DEFAULT_DATASET_ROOT / "CloseFridge/20250816/lerobot/meta/info.json",
    )
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    missing.extend(str(path) for path in required_files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"H3-E H100 authority assets missing: {missing}")

    config_sha = _sha256_file(DEFAULT_CONFIG)
    model_metadata_sha = _sha256_file(DEFAULT_CHECKPOINT / "model/.metadata")
    if config_sha != STAGE_A_CONFIG_SHA256 or model_metadata_sha != STAGE_A_MODEL_METADATA_SHA256:
        raise ValueError(
            f"H3-E Stage-A H100 authority digest mismatch: config={config_sha}, model_metadata={model_metadata_sha}"
        )
    return {
        "config_sha256": config_sha,
        "model_metadata_sha256": model_metadata_sha,
        "stage_a_checkpoint": str(DEFAULT_CHECKPOINT),
        "source_root": str(DEFAULT_DATASET_ROOT),
        "cache_root": str(CACHE_ROOT),
        "edge": str(DEFAULT_EDGE),
        "vae": str(DEFAULT_VAE),
    }


def load_stage_a_config():
    for key, expected in (
        ("EDGE_POLICY_CHECKPOINT", DEFAULT_EDGE),
        ("WAN_VAE_PATH", DEFAULT_VAE),
        ("ROBOCASA_ROOT", DEFAULT_DATASET_ROOT),
    ):
        if key in os.environ and Path(os.environ[key]).resolve() != expected:
            raise ValueError(f"H3-E {key} 必须匹配 Stage-A 冻结资产")
        os.environ[key] = str(expected)
    os.environ["BASE_CHECKPOINT_PATH"] = str(DEFAULT_CHECKPOINT)
    return load_experiment_from_toml(RECIPE)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("fresh", "resume"), required=True)
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--job-name", default="h3e_edge_local_h100")
    result.add_argument("--expected-root", required=True)
    result.add_argument("--expected-child", required=True)
    return result


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def lock_pair(expected_root: str, expected_child: str) -> dict[str, str]:
    child = Path(__file__).resolve().parents[1]
    root_sha, child_sha = _git(ROOT_WORKTREE, "rev-parse", "HEAD"), _git(child, "rev-parse", "HEAD")
    gitlink = _git(ROOT_WORKTREE, "ls-tree", "HEAD", "cosmos-framework").split()[2]
    if (
        len(expected_root) != 40
        or len(expected_child) != 40
        or (root_sha, child_sha, gitlink) != (expected_root, expected_child, expected_child)
        or _git(child, "status", "--porcelain")
        or any(line != "?? artifacts/v3/" for line in _git(ROOT_WORKTREE, "status", "--porcelain").splitlines())
    ):
        raise ValueError("H3-E formal root/child/Gitlink 或工作树不匹配")
    return {"root": root_sha, "child": child_sha, "gitlink": gitlink}


def _paths(output_root: Path) -> SmokePaths:
    return SmokePaths(
        DEFAULT_CHECKPOINT,
        DEFAULT_CONFIG,
        DEFAULT_CACHE,
        DEFAULT_DATASET_ROOT,
        DEFAULT_EDGE,
        DEFAULT_VAE,
        output_root,
    )


def config_digest() -> str:
    authority = {
        "stage_a_config_sha256": _sha256_file(DEFAULT_CONFIG),
        "stage_a_model_metadata_sha256": _sha256_file(DEFAULT_CHECKPOINT / "model/.metadata"),
        "source": str(DEFAULT_DATASET_ROOT),
        "cache": str(CACHE_ROOT),
        "geometry": [8, 8, 2, 16, 4, 32, 33, 15, 64],
        "optimizer": [*SELECTED_KEYS, "FusedAdam", 5e-5],
        "manifest_digest": MANIFEST_DIGEST,
    }
    return hashlib.sha256(json.dumps(authority, sort_keys=True).encode()).hexdigest()


def overlay_h100_config(config: Any, *, phase: str, job_name: str) -> None:
    if phase not in ("fresh", "resume") or not job_name or "/" in job_name or job_name in (".", ".."):
        raise ValueError("H3-E phase/job-name 不合法")
    from examples.psm_wma_robocasa_local_s1 import overlay_local_config

    overlay_local_config(config)
    config.model.config.parallelism.data_parallel_shard_degree = 8
    config.model.config.parallelism.data_parallel_replicate_degree = 1
    config.optimizer.keys_to_select = list(SELECTED_KEYS)
    config.optimizer.lr_multipliers = {key: 5.0 for key in ("action2llm", "llm2action", "action_modality_embed")}
    config.trainer.type = GroupedLocalMemoryTrainer
    config.trainer.grad_accum_iter = 2
    config.trainer.max_iter = 1 if phase == "fresh" else 2
    config.trainer.callbacks = {}
    config.trainer.run_validation = False
    config.checkpoint.load_path = str(DEFAULT_CHECKPOINT)
    config.checkpoint.load_training_state = False
    config.checkpoint.keys_to_skip_loading = ["net_ema.", "local_memory"]
    config.checkpoint.strict_resume = True
    config.checkpoint.save_iter = 1
    config.job.project = "psm_wma_v3"
    config.job.group = "h3e_edge_local_h100"
    config.job.name = job_name
    if (
        config.optimizer.optimizer_type != "FusedAdam"
        or float(config.optimizer.lr) != 5e-5
        or config.model.config.parallelism.fsdp_master_dtype != "float32"
        or config.model.config.parallelism.fsdp_reduce_dtype != "bfloat16"
        or config.model.config.precision != "bfloat16"
        or config.model.config.max_action_dim != 64
        or config.model.config.num_embodiment_domains != 32
        or config.model.config.tokenizer.encode_exact_durations != [33]
    ):
        raise ValueError("H3-E Edge/H100/optimizer 精度合同不匹配")


def make_catalog() -> tuple[RoboCasaLeRobotDataset, RoboCasaEpisodeCatalog]:
    dataset = RoboCasaLeRobotDataset(
        root=str(DEFAULT_DATASET_ROOT),
        fps=20,
        chunk_length=32,
        split="train",
        split_seed=42,
        split_val_ratio=0.01,
        mode="wam",
        task_names=DEFAULT_ALL_ATOMIC_TASKS,
        use_state=True,
        use_base_action=True,
        base_encoding="raw",
        camera_set="left_wrist",
        action_normalization=None,
    )
    catalog = RoboCasaEpisodeCatalog.from_stage_a_dataset(
        dataset, source_root=DEFAULT_DATASET_ROOT, cache_root=CACHE_ROOT
    )
    if len(catalog.episodes) != 9036 or catalog.manifest_digest != MANIFEST_DIGEST:
        raise ValueError("H3-E 18 类 train catalog 数量或 digest 不匹配")
    partitions = [set(item.uid for item in catalog.rank_episodes(rank)) for rank in range(8)]
    if set.union(*partitions) != set(catalog.by_uid) or sum(map(len, partitions)) != len(catalog.episodes):
        raise ValueError("H3-E 8-rank catalog 分区重复或缺失")
    return dataset, catalog


class GroupedTriggerLoader:
    """只驱动原生 trainer 的两次 GA；样本始终由 H3-B binder 供应。"""

    def __init__(self, max_iter: int) -> None:
        self.max_iter, self.start = max_iter, 0

    def set_start_iteration(self, fetched: int) -> None:
        if not 0 <= fetched <= self.max_iter * 2:
            raise ValueError("H3-E trigger resume 偏移越界")
        self.start = fetched

    def __iter__(self):
        for _ in range(self.start, self.max_iter * 2):
            yield {}

    def __len__(self) -> int:
        return self.max_iter * 2 - self.start


def preflight_native_batch(
    dataset: RoboCasaLeRobotDataset,
    catalog: RoboCasaEpisodeCatalog,
    paths: SmokePaths,
    *,
    digest: str,
) -> dict[str, Any]:
    """Materialize rank0's first grouped member through the production catalog/binder path."""
    transform, resolution = build_stage_a_action_transform(paths)
    binder = StageARoboCasaEpisodeBinder(
        dataset,
        catalog,
        source_root=DEFAULT_DATASET_ROOT,
        cache_root=CACHE_ROOT,
        transform=transform,
        resolution=resolution,
        config_digest=digest,
    )
    planner = RankLocalGroupedPlanner(catalog, rank=0, world_size=8, seed=0)
    plan = planner.plan_window(planner.initial_frontier())
    segments = materialize_member(plan.members[0], binder.producer_for)
    payloads = tuple(segment.consumer_payload[0][0] for segment in segments)
    batch = collate_grouped_native_batch(payloads)
    if (
        len(segments) != 8
        or len(batch["sequence_plan"]) != 8
        or any(item[0].shape != (33, 64) for item in batch["action"])
        or any(item[0].shape != (33, 15) for item in batch["action_raw"])
    ):
        raise ValueError("H3-E Stage-A grouped 8-slot native batch ABI 不匹配")
    return {
        "native_batch": 8,
        "preflight_uids": [request.episode.uid for request in plan.members[0]],
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    pair = lock_pair(args.expected_root, args.expected_child)
    output_root = args.output_root.expanduser().resolve()
    if output_root == ROOT_WORKTREE or ROOT_WORKTREE in output_root.parents:
        raise ValueError("H3-E 产物必须位于 root worktree 之外")
    job = output_root / "psm_wma_v3/h3e_edge_local_h100" / args.job_name
    if args.phase == "fresh" and job.exists():
        raise FileExistsError("fresh job 已存在；禁止覆盖原始 Evidence")
    if args.phase == "resume" and not (job / "checkpoints/latest_checkpoint.txt").is_file():
        raise FileNotFoundError("resume 缺少同 job latest_checkpoint.txt")
    paths = _paths(output_root)
    asset_authority = validate_h100_asset_authority()
    contract = read_stage_a_contract(paths)
    config = load_stage_a_config()
    overlay_h100_config(config, phase=args.phase, job_name=args.job_name)
    dataset, catalog = make_catalog()
    digest = config_digest()
    native = preflight_native_batch(dataset, catalog, paths, digest=digest)
    return {
        "pair": pair,
        "phase": args.phase,
        "job": str(job),
        "catalog_episodes": len(catalog.episodes),
        "manifest_digest": catalog.manifest_digest,
        "config_digest": digest,
        "stage_a_dcp_keys": contract["dcp_keys"],
        "asset_authority": asset_authority,
        **native,
        "geometry": [8, 8, 2, 16, 4],
        "selector": list(SELECTED_KEYS),
    }


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def committed_digest(trainer: GroupedLocalMemoryTrainer) -> str:
    state = snapshot_grouped_local_state(
        trainer._grouped_window,
        iteration=trainer._grouped_completed_iteration,
        config_digest=trainer._grouped_config_digest,
    )
    digest = hashlib.sha256(
        repr(
            (
                state["iteration"],
                state["manifest_digest"],
                state["config_digest"],
                state["frontier"],
                state["scheduler"],
            )
        ).encode()
    )
    for slot, (identity, provenance, fast) in sorted(state["sidecar"].items()):
        digest.update(repr((slot, identity, provenance)).encode())
        for value in fast:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_completed_checkpoint(job: Path, *, rank: int, iteration: int, phase: str, resumed: bool) -> Path:
    expected = f"iter_{iteration:09d}"
    checkpoint = job / "checkpoints" / expected
    if (
        (job / "checkpoints/latest_checkpoint.txt").read_text().strip() != expected
        or any(not (checkpoint / key / ".metadata").is_file() for key in ("model", "optim", "scheduler", "trainer"))
        or not (checkpoint / "dataloader" / f"rank_{rank}.pkl").is_file()
        or (phase == "resume" and not resumed)
    ):
        raise RuntimeError("H3-E DCP/同 job resume 证据不完整")
    return checkpoint


class SmokeObserver:
    def __init__(self, path: Path, model: torch.nn.Module) -> None:
        self.path, self.model = path, model
        self.counts: dict[str, int] = {}

    def __call__(
        self, *, phase: str, iteration: int, member: int, index: int | None, loss: torch.Tensor | None, trainer: Any
    ) -> None:
        torch.cuda.synchronize()
        self.counts[phase] = self.counts.get(phase, 0) + 1
        event: dict[str, Any] = {
            "phase": phase,
            "iteration": iteration,
            "member": member,
            "index": index,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if loss is not None:
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("H3-E native loss 非有限或非标量")
            event["native_loss"] = float(loss.detach())
        if phase == "pre_optimizer":
            selected = [
                (name, _local(parameter.grad))
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not selected or any(not torch.isfinite(gradient).all() for _, gradient in selected):
                raise FloatingPointError("H3-E selected gradients 缺失或非有限")
            event["grad_norms"] = {
                name: float(gradient.float().norm()) for name, gradient in selected if "local_memory" in name
            }
        if phase == "post_commit":
            event["completed_iteration"] = trainer._grouped_completed_iteration + 1
            event["frontier_epoch"] = trainer._grouped_window.live.frontier.epoch
        with self.path.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or not torch.cuda.is_available():
        raise RuntimeError("H3-E 必须由8-rank CUDA torchrun 启动")
    rank = int(os.environ["RANK"])
    if not 0 <= rank < 8 or "H100" not in torch.cuda.get_device_name(int(os.environ["LOCAL_RANK"])):
        raise RuntimeError("H3-E 每个 local rank 必须绑定 H100")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root.expanduser().resolve())
    os.environ["EDGE_POLICY_CHECKPOINT"] = str(DEFAULT_EDGE)
    os.environ["WAN_VAE_PATH"] = str(DEFAULT_VAE)
    config = load_stage_a_config()
    overlay_h100_config(config, phase=args.phase, job_name=args.job_name)
    job = Path(report["job"])
    job.mkdir(parents=True, exist_ok=True)
    result_path = job / f"h3e_{args.phase}_rank{rank}.json"
    if result_path.exists():
        raise FileExistsError(f"H3-E 不覆盖已有 rank Evidence：{result_path}")
    result: dict[str, Any] = dict(report, rank=rank, result="FAIL")
    try:
        distributed.init()
        config.validate()
        config.freeze()
        trainer = GroupedLocalMemoryTrainer(config)
        with model_init():
            model = instantiate(config.model)
        original = model.init_optimizer_scheduler

        def checked_optimizer(optimizer_config, scheduler_config):
            optimizer, scheduler = original(optimizer_config, scheduler_config)
            named = dict(model.named_parameters())
            selected = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            selected_names = {name for name, parameter in named.items() if id(parameter) in selected}
            local_names = {name for name in named if name.startswith("net.local_memory")}
            if (
                not local_names <= selected_names
                or not all(any(key in name for name in selected_names) for key in HOST_KEYS)
                or sum(named[name].numel() for name in local_names) != LOCAL_PARAMS
                or not all(any(key in name for key in SELECTED_KEYS) for name in selected_names)
                or any(parameter.requires_grad for name, parameter in named.items() if name not in selected_names)
            ):
                raise ValueError("H3-E optimizer Edge+Local inventory 不匹配")
            result["selected_names"] = sorted(selected_names)
            result["selected_local_params"] = LOCAL_PARAMS
            return optimizer, scheduler

        model.init_optimizer_scheduler = checked_optimizer
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
        observer = SmokeObserver(job / f"h3e_{args.phase}_rank{rank}_cuda.jsonl", model)
        trainer.grouped_observer = observer
        trainer.train(model, GroupedTriggerLoader(config.trainer.max_iter), None)
        if observer.counts != {"native_forward": 32, "native_backward": 32, "pre_optimizer": 1, "post_commit": 1}:
            raise RuntimeError(f"H3-E grouped 事件数量错误：{observer.counts}")
        if trainer._grouped_completed_iteration != config.trainer.max_iter:
            raise RuntimeError("H3-E 完成的 optimizer iteration 不匹配")
        checkpoint = verify_completed_checkpoint(
            job,
            rank=rank,
            iteration=config.trainer.max_iter,
            phase=args.phase,
            resumed=trainer._resume_required,
        )
        frontier = trainer._grouped_window.live.frontier
        next_plan = planner.plan_window(frontier)
        result.update(
            result="PASS",
            events=observer.counts,
            checkpoint=str(checkpoint),
            committed_digest=committed_digest(trainer),
            next_identities=[str(request.identity) for request in next_plan.members[0]],
        )
    except Exception:
        result["traceback"] = traceback.format_exc()
        raise
    finally:
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
        print(f"H3-E FAIL: {error}", file=sys.stderr)
        raise

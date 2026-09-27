"""V3 B2-C 单段真实 S1 harness；仅经显式非 preflight 命令运行 CUDA。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.distributed.checkpoint import FileSystemReader

from cosmos_framework.model.generator.mot.local_memory_native_segment import (
    NativeConsumerResult,
    SingleSegmentNativeGradientRelay,
)
from cosmos_framework.model.generator.mot.local_memory_segment import GAWindowPlan, SegmentBatch, SegmentIdentity
from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader
from cosmos_framework.model.generator.mot.robocasa_segment_producer import RoboCasaSegmentProducer

TASK = "CloseFridge"
DATE = "20250816"
EPISODE_INDEX = 0
EPISODE_ID = "ep_000000"
CURSOR = 0
T = 16
LOCAL_PARAMS = 165_312
LR = 5e-5
STAGE_A_ROOT = Path(
    "/disk/rl/worktrees/psm_wma-v3/artifacts/v3/stage_a_edge_raw15_run02/psm_wma_v3/edge_robocasa/smoke"
)
DEFAULT_CHECKPOINT = STAGE_A_ROOT / "checkpoints/iter_000000001"
DEFAULT_CONFIG = STAGE_A_ROOT / "config.yaml"
DEFAULT_CACHE = Path(
    "/disk/rl/starVLA/playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic/CloseFridge/20250816/lerobot/ep_000000.h5"
)
DEFAULT_DATASET_ROOT = Path("/disk/rl/data/robocasa_v30")
DEFAULT_EDGE = Path("/disk/rl/models/Cosmos3-Edge-Policy-DROID")
DEFAULT_VAE = Path("/disk/rl/models/wan22_vae/Wan2.2_VAE.pth")
B2B_ROOT = "81fa515593e7cd8e2d4f7d226efb915b17be3b5b"
B2B_CHILD = "bf6c80e679812b7d2881d6a54aa0b518299e3869"
ROOT_WORKTREE = Path("/disk/rl/worktrees/psm_wma-v3")


@dataclass(frozen=True)
class SmokePaths:
    checkpoint: Path
    config: Path
    cache: Path
    dataset_root: Path
    edge: Path
    vae: Path
    output: Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preflight", action="store_true", help="仅执行 CPU 数据与配置预检")
    result.add_argument("--output", type=Path, required=True, help="必须是尚不存在的唯一目录")
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    result.add_argument("--edge", type=Path, default=DEFAULT_EDGE)
    result.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    return result


def paths_from_args(args: argparse.Namespace) -> SmokePaths:
    return SmokePaths(
        *(
            getattr(args, key).expanduser().resolve()
            for key in ("checkpoint", "config", "cache", "dataset_root", "edge", "vae", "output")
        )
    )


def validate_frozen_paths(paths: SmokePaths) -> None:
    for key, value, expected in (
        ("checkpoint", paths.checkpoint, DEFAULT_CHECKPOINT),
        ("config", paths.config, DEFAULT_CONFIG),
        ("cache", paths.cache, DEFAULT_CACHE),
        ("dataset_root", paths.dataset_root, DEFAULT_DATASET_ROOT),
        ("edge", paths.edge, DEFAULT_EDGE),
        ("vae", paths.vae, DEFAULT_VAE),
    ):
        if value != expected.resolve():
            raise ValueError(f"B2-C 冻结资产路径不匹配：{key}={value}")
    if paths.checkpoint.parent.parent != paths.config.parent:
        raise ValueError("Stage-A DCP 与 config.yaml 必须属于同一 smoke run")


def _require(path: Path, *, directory: bool = False) -> None:
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"B2-C 必需资产缺失：{path}")


def read_stage_a_contract(paths: SmokePaths) -> dict[str, Any]:
    from examples.psm_wma_robocasa_native import check_edge_checkpoint

    _require(paths.config)
    _require(paths.checkpoint / "model/.metadata")
    _require(paths.cache)
    _require(paths.vae)
    _require(paths.edge / "config.json")
    if check_edge_checkpoint(str(paths.edge)) != paths.edge:
        raise ValueError("Stage-A Edge processor/checkpoint 身份不匹配")
    _require(paths.dataset_root / TASK / DATE / "lerobot/meta/info.json")
    config = yaml.safe_load(paths.config.read_text())
    dataset = config["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]
    model = config["model"]["config"]
    expected = {
        "use_base_action": True,
        "base_encoding": "raw",
        "camera_set": "left_wrist",
        "use_state": True,
        "fps": 20,
        "chunk_length": 32,
        "action_normalization": None,
        "root": str(paths.dataset_root),
        "task_names": [TASK],
        "max_action_dim": 64,
    }
    if any(dataset.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage-A config 的 RoboCasa raw15 合同不匹配")
    if (
        model["tokenizer"]["encode_exact_durations"] != [33]
        or model["max_action_dim"] != 64
        or model["num_embodiment_domains"] != 32
        or model["vlm_config"]["tokenizer"]["tokenizer_type"] != str(paths.edge)
        or model["tokenizer"]["vae_path"] != str(paths.vae)
        or model["activation_checkpointing"]["mode"] != "selective"
        or model["compile"]["enabled"]
        or model["ema"]["enabled"]
        or model["lbl"]["coeff_gen"] is not None
        or model["lbl"]["coeff_und"] is not None
        or config["optimizer"]["optimizer_type"] != "FusedAdam"
        or float(config["optimizer"]["lr"]) != LR
    ):
        raise ValueError("Stage-A config 的 Edge/资源合同不匹配")
    info = json.loads((paths.dataset_root / TASK / DATE / "lerobot/meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("B2-C 需要 RoboCasa LeRobot v3.0")
    metadata = FileSystemReader(paths.checkpoint / "model").read_metadata().state_dict_metadata
    keys = set(metadata)
    required = {
        "net.action2llm.fc.weight",
        "net.llm2action.fc.weight",
        "net.action_modality_embed",
        "net.vae2llm.weight",
        "net.llm2vae.weight",
    }
    if not required <= keys or any("local_memory" in key or not key.startswith("net.") for key in keys):
        raise ValueError("Stage-A DCP 缺少 host/action 键或已混入 Local 参数")
    return {"config": config, "dcp_keys": len(keys), "dataset_info": info}


def validate_warm_start_keys(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    """仅允许新 Local 键缺失；禁用 EMA 后不得遗漏任何 host 键。"""
    from cosmos_framework.checkpoint.dcp import ModelWrapper

    target = ModelWrapper(model).state_dict()
    metadata = FileSystemReader(checkpoint / "model").read_metadata().state_dict_metadata
    source_keys, target_keys = set(metadata), set(target)
    missing = target_keys - source_keys
    unexpected = source_keys - target_keys
    if any("local_memory" not in key and not key.startswith("net_ema.") for key in missing):
        raise ValueError(f"Stage-A DCP 缺少 host 权重：{sorted(missing)[:8]}")
    if any(not key.startswith("net_ema.") for key in unexpected):
        raise ValueError(f"Stage-A DCP 包含意外 host 权重：{sorted(unexpected)[:8]}")
    for key in source_keys & target_keys:
        size = getattr(metadata[key], "size", None)
        if size is not None and tuple(size) != tuple(target[key].shape):
            raise ValueError(f"Stage-A DCP 形状不匹配：{key}")
    return {"missing_local": sorted(missing), "disabled_ema": sorted(unexpected), "host_keys": len(source_keys)}


def load_stage_a_host(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    """复用训练侧 ModelWrapper + DCP planner，严格载入全部 host 权重。"""
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, ModelWrapper

    report = validate_warm_start_keys(model, checkpoint)
    wrapper = ModelWrapper(model)
    state = wrapper.state_dict()
    dcp.load(
        state_dict=state,
        storage_reader=FileSystemReader(checkpoint / "model"),
        planner=CustomLoadPlanner(keys_to_skip_loading=["local_memory"]),
        no_dist=True,
    )
    result = wrapper.load_state_dict(state)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"DCP load_state_dict 未完整写回 host：{result}")
    return report


def load_episode_segment(paths: SmokePaths) -> tuple[SegmentBatch, SegmentIdentity, dict[str, Any]]:
    """从精确 v3 episode0 取得完整 raw15 与 16 个官方 RGB payload。"""
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

    shard = paths.dataset_root / TASK / DATE / "lerobot"
    dataset = RoboCasaLeRobotDataset(
        root=str(shard),
        fps=20,
        chunk_length=32,
        split="full",
        mode="wam",
        task_names=(TASK,),
        use_state=True,
        use_base_action=True,
        base_encoding="raw",
        camera_set="left_wrist",
        action_normalization=None,
    )
    if dataset._all_shard_roots != [str(shard)]:
        raise ValueError("RoboCasa source shard 不匹配")
    spans = [
        (span_index, source, start, length)
        for span_index, (source, start, length, episode) in enumerate(dataset._episode_records)
        if episode == EPISODE_INDEX
    ]
    if len(spans) != 1 or spans[0][1] != 0:
        raise ValueError("RoboCasa episode0 必须有唯一的官方样本 span")
    span_index, source, row_start, valid_length = spans[0]
    flat_start = 0 if span_index == 0 else dataset._episode_cum_ends[span_index - 1]
    lerobot = dataset._get_dataset(source)
    frames = int(lerobot.meta.episodes["length"][EPISODE_INDEX])
    if valid_length != frames - 32 or valid_length < T:
        raise ValueError("episode0 无法提供精确 cursor0 T16、chunk32 窗口")
    rows = lerobot.hf_dataset[row_start : row_start + frames]
    if list(rows["episode_index"]) != [EPISODE_INDEX] * frames or [int(i) for i in rows["frame_index"]] != list(
        range(frames)
    ):
        raise ValueError("LeRobot episode0 行身份或 source timestep 不连续")
    raw12 = torch.stack(rows["action"]).float()
    if raw12.shape != (frames, 12):
        raise ValueError("LeRobot 原始 action 必须为 [F,12]")
    raw15 = torch.cat((raw12[:, :5], dataset._build_frame_wise_action(raw12)), dim=-1)
    if raw15.shape != (frames, 15) or not torch.isfinite(raw15).all():
        raise ValueError("官方转换的完整 episode raw15 非法")
    payloads = []
    for step in range(T):
        flat = flat_start + step
        if dataset._resolve_index(flat) != (source, row_start + step, EPISODE_INDEX, step):
            raise ValueError(f"找不到 episode0 consumer step {step} 的官方锚点")
        payload = dataset[flat]
        action, video = payload.get("action"), payload.get("video")
        if not isinstance(action, torch.Tensor) or action.shape != (33, 15):
            raise ValueError(f"consumer {step} action 不是含 state 的 raw15/33")
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 4
            or video.shape[0] != 3
            or video.shape[1] != 33
            or video.shape[3] != 2 * video.shape[2]
        ):
            raise ValueError(f"consumer {step} 必须走官方 RGB/33 帧 policy 输入")
        if not torch.equal(action[1:], raw15[step : step + 32]):
            raise ValueError(f"consumer {step} 的重叠 raw15 transition 不一致")
        if "video_latent" in payload:
            raise ValueError("cached latent 不得进入主 policy payload")
        payloads.append(payload)
    reader = RoboCasaLatentReader(paths.cache, expected_episode_id=EPISODE_ID, expected_source_frames=frames)
    producer = RoboCasaSegmentProducer(
        reader,
        episode_id=EPISODE_ID,
        category="robocasa",
        raw15=raw15,
        payload_at=lambda step: payloads[step],
        manifest_digest="b2c-stage-a",
        config_digest="b2c-stage-a-config",
        source_digest="b2c-closefridge-ep0",
    )
    identity = producer.identity(slot_id=0, cursor=CURSOR, segment_id=0)
    segment = producer.produce(identity)
    if segment.consumer_step.tolist() != [list(range(T))] or not bool(segment.consumer_valid.all()):
        raise ValueError("B2-C cursor0 必须恰有 16 个有效 consumer")
    return segment, identity, {"frames": frames, "raw_action_dim": 15, "payloads": len(payloads)}


def native_callback(model: Any, trace: list[dict[str, Any]], state: dict[str, Any]):
    """B2-B 单 consumer ABI：仅 policy 原生 loss，缓存 latent 只供 B0 evidence。"""
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    from cosmos_framework.utils import misc

    def forward(payload: dict[str, Any], leaf: torch.Tensor | None, index: int) -> NativeConsumerResult:
        if index not in range(T) or (leaf is None) != (index == 0) or "video_latent" in payload:
            raise ValueError("原生 consumer 的 S0/Local 或 RGB 权威不匹配")
        if index > 1:
            # B2-B 串行循环保证进入下一个 callback 时上一个 backward 已结束。
            record_cuda(trace, "consumer_backward", index - 1)
        state["phase"], state["consumer"] = "native_forward", index
        torch.cuda.reset_peak_memory_stats()
        batch = misc.to(custom_collate_fn([payload]), device="cuda")
        diagnostics, loss = model.training_step(batch, iteration=0, _local_memory_prefixes=(leaf,))
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("原生 total policy loss 必须为标量")
        if any(key.startswith("aux_loss_") for key in diagnostics):
            raise ValueError("B2-C 串行 relay 不支持 sample-coupled auxiliary loss")
        record_cuda(trace, "consumer_forward", index)
        if leaf is not None:
            state["phase"] = "native_backward"
        return NativeConsumerResult(loss, output={"loss": loss.detach(), "index": index})

    return forward


def record_cuda(trace: list[dict[str, Any]], phase: str, consumer: int | None = None) -> None:
    torch.cuda.synchronize()
    trace.append(
        {
            "phase": phase,
            "consumer": consumer,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )


def local_optimizer(model: torch.nn.Module, config: Any):
    """沿用框架优化器参数选择器与 Stage-A FusedAdam，冻结全部 host。"""
    from cosmos_framework.utils.generator.optimizer import build_optimizer

    if config.optimizer.optimizer_type != "FusedAdam" or float(config.optimizer.lr) != LR:
        raise ValueError("Stage-A optimizer 类型或 LR 不匹配")
    for name, parameter in model.named_parameters():
        if not name.startswith("net.local_memory"):
            parameter.requires_grad_(False)
    container = build_optimizer(
        model,
        optimizer_type="FusedAdam",
        fused=True,
        lr=LR,
        betas=list(config.optimizer.betas),
        eps=float(config.optimizer.eps),
        weight_decay=float(config.optimizer.weight_decay),
        keys_to_select=["local_memory"],
        lr_multipliers={},
    )
    if len(container.optimizers) != 1:
        raise ValueError("B2-C 只允许一个 Local-only optimizer")
    selected = [parameter for group in container.optimizers[0].param_groups for parameter in group["params"]]
    named = dict(model.named_parameters())
    local = {id(parameter) for name, parameter in named.items() if name.startswith("net.local_memory")}
    if sum(parameter.numel() for parameter in selected) != LOCAL_PARAMS or {id(p) for p in selected} != local:
        raise ValueError("Local optimizer inventory 必须精确为 165312")
    if any(parameter.requires_grad for name, parameter in named.items() if not name.startswith("net.local_memory")):
        raise ValueError("4090 profile 必须冻结全部 host/reasoner 参数")
    return container.optimizers[0]


def overlay_local_config(config: Any) -> None:
    """只在运行内存中叠加 B2-C 资源 profile，不改 Stage-A recipe。"""
    config.model.config.local_memory_enabled = True
    for key, value in (
        ("local_memory_dim", 32),
        ("local_memory_evidence_dim", 256),
        ("local_memory_action_dim", 15),
        ("local_memory_ttt_dim", 64),
        ("local_memory_fast_hidden_dim", 256),
        ("local_memory_inner_lr", 0.1),
        ("local_memory_ttt_tbptt_steps", T),
        ("local_memory_k_local", 4),
    ):
        setattr(config.model.config, key, value)
    config.model.config.ema.enabled = False
    config.model.config.compile.enabled = False
    if config.model.config.activation_checkpointing.mode != "selective":
        raise ValueError("必须保留 selective activation checkpointing")
    config.optimizer.keys_to_select = ["local_memory"]
    config.optimizer.lr = LR
    config.optimizer.lr_multipliers = {}


def lock_implementation_pair() -> dict[str, str]:
    """运行时核对 V3 root Gitlink 与当前 child 源码，禁止混用提交。"""
    child = Path(__file__).resolve().parents[1]

    def git(path: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    root_sha = git(ROOT_WORKTREE, "rev-parse", "HEAD")
    child_sha = git(child, "rev-parse", "HEAD")
    gitlink = git(ROOT_WORKTREE, "ls-tree", "HEAD", "cosmos-framework").split()[2]
    if child_sha != gitlink:
        raise ValueError("V3 root Gitlink 与 B2-C child 源码不一致")
    if git(child, "status", "--porcelain"):
        raise ValueError("B2-C child 工作树必须干净，避免 pair lock 与实际源码不一致")
    return {"root": root_sha, "child": child_sha, "gitlink": gitlink}


def _witness(model: torch.nn.Module) -> dict[str, Any]:
    names = (
        "net.local_memory_runtime.encoder.visual_proj.weight",
        "net.local_memory_runtime.core.slot_queries",
        "net.local_memory_runtime.core.w0_fast_in_weight",
        "net.local_memory2llm.weight",
        "net.local_memory_modality_embed",
    )
    selected = dict(model.named_parameters())
    witness = {}
    for name in names:
        gradient = selected[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(torch.count_nonzero(gradient)):
            raise ValueError(f"Local 梯度 witness 缺失、非有限或全零：{name}")
        witness[name] = {"norm": float(gradient.float().norm()), "nonzero": int(torch.count_nonzero(gradient))}
    if any(
        parameter.grad is not None for name, parameter in selected.items() if not name.startswith("net.local_memory")
    ):
        raise ValueError("冻结的 host 参数出现梯度")
    return witness


def execute_cuda(
    paths: SmokePaths,
    trace: list[dict[str, Any]],
    state: dict[str, Any],
    segment: SegmentBatch,
    identity: SegmentIdentity,
) -> dict[str, Any]:
    """只供审核后的 ds 独立进程调用；无 trainer、DCP save 或自动降级。"""
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

    if os.environ.get("WORLD_SIZE", "1") != "1" or not torch.cuda.is_available():
        raise RuntimeError("B2-C 仅允许有 CUDA 的单 rank 独立进程")
    if os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0":
        raise RuntimeError("B2-C 仅允许 rank0/local_rank0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    state["phase"] = "cuda_init"
    import torch.distributed as dist

    distributed.init(store=dist.FileStore(str(paths.output / "process_group.store"), world_size=1), backend="nccl")
    record_cuda(trace, "cuda_init")
    config = LazyConfig.load(str(paths.config))
    overlay_local_config(config)
    record_cuda(trace, "processor_config_ready")
    state["phase"] = "model_materialize"
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=torch.preserve_format)
    model.on_train_start()
    record_cuda(trace, "model_materialized")
    state["phase"] = "dcp_host_load"
    key_report = load_stage_a_host(model, paths.checkpoint)
    record_cuda(trace, "stage_a_dcp_host_loaded")
    record_cuda(trace, "wan_vae_tokenizer_ready")
    state["phase"] = "optimizer_ready"
    optimizer = local_optimizer(model, config)
    record_cuda(trace, "local_optimizer_ready")
    relay = SingleSegmentNativeGradientRelay(model, optimizer)
    plan = GAWindowPlan((identity.member,), (T,))
    before_host = (
        next(
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("net.") and "local_memory" not in name
        )
        .detach()
        .flatten()[:128]
        .clone()
    )
    before_local = model.net.local_memory2llm.weight.detach().clone()
    real_step, real_relay, real_commit = optimizer.step, relay._relay_gradients, relay.adapter.commit
    witnesses: dict[str, Any] = {}
    step_calls = 0

    def step_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_calls
        state["phase"] = "optimizer_step"
        step_calls += 1
        if step_calls != 1 or relay.sidecar.snapshot() or relay.scheduler._committed:
            raise RuntimeError("optimizer step 前不得发布 fast state/frontier")
        witnesses.update(_witness(model))
        result = real_step(*args, **kwargs)
        record_cuda(trace, "optimizer_step")
        return result

    def relay_and_record(prefixes, gradients) -> None:
        state["phase"] = "relay_backward"
        record_cuda(trace, "consumer_backward", T - 1)
        real_relay(prefixes, gradients)
        record_cuda(trace, "relay_backward")

    def commit_and_record(*args: Any, **kwargs: Any) -> None:
        state["phase"] = "fast_state_commit"
        after_host = (
            next(
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith("net.") and "local_memory" not in name
            )
            .detach()
            .flatten()[:128]
        )
        if (
            step_calls != 1
            or not torch.equal(before_host, after_host)
            or torch.equal(before_local, model.net.local_memory2llm.weight)
        ):
            raise ValueError("Local-only step 或冻结 host witness 不匹配")
        real_commit(*args, **kwargs)
        record_cuda(trace, "fast_state_commit")

    optimizer.step = step_once
    relay._relay_gradients = relay_and_record
    relay.adapter.commit = commit_and_record
    state["phase"] = "b0_scan"
    prepared = relay.prepare(segment, identity, plan)
    record_cuda(trace, "b0_scan_completed")
    callback = native_callback(model, trace, state)
    outcome = relay.execute_prepared(prepared, callback)
    committed = relay.sidecar.snapshot()
    if len(committed) != 1 or relay.scheduler._committed != {0: identity}:
        raise ValueError("cursor0 fast-state/frontier 必须恰好提交一次")
    if any(
        value.dtype != torch.float32 or value.grad_fn is not None or not torch.isfinite(value).all()
        for value in committed[0][2]
    ):
        raise ValueError("提交的 fast state 必须是有限、脱图的 fp32")
    return {
        "dcp": key_report,
        "valid_count": outcome.valid_count,
        "mean_loss": float(outcome.mean_loss),
        "losses": [float(item["loss"]) for item in outcome.outputs],
        "optimizer_steps": step_calls,
        "local_gradient_witness": witnesses,
        "host_unchanged": True,
        "fast_state_committed": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    state: dict[str, Any] = {"phase": "arguments", "consumer": None}
    trace: list[dict[str, Any]] = []
    paths: SmokePaths | None = None
    result: dict[str, Any] = {
        "status": "FAIL",
        "preflight": args.preflight,
        "task": TASK,
        "episode_index": 0,
        "cursor": 0,
    }
    try:
        paths = paths_from_args(args)
        if paths.output.exists():
            raise FileExistsError(f"输出目录必须唯一且尚不存在：{paths.output}")
        paths.output.mkdir(parents=True, exist_ok=False)
        validate_frozen_paths(paths)
        result["paths"] = {
            key: str(getattr(paths, key)) for key in ("checkpoint", "config", "cache", "dataset_root", "edge", "vae")
        }
        result["authority"] = {"b2b_root": B2B_ROOT, "b2b_child": B2B_CHILD}
        result["command"] = sys.argv if argv is None else [sys.argv[0], *argv]
        result["environment"] = {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "COSMOS_DEVICE", "WORLD_SIZE", "RANK", "LOCAL_RANK")
        }
        state["phase"] = "stage_a_contract"
        contract = read_stage_a_contract(paths)
        state["phase"] = "episode_source"
        segment, identity, episode = load_episode_segment(paths)
        result["preflight_evidence"] = {"dcp_keys": contract["dcp_keys"], **episode}
        if not args.preflight:
            result["pair_lock"] = lock_implementation_pair()
            result["runtime"] = execute_cuda(paths, trace, state, segment, identity)
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": state["phase"],
            "consumer": state["consumer"],
        }
        result["traceback"] = traceback.format_exc()
        if not args.preflight and torch.cuda.is_initialized():
            try:
                record_cuda(trace, "failure", state["consumer"])
            except Exception as memory_error:
                result["memory_trace_error"] = str(memory_error).splitlines()[0]
    finally:
        if paths is not None and paths.output.is_dir():
            (paths.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            (paths.output / "cuda_memory_trace.json").write_text(json.dumps(trace, indent=2))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(paths.output) if paths else None,
                "error": None
                if result.get("error") is None
                else {
                    **result["error"],
                    "message": result["error"]["message"].splitlines()[0],
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

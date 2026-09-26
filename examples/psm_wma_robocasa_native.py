"""PSM-WMA V3 Edge-Policy-DROID raw15 入口；组合官方训练与仿真命令。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "examples/toml/sft_config/psm_wma_robocasa_native_smoke.toml"
DATASET_NODE = "dataloader_train.dataloader.datasets.robocasa.dataset"


def check_edge_checkpoint(value: str | None) -> Path:
    if not value or not value.strip():
        raise ValueError("必须设置非空 EDGE_POLICY_CHECKPOINT，指向本地 Cosmos3-Edge-Policy-DROID")
    root = require_path(Path(value), "EDGE_POLICY_CHECKPOINT", directory=True)
    config = json.loads(require_path(root / "config.json", "Edge config.json").read_text())
    model = config.get("model", {}).get("config", {})
    expected = dict(action_gen=True, vision_gen=True, max_action_dim=64, num_embodiment_domains=32)
    for key, expected_value in expected.items():
        if model.get(key) != expected_value:
            raise ValueError(f"Edge-Policy-DROID 配置不匹配：{key}")
    if (
        model.get("tokenizer", {}).get("encode_exact_durations") != [33]
        or model.get("vlm_config", {}).get("model_name") != "nvidia/Cosmos3-Edge-Policy-DROID"
        or config.get("text_config", {}).get("hidden_size") != 2048
    ):
        raise ValueError("必须使用 Cosmos3-Edge-Policy-DROID 的配置与 tokenizer")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
    ):
        require_path(root / name, f"本地 Edge tokenizer {name}")
    return root


def check_droid_dcp(base: Path) -> None:
    # 仅读 CPU 元数据，不加载模型张量；形状检查可在启动前拒绝 Nano/无动作头基座。
    from torch.distributed.checkpoint import FileSystemReader

    require_path(base / "model/.metadata", "基座 DCP model/.metadata")
    policy = json.loads(require_path(base / "checkpoint.json", "DROID policy 元信息").read_text()).get("policy", {})
    if policy != dict(action_chunk_size=32, conditioning_fps=15.0, domain_name="droid_lerobot"):
        raise ValueError("基座 DCP 必须来自 Edge-Policy-DROID（32 chunk / 15fps / droid_lerobot）")
    metadata = FileSystemReader(base / "model").read_metadata().state_dict_metadata
    expected = {
        "net.action2llm.fc.weight": (32, 131072),
        "net.action2llm.bias.weight": (32, 2048),
        "net.llm2action.fc.weight": (32, 131072),
        "net.llm2action.bias.weight": (32, 64),
        "net.action_modality_embed": (2048,),
        "net.vae2llm.weight": (2048, 192),
        "net.llm2vae.weight": (192, 2048),
    }
    for key, shape in expected.items():
        if tuple(getattr(metadata.get(key), "size", ())) != shape:
            raise ValueError(f"Edge DCP 缺少或形状不符：{key}，应为 {shape}")


def require_path(value: Path | None, label: str, *, directory: bool = False) -> Path:
    if value is None:
        raise ValueError(f"缺少 {label}")
    path = value.expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} 不存在或类型错误：{path}")
    return path


def check_dataset(root: Path | None, task: str) -> Path:
    root = require_path(root, "--dataset-root / ROBOCASA_ROOT", directory=True)
    exports = sorted((root / task).glob("*/lerobot"))
    if len(exports) != 1:
        raise ValueError(f"{root / task} 必须恰有一个 <date>/lerobot，实际 {len(exports)} 个")
    meta = exports[0] / "meta"
    info = json.loads(require_path(meta / "info.json", "数据元信息").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"需要 LeRobot v3.0：{meta} 的版本为 {info.get('codebase_version')!r}")
    require_path(meta / "tasks.parquet", "v3.0 tasks.parquet")
    if not any((meta / "episodes").glob("*/*.parquet")):
        raise ValueError(f"缺少 v3.0 episode 元信息：{meta / 'episodes'}")
    return root


def check_raw15_config(config: Path) -> None:
    # 复用框架已有的 PyYAML；不导入训练模块或实例化模型。
    import yaml

    try:
        with config.open() as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"无法解析训练 YAML：{config}") from exc
    try:
        dataset = data["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]
        durations = data["model"]["config"]["tokenizer"]["encode_exact_durations"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"缺少官方 RoboCasa 训练配置节点：{config}") from exc
    if not isinstance(dataset, dict):
        raise ValueError(f"RoboCasa dataset 节点必须是映射：{config}")
    expected = dict(
        use_base_action=True,
        base_encoding="raw",
        camera_set="left_wrist",
        use_state=True,
        fps=20,
        chunk_length=32,
        action_normalization=None,
    )
    for key, value in expected.items():
        if key not in dataset or dataset[key] != value:
            raise ValueError(f"raw15 配置不匹配：{key} 应为 {value!r}，实际 {dataset.get(key)!r}")
    if durations != [33]:
        raise ValueError(f"raw15 VAE encode_exact_durations 应为 [33]，实际 {durations!r}")
    model = data["model"]["config"]
    for key, value in dict(action_gen=True, vision_gen=True, max_action_dim=64, num_embodiment_domains=32).items():
        if model.get(key) != value:
            raise ValueError(f"Edge 模型配置不匹配：{key}")
    tokenizer = model.get("vlm_config", {}).get("tokenizer", {})
    local = check_edge_checkpoint(os.environ.get("EDGE_POLICY_CHECKPOINT"))
    if tokenizer.get("repository") is not None or tokenizer.get("tokenizer_type") != str(local):
        raise ValueError("训练配置 tokenizer 必须匹配本地 EDGE_POLICY_CHECKPOINT，禁止联网 repo")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("config", "train", "server", "export", "eval"))
    p.add_argument("--print-command", action="store_true", help="只检查输入并打印命令，不加载模型")
    p.add_argument("--output-root", type=Path, default=Path("outputs/psm_wma_v3_edge"))
    p.add_argument("--dataset-root", type=Path, default=os.environ.get("ROBOCASA_ROOT"))
    p.add_argument("--base-checkpoint", type=Path, default=os.environ.get("BASE_CHECKPOINT_PATH"))
    p.add_argument("--vae", type=Path, default=os.environ.get("WAN_VAE_PATH"))
    p.add_argument("--task", default="CloseFridge")
    p.add_argument("--steps", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--checkpoint", type=Path, help="默认读取本次 smoke 的最后一步 DCP")
    p.add_argument("--config", type=Path, help="默认使用本次训练生成的 config.yaml")
    p.add_argument(
        "--eval-dataset",
        type=Path,
        default=os.environ.get("ROBOCASA_EVAL_DATASET"),
        help="原始 v2.1 单任务 lerobot 目录，须包含 extras/dataset_meta.json",
    )
    p.add_argument("--sim-python", default=os.environ.get("SIM_PYTHON", sys.executable))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-url", help="默认 http://127.0.0.1:<port>")
    p.add_argument("--seed", type=int, default=42)
    return p


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.task):
        raise ValueError("--task 必须是单个任务名，不接受路径或 Hydra 表达式")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port 必须在 1..65535")
    output = args.output_root.expanduser().resolve()
    job = output / "psm_wma_v3/edge_robocasa/smoke"
    env = {"PYTHONPATH": str(REPO)}
    python = sys.executable

    if args.stage in ("train", "config"):
        root = check_dataset(args.dataset_root, args.task)
        base = require_path(args.base_checkpoint, "--base-checkpoint / BASE_CHECKPOINT_PATH", directory=True)
        edge = check_edge_checkpoint(os.environ.get("EDGE_POLICY_CHECKPOINT"))
        check_droid_dcp(base)
        vae = require_path(args.vae, "--vae / WAN_VAE_PATH")
        if (job / "checkpoints").exists():
            raise ValueError(f"输出中已有 checkpoint；请使用新的 --output-root，避免 smoke 自动续训：{job}")
        env.update(
            ROBOCASA_ROOT=str(root),
            BASE_CHECKPOINT_PATH=str(base),
            WAN_VAE_PATH=str(vae),
            EDGE_POLICY_CHECKPOINT=str(edge),
            IMAGINAIRE_OUTPUT_ROOT=str(output),
        )
        command = [python]
        if args.stage == "train":
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1"]
        command += ["-m", "cosmos_framework.scripts.train", f"--sft-toml={RECIPE}"]
        if args.stage == "config":
            command += ["--dryrun"]
            env.update(CUDA_VISIBLE_DEVICES="", COSMOS_DEVICE="cpu")
        command += [
            "--",
            f"trainer.max_iter={args.steps}",
            f"trainer.seed={args.seed}",
            f"{DATASET_NODE}.task_names=[{args.task}]",
            "dataloader_train.dataloader.num_workers=1",
            "dataloader_train.dataloader.prefetch_factor=1",
            "trainer.callbacks.training_stats.log_freq=1",
        ]
        return command, env

    if args.stage in ("server", "export"):
        config = require_path(args.config or job / "config.yaml", "训练 config.yaml")
        check_raw15_config(config)
        checkpoint = require_path(
            args.checkpoint or job / f"checkpoints/iter_{args.steps:09d}", "训练 DCP", directory=True
        )
        # net/net_ema 共存在 model 分片，不能误要求单独的 model_ema 目录。
        require_path(checkpoint / "model/.metadata", "DCP model/.metadata")
        require_path(checkpoint / "trainer/.metadata", "DCP trainer/.metadata")
        command = [
            python,
            "-m",
            f"cosmos_framework.scripts.{'export_model' if args.stage == 'export' else 'action_policy_server_robocasa'}",
            "--checkpoint-path",
            str(checkpoint),
            "--config-file",
            str(config),
        ]
        if args.stage == "export":
            command += ["--output-dir", str(output / "export")]
        else:
            command += [
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--output-dir",
                str(output / "server"),
                "--dump-dir",
                str(output / "server/dumps"),
                "--raw-action-dim",
                "15",
                "--action-chunk-size",
                "32",
                "--fps",
                "20",
                "--seed",
                str(args.seed),
                "--num-steps",
                "30",
                "--http-400-on-error",
                "--no-guardrails",
            ]
            # 官方 CLI 没有 normalization=none；不传 stats 即保持原动作值。
        return command, env

    dataset = require_path(args.eval_dataset, "--eval-dataset / ROBOCASA_EVAL_DATASET", directory=True)
    metadata = require_path(dataset / "extras/dataset_meta.json", "官方仿真环境元信息")
    env_args = json.loads(metadata.read_text()).get("env_args", {})
    if env_args.get("env_name") != args.task:
        raise ValueError(f"评测任务不匹配：请求 {args.task}，元信息为 {env_args.get('env_name')!r}")
    env["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    command = [
        args.sim_python,
        str(REPO / "cosmos_framework/simulation/robocasa/closed_loop_eval.py"),
        "--dataset-dir",
        str(dataset),
        "--server-url",
        args.server_url or f"http://127.0.0.1:{args.port}",
        "--output-dir",
        str(output / "eval" / args.task),
        "--num-test-episodes",
        "1",
        "--action-horizon",
        "32",
        "--camera-set",
        "left_wrist",
        "--use-state",
        "--use-base-action",
        "--base-encoding",
        "raw",
        "--success-latch",
        "1",
        "--image-size",
        "256",
        "--cam-size",
        "256",
        "--seed",
        str(args.seed),
    ]
    return command, env


def main() -> int:
    p = parser()
    args = p.parse_args()
    try:
        command, overrides = build_command(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {"stage": args.stage, "cwd": str(REPO), "env": overrides, "command": command}, ensure_ascii=False, indent=2
        ),
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0
    return subprocess.run(command, cwd=REPO, env={**os.environ, **overrides}, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only exact-window cache versus current full Wan VAE parity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    ExactWindowEpisodeRecord,
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
    CorrectedRoboCasaPolicyContract,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import (
    RoboCasaExactWindowSourceReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    _IMAGE_FEATURES,
    RoboCasaLeRobotDataset,
)
from cosmos_framework.data.generator.action.utils.transforms import VideoResize
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item


@dataclass(frozen=True)
class ProbeDependencies:
    decode: Callable[..., torch.Tensor] = decode_video_frames
    tokenizer_factory: Callable[..., Any] = Wan2pt2VAEInterface
    normalize: Callable[..., torch.Tensor] = normalize_uint8_item
    resize_factory: Callable[..., Any] = VideoResize
    compose: Callable[..., torch.Tensor] = RoboCasaLeRobotDataset._compose_left_wrist
    convert: Callable[..., torch.Tensor] = RoboCasaLeRobotDataset._convert_video
    crop: Callable[..., list[torch.Tensor]] = OmniMoTModel._remove_padding_from_latent
    allow_cpu_for_test: bool = False


def select_starts(record: ExactWindowEpisodeRecord, explicit: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if record.window_count < 3:
        raise ValueError(f"episode 至少需要 3 个 exact windows：{record.key}")
    starts = explicit if explicit is not None else (0, (record.window_count - 1) // 2, record.window_count - 1)
    if not starts or any(type(start) is not int or start not in record.window_starts for start in starts):
        raise ValueError(f"start 不在 exact cache：{record.key}/{starts}")
    return tuple(dict.fromkeys(starts))


def select_episodes(
    catalog: RoboCasaExactWindowCacheCatalog,
    task_classes: tuple[str, ...],
    episode_index: int | None,
    starts: tuple[int, ...] | None,
    num_task_classes: int,
) -> tuple[tuple[ExactWindowEpisodeRecord, tuple[int, ...]], ...]:
    if num_task_classes < 1 or (episode_index is not None and len(task_classes) != 1):
        raise ValueError("num_task_classes 必须为正；episode_index 必须配合单个 task_class")
    eligible = [record for record in catalog.episodes if record.window_count >= 3]
    if task_classes:
        unknown = set(task_classes) - {record.key.task_class for record in eligible}
        if unknown:
            raise ValueError(f"task_class 无 eligible episode：{sorted(unknown)}")
        chosen = []
        for task in dict.fromkeys(task_classes):
            matches = [
                record
                for record in eligible
                if record.key.task_class == task
                and (episode_index is None or record.key.episode_index == episode_index)
            ]
            if not matches:
                raise ValueError(f"task/episode 不在 exact cache：{task}/{episode_index}")
            chosen.append(matches[0])
    else:
        if episode_index is not None or starts is not None:
            raise ValueError("--episode-index/--starts 需要显式 --task-class")
        chosen = []
        for record in eligible:
            if record.key.task_class not in {item.key.task_class for item in chosen}:
                chosen.append(record)
            if len(chosen) == num_task_classes:
                break
        if len(chosen) != num_task_classes:
            raise ValueError("eligible task class 数量不足")
    return tuple((record, select_starts(record, starts)) for record in chosen)


def validate_final_coverage(selection: tuple[tuple[ExactWindowEpisodeRecord, tuple[int, ...]], ...]) -> None:
    if len({record.key.task_class for record, _ in selection}) < 3 or sum(len(starts) for _, starts in selection) < 9:
        raise ValueError("thresholded Gate 要求至少 3 task classes / 9 exact windows")


def resolve_vae_config(contract: CorrectedRoboCasaPolicyContract, vae_path: Path) -> dict[str, Any]:
    candidate = deepcopy(EDGE_MODEL_CONFIG["tokenizer"])
    resolved = contract.resolve_tokenizer_config(candidate)
    resolved["bucket_name"] = ""
    resolved["vae_path"] = str(vae_path)
    if resolved.get("use_streaming_encode") is not False:
        raise ValueError("full encode 必须禁用 streaming")
    contract.validate_tokenizer_config(resolved)
    return resolved


def _local_video_path(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"video 路径必须在 source-root 内：{relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"video 文件缺失或越界：{path}")
    return path


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_rows(
    source: RoboCasaExactWindowSourceReader, record: ExactWindowEpisodeRecord
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    key = record.key
    bound = source._bound[key]
    path = source.source_root / bound.data_file
    table = pq.read_table(
        path, columns=["index", "timestamp", "episode_index"], filters=[("episode_index", "=", key.episode_index)]
    )
    order = sorted(range(table.num_rows), key=lambda i: table["index"][i].as_py())
    indices = tuple(table["index"][i].as_py() for i in order)
    episodes = tuple(table["episode_index"][i].as_py() for i in order)
    timestamps = tuple(table["timestamp"][i].as_py() for i in order)
    if (
        len(indices) != bound.length
        or indices != tuple(range(bound.dataset_from_index, bound.dataset_to_index))
        or episodes != (key.episode_index,) * bound.length
        or any(not isinstance(ts, (int, float)) or not math.isfinite(ts) for ts in timestamps)
    ):
        raise ValueError(f"source 整 episode row/timestamp 与 Phase1B binding 不一致：{key}")
    return indices, tuple(float(ts) for ts in timestamps)


def verify_witnesses(
    reader: RoboCasaExactWindowEpisodeReader,
    key: ExactWindowEpisodeKey,
    starts: tuple[int, ...],
    rows: tuple[int, ...],
) -> list[dict[str, Any]]:
    result = []
    for start in starts:
        identity = reader.read_identity(key, start)
        expected_frames = tuple(range(start, start + 17))
        if (
            identity.key != key
            or identity.start_frame != start
            or identity.global_row_indices != rows[start : start + 17]
            or identity.window_frame_indices != expected_frames
            or identity.latent_source_frame_indices != expected_frames[::4]
        ):
            raise ValueError(f"exact window witness 不匹配：{key}/{start}")
        result.append(
            {
                "start_frame": start,
                "global_row_indices": list(identity.global_row_indices),
                "window_frame_indices": list(identity.window_frame_indices),
                "latent_source_frame_indices": list(identity.latent_source_frame_indices),
            }
        )
    return result


def video_paths(source: RoboCasaExactWindowSourceReader, key: ExactWindowEpisodeKey) -> dict[str, tuple[Path, float]]:
    meta = source.meta
    if meta.episodes is None:
        raise ValueError("source 缺少 episode metadata")
    result = {}
    for camera in ("left", "wrist"):
        feature = _IMAGE_FEATURES[camera]
        field = f"videos/{feature}/from_timestamp"
        if field not in meta.episodes.column_names:
            raise ValueError(f"episode metadata 缺少 {field}")
        from_ts = float(meta.episodes[field][key.episode_index])
        if not math.isfinite(from_ts):
            raise ValueError(f"video from_timestamp 非有限：{field}")
        result[feature] = (
            _local_video_path(source.source_root, meta.get_video_file_path(key.episode_index, feature)),
            from_ts,
        )
    return result


def _validate_frames(frames: torch.Tensor, count: int, feature: str) -> None:
    if (
        not isinstance(frames, torch.Tensor)
        or tuple(frames.shape) != (count, 3, 256, 256)
        or not frames.is_floating_point()
        or not bool(torch.isfinite(frames).all())
        or bool((frames < 0).any())
        or bool((frames > 1).any())
    ):
        raise ValueError(f"decoded camera shape/dtype/range 无效：{feature}")


def _geometry(
    shape: tuple[int, ...], image_size: torch.Tensor, latent_shape: tuple[int, ...], factor: int, count: int
) -> list[int]:
    if len(shape) != 4 or shape[:2] != (3, count):
        raise ValueError(f"resize 后 C/T 不匹配：{shape}")
    size = [int(x) for x in image_size.flatten().tolist()]
    h, w = shape[-2:]
    if (
        len(size) != 4
        or size[:2] != [h, w]
        or any(value <= 0 for value in size)
        or h % factor
        or w % factor
        or latent_shape != (5, 48, h // factor, w // factor)
    ):
        raise ValueError(f"canvas/latent geometry 不匹配：{shape}/{size}/{latent_shape}")
    return size


def reconstruct_episode(
    source: RoboCasaExactWindowSourceReader,
    key: ExactWindowEpisodeKey,
    timestamps: tuple[float, ...],
    paths: dict[str, tuple[Path, float]],
    tolerance_s: float,
    video_backend: str | None,
    deps: ProbeDependencies,
) -> tuple[torch.Tensor, dict[str, Any]]:
    proxy = object.__new__(RoboCasaLeRobotDataset)
    proxy._image_features = _IMAGE_FEATURES
    proxy._skip_video_loading = False
    sample = {}
    for feature, (path, from_ts) in paths.items():
        kwargs = {"backend": video_backend} if video_backend is not None else {}
        frames = deps.decode(path, [from_ts + ts for ts in timestamps], tolerance_s, **kwargs)
        _validate_frames(frames, len(timestamps), feature)
        sample[feature] = frames
    composite = deps.compose(proxy, sample)
    if tuple(composite.shape) != (len(timestamps), 3, 256, 512) or not bool(torch.isfinite(composite).all()):
        raise ValueError(f"official composite shape/finite 无效：{key}")
    uint8 = deps.convert(proxy, composite)
    if (
        not isinstance(uint8, torch.Tensor)
        or uint8.dtype != torch.uint8
        or tuple(uint8.shape) != (3, len(timestamps), 256, 512)
    ):
        raise ValueError(f"official uint8 conversion 无效：{key}")
    transformed = deps.resize_factory(pad_keys=["video"], keep_aspect_ratio=True)({"video": uint8}, resolution=None)
    video = transformed["video"]
    if video.dtype != torch.uint8:
        raise ValueError("VideoResize 未保持 uint8")
    return video, {
        "composite_shape": list(composite.shape),
        "resized_shape": list(video.shape),
        "image_size": transformed["image_size"],
    }


def _fp32(value: torch.Tensor, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != shape
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} 必须是 finite fp32 {shape}")
    return value


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if (
        left.shape != right.shape
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
    ):
        raise ValueError("parity metrics 输入 shape/dtype/finite 不一致")
    diff = (left - right).double().abs()
    return {
        "exact_equal": bool(torch.equal(left, right)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(diff.square().mean().sqrt()),
        "finite": True,
        "numel": diff.numel(),
    }


def compare_window(
    cache: torch.Tensor, online: torch.Tensor, image_size: list[int], factor: int, deps: ProbeDependencies
) -> dict[str, Any]:
    shape = tuple(cache.shape)
    _fp32(cache, shape, "cache")
    _fp32(online, shape, "online")
    if len(shape) != 4 or shape[:2] != (5, 48):
        raise ValueError("padded latent shape 无效")
    pre = metrics(cache, online)
    temporal = [metrics(cache[t], online[t]) for t in range(5)]
    proxy = SimpleNamespace(tokenizer_vision_gen=SimpleNamespace(spatial_compression_factor=factor))
    frame_size = [torch.tensor(image_size, dtype=torch.long)]
    cache_model = cache.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    online_model = online.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    cropped_cache = deps.crop(proxy, [cache_model], frame_size)[0]
    cropped_online = deps.crop(proxy, [online_model], frame_size)[0]
    if cropped_cache.ndim != 5 or cropped_cache.shape[:3] != (1, 48, 5):
        raise ValueError("native crop layout 无效")
    post = metrics(cropped_cache, cropped_online)
    z0 = metrics(cropped_cache[0, :, 0], cropped_online[0, :, 0])
    return {
        "pre_crop": pre,
        "temporal": temporal,
        "post_crop": post,
        "z0": z0,
        "cropped_shape": list(cropped_cache.shape),
    }


def encode_window(
    video: torch.Tensor, tokenizer: Any, device: torch.device, shape: tuple[int, ...], deps: ProbeDependencies
) -> torch.Tensor:
    if video.dtype != torch.uint8 or video.shape != (3, 17, shape[2] * 16, shape[3] * 16):
        raise ValueError("online encode 输入 window shape/dtype 无效")
    normalized = deps.normalize(video.unsqueeze(0), {"device": device, "dtype": torch.float32})
    model = tokenizer.encode(normalized).contiguous().float()
    _fp32(model, (1, 48, 5, shape[2], shape[3]), "online model latent")
    return model.squeeze(0).permute(1, 0, 2, 3).contiguous().cpu()


def prepare_tokenizer_device(tokenizer: Any, device: torch.device) -> None:
    if tokenizer.use_streaming_encode or tokenizer._keep_encoder_cache:
        raise ValueError("Wan tokenizer 必须使用 normal full encode")
    wan = tokenizer.model
    scale_mean, scale_inv_std = wan.scale
    wan.model.to(device)
    wan.scale = (scale_mean.to(device), scale_inv_std.to(device))
    wan.model.eval()


def gate_result(windows: list[dict[str, Any]], threshold: float | None, dry_run: bool) -> tuple[str, bool | None]:
    if dry_run:
        return "DRY_RUN_PASS", None
    if threshold is None:
        return "OBSERVATIONAL_NO_THRESHOLD", None
    passed = all(
        all(window["metrics"][part]["max_abs"] <= threshold for part in ("pre_crop", "post_crop", "z0"))
        for window in windows
    )
    return ("PASS", True) if passed else ("FAIL", False)


def run(
    args: argparse.Namespace,
    *,
    deps: ProbeDependencies | None = None,
    catalog_factory: Callable[..., Any] = RoboCasaExactWindowCacheCatalog,
    source_factory: Callable[..., Any] = RoboCasaExactWindowSourceReader,
    reader_factory: Callable[..., Any] = RoboCasaExactWindowEpisodeReader,
) -> dict[str, Any]:
    deps = deps or ProbeDependencies()
    if not math.isfinite(args.tolerance_s) or args.tolerance_s <= 0:
        raise ValueError("tolerance_s 必须是有限正数")
    if args.max_abs_threshold is not None and (not math.isfinite(args.max_abs_threshold) or args.max_abs_threshold < 0):
        raise ValueError("max_abs_threshold 必须是有限非负数")
    if not args.vae_path.is_file():
        raise FileNotFoundError(f"本地 Wan VAE 文件缺失：{args.vae_path}")
    catalog = catalog_factory(args.cache_root)
    reader = reader_factory(catalog)
    source = source_factory(catalog, args.source_root)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog)
    config = resolve_vae_config(contract, args.vae_path)
    selection = select_episodes(catalog, tuple(args.task_class), args.episode_index, args.starts, args.num_task_classes)
    if args.max_abs_threshold is not None:
        validate_final_coverage(selection)
    torch.backends.cudnn.benchmark = False
    report: dict[str, Any] = {
        "schema": "robocasa_exact_window_real_vae_parity_v1",
        "authority": {
            "cache_manifest_sha256": catalog.manifest_sha256,
            "corpus_digest": catalog.corpus_digest,
            "source_binding_digest": source.source_binding_digest,
            "runtime_source_root": str(args.source_root),
            "runtime_vae_path": str(args.vae_path),
            "vae_sha256": _sha256_file(args.vae_path) if args.hash_vae else None,
            "tokenizer_class": "Wan2pt2VAEInterface",
            "resolved_vae_contract": config,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": args.device,
            "video_backend": args.video_backend,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "no_fallback": True,
            "read_only": True,
        },
        "selection": [],
        "windows": [],
        "aggregate": {},
        "gate": {"dry_run": args.dry_run, "threshold": args.max_abs_threshold},
    }
    prepared = []
    factor = int(config["spatial_compression_factor"])
    for record, starts in selection:
        key = record.key
        rows, timestamps = episode_rows(source, record)
        witnesses = verify_witnesses(reader, key, starts, rows)
        paths = video_paths(source, key)
        report["selection"].append(
            {
                "task_class": key.task_class,
                "episode_index": key.episode_index,
                "starts": list(starts),
                "witnesses": witnesses,
            }
        )
        prepared.append((record, starts, timestamps, paths))
    if args.dry_run:
        static = VideoResize(pad_keys=["video"], keep_aspect_ratio=True)(
            {"video": torch.zeros((3, 1, 256, 512), dtype=torch.uint8)}, resolution=None
        )
        report["expected_geometry"] = {
            "resized_shape": list(static["video"].shape),
            "image_size": _geometry(
                tuple(static["video"].shape), static["image_size"], catalog.latent_shape, factor, 1
            ),
            "cached_padded_latent_shape": list(catalog.latent_shape),
        }
    else:
        device = torch.device(args.device)
        if not deps.allow_cpu_for_test and (device.type != "cuda" or not torch.cuda.is_available()):
            raise ValueError("真实 encode 需要可用 CUDA；dry-run 不需要")
        tokenizer = deps.tokenizer_factory(**config)
        prepare_tokenizer_device(tokenizer, device)
        with torch.inference_mode():
            for record, starts, timestamps, paths in prepared:
                video, geometry = reconstruct_episode(
                    source, record.key, timestamps, paths, args.tolerance_s, args.video_backend, deps
                )
                geometry["image_size"] = _geometry(
                    tuple(video.shape), geometry["image_size"], catalog.latent_shape, factor, len(timestamps)
                )
                for start in starts:
                    cache = reader.read_window(record.key, start)
                    online = encode_window(video[:, start : start + 17], tokenizer, device, catalog.latent_shape, deps)
                    result = compare_window(cache, online, geometry["image_size"], factor, deps)
                    report["windows"].append(
                        {
                            "task_class": record.key.task_class,
                            "episode_index": record.key.episode_index,
                            "start_frame": start,
                            "geometry": {
                                **geometry,
                                "camera_keys": [_IMAGE_FEATURES["left"], _IMAGE_FEATURES["wrist"]],
                                "episode_frame_count": len(timestamps),
                                "cached_padded_latent_shape": list(catalog.latent_shape),
                                "cropped_shape": result.pop("cropped_shape"),
                            },
                            "metrics": result,
                        }
                    )
        windows = report["windows"]
        if windows:
            worst = max(windows, key=lambda item: item["metrics"]["pre_crop"]["max_abs"])
            temporal_worst = max((item["metrics"]["temporal"][t]["max_abs"], t) for item in windows for t in range(5))
            total = sum(item["metrics"]["pre_crop"]["numel"] for item in windows)
            report["aggregate"] = {
                "global_max_abs": worst["metrics"]["pre_crop"]["max_abs"],
                "global_mean_abs": sum(
                    item["metrics"]["pre_crop"]["mean_abs"] * item["metrics"]["pre_crop"]["numel"] for item in windows
                )
                / total,
                "worst_window": {key: worst[key] for key in ("task_class", "episode_index", "start_frame")},
                "worst_temporal_index": temporal_worst[1],
            }
    status, passed = gate_result(report["windows"], args.max_abs_threshold, args.dry_run)
    report["gate"].update(status=status, parity_gate_pass=passed)
    return report


def _parse_starts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--starts 需要逗号分隔整数") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--vae-path", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-class", action="append", default=[])
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--starts", type=_parse_starts)
    parser.add_argument("--num-task-classes", type=int, default=1)
    parser.add_argument("--video-backend")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-abs-threshold", type=float)
    parser.add_argument("--hash-vae", action="store_true")
    args = parser.parse_args()
    report = run(args)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{report['gate']['status']}: {args.output_json}")
    return 1 if report["gate"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

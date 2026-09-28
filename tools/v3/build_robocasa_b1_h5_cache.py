# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: OpenMDW-1.1

"""Build B1 frozen episode-level H5 latent caches from a RoboCasa v3.0 source.
Output contract (authority: `RoboCasaLatentReader`):
  ///lerobot/ep_XXXXXX.h5

- root attrs: `episode_id`, `frame_count`, `temporal_compression_factor=4`,
`source_frame_to_latent_policy="causal_endpoint"`
- two cameras: `observation.images.robot0_agentview_left` +
`observation.images.robot0_eye_in_hand`
- per camera:
  `latents/{cam}`: fp16 [N, 48, 16, 16]
  `indices/latent_source_frame_indices/{cam}`: integer [N] == 0,4,8,... (+ terminal F-1)
  `valid/{cam}`: bool [N], all true
Episode enumeration is the production train-split authority: it reuses
`RoboCasaLeRobotDataset` (frozen Stage-A loader contract) and covers its 9036
train records. No new manifest authority is produced; raw15 stays in the V3
source loader and is never read back from the H5 cache.
Modes:
  static -- deterministic CPU-only latents (schema/round-trip validation)
  build  -- real Wan2.2 VAE encode (GPU path; executed in a later Gate)
Atomic publication: write to `*.tmp` -> validate with `RoboCasaLatentReader`
-> `os.replace`. Resume only skips an existing H5 that validates; invalid or
partial caches are deleted and rebuilt, never trusted as complete.
Multi-process / multi-GPU sharding is prepared via `--workers/--worker`
(record-ordinal modulo split) but is not executed by this Gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

LEFT_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
CAMERAS = (LEFT_CAMERA, WRIST_CAMERA)
TEMPORAL_COMPRESSION_FACTOR = 4
SOURCE_FRAME_TO_LATENT_POLICY = "causal_endpoint"

# Frozen Stage-A loader contract; identical to the Route-B Stage-2 acceptance.

FROZEN_LOADER_KWARGS = {
    "fps": 20,
    "chunk_length": 32,
    "split_seed": 42,
    "split_val_ratio": 0.01,
    "split": "train",
    "mode": "wam",
    "use_state": True,
    "use_base_action": True,
    "base_encoding": "raw",
    "camera_set": "left_wrist",
    "action_normalization": None,
}


@dataclass(frozen=True)


class EpisodeSpec:
    shard: Path
    task: str
    date: str
    episode_index: int
    frame_count: int
    source_row_start: int
    valid_count: int
    @property
    def episode_id(self) -> str:
        return f"ep_{self.episode_index:06d}"
    @property
    def full_id(self) -> str:
        """Globally-unique identity: episode_index is shard-local, task/date disambiguate."""
        return f"{self.task}/{self.date}/ep_{self.episode_index:06d}"


def endpoint_vector(frame_count: int) -> list[int]:
    """0,4,8,... plus a terminal `frame_count - 1` when the grid misses it."""
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError(f"frame_count must be a positive int, got {frame_count!r}")
    endpoints = list(range(0, frame_count, TEMPORAL_COMPRESSION_FACTOR))
    if endpoints[-1] != frame_count - 1:
        endpoints.append(frame_count - 1)
    return endpoints


def episode_output_path(output_root: Path, task: str, date: str, episode_index: int) -> Path:
    """Exact relative layout: ///lerobot/ep_XXXXXX.h5."""
    if not task or not date:
        raise ValueError("task and date must be non-empty")
    return output_root / task / date / "lerobot" / f"ep_{episode_index:06d}.h5"


def load_shard_episode_meta(shard: Path) -> dict[int, dict]:
    """Read v3.0 per-episode metadata (length + per-camera video chunk/file index)."""
    tables = []
    for parquet in sorted(shard.glob("meta/episodes/chunk-*/file-*.parquet")):
        tables.append(pq.read_table(parquet))
    if not tables:
        raise FileNotFoundError(f"no episodes metadata under {shard}")
    meta: dict[int, dict] = {}
    for table in tables:
        data = table.to_pydict()
        for i in range(table.num_rows):
            ep = int(data["episode_index"][i])
            length = int(data["length"][i])
            from_index = int(data["dataset_from_index"][i])
            to_index = int(data["dataset_to_index"][i])
            if length != to_index - from_index:
                raise ValueError(f"episode {ep} length != dataset span ({shard})")
            entry: dict = {"length": length, "from_index": from_index, "video": {}}
            for cam in CAMERAS:
                prefix = f"videos/{cam}"
                if f"{prefix}/chunk_index" not in data or f"{prefix}/file_index" not in data:
                    raise ValueError(f"episode {ep} missing camera columns {prefix} ({shard})")
                entry["video"][cam] = {
                    "chunk_index": int(data[f"{prefix}/chunk_index"][i]),
                    "file_index": int(data[f"{prefix}/file_index"][i]),
                }
            if ep in meta:
                raise ValueError(f"duplicate episode_index {ep} ({shard})")
            meta[ep] = entry
    return meta


def enumerate_train_episodes(
    source_root: Path,
    task_names: list[str] | tuple[str, ...] | None = None,
) -> list[EpisodeSpec]:
    """Deterministic production train-split enumeration (authority)."""
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
        DEFAULT_ALL_ATOMIC_TASKS,
        RoboCasaLeRobotDataset,
    )

    if task_names is None:
        task_names = DEFAULT_ALL_ATOMIC_TASKS
    ds = RoboCasaLeRobotDataset(
        root=str(source_root),
        task_names=list(task_names),
        enable_fast_init=False,
        **FROZEN_LOADER_KWARGS,
    )
    shards = [Path(path).resolve() for path in ds._all_shard_roots]
    meta_cache: dict[Path, dict[int, dict]] = {}
    specs: list[EpisodeSpec] = []
    for source_index, row_start, valid_count, episode_index in ds._episode_records:
        shard = shards[source_index]
        relative = shard.relative_to(source_root.resolve())
        if len(relative.parts) != 3 or relative.parts[-1] != "lerobot":
            raise ValueError(f"shard must be //lerobot: {relative}")
        task, date, _ = relative.parts
        if task not in task_names:
            raise ValueError(f"unexpected task {task} outside task_names")
        if shard not in meta_cache:
            meta_cache[shard] = load_shard_episode_meta(shard)
        entry = meta_cache[shard].get(episode_index)
        if entry is None:
            raise ValueError(f"episode ep_{episode_index:06d} missing from shard metadata ({shard})")
        frame_count = entry["length"]
        if valid_count != frame_count - FROZEN_LOADER_KWARGS["chunk_length"]:
            raise ValueError(f"ep_{episode_index:06d} valid_count {valid_count} != frame_count-32 {frame_count - 32}")
        specs.append(
            EpisodeSpec(
                shard=shard,
                task=task,
                date=date,
                episode_index=episode_index,
                frame_count=frame_count,
                source_row_start=row_start,
                valid_count=valid_count,
            )
        )
    return specs


def decode_episode_frames(mp4_path: Path, start_frame: int, num_frames: int, fps: int) -> torch.Tensor:
    from lerobot.datasets.video_utils import decode_video_frames

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    timestamps = [(start_frame + k) / fps for k in range(num_frames)]
    frames = decode_video_frames(str(mp4_path), timestamps, tolerance_s=1e-4)
    if frames.ndim != 4 or frames.shape[0] != num_frames:
        raise ValueError(f"decoded frame count mismatch: expected {num_frames}, got {tuple(frames.shape)}")
    return (frames * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)


def encode_episode(vae: object, frames_uint8: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    """Wan2.2 VAE RGB encoding contract (normalize div_(127.5).sub_(1.0), causal prefix)."""
    if frames_uint8.ndim != 4 or frames_uint8.dtype != torch.uint8:
        raise ValueError("frames_uint8 must be uint8 [F, 3, H, W]")
    frame_count = int(frames_uint8.shape[0])
    endpoints = endpoint_vector(frame_count)
    x = frames_uint8.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1,3,F,H,W]
    x = x.to(torch.float32).div_(127.5).sub_(1.0)
    if (frame_count - 1) % TEMPORAL_COMPRESSION_FACTOR != 0:
        padded = TEMPORAL_COMPRESSION_FACTOR * ((frame_count - 1) // TEMPORAL_COMPRESSION_FACTOR + 1) + 1
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - frame_count))
    latent = vae.encode(x)  # [1,48,N,16,16]
    latent = latent.squeeze(0).permute(1, 0, 2, 3).detach().cpu()  # [N,48,16,16]
    if latent.shape[0] != len(endpoints):
        raise RuntimeError(f"latent N={latent.shape[0]} != endpoints {len(endpoints)} (frame_count={frame_count})")
    if not torch.isfinite(latent).all():
        raise FloatingPointError("VAE produced non-finite latents")
    return latent.to(torch.float16), endpoints


def static_latents(episode_id: str, n: int) -> dict[str, torch.Tensor]:
    """Deterministic CPU-only latents for schema/round-trip validation."""
    seed = int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return {
        cam: torch.from_numpy(rng.standard_normal((n, 48, 16, 16), dtype=np.float32)).to(torch.float16)
        for cam in CAMERAS
    }


def write_episode_h5(
    path: Path,
    *,
    episode_id: str,
    frame_count: int,
    latents: dict[str, torch.Tensor],
    endpoints: dict[str, list[int]],
) -> None:
    """Atomic write: temp -> validate -> rename."""
    if set(latents) != set(CAMERAS) or set(endpoints) != set(CAMERAS):
        raise ValueError("both cameras must provide latents/endpoints")
    canonical = endpoint_vector(frame_count)
    for cam in CAMERAS:
        if endpoints[cam] != canonical:
            raise ValueError(f"{cam} endpoints differ from canonical 4-grid+terminal")
    if {tuple(endpoints[cam]) for cam in CAMERAS} != {tuple(canonical)}:
        raise ValueError("both cameras must share exactly the same endpoint vector")
    for cam in CAMERAS:
        z = latents[cam]
        if not isinstance(z, torch.Tensor) or z.ndim != 4 or z.shape[1:] != (48, 16, 16):
            raise ValueError(f"{cam} latents must be [N,48,16,16]")
        if z.shape[0] != len(canonical):
            raise ValueError(f"{cam} latent N={z.shape[0]} != endpoints {len(canonical)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["episode_id"] = episode_id
            handle.attrs["frame_count"] = int(frame_count)
            handle.attrs["temporal_compression_factor"] = TEMPORAL_COMPRESSION_FACTOR
            handle.attrs["source_frame_to_latent_policy"] = SOURCE_FRAME_TO_LATENT_POLICY
            for cam in CAMERAS:
                z = latents[cam].to(torch.float16).contiguous().numpy()
                idx = np.asarray(endpoints[cam], dtype=np.int64)
                valid = np.ones(len(idx), dtype=bool)
                handle.create_dataset(f"latents/{cam}", data=z, dtype="float16", chunks=True)
                handle.create_dataset(f"indices/latent_source_frame_indices/{cam}", data=idx, dtype="int64")
                handle.create_dataset(f"valid/{cam}", data=valid, dtype="bool")
        verify_episode_h5(temporary, episode_id=episode_id, frame_count=frame_count)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_episode_h5(path: Path, *, episode_id: str, frame_count: int) -> None:
    """Round-trip the canonical reader; raises on any contract violation."""
    from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader

    reader = RoboCasaLatentReader(
        str(path),
        expected_episode_id=episode_id,
        expected_source_frames=frame_count,
    )
    expected = endpoint_vector(frame_count)
    if tuple(reader.endpoint_indices) != tuple(expected):
        raise ValueError(f"reader endpoint mismatch: {path}")
    if reader._summaries.shape != (len(expected), 96):
        raise ValueError(f"reader visual96 shape mismatch: {tuple(reader._summaries.shape)}")
    if not torch.isfinite(reader._summaries).all():
        raise FloatingPointError("reader visual96 contains non-finite values")


def is_complete_episode(path: Path, *, episode_id: str, frame_count: int) -> bool:
    """True only when the existing H5 validates fully (resume gate)."""
    try:
        verify_episode_h5(path, episode_id=episode_id, frame_count=frame_count)
        return True
    except BaseException:
        return False


def build_cache(
    source_root: Path,
    output_root: Path,
    *,
    mode: str,
    vae_path: Path | None,
    device: str,
    task_names: list[str] | None,
    episode_filter: set[str],
    limit: int,
    workers: int,
    worker: int,
    vae: object | None = None,
) -> dict:
    """Build (or resume) the cache for the frozen train split.
    `vae` may be supplied by the caller (e.g. an already-loaded
    `Wan2pt2VAEInterface`) to keep a single VAE instance across episodes.
    """
    if mode not in ("static", "build"):
        raise ValueError(f"mode must be 'static' or 'build', got {mode!r}")
    specs = enumerate_train_episodes(source_root, task_names)
    if len(specs) != 9036:
        raise RuntimeError(
            f"train enumeration must be exactly 9036, got {len(specs)}; refusing to build an unexpected split"
        )
    reports: dict = {"records_total": len(specs), "built": [], "skipped": [], "failed": []}
    t0 = time.time()
    if mode == "build" and vae_path is None:
        raise SystemExit("build mode requires --vae-path")
    if mode == "build" and device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {device} but no CUDA available")
    torch_device = torch.device(device)
    if vae is None and mode == "build":
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
            Wan2pt2VAEInterface,
        )

        vae = Wan2pt2VAEInterface(
            vae_path=str(vae_path),
            encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        )
        vae.model.model.to(torch_device)
        scale_mean, scale_inv_std = vae.model.scale
        vae.model.scale = (scale_mean.to(torch_device), scale_inv_std.to(torch_device))
    meta_cache: dict[Path, dict[int, dict]] = {}
    built_count = 0
    for ordinal, spec in enumerate(specs):
        if ordinal % workers != worker:
            continue
        if episode_filter and spec.episode_id not in episode_filter:
            continue
        if limit and (len(reports["built"]) + len(reports["skipped"])) >= limit:
            break
        cache_path = episode_output_path(output_root, spec.task, spec.date, spec.episode_index)
        if cache_path.exists():
            if is_complete_episode(cache_path, episode_id=spec.episode_id, frame_count=spec.frame_count):
                reports["skipped"].append(str(cache_path))
                continue
            cache_path.unlink()
        if spec.shard not in meta_cache:
            meta_cache[spec.shard] = load_shard_episode_meta(spec.shard)
        entry = meta_cache[spec.shard][spec.episode_index]
        try:
            if mode == "static":
                endpoints = endpoint_vector(spec.frame_count)
                n = len(endpoints)
                latents = static_latents(spec.full_id, n)
                ep_endpoints = {cam: endpoints for cam in CAMERAS}
            else:
                latents = {}
                ep_endpoints = {}
                for cam in CAMERAS:
                    video = entry["video"][cam]
                    mp4 = (
                        spec.shard
                        / "videos"
                        / cam
                        / f"chunk-{video['chunk_index']:03d}"
                        / f"file-{video['file_index']:03d}.mp4"
                    )
                    if not mp4.is_file():
                        raise FileNotFoundError(f"missing video: {mp4}")
                    frames = decode_episode_frames(
                        mp4, entry["from_index"], spec.frame_count, FROZEN_LOADER_KWARGS["fps"]
                    )
                    z, ep_endpoints[cam] = encode_episode(vae, frames, torch_device)
                    latents[cam] = z
            write_episode_h5(
                cache_path,
                episode_id=spec.episode_id,
                frame_count=spec.frame_count,
                latents=latents,
                endpoints=ep_endpoints,
            )
            reports["built"].append(str(cache_path))
            built_count += 1
        except BaseException as error:  # noqa: BLE001
            reports["failed"].append({"path": str(cache_path), "error": str(error)})
            if cache_path.exists():
                cache_path.unlink()
    reports["elapsed_s"] = round(time.time() - t0, 1)
    reports["built_count"] = built_count
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default="/mnt/data1/data_v2_0617/robocasa365_official_v30")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["static", "build"], default="static")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--episode-filter", nargs="*", default=[], help="tiny smoke: ep_XXXXXX subset")
    parser.add_argument("--limit", type=int, default=0, help="0 = all train records")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("/tmp/b1_h5_cache_report.json"))
    args = parser.parse_args()
    if args.worker < 0 or args.worker >= args.workers:
        raise SystemExit(f"--worker {args.worker} out of range (workers={args.workers})")
    reports = build_cache(
        source_root=args.source_root,
        output_root=args.output_root,
        mode=args.mode,
        vae_path=args.vae_path,
        device=args.device,
        task_names=args.tasks or None,
        episode_filter=set(args.episode_filter),
        limit=args.limit,
        workers=args.workers,
        worker=args.worker,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

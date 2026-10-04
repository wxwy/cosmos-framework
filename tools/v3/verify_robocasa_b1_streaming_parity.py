# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare frozen offline B1 RoboCasa latents with online Wan causal streaming endpoints.

This is an execution probe, not a second evidence implementation.  It reuses the canonical
B1 builder, endpoint and visual96 authorities and reports numeric differences without inventing
an acceptance tolerance.  A later Gate may pass --max-fp16-abs once empirical parity is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.model.generator.mot.robocasa_latent_evidence import (
    CAMERAS,
    LEFT_CAMERA,
    TEMPORAL_COMPRESSION_FACTOR,
    WRIST_CAMERA,
    latent_to_visual96,
)
from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item
from tools.v3.build_robocasa_b1_h5_cache import (
    FROZEN_LOADER_KWARGS,
    decode_episode_frames,
    encode_episode,
    enumerate_train_episodes,
    load_shard_episode_meta,
    load_wan_vae,
    video_local_start_frame,
)


def _find_spec(source_root: Path, full_id: str):
    matches = [spec for spec in enumerate_train_episodes(source_root) if spec.full_id == full_id]
    if len(matches) != 1:
        raise ValueError(f"--full-id must resolve exactly one frozen train episode, got {len(matches)}")
    return matches[0]


def _load_episode_rgb(spec) -> dict[str, torch.Tensor]:
    entry = load_shard_episode_meta(spec.shard)[spec.episode_index]
    frames: dict[str, torch.Tensor] = {}
    fps = FROZEN_LOADER_KWARGS["fps"]
    for camera in CAMERAS:
        video = entry["video"][camera]
        mp4 = (
            spec.shard
            / "videos"
            / camera
            / f"chunk-{video['chunk_index']:03d}"
            / f"file-{video['file_index']:03d}.mp4"
        )
        if not mp4.is_file():
            raise FileNotFoundError(f"missing video: {mp4}")
        frames[camera] = decode_episode_frames(
            mp4,
            video_local_start_frame(video, fps),
            spec.frame_count,
            fps,
        )
    return frames


def _stream_chunk(
    frames: dict[str, torch.Tensor],
    start: int,
    count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for camera in CAMERAS:
        value = frames[camera][start : start + count]
        if value.shape != (count, 3, 256, 256) or value.dtype != torch.uint8:
            raise ValueError(f"{camera} chunk shape/dtype drift: {tuple(value.shape)} {value.dtype}")
        rows.append(value.permute(1, 0, 2, 3))
    pixels = torch.stack(rows, dim=0)  # [2,3,T,256,256]
    return normalize_uint8_item(pixels, {"device": device, "dtype": torch.float32})


def verify(
    *,
    source_root: Path,
    full_id: str,
    vae_path: Path,
    device: str,
    max_endpoints: int,
    max_fp16_abs: float | None,
) -> dict:
    if max_endpoints <= 0:
        raise ValueError("--max-endpoints must be positive")
    torch_device = torch.device(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} requested but CUDA is unavailable")

    spec = _find_spec(source_root, full_id)
    frames = _load_episode_rgb(spec)
    vae = load_wan_vae(vae_path, torch_device)

    offline: dict[str, torch.Tensor] = {}
    offline_endpoints: list[int] | None = None
    for camera in CAMERAS:
        latent, endpoints = encode_episode(vae, frames[camera], torch_device)
        offline[camera] = latent
        if offline_endpoints is None:
            offline_endpoints = endpoints
        elif endpoints != offline_endpoints:
            raise RuntimeError("offline B1 camera endpoint vectors disagree")
    assert offline_endpoints is not None

    grid_endpoints = [
        endpoint for endpoint in offline_endpoints if endpoint % TEMPORAL_COMPRESSION_FACTOR == 0
    ][:max_endpoints]
    if not grid_endpoints or grid_endpoints[0] != 0:
        raise RuntimeError("offline B1 endpoint authority does not start at zero")

    vae.restore_encoder_stream_state(vae.new_encoder_stream_state())
    streamed: dict[int, torch.Tensor] = {}
    try:
        prime = vae.encode_streaming(_stream_chunk(frames, 0, 1, device=torch_device))
        if tuple(prime.shape) != (2, 48, 1, 16, 16):
            raise RuntimeError(f"streaming prime shape drift: {tuple(prime.shape)}")
        streamed[0] = prime[:, :, 0].detach().cpu().to(torch.float16).contiguous()

        for endpoint in grid_endpoints[1:]:
            start = endpoint - (TEMPORAL_COMPRESSION_FACTOR - 1)
            chunk = vae.encode_streaming(
                _stream_chunk(
                    frames,
                    start,
                    TEMPORAL_COMPRESSION_FACTOR,
                    device=torch_device,
                )
            )
            if tuple(chunk.shape) != (2, 48, 1, 16, 16):
                raise RuntimeError(f"streaming endpoint {endpoint} shape drift: {tuple(chunk.shape)}")
            streamed[endpoint] = chunk[:, :, 0].detach().cpu().to(torch.float16).contiguous()
    finally:
        vae.clear_encoder_cache()

    rows = []
    worst_fp16 = 0.0
    worst_visual = 0.0
    endpoint_to_offline_index = {endpoint: index for index, endpoint in enumerate(offline_endpoints)}
    for endpoint in grid_endpoints:
        index = endpoint_to_offline_index[endpoint]
        stream_pair = streamed[endpoint]
        offline_left = offline[LEFT_CAMERA][index]
        offline_wrist = offline[WRIST_CAMERA][index]
        camera_rows = {}
        for row, camera in enumerate(CAMERAS):
            expected = offline[camera][index]
            actual = stream_pair[row]
            diff = (actual.float() - expected.float()).abs()
            camera_rows[camera] = {
                "max_abs_fp16": float(diff.max().item()),
                "mean_abs_fp16": float(diff.mean().item()),
                "exact_fp16": bool(torch.equal(actual, expected)),
            }
            worst_fp16 = max(worst_fp16, camera_rows[camera]["max_abs_fp16"])

        expected_visual = latent_to_visual96(offline_left, offline_wrist)
        actual_visual = latent_to_visual96(stream_pair[0], stream_pair[1])
        visual_diff = (actual_visual - expected_visual).abs()
        visual_max = float(visual_diff.max().item())
        worst_visual = max(worst_visual, visual_max)
        rows.append(
            {
                "endpoint": endpoint,
                "cameras": camera_rows,
                "visual96_max_abs": visual_max,
                "visual96_mean_abs": float(visual_diff.mean().item()),
                "visual96_exact": bool(torch.equal(actual_visual, expected_visual)),
            }
        )

    result = {
        "full_id": full_id,
        "frame_count": spec.frame_count,
        "device": device,
        "checked_endpoints": grid_endpoints,
        "worst_fp16_max_abs": worst_fp16,
        "worst_visual96_max_abs": worst_visual,
        "rows": rows,
        "status": "OBSERVED",
    }
    if max_fp16_abs is not None:
        if max_fp16_abs < 0:
            raise ValueError("--max-fp16-abs must be non-negative")
        result["max_fp16_abs_threshold"] = max_fp16_abs
        result["status"] = "PASS" if worst_fp16 <= max_fp16_abs else "FAIL"
        if result["status"] != "PASS":
            raise RuntimeError(
                f"B1 offline/streaming fp16 parity failed: max_abs={worst_fp16} > {max_fp16_abs}"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--full-id", required=True, help="exact task/date/ep_XXXXXX frozen train identity")
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-endpoints", type=int, default=8)
    parser.add_argument(
        "--max-fp16-abs",
        type=float,
        default=None,
        help="optional frozen acceptance threshold; omit during first observational parity run",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = verify(
        source_root=args.source_root,
        full_id=args.full_id,
        vae_path=args.vae_path,
        device=args.device,
        max_endpoints=args.max_endpoints,
        max_fp16_abs=args.max_fp16_abs,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

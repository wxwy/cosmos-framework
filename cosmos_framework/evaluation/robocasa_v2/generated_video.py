"""Durable generated-video receipts; used only by explicit strict all media."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

MAX_VIDEO_BYTES = 512 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_video(path, *, frames, size):
    """Decode every frame, independently of Core/SR result validation."""
    import imageio_ffmpeg

    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError("ordinary bounded completed generated MP4 required")
    # read_frames parses FFmpeg's stream metadata from stderr. Suppressing INFO
    # here would hide that header and reject an otherwise valid completed MP4.
    stream = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(stream)
        if tuple(metadata["size"]) != tuple(size):
            raise ValueError("generated MP4 dimensions changed")
        count = 0
        for frame in stream:
            if len(frame) != size[0] * size[1] * 3:
                raise ValueError("incomplete generated video frame")
            count += 1
            if count > frames:
                raise ValueError("unexpected extra generated video frames")
        if count != frames:
            raise ValueError("generated MP4 frame count mismatch")
    finally:
        stream.close()
    return {"frames": frames, "size": list(size), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def write_generated(directory, envelope, fps):
    import base64
    import io

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    index, video, binding = envelope["index"], envelope["video"], envelope["binding"]
    if type(index) is not int or not 0 <= index <= 100000:
        raise ValueError("bounded generated-video index required")
    if not isinstance(video, list) or not 1 <= len(video) <= 65:
        raise ValueError("bounded generated-video frames required")
    target = Path(directory) / "generated"
    target.mkdir(exist_ok=True, parents=True)
    final = target / f"replan_{index:04d}.mp4"
    receipt_path = target / f"replan_{index:04d}.receipt.json"
    if final.exists() or final.is_symlink() or receipt_path.exists():
        raise ValueError("refusing to overwrite a generated video or receipt")
    temporary = target / f".replan_{index:04d}.{uuid.uuid4().hex}.tmp.mp4"
    try:
        size = None
        with imageio.get_writer(str(temporary), fps=fps, macro_block_size=1, ffmpeg_log_level="error") as writer:
            for encoded in video:
                raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
                with Image.open(io.BytesIO(raw)) as frame:
                    if frame.width * frame.height > 2 * 1024**2:
                        raise ValueError("oversized generated frame")
                    if size is None:
                        size = frame.size
                    if frame.size != size:
                        raise ValueError("inconsistent generated frame dimensions")
                    writer.append_data(np.asarray(frame.convert("RGB")))
        details = inspect_video(temporary, frames=len(video), size=size)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, final)
        sync_dir(target)
        receipt = {
            "status": "VIDEO_SAVED",
            "index": index,
            "binding": binding,
            "binding_sha256": sha(binding),
            "path": f"generated/{final.name}",
            **details,
        }
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def read_receipt(directory, index, binding):
    directory = Path(directory)
    path = directory / "generated" / f"replan_{index:04d}.receipt.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("bounded ordinary Encoder receipt required")
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict):
        raise ValueError("Encoder receipt object required")
    expected_path = f"generated/replan_{index:04d}.mp4"
    video = directory / expected_path
    if (
        receipt.get("status") != "VIDEO_SAVED"
        or receipt.get("index") != index
        or receipt.get("binding") != binding
        or receipt.get("binding_sha256") != sha(binding)
        or receipt.get("path") != expected_path
        or type(receipt.get("frames")) is not int
        or not 1 <= receipt["frames"] <= 65
        or not isinstance(receipt.get("size"), list)
        or len(receipt["size"]) != 2
        or any(type(n) is not int or n <= 0 for n in receipt["size"])
        or video.is_symlink()
        or not video.is_file()
        or type(receipt.get("bytes")) is not int
        or not 0 < receipt["bytes"] <= MAX_VIDEO_BYTES
        or video.stat().st_size != receipt["bytes"]
        or file_sha(video) != receipt.get("sha256")
    ):
        raise ValueError("Encoder receipt/file binding mismatch")
    return receipt

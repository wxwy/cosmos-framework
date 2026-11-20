# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""File identities shared by SafeLocal checkpoint publication and inference."""

import hashlib
from pathlib import Path

from torch.distributed.checkpoint.filesystem import FileSystemReader

SAFE_LOCAL_PROFILE = "psm_v3_sync_local_v1"
SAFE_LOCAL_COMPLETE = "complete.v1.json"


def file_witness(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint file missing, nonregular or symlinked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def component_witness(directory, *, expected_metadata=None):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("checkpoint component symlink is forbidden")
    metadata_witness = file_witness(directory / ".metadata")
    if expected_metadata is not None and metadata_witness != expected_metadata:
        raise ValueError("DCP model metadata differs from its pinned publication witness")
    metadata = FileSystemReader(str(directory)).read_metadata()
    files = {}
    for entry in (metadata.storage_data or {}).values():
        relative = Path(entry.relative_path)
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("DCP shard path is not a local component file")
        if path.is_symlink() or not path.is_file() or entry.offset < 0 or entry.length < 0:
            raise ValueError("missing/invalid DCP shard")
        size = path.stat().st_size
        if size < entry.offset + entry.length:
            raise ValueError("truncated DCP shard")
        files[str(relative)] = size
    return {"metadata": metadata_witness, "files": files}

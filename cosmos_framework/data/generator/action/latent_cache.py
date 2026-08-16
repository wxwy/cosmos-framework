"""R12 Cosmos latent-cache loading for LIBERO action windows."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class R12CosmosLatentCache:
    """Load and slice the validated R12 ``episode_*.pt`` latent artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        parity_path: str | Path | None = None,
        max_cached_episodes: int = 8,
        temporal_compression_factor: int = 4,
    ) -> None:
        self.root = Path(root)
        self.max_cached_episodes = int(max_cached_episodes)
        self.temporal_compression_factor = int(temporal_compression_factor)
        self._episodes: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.enabled = self._parity_passed(parity_path)

    @staticmethod
    def _parity_passed(path: str | Path | None) -> bool:
        if path is None:
            return False
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return str(payload.get("status", "")).upper() == "PASS"

    def _load_episode(self, episode_index: int) -> dict[str, Any]:
        episode_index = int(episode_index)
        if episode_index in self._episodes:
            self._episodes.move_to_end(episode_index)
            return self._episodes[episode_index]
        paths = sorted(self.root.glob(f"episode_{episode_index:06d}.pt"))
        if not paths:
            paths = sorted((self.root / "episodes").glob(f"episode_{episode_index:06d}.pt"))
        if not paths:
            paths = sorted(self.root.glob(f"episode_{episode_index}.pt"))
        if not paths:
            paths = sorted((self.root / "episodes").glob(f"episode_{episode_index}.pt"))
        if not paths:
            raise FileNotFoundError(f"R12 latent cache episode not found: {episode_index}")
        item = torch.load(paths[0], map_location="cpu", weights_only=False)
        self._episodes[episode_index] = item
        self._episodes.move_to_end(episode_index)
        while len(self._episodes) > self.max_cached_episodes:
            self._episodes.popitem(last=False)
        return item

    def get_window(self, episode_index: int, start_frame: int, window_frames: int = 17) -> torch.Tensor | None:
        """Return ``[T_latent,C,H,W]`` for a 17-frame source window.

        R12 stores source-frame indices explicitly; selecting by those indices
        prevents accidental positional slicing when an episode is padded.
        """
        if not self.enabled:
            return None
        try:
            item = self._load_episode(episode_index)
        except FileNotFoundError:
            # Partial cache coverage falls back to the online VAE path.
            return None
        windows = item.get("windows")
        if isinstance(windows, dict):
            window = windows.get(str(int(start_frame)))
            if not isinstance(window, dict):
                return None
            latent = window.get("latent")
            if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or latent.shape[0] != 5:
                return None
            return latent.contiguous()
        indices = item.get("indices", {}).get("source_frame_indices")
        latents = item.get("latents", {}).get("cosmos_concat_view")
        if not isinstance(indices, torch.Tensor) or not isinstance(latents, torch.Tensor):
            return None
        # The VAE's 4n+1 padding anchors each arbitrary 17-frame window at the
        # preceding compression boundary (e.g. source frames 1..17 use 0,4,..16).
        aligned_start = int(start_frame) - int(start_frame) % self.temporal_compression_factor
        expected = torch.arange(
            aligned_start, aligned_start + int(window_frames), self.temporal_compression_factor
        )
        positions = torch.searchsorted(indices.to(dtype=torch.long), expected)
        if positions.numel() != 5 or torch.any(positions >= indices.numel()):
            return None
        if not torch.equal(indices[positions].to(dtype=torch.long), expected):
            return None
        selected = latents[positions.tolist()]
        if selected.ndim != 4 or selected.shape[0] != 5:
            return None
        return selected.contiguous()

"""Spawn-safe synthetic raw cache stand-in; no real source/cache or models."""

import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IMPORTED_CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES")
IMPORTED_RANK = os.environ.get("RANK")


@dataclass(frozen=True)
class SyntheticRecipe:
    fault: str = ""
    witness: str = ""

    def reader(self):
        return {}

    def read(self, job, reader):
        if self.fault == "ignore_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if self.witness:
            Path(self.witness, f"ready_{os.getpid()}").write_text(self.fault)
        if self.fault == "crash":
            os._exit(17)
        if self.fault == "error":
            raise OSError("injected synthetic raw cache IO error")
        if self.fault == "timeout":
            time.sleep(20)
        if self.fault == "ignore_term":
            time.sleep(20)
        # These draws are deliberately worker-local, never payload semantics.
        random.random()
        np.random.rand()
        torch.rand(())
        rows = [(flat, dict(item)) for flat, item in job.sources]
        if self.fault == "order":
            rows.reverse()
        if self.fault == "abi":
            rows[0][1]["action"] = torch.zeros(12)
        if self.fault == "source":
            rows[0][1]["task_class"] = "foreign"
        if self.fault == "large_packet":
            rows[0][1]["padding"] = torch.zeros(20000)
        for _, item in rows:
            if IMPORTED_CUDA_VISIBLE != "" or IMPORTED_RANK is not None:
                raise AssertionError("worker imports must already be CPU-only and rank-free")
            if torch.cuda.is_initialized() or torch.utils.data.get_worker_info() is None:
                raise AssertionError("raw read must be a CPU DataLoader worker")
            if torch.get_num_threads() != 1 or os.environ.get("RANK") is not None:
                raise AssertionError("worker thread/rank environment not isolated")
        return tuple(rows)

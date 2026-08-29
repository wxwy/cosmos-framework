"""Startup provenance for one fixed-weight R08 Gate B capture."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from cosmos_framework.utils.callback import Callback


class R08GateBProvenanceCallback(Callback):
    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)

    def on_train_start(self, model) -> None:
        del model
        root = Path.cwd().parent
        submodule = Path.cwd()
        git = lambda repo, arg: subprocess.check_output(["git", "-C", str(repo), "rev-parse", arg], text=True).strip()
        payload = {
            "schema_version": "r08_gate_b_capture_provenance_v1",
            "root_revision": git(root, "HEAD"),
            "submodule_revision": git(submodule, "HEAD"),
            "gitlink_revision": git(root, "HEAD:cosmos-framework"),
            "history_mode": os.environ.get("PSM_R08_HISTORY_MODE"),
            "capture_only": os.environ.get("PSM_R08_GATE_B_CAPTURE_ONLY") == "1",
            "checkpoint_path": os.environ.get("PSM_R08_GATE_B_CHECKPOINT_PATH"),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Round-trip consistency check: dataset 7D action --source-conversions--> 7D action.

Question (user): does the dataset's raw 7D action, passed through cosmos's pose
conversion AND the conversion-back logic, reproduce the original?

The conversion chain under test — BOTH legs are the *actual source-code functions*
(no reimplementation):

  Forward (training data conversion, source:
      cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py
      LIBEROLeRobotDataset._build_frame_wise_action):
      raw7 [dpos(3), drot_axisangle(3), gripper] ->
          convert_rotation(axisangle -> matrix) ->
          convert_rotation(matrix -> rot6d)        [libero_rotation_format("6d")=="rot6d"]
      -> 10D [dpos(3), rot6d(6), gripper]

  Backward (eval conversion, source:
      cosmos_framework/simulation/libero/closed_loop_eval.py
      _framewise_action_to_delta -> _rotation_repr_to_mat):
      10D [dpos(3), rot6d(6), gripper] ->
          convert_rotation(rot6d -> matrix, normalize_matrix=True) ->
          R.from_matrix(...).as_rotvec()
      -> 7D [dpos(3), drot_rotvec(3), gripper]

Run (RLinf python — has torch/scipy/pandas + repo source):
  cd /disk/rl/psm_wma/cosmos-framework
  /disk/rl/RLinf/.venv/bin/python examples/verify_action_roundtrip.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIBERO_ROOT = Path("/disk/data/LIBERO_LeRobot_v3/libero_10_task0")
_ACTION_FEATURE = "action"


def _load_parquet() -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((LIBERO_ROOT / "data").glob("chunk-*/*.parquet"))
    tbl = pq.read_table(files[0])
    for f in files[1:]:
        tbl = pa.concat_tables([tbl, pq.read_table(f)])
    return {
        "ep": np.stack(tbl.column("episode_index").to_numpy()),
        "action": np.stack(tbl.column(_ACTION_FEATURE).to_numpy()),
    }


def main() -> None:
    sys.path.insert(0, str(REPO))

    # --- source functions -------------------------------------------------
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import (
        LIBEROLeRobotDataset,
    )
    from cosmos_framework.simulation.libero import closed_loop_eval as ev
    from scipy.spatial.transform import Rotation as R

    d = _load_parquet()
    eps = np.unique(d["ep"])
    print(f"[data] {len(eps)} episodes in task0 parquet")

    # run the roundtrip across ALL stove-task episodes for coverage
    rot_coord_diff = []
    rot_angle_deg = []
    pos_maxdiff = []
    grip_maxdiff = []
    worst = None  # (ep, idx, raw, out)

    shim = SimpleNamespace(_rotation_space="6d")  # only attr the source method reads
    fwd = LIBEROLeRobotDataset._build_frame_wise_action  # unbound -> runs exact source body

    for ep in eps:
        raw7 = d["action"][d["ep"] == ep].astype(np.float32)  # [N,7]
        a10 = fwd(shim, raw7).numpy()  # [N,10]  forward: SOURCE method
        out7 = np.stack([ev._framewise_action_to_delta(a10[i], "6d") for i in range(len(a10))])

        pos_maxdiff.append(float(np.abs(out7[:, :3] - raw7[:, :3]).max()))
        grip_maxdiff.append(float(np.abs(out7[:, 6] - raw7[:, 6]).max()))
        rot_coord_diff.append(float(np.abs(out7[:, 3:6] - raw7[:, 3:6]).max()))

        # physical rotation difference (convention-robust): R(raw) vs R(roundtrip)
        rel = (R.from_rotvec(raw7[:, 3:6]).inv() * R.from_rotvec(out7[:, 3:6])).as_rotvec()
        ang = np.degrees(np.linalg.norm(rel, axis=1))
        rot_angle_deg.append(float(ang.max()))
        if ang.max() > 1e-4 and (worst is None or ang.max() > worst[0]):
            i = int(np.argmax(ang))
            worst = (ang.max(), ep, i, raw7[i].copy(), out7[i].copy())

    print(
        f"[roundtrip] episodes={len(eps)}  steps={sum(len(d['action'][d['ep']==e]) for e in eps)}\n"
        f"  pos     max |roundtrip - raw| = {max(pos_maxdiff):.3e}\n"
        f"  gripper max |roundtrip - raw| = {max(grip_maxdiff):.3e}\n"
        f"  rot     max |coord diff|      = {max(rot_coord_diff):.4f}  (见坐标约定说明)\n"
        f"  rot     max physical angle err = {max(rot_angle_deg):.4f} deg"
    )

    if worst is None:
        print("[roundtrip] VERDICT: 完全一致 —— 正/反向转化是互逆的，往返无损 (物理误差 < 1e-4 deg)")
    else:
        ang, ep, i, raw, out = worst
        print(
            f"[roundtrip] worst coord mismatch ep={ep} idx={i} angle={ang:.4f} deg\n"
            f"    raw=[{np.round(raw, 5)}]\n"
            f"    out=[{np.round(out, 5)}]\n"
            f"    ||raw rot||={np.linalg.norm(raw[3:6]):.4f}  ||out rot||={np.linalg.norm(out[3:6]):.4f}"
        )
        print("[roundtrip] VERDICT: 存在数值级差异，见上方详情")

    # coordinate-convention probe: fraction of raw rotvecs whose norm > pi (impossible for scipy rotvec)
    all_raw = d["action"][:, 3:6]
    norms = np.linalg.norm(all_raw, axis=1)
    print(
        f"[convention] raw rotvec norms: min={norms.min():.4f} max={norms.max():.4f} "
        f"#(norm>pi)={int((norms > np.pi).sum())}/{len(norms)}"
    )


if __name__ == "__main__":
    main()

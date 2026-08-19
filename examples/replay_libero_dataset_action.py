# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Replay LIBERO dataset actions in the sim to diagnose action/coordinate correspondence.

User observation: the model's *predicted video* (mp4_pred) shows the arm moving toward
the stove, but the *executed action* drives the arm in a visually-unrelated direction.
Hypotheses:
  H1. action head hasn't learned the task yet (SFT too early),
  H2. sim execution of the denormalized action is wrong (coordinate correspondence),
  H3. the dataset's stored actions themselves don't reproduce the recorded video.

This script tests H2/H3 directly by replaying a *dataset* episode's actions in the
LIBERO sim and comparing the replayed EEF trajectory against the dataset's recorded
`observation.state`. Two replay modes:
  Level A (raw):    stored 7D action [pos_delta, axis_angle_delta, gripper[0,1]] fed to
                    env.step with only the gripper remapped [0,1]->[-1,1] (same as the
                    eval's `_remap_gripper` continuous mapping).
  Level B (eval):   the exact pipeline closed_loop_eval uses for model outputs:
                    stored 7D -> rot6d (training conversion) -> _framewise_action_to_delta
                    -> _remap_gripper -> env.step.

Verdicts:
  Level A reproduces the dataset EEF trajectory  -> sim+data self-consistent -> the eval
      failure is H1 (learning/denormalization), not H2/H3.
  Level A diverges immediately                  -> the stored action does not match the
      env's controller frame -> H2 (execution) or H3 (data conversion).
  Level A ok but Level B diverges               -> the eval's rot6d conversion is wrong.

Replay run (RLinf python, has LIBERO sim):
  cd /disk/rl/psm_wma/cosmos-framework
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  NO_PROXY=localhost,127.0.0.1 \
  /disk/rl/RLinf/.venv/bin/python examples/replay_libero_dataset_action.py --episode 0

Reference-video decode (AV1) must run in the cosmos venv (has PyAV/libdav1d):
  .venv/bin/python examples/replay_libero_dataset_action.py --decode-ref --episode 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIBERO_ROOT = Path("/disk/data/LIBERO_LeRobot_v3/libero_10_task0")
STOVE_TASK = "turn on the stove and put the moka pot on it"

_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_IMAGE_VIDEO = "observation.images.image"


# --------------------------------------------------------------------------
# dataset access (lazy imports: pyarrow only, works in both venvs)
# --------------------------------------------------------------------------
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
        "state": np.stack(tbl.column(_STATE_FEATURE).to_numpy()),
    }


def _stove_episodes() -> list[int]:
    """Global libero_10 episode indices whose task is the stove task (sorted)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    mt = pq.read_table(LIBERO_ROOT / "meta/episodes/chunk-000/file-000.parquet")
    mt2 = pq.read_table(LIBERO_ROOT / "meta/episodes/chunk-000/file-001.parquet")
    m = pa.concat_tables([mt, mt2]).to_pydict()
    return sorted(e for e, t in zip(m["episode_index"], m["tasks"]) if t[0] == STOVE_TASK)


def _episode_video_range(episode: int) -> tuple[Path, float, float]:
    """Map a global episode index to (video file, from_timestamp, to_timestamp)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    mt = pq.read_table(LIBERO_ROOT / "meta/episodes/chunk-000/file-000.parquet")
    mt2 = pq.read_table(LIBERO_ROOT / "meta/episodes/chunk-000/file-001.parquet")
    m = pa.concat_tables([mt, mt2]).to_pydict()
    idx = m["episode_index"].index(episode)
    c = m[f"videos/{_IMAGE_VIDEO}/chunk_index"][idx]
    f = m[f"videos/{_IMAGE_VIDEO}/file_index"][idx]
    path = LIBERO_ROOT / "videos" / _IMAGE_VIDEO / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
    from_ts = m[f"videos/{_IMAGE_VIDEO}/from_timestamp"][idx]
    to_ts = m[f"videos/{_IMAGE_VIDEO}/to_timestamp"][idx]
    return path, from_ts, to_ts


# --------------------------------------------------------------------------
# reference-video decode (cosmos venv: av / cv2)
# --------------------------------------------------------------------------
def _decode_ref(episode: int, out_dir: Path) -> None:
    import av
    import cv2

    path, from_ts, to_ts = _episode_video_range(episode)
    print(f"decoding reference video: {path}  frames in [{from_ts:.2f}, {to_ts:.2f}]s")
    frames = []
    with av.open(str(path)) as cont:
        for frame in cont.decode(video=0):
            t = float(frame.time)
            if t < from_ts - 0.03:
                continue
            if t > to_ts + 0.03:
                break
            frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24")))
    print(f"decoded {len(frames)} frames  size={frames[0].shape}")
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_dir / "reference.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"saved {out_dir / 'reference.mp4'}")


# --------------------------------------------------------------------------
# sim replay (RLinf venv)
# --------------------------------------------------------------------------
def _sim_imports():
    sys.path.insert(0, str(REPO))
    sys.path.append(f"{REPO}/.venv/lib/python3.13/site-packages")
    from cosmos_framework.simulation.libero import closed_loop_eval as ev

    ev._import_libero()  # sets module globals (benchmark, get_libero_path, OffScreenRenderEnv)
    return ev


def _eef_state(pos: np.ndarray, quat: np.ndarray, gripper: float) -> np.ndarray:
    """sim obs -> dataset 8D state [pos(3), axis_angle(3), gripper, gripper]."""
    from scipy.spatial.transform import Rotation as R

    aa = R.from_quat(np.asarray(quat, dtype=np.float32)).as_rotvec()
    return np.concatenate([np.asarray(pos, dtype=np.float32), aa.astype(np.float32), [gripper, gripper]])


def _match_sim_demo(ev, task_suite, task_id: int, dataset_state0: np.ndarray, env_args: dict) -> tuple[int, float]:
    """Find sim demo whose reset EEF state is closest to the dataset episode's first state."""
    init_states = task_suite.get_task_init_states(task_id)
    env, _ = ev._get_libero_env(task_suite.get_task(task_id), **env_args)
    best_idx, best_dist = -1, np.inf
    try:
        for d in range(len(init_states)):
            env.reset()
            obs = env.set_init_state(np.array(init_states[d]))
            s = _eef_state(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"][0])
            dist = float(np.linalg.norm(s[:7] - dataset_state0[:7]))
            if dist < best_dist:
                best_dist, best_idx = dist, d
    finally:
        env.close()
    return best_idx, best_dist


def _obs_eef7(obs: dict) -> np.ndarray:
    """sim obs -> [pos(3), axis_angle(3), gripper] for trajectory comparison."""
    from scipy.spatial.transform import Rotation as R

    pos = np.asarray(obs["robot0_eef_pos"], np.float32)
    aa = R.from_quat(np.asarray(obs["robot0_eef_quat"], np.float32)).as_rotvec().astype(np.float32)
    return np.concatenate([pos, aa, [float(obs["robot0_gripper_qpos"][0])]])


def _replay(ev, env, actions7: np.ndarray, mode: str, obs0: dict) -> dict:
    """Replay dataset actions in the env (init state already set). Returns traj + frames.

    ``traj[0]``/``frames[0]`` are the initial state (pre-action), so frame i and
    dataset state i are 1:1 aligned across the whole episode.
    """
    rot = ev._infer_rotation_space(10, "6d")
    traj, frames = [], []
    obs = obs0
    traj.append(_obs_eef7(obs))
    frames.append(ev._get_libero_image(obs, "agentview", flip_images=False, rotate_180=False))
    for a7 in actions7:
        if mode == "A":
            cmd = np.asarray(a7, dtype=np.float32).copy()
            cmd[-1] = _gripper_zero_one_to_env(cmd[-1])
        else:  # B: dataset 7D -> 10D (training conversion) -> eval 7D pipeline
            a10 = _build_frame_wise_action(a7)
            cmd = ev._framewise_action_to_delta(a10, rot)
            cmd[-1] = ev._remap_gripper(cmd.tolist(), "zero_one")[-1]
        obs, _, done, info = env.step(cmd.tolist())
        traj.append(_obs_eef7(obs))
        frames.append(ev._get_libero_image(obs, "agentview", flip_images=False, rotate_180=False))
        if isinstance(info, dict) and info.get("success"):
            break
    return {"traj": np.array(traj), "frames": frames, "steps": len(traj)}


def _gripper_zero_one_to_env(g: float) -> float:
    """[0,1] (1=open) -> [-1,1] (negative=open); same continuous map as eval _remap_gripper."""
    return max(-1.0, min(1.0, float(g) * 2.0 - 1.0)) * -1.0


def _build_frame_wise_action(raw7: np.ndarray) -> np.ndarray:
    """Mirror LIBEROLeRobotDataset._build_frame_wise_action: [7]->[10] (rot6d)."""
    from cosmos_framework.data.generator.action.libero_pose_utils import libero_rotation_format
    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

    trans = raw7[:3]
    rotmat = convert_rotation(raw7[3:6], input_format="axisangle", output_format="matrix")
    rot6d = convert_rotation(rotmat, input_format="matrix", output_format=libero_rotation_format("6d"))
    return np.concatenate([trans, np.asarray(rot6d, dtype=np.float32), raw7[6:7]], axis=0)


def _traj_mae(traj: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, float]:
    """Compare replay traj[i] vs dataset state[i] (both start at the initial state).

    Returns absolute MAE (pos, rot, gripper) and relative (start-anchored) pos MAE.
    """
    n = min(len(traj), len(ref))
    a = traj[:n]
    r = ref[:n]
    mae_pos = float(np.abs(a[:, :3] - r[:, :3]).mean())
    mae_aa = float(np.abs(a[:, 3:6] - r[:, 3:6]).mean())
    mae_g = float(np.abs(a[:, 6:7] - r[:, 6:7]).mean())
    # start-anchored displacement comparison (insensitive to initial-state offset)
    da = a[:] - a[0]
    dr = r[:] - r[0]
    rel_pos = float(np.abs(da[:, :3] - dr[:, :3]).mean())
    return mae_pos, mae_aa, mae_g, rel_pos


def _load_mp4_frames(path: Path) -> list[np.ndarray]:
    """Read an MP4's frames as RGB uint8 arrays (cv2, works on mp4v output)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _save_compare(replay_frames: list[np.ndarray], ref_frames: list[np.ndarray], out_path: Path, fps: int = 20) -> int:
    """Side-by-side [replay | dataset reference], frame-aligned, saved as MP4."""
    import cv2

    n = min(len(replay_frames), len(ref_frames))
    if n == 0:
        return 0
    h, w = np.asarray(replay_frames[0]).shape[:2]
    rw = np.asarray(ref_frames[0]).shape[1]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + rw, h))
    try:
        for i in range(n):
            left = np.asarray(replay_frames[i])
            right = np.asarray(ref_frames[i])
            if left.shape != right.shape:
                right = cv2.resize(right, (w, h))
            combined = np.hstack([left, right])
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0, help="index into the sorted stove-task episode list (0=first)")
    parser.add_argument("--output_dir", type=str, default="results/replay_dataset_action")
    parser.add_argument("--max_steps", type=int, default=0, help="0 = full episode")
    parser.add_argument("--decode-ref", action="store_true", help="only decode the dataset reference video (cosmos venv)")
    args = parser.parse_args()

    stove_eps = _stove_episodes()
    if args.episode >= len(stove_eps):
        args.episode = args.episode % len(stove_eps)
    episode = stove_eps[args.episode]
    out_dir = Path(args.output_dir) / f"ep{episode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.decode_ref:
        _decode_ref(episode, out_dir)
        return

    ev = _sim_imports()
    d = _load_parquet()
    ref_states = d["state"][d["ep"] == episode]
    actions7 = d["action"][d["ep"] == episode]
    max_steps = args.max_steps or len(actions7)
    actions7 = actions7[:max_steps]
    print(f"[dataset] stove episodes={len(stove_eps)}  picked global episode={episode}  steps={len(actions7)}")
    print(f"[dataset] state0=[{np.round(ref_states[0], 4)}]")
    vpath, vfrom, vto = _episode_video_range(episode)
    print(f"[dataset] video = {vpath}  [{vfrom:.2f}s, {vto:.2f}s]")

    from libero.libero import benchmark  # noqa: E402

    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_id = 2  # sim task_id for the stove task (verified: language == STOVE_TASK)
    env_args = {"resolution": 256, "seed": 0, "render_gpu_device_id": -1}

    best_demo, best_dist = _match_sim_demo(ev, task_suite, task_id, ref_states[0], env_args)
    n_demos = len(task_suite.get_task_init_states(task_id))
    print(f"[sim] matched demo #{best_demo}/{n_demos}  init-eef dist={best_dist:.4f}")

    ref_mp4 = out_dir / "reference.mp4"
    ref_frames = _load_mp4_frames(ref_mp4) if ref_mp4.exists() else []
    if ref_frames:
        print(f"[ref] loaded {len(ref_frames)} reference frames from {ref_mp4}")
    else:
        print("[ref] WARN: reference.mp4 不存在，先跑 .venv/bin/python examples/replay_libero_dataset_action.py --decode-ref")

    init = np.array(task_suite.get_task_init_states(task_id)[best_demo])
    for mode in ("A", "B"):
        env, _ = ev._get_libero_env(task_suite.get_task(task_id), **env_args)
        env.reset()
        obs0 = env.set_init_state(init)
        r = _replay(ev, env, actions7, mode, obs0)
        env.close()
        np.save(out_dir / f"traj_{mode}.npy", r["traj"])
        ev._save_mp4(r["frames"], out_dir / f"replay_{mode}.mp4", 20)
        mae_pos, mae_aa, mae_g, rel_pos = _traj_mae(r["traj"], ref_states)
        print(
            f"[Level {mode}] steps={r['steps']}  "
            f"pos MAE={mae_pos:.4f}  rot MAE={mae_aa:.4f}  grip MAE={mae_g:.4f}  "
            f"anchored pos MAE={rel_pos:.4f}"
        )
        if ref_frames:
            n_cmp = _save_compare(r["frames"], ref_frames, out_dir / f"compare_{mode}.mp4")
            print(f"[Level {mode}] saved compare_{mode}.mp4 ({n_cmp} frames: 左=replay, 右=数据集参考)")

    np.save(out_dir / "ref_states.npy", ref_states)
    json.dump(
        {"episode": episode, "best_demo": best_demo, "init_dist": best_dist, "n_steps": len(actions7),
         "n_demos": n_demos},
        open(out_dir / "meta.json", "w"), indent=2,
    )
    print(f"\n结果目录: {out_dir}")
    print(
        "判读: 若 Level A 的 pos/anchored-pos MAE 很小(≲1cm) => 数据 action 与仿真自洽，"
        "闭环失败属 H1(策略没学会/反归一化)；\n"
        "      若 Level A 立即发散 => 数据 action 或坐标对应有问题(H2/H3)；\n"
        "      若 A 好而 B 发散 => eval 的 rot6d 转换有 bug。"
    )


if __name__ == "__main__":
    main()

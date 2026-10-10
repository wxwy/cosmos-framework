# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Closed-loop evaluation for RoboCasa using the Action HTTP inference server.

# Example (mobile-base recipe: 15-D raw base action, agentview_left | eye_in_hand composite):
MUJOCO_GL=egl PYTHONPATH=. python cosmos_framework/simulation/robocasa/closed_loop_eval.py \
  --server-url http://localhost:8900 \
  --dataset-dir /path/to/robocasa/datasets/v1.0/target/atomic/CloseFridge/<date>/lerobot \
  --output-dir results/robocasa_closed_loop/CloseFridge \
  --num-test-episodes 50 \
  --action-horizon 16 \
  --camera-set left_wrist \
  --use-state \
  --use-base-action --base-encoding raw \
  --local-memory-mode required \
  --success-latch 1 --seed 0 --image-size 256 --cam-size 256

TWO PYTHON ENVIRONMENTS ARE REQUIRED. robosuite/robocasa and cosmos-framework cannot share one
venv (conflicting numpy/mujoco pins), so this script -- which only drives the simulator -- runs
in the robosuite/robocasa venv, while the policy is served by
``cosmos_framework.scripts.action_policy_server_robocasa`` from the cosmos-framework venv. Only
empty namespace packages are imported from ``cosmos_framework`` here, so ``PYTHONPATH=.`` is
enough and cosmos-framework need not be installed in the simulator venv.

The evaluation contract -- ``--action-horizon``, ``--camera-set``, ``--use-state``,
``--base-encoding`` -- MUST match the training recipe. A mismatch does not raise; it silently
degrades the policy.

Protocol: env args are rebuilt from the dataset's ``extras/dataset_meta.json`` so they match how
the demonstrations were recorded, held-out scenes come from the official ``target`` object split
and official layout/style combinations, success uses the environment's own ``_check_success``
with first-success latching, and the rollout horizon is RoboCasa's official per-task value.

Writes ``results.json`` (one record per rollout) into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import robocasa  # noqa: F401  (registers RoboCasa envs like CloseToasterOvenDoor; does NOT pull lerobot)
import robosuite
from PIL import Image

from cosmos_framework.inference.robocasa_local_memory_contract import validate_local_memory_eval_contract
from cosmos_framework.simulation.robocasa.eval_utils import (
    canonicalize_raw15_for_env,
    decode_10d_to_env12,
    decode_15d_to_env12,
    decode_20d_to_env12,
    decode_pred_video,
    predict,
    reset_local_memory,
)
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient
from cosmos_framework.utils.rollout_evidence import atomic_json, run_recorded_episode, save_partial_video

CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def get_env_metadata(dsdir: str) -> dict:
    with open(Path(dsdir) / "extras" / "dataset_meta.json") as f:
        return json.load(f)["env_args"]


CAMERA_SET = "wrist_lr"  # set from --camera-set in main(); switches compose layout
USE_BASE_ACTION = False  # set from --use-base-action; widens the state token to the mobile contract
BASE_ENCODING = "ego"  # set from --base-encoding; "ego" = 20D pose delta, "raw" = 15D base_motion
_ACTION_DIM_BY_ENCODING = {"ego": 20, "raw": 15}


def _upright(img: np.ndarray) -> np.ndarray:
    """Flip a raw robosuite camera observation to match the recorded dataset.

    MuJoCo renders with the OpenGL convention (origin bottom-left), so
    ``obs["<cam>_image"]`` comes back vertically mirrored, while the RoboCasa
    LeRobot videos the policy was trained on are stored upright. Feeding the
    unflipped frame does not raise -- the policy simply acts on an upside-down
    world and success collapses to ~0, so this must not be made optional.
    """
    return img[::-1]


def compose(obs) -> np.ndarray:
    if CAMERA_SET == "left_wrist":
        # LIBERO-style: [agentview_left | wrist], full-res, horizontal -> 256x512.
        # Mirrors RoboCasaLeRobotDataset._compose_left_wrist (left | wrist, no downscale).
        l = _upright(obs[f"{CAMS[0]}_image"])
        w = _upright(obs[f"{CAMS[2]}_image"])
        return np.concatenate([l, w], axis=1)  # [H, 2W, 3]
    if CAMERA_SET == "lrw":
        # All three cameras side by side at full res -> 256x768.
        # Mirrors RoboCasaLeRobotDataset._compose_lrw (left | right | wrist).
        l = _upright(obs[f"{CAMS[0]}_image"])
        r = _upright(obs[f"{CAMS[1]}_image"])
        w = _upright(obs[f"{CAMS[2]}_image"])
        return np.concatenate([l, r, w], axis=1)  # [H, 3W, 3]
    w = _upright(obs[f"{CAMS[2]}_image"])
    l = _upright(obs[f"{CAMS[0]}_image"])
    r = _upright(obs[f"{CAMS[1]}_image"])
    h, wd = w.shape[:2]
    hh, hw = h // 2, wd // 2
    lh = np.asarray(Image.fromarray(l).resize((hw, hh), Image.BILINEAR))
    rh = np.asarray(Image.fromarray(r).resize((hw, hh), Image.BILINEAR))
    return np.concatenate([w, np.concatenate([lh, rh], axis=1)], axis=0)


def build_state_token(obs) -> list[float]:
    """Current EEF proprioception -> 10D ``[pos(3), rot6d(6), gripper(1)]`` token, matching
    ``RoboCasaLeRobotDataset._build_initial_state``. The stored observation.state eef fields
    map to robosuite obs keys (robocasa lerobot_utils): end_effector_position_relative =
    robot0_base_to_eef_pos, end_effector_rotation_relative = robot0_base_to_eef_quat (xyzw),
    gripper_qpos = robot0_gripper_qpos. So these obs are already in the (base) frame the model
    was trained on — no manual transform needed."""
    import robosuite.utils.transform_utils as T

    pos = np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float32).reshape(3)
    quat = np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float32).reshape(4)  # xyzw
    m = T.quat2mat(quat)  # [3,3]; robosuite quat is xyzw
    rot6d = np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)  # [col0, col1] (matches convert_rotation)
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    grip = np.array([qpos[0] - qpos[1]], dtype=np.float32)  # signed finger opening
    token = np.concatenate([pos, rot6d, grip]).astype(np.float32)  # [10]
    if USE_BASE_ACTION:
        # The conditioning token must match the action width (20D ego / 15D raw). The base
        # block is zero-filled exactly as in training (RoboCasaLeRobotDataset.__getitem__):
        # under "ego" the absolute base pose is world-frame and carries no transferable
        # signal, and under "raw" the observation has no base velocity to put there at all.
        pad = _ACTION_DIM_BY_ENCODING[BASE_ENCODING] - 10
        token = np.concatenate([np.zeros(pad, dtype=np.float32), token])
    return token.tolist()


# RoboCasa365 "Atomic-Seen" split (leaderboard protocol). Authoritative source is the
# installed robocasa's ``robocasa/utils/env_utils.py::create_env`` with ``split="target"``:
#
#     obj_instance_split = "target"
#     layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))   # 10 target kitchens
#     robots = "PandaOmron"  (composite controller default -> HYBRID_MOBILE_BASE)
#
# and ``docs/benchmarking/benchmarking_overview.md``: "For all experiments, we randomly
# sample 50 scenarios to run evaluation rollouts on." Horizons come from
# ``dataset_registry.py`` (v1.0.1 bumped every horizon by 1.5x).
#
# NOTE: ``robocasa/utils/eval_utils.py::create_eval_env`` still defaults to PandaMobile +
# OSC_POSE + obj_instance_split="B" + the 5 off-diagonal combos above. That is the older
# v0.x generalization protocol, NOT Atomic-Seen -- do not use it for leaderboard numbers.
ATOMIC_SEEN_LAYOUT_STYLE = tuple(zip(range(1, 11), range(1, 11)))
ATOMIC_SEEN_OBJ_SPLIT = "target"
ATOMIC_SEEN_NUM_ROLLOUTS = 50

# The three RoboCasa365 target splits are defined authoritatively by
# ``dataset_registry.TARGET_TASKS`` -- 18 atomic_seen + 16 composite_seen + 16
# composite_unseen = the "50 target tasks" the leaderboard reports. Do NOT infer the
# composite split from which tasks have a pretrain entry: that gives 17/16, not 16/16.
# FIXED composite subset -- 12 seen + 6 unseen = 18 tasks, matching atomic_seen's task count
# so the three splits cost the same and the numbers sit on the same scale.
#


def official_horizon(task: str):
    """Official per-task rollout horizon from robocasa's registry (the source of truth).

    Returns ``None`` when the registry cannot be read, so the caller falls back to
    ``--max-steps``. Mirroring the horizons here would just be a copy that silently
    goes stale (v1.0.1 already rebased every one of them by 1.5x), so the miss is
    reported instead."""
    try:
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        return int(get_task_horizon(task))
    except Exception as exc:
        print(
            f"[eval] WARNING: could not read the official horizon for {task} from robocasa "
            f"({type(exc).__name__}: {exc}); falling back to --max-steps",
            flush=True,
        )
        return None


def make_env(dataset_dir: str, cam_size: int, *, obj_split=None, layout_style=None, seed=None):
    env_meta = get_env_metadata(dataset_dir)
    ek = dict(env_meta["env_kwargs"])
    ek["env_name"] = env_meta["env_name"]
    if seed is not None:
        # Fixed seed -> reproducible test-scene sequence (self.rng.choice over layout/style +
        # object/placement sampling). Lets different checkpoints see the SAME 20 scenes/task.
        ek["seed"] = seed
    ek["has_renderer"] = False
    ek["has_offscreen_renderer"] = True
    ek["use_camera_obs"] = True
    ek["camera_names"] = CAMS
    ek["camera_widths"] = cam_size
    ek["camera_heights"] = cam_size
    ek["control_freq"] = 20
    ek["ignore_done"] = True
    # Held-out test protocol: override object split + layout/style set so env.reset() (test
    # from the demo model.xml, so exact reproduction is unaffected by these overrides.
    if obj_split is not None:
        ek["obj_instance_split"] = obj_split
    if layout_style is not None:
        ek["layout_and_style_ids"] = [list(ls) for ls in layout_style]
        ek["layout_ids"] = None
        ek["style_ids"] = None
    ek.pop("renderer", None)
    return robosuite.make(**ek)


def check_success(env) -> bool:
    try:
        return bool(env._check_success())
    except Exception:
        return False


def run_policy(
    env,
    *,
    server_url,
    image_size,
    action_horizon,
    max_steps,
    latch,
    timeout,
    use_state=False,
    save_png=None,
    save_video=None,
    gen_video_path=None,
    video_fps=20,
    local_memory_client: RoboCasaLocalMemoryClient | None = None,
    on_prediction: Callable[[dict, float], None] | None = None,
    on_action: Callable[[dict], None] | None = None,
) -> tuple[bool, int, str]:
    obs = env.reset()
    if local_memory_client is not None:
        local_memory_client.begin(0, image_size=image_size)
    rollout_error: BaseException | None = None
    frames, gen_frames = [], []
    try:
        try:
            prompt = env.get_ep_meta().get("lang", "") or ""
        except Exception:
            prompt = ""
        init_comp = compose(obs).astype(np.uint8)
        if save_png is not None:
            Image.fromarray(init_comp).save(save_png)
        frames = [init_comp]
        gen_frames = []
        streak, queue, success, done_steps = 0, [], False, max_steps
        for step in range(max_steps):
            if not queue:
                comp = compose(obs)
                state_token = build_state_token(obs) if use_state else None
                local_payload = local_memory_client.payload(0) if local_memory_client is not None else None
                prediction_t0 = time.monotonic()
                result = predict(
                    server_url,
                    comp,
                    prompt,
                    image_size,
                    timeout,
                    state=state_token,
                    local_memory=local_payload,
                )
                prediction_ms = (time.monotonic() - prediction_t0) * 1000.0
                if on_prediction is not None:
                    on_prediction(result, prediction_ms)
                if local_memory_client is not None:
                    status = result.get("local_memory")
                    if not isinstance(status, dict):
                        raise RuntimeError("required Local-TTT server response is missing local_memory status")
                    local_memory_client.acknowledge(0, status)
                if gen_video_path is not None and result.get("video"):
                    gen_frames.extend(decode_pred_video(result["video"]))
                acts = result["action"]
                queue = acts[:action_horizon] if action_horizon > 0 else list(acts)

            a = np.asarray(queue.pop(0))
            pre_composite = compose(obs).astype(np.uint8) if local_memory_client is not None else None
            if USE_BASE_ACTION:
                want = _ACTION_DIM_BY_ENCODING[BASE_ENCODING]
                if a.shape[-1] != want:
                    raise ValueError(
                        f"--base-encoding={BASE_ENCODING} expects {want}D actions but the server "
                        f"returned {a.shape[-1]}D. Check that the flag matches the checkpoint."
                    )
                if local_memory_client is not None:
                    env_action, canonical_raw15 = canonicalize_raw15_for_env(a, False)
                else:
                    env_action = (decode_15d_to_env12 if BASE_ENCODING == "raw" else decode_20d_to_env12)(a, False)
            else:
                if a.shape[-1] != 10:
                    raise ValueError(
                        f"expected the 10D fixed-base contract but the server returned "
                        f"{a.shape[-1]}D; pass --use-base-action (and --base-encoding)."
                    )
                env_action = decode_10d_to_env12(a, False)

            obs, reward, env_done, info = env.step(env_action)
            if on_action is not None:
                on_action(
                    {
                        "source_step": step,
                        "predicted_action": a.tolist(),
                        "predicted_raw15": a.tolist() if a.shape[-1] == 15 else None,
                        "canonical_executed_raw15": canonical_raw15.tolist()
                        if local_memory_client is not None
                        else None,
                        "submitted_env12": np.asarray(env_action).tolist(),
                        "reward": float(reward),
                        "env_done": bool(env_done),
                        "evidence_completed": True,
                    }
                )
            if local_memory_client is not None:
                assert pre_composite is not None
                local_memory_client.record_completed(0, pre_composite, canonical_raw15)
            frames.append(compose(obs).astype(np.uint8))
            if check_success(env):
                streak += 1
                if streak >= latch:
                    success, done_steps = True, step + 1
                    break
            else:
                streak = 0

        return success, done_steps, prompt
    except BaseException as error:
        rollout_error = error
        raise
    finally:
        media_error = None
        try:
            save_partial_video(frames, save_video, video_fps)
            save_partial_video(gen_frames, gen_video_path, video_fps)
        except Exception as error:
            media_error = error
            print(f"[eval] partial-video failure: {type(error).__name__}: {error}", flush=True)
        if local_memory_client is not None:
            session_id = local_memory_client.end(0)
            if session_id is not None:
                try:
                    reset_local_memory(server_url, session_id, timeout)
                except Exception as cleanup_error:
                    if rollout_error is None:
                        raise
                    print(
                        f"[eval] WARNING: Local-TTT cleanup failed after rollout error: {cleanup_error}",
                        flush=True,
                    )
        if media_error is not None and rollout_error is None:
            raise media_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8912")
    ap.add_argument("--dataset-dir", required=True, help="v2.1 lerobot dir WITH extras/")
    ap.add_argument(
        "--num-test-episodes", type=int, default=0, help="test-scene episodes (random reset, same env_args)"
    )
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument(
        "--save-gen-video",
        action="store_true",
        help="save the model's GENERATED video (flow-matching vision branch, from the "
        "first inference) per rollout, for comparing video-generation quality",
    )
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--evaluation-run-digest", default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--cam-size", type=int, default=256)
    ap.add_argument(
        "--success-latch",
        type=int,
        default=1,
        help="consecutive _check_success() steps to declare success; 1 = official "
        "run_random_rollouts (first success). Higher only to reject transient flukes.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="fixed env seed for reproducible test scenes (same scenes across checkpoints); "
        "omit for official nondeterministic sampling",
    )
    ap.add_argument(
        "--camera-set",
        default="wrist_lr",
        choices=["wrist_lr", "left_wrist", "lrw"],
        help="wrist_lr = 3-cam squished (384x256); left_wrist = LIBERO-style [left|wrist] 256x512; "
        "lrw = 3-cam full-res side by side [left|right|wrist] 256x768",
    )
    ap.add_argument(
        "--base-encoding",
        choices=("ego", "raw"),
        default="ego",
        help="mobile-base action contract: 'ego' (20D state-derived pose delta, "
        "the original) or 'raw' (15D native base_motion command)",
    )
    ap.add_argument(
        "--use-base-action",
        action="store_true",
        help="20D mobile-base checkpoint: widen the state token and decode base_motion",
    )
    ap.add_argument(
        "--use-state",
        action="store_true",
        help="send EEF proprioception (robot0_base_to_eef_pos/quat + gripper_qpos) as the "
        "clean conditioning token; MUST match a use_state=True trained checkpoint",
    )
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument(
        "--local-memory-mode",
        choices=("off", "required"),
        default="off",
        help="V3 online Local-TTT. Formal iter500 evaluation must use required.",
    )
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    global CAMERA_SET, USE_BASE_ACTION, BASE_ENCODING
    CAMERA_SET = args.camera_set
    USE_BASE_ACTION = args.use_base_action
    BASE_ENCODING = args.base_encoding
    validate_local_memory_eval_contract(
        local_memory_mode=args.local_memory_mode,
        camera_set=args.camera_set,
        use_base_action=args.use_base_action,
        base_encoding=args.base_encoding,
        use_state=args.use_state,
        action_horizon=args.action_horizon,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.dataset_dir,
        args.cam_size,
        obj_split=ATOMIC_SEEN_OBJ_SPLIT,
        layout_style=ATOMIC_SEEN_LAYOUT_STYLE,
        seed=args.seed,
    )
    print(f"[eval] official protocol: rollouts={args.num_test_episodes} seed={args.seed}", flush=True)

    # Effective rollout horizon: official per-task (robocasa registry) or the fixed --max-steps.
    task_name = get_env_metadata(args.dataset_dir)["env_name"]
    h = official_horizon(task_name)
    eff_max_steps = h if h else args.max_steps
    print(f"[eval] {task_name}: official horizon {eff_max_steps}", flush=True)

    # wait for server
    import requests

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            info_response = requests.get(f"{args.server_url}/info", timeout=5)
            if info_response.ok:
                info = info_response.json()
                server_mode = (info.get("local_memory") or {}).get("mode", "off")
                if server_mode != args.local_memory_mode:
                    raise RuntimeError(
                        f"eval/server Local-TTT mode mismatch: eval={args.local_memory_mode} server={server_mode}"
                    )
                print(f"[train-scene] server ready at {args.server_url} local_memory={server_mode}", flush=True)
                break
        except requests.RequestException:
            time.sleep(3)
    else:
        raise RuntimeError("server not ready")

    results = []
    local_memory_client = RoboCasaLocalMemoryClient(enabled=True) if args.local_memory_mode == "required" else None

    # Keep the official sequential environment reset/rollout protocol.
    try:
        for t in range(args.num_test_episodes):
            result = run_recorded_episode(
                run_policy,
                env,
                output_dir=out,
                episode=t,
                run_digest=args.evaluation_run_digest,
                server_url=args.server_url,
                image_size=args.image_size,
                action_horizon=args.action_horizon,
                max_steps=eff_max_steps,
                latch=args.success_latch,
                timeout=args.timeout,
                use_state=args.use_state,
                local_memory_client=local_memory_client,
                save_png=str(out / f"rollout{t:02d}_init.png"),
                save_video=str(out / f"rollout{t:02d}.mp4"),
                gen_video_path=str(out / f"rollout{t:02d}_generated.mp4") if args.save_gen_video else None,
            )
            results.append(result)
            atomic_json(out / "results.json", results)
            print(
                f"[eval] rollout {t:02d} outcome={result['outcome']} steps={result['steps']} error={result['error']}",
                flush=True,
            )
    finally:
        env.close()

    n_ok = sum(1 for r in results if r["policy"])
    print(f"\n{task_name}: {n_ok}/{len(results)} successful rollouts", flush=True)
    atomic_json(out / "results.json", results)
    if any(row["error"] is not None for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

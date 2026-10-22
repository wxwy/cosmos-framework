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

The simulator and policy server may use one compatible interpreter or separate environments.
The manual launcher attests the selected simulator's actual imports and pinned vendor identity
before environment creation. Only empty namespace packages are imported from ``cosmos_framework``
here, so ``PYTHONPATH=.`` is enough and cosmos-framework need not be installed in the simulator venv.

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
import importlib.util
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
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
    # The strict launcher re-attests source identity before these vendor imports.
    import robocasa  # noqa: F401 (registers task environments)
    import robosuite

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


def verify_runtime(args, task_name: str) -> dict | None:
    """Recheck the actual Simulator interpreter before importing vendor/env code."""
    required = (args.runtime_probe_script, args.runtime_lock_file, args.expected_runtime_digest, args.runtime_tasks)
    if not any(required) and args.runtime_asset_manifest is None:
        return None  # Direct legacy entry; the unified launcher always supplies all fields.
    if not all(required):
        raise ValueError("runtime identity requires probe, lock, digest and the complete selected task set")
    tasks = args.runtime_tasks.split(",")
    if not all(tasks) or len(tasks) != len(set(tasks)) or task_name not in tasks:
        raise ValueError("runtime selected task identity mismatch")
    path = Path(args.runtime_probe_script).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("psm_robocasa_runtime_probe", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned runtime probe")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    runtime = probe.probe_runtime(
        lock_path=Path(args.runtime_lock_file), asset_manifest=args.runtime_asset_manifest, tasks=tasks
    )
    if runtime.get("digest") != args.expected_runtime_digest:
        raise ValueError("Simulator runtime changed after preflight")
    horizons = runtime.get("protocol", {}).get("official_horizons", {})
    if set(horizons) != set(tasks) or any(type(value) is not int or value <= 0 for value in horizons.values()):
        raise ValueError("runtime is missing authoritative official task horizons")
    return runtime


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
    generated_video_mode=None,
    generated_run_digest=None,
    generated_attempt_id=None,
    video_fps=20,
    local_memory_client: RoboCasaLocalMemoryClient | None = None,
    on_prediction: Callable[[dict, float], None] | None = None,
    on_action: Callable[[dict], None] | None = None,
    on_cleanup: Callable[[dict], None] | None = None,
) -> tuple[bool, int, str]:
    rollout_error: BaseException | None = None
    frames, generated = [], []
    mode = generated_video_mode or ("all" if gen_video_path is not None else "none")
    if mode not in {"none", "first", "all"} or (mode != "none" and gen_video_path is None):
        raise ValueError("generated video requires a valid mode and output path")
    try:
        obs = env.reset()
        if local_memory_client is not None:
            local_memory_client.begin(0, image_size=image_size)
        try:
            prompt = env.get_ep_meta().get("lang", "") or ""
        except Exception:
            prompt = ""
        init_comp = compose(obs).astype(np.uint8)
        if save_png is not None:
            with open(save_png, "xb") as handle:
                Image.fromarray(init_comp).save(handle, format="PNG")
        frames = [init_comp] if save_video is not None else []
        replans = 0
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
                if mode == "all" or (mode == "first" and replans == 0):
                    if not result.get("video"):
                        raise ValueError("enabled generated video is missing from model prediction")
                    generated_frames = list(decode_pred_video(result["video"]))
                    if not generated_frames:
                        raise ValueError("enabled generated video contains no decoded frames")
                    base = Path(gen_video_path)
                    target = base.with_name(f"{base.stem}_replan{replans:04d}{base.suffix}") if mode == "all" else base
                    save_partial_video(generated_frames, target, video_fps)
                    generated.append(
                        {"replan": replans, "source_step": step, "file": target.name, "frames": len(generated_frames)}
                    )
                    atomic_json(
                        base.with_name(base.stem + "_manifest.json"),
                        {
                            "schema": "psm_generated_video_v1",
                            "evaluation_run_digest": generated_run_digest,
                            "evaluation_attempt_id": generated_attempt_id,
                            "mode": mode,
                            "status": "IN_PROGRESS",
                            "videos": generated,
                        },
                    )
                replans += 1
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
            if save_video is not None:
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
        except Exception as error:
            media_error = error
            print(f"[eval] partial-video failure: {type(error).__name__}: {error}", flush=True)
        cleanup_error = None
        session_id = None
        try:
            if local_memory_client is not None:
                session_id = local_memory_client.end(0)
                if session_id is not None:
                    reset_local_memory(server_url, session_id, timeout)
        except Exception as error:
            cleanup_error = error
            print(f"[eval] Local-TTT cleanup failure: {type(error).__name__}: {error}", flush=True)
        witness = {
            "session_id": session_id,
            "reset_ok": cleanup_error is None,
            "rollout_error": repr(rollout_error) if rollout_error is not None else None,
            "media_error": repr(media_error) if media_error is not None else None,
            "cleanup_error": repr(cleanup_error) if cleanup_error is not None else None,
        }
        try:
            if on_cleanup is not None:
                on_cleanup(witness)
        except Exception as error:
            cleanup_error = cleanup_error or error
        if rollout_error is None:
            if media_error is not None:
                raise media_error
            if cleanup_error is not None:
                raise cleanup_error


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
        help="legacy alias for --save-generated-video first (per episode)",
    )
    ap.add_argument("--save-generated-video", choices=("none", "first", "all"), default=None)
    ap.add_argument("--no-save-rollout-mp4", action="store_true")
    ap.add_argument("--no-save-init-png", action="store_true")
    ap.add_argument("--runtime-probe-script", type=Path)
    ap.add_argument("--runtime-lock-file", type=Path)
    ap.add_argument("--runtime-asset-manifest", type=Path)
    ap.add_argument("--expected-runtime-digest")
    ap.add_argument("--runtime-tasks")
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--evaluation-run-digest", default=None)
    ap.add_argument("--evaluation-attempt-id", default=None)
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
    ap.add_argument("--progress-file", type=Path, default=None)
    args = ap.parse_args()
    if args.save_gen_video and args.save_generated_video not in (None, "first"):
        raise ValueError("--save-gen-video conflicts with --save-generated-video")
    video_mode = args.save_generated_video or ("first" if args.save_gen_video else "none")

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
    task_name = get_env_metadata(args.dataset_dir)["env_name"]
    if args.evaluation_attempt_id is not None and (
        args.evaluation_attempt_id != f"{task_name}/{out.name}" or out.parent.name != task_name
    ):
        raise ValueError("evaluation attempt identity must match the output namespace")
    out.mkdir(parents=True, exist_ok=True)

    runtime = verify_runtime(args, task_name)
    if runtime is not None:
        atomic_json(out / "runtime_attestation.json", runtime)
    # The verified registry already incorporates RoboCasa 1.0.1's 1.5x horizon update.
    h = runtime["protocol"]["official_horizons"][task_name] if runtime is not None else official_horizon(task_name)
    eff_max_steps = h if h else args.max_steps
    env = make_env(
        args.dataset_dir,
        args.cam_size,
        obj_split=ATOMIC_SEEN_OBJ_SPLIT,
        layout_style=ATOMIC_SEEN_LAYOUT_STYLE,
        seed=args.seed,
    )
    print(f"[eval] official protocol: rollouts={args.num_test_episodes} seed={args.seed}", flush=True)

    # Effective rollout horizon: official per-task (robocasa registry) or the fixed --max-steps.
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
        with (args.progress_file or out / "episode_progress.jsonl").open(
            "x", encoding="utf-8", buffering=1
        ) as progress:

            def emit(event, episode):
                progress.write(
                    json.dumps(
                        {
                            "schema": "psm_episode_progress_v1",
                            "event": event,
                            "episode": episode,
                            "evaluation_run_digest": args.evaluation_run_digest,
                            "evaluation_attempt_id": args.evaluation_attempt_id,
                        }
                    )
                    + "\n"
                )
                progress.flush()

            for t in range(args.num_test_episodes):
                emit("start", t)
                result = run_recorded_episode(
                    run_policy,
                    env,
                    output_dir=out,
                    episode=t,
                    run_digest=args.evaluation_run_digest,
                    attempt_id=args.evaluation_attempt_id,
                    server_url=args.server_url,
                    image_size=args.image_size,
                    action_horizon=args.action_horizon,
                    max_steps=eff_max_steps,
                    latch=args.success_latch,
                    timeout=args.timeout,
                    use_state=args.use_state,
                    local_memory_client=local_memory_client,
                    on_cleanup=lambda witness, episode=t: atomic_json(
                        out / f"rollout{episode:02d}_cleanup.json", witness
                    ),
                    save_png=str(out / f"rollout{t:02d}_init.png") if not args.no_save_init_png else None,
                    save_video=str(out / f"rollout{t:02d}.mp4") if not args.no_save_rollout_mp4 else None,
                    gen_video_path=str(out / f"rollout{t:02d}_generated.mp4") if video_mode != "none" else None,
                    generated_video_mode=video_mode,
                    generated_run_digest=args.evaluation_run_digest,
                    generated_attempt_id=args.evaluation_attempt_id,
                )
                generated_manifest = out / f"rollout{t:02d}_generated_manifest.json"
                result["generated_videos"] = []
                if generated_manifest.exists():
                    generated = json.loads(generated_manifest.read_text())
                    generated["status"] = "FAILED" if result["error"] is not None else "COMPLETE"
                    result["generated_videos"] = [video["file"] for video in generated["videos"]]
                    atomic_json(generated_manifest, generated)
                atomic_json(out / f"rollout{t:02d}_episode.json", result)
                results.append(result)
                atomic_json(out / "results.json", results)
                print(
                    f"[eval] rollout {t:02d} outcome={result['outcome']} steps={result['steps']} error={result['error']}",
                    flush=True,
                )
                emit("end", t)
                if result["error"] is not None:
                    break  # Do not register another episode after a failed cleanup/rollout.
    finally:
        env.close()

    n_ok = sum(1 for r in results if r["policy"])
    print(f"\n{task_name}: {n_ok}/{len(results)} successful rollouts", flush=True)
    atomic_json(out / "results.json", results)
    if any(row["error"] is not None for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

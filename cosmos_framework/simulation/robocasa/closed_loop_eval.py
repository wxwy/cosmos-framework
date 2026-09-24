"""Closed-loop RoboCasa evaluation for Cosmos action policies with online Local-TTT.

Official RoboCasa contracts used here:
- gym.make("robocasa/<task>", split=..., seed=...)
- get_task_horizon(task)
- success from info["success"]
- robocasa.utils.env_utils.convert_action for the simulator command.

PSM-WMA uses canonical 20-D model actions:
base ego delta9 + control_mode1 + EEF/gripper10.
This module decodes them into RoboCasa's 12-D controller action and records
Local-TTT evidence only after the environment step was actually executed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


def _to_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def compose_left_wrist(obs: dict[str, Any]) -> np.ndarray:
    left = _to_uint8(obs["video.robot0_agentview_left"])
    wrist = _to_uint8(obs["video.robot0_eye_in_hand"])
    if left.shape != wrist.shape:
        raise ValueError(f"left/wrist camera shapes differ: {left.shape} vs {wrist.shape}")
    return np.concatenate([left, wrist], axis=1)


def state16_from_observation(obs: dict[str, Any]) -> np.ndarray:
    parts = (
        np.asarray(obs["state.base_position"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_rotation"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.gripper_qpos"], dtype=np.float32).reshape(-1),
    )
    state = np.concatenate(parts)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(f"RoboCasa state must be finite [16], got {state.shape}")
    return state


def _rotation_matrix(value: np.ndarray, fmt: str) -> np.ndarray:
    matrix = convert_rotation(np.asarray(value, dtype=np.float32), input_format=fmt, output_format="matrix")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid rotation matrix from {fmt}: {matrix.shape}")
    return matrix


def canonical20_to_env12(
    action20: np.ndarray | list[float],
    *,
    base_decode_mode: str = "velocity",
    fps: float = 20.0,
) -> np.ndarray:
    """Decode canonical model action into RoboCasa's official flat env action.

    Canonical20:
      base ego translation3 + relative base rot6d + control1
      + EEF delta translation3 + EEF delta rot6d + gripper1.

    Env12:
      EEF xyz3 + EEF axis-angle3 + gripper1
      + base_motion4 + control_mode1.

    PandaOmron's base controller is velocity controlled. Training stores the
    observed per-step base state delta, so velocity mode divides by dt (i.e.
    multiplies by fps). delta sends per-step values directly; zero disables
    mobile-base motion for manipulation-only smoke tests.
    """
    action = np.asarray(action20, dtype=np.float32).reshape(-1)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError("canonical RoboCasa action must be finite [20]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if base_decode_mode not in {"velocity", "delta", "zero"}:
        raise ValueError("base_decode_mode must be velocity, delta, or zero")

    base_xyz = action[0:3]
    base_rot = _rotation_matrix(action[3:9], "rot6d")
    base_yaw = math.atan2(float(base_rot[1, 0]), float(base_rot[0, 0]))
    if base_decode_mode == "zero":
        base_motion = np.zeros(4, dtype=np.float32)
    else:
        base_motion = np.asarray([base_xyz[0], base_xyz[1], base_yaw, base_xyz[2]], dtype=np.float32)
        if base_decode_mode == "velocity":
            base_motion *= float(fps)

    eef_xyz = action[10:13]
    eef_rot = np.asarray(
        convert_rotation(action[13:19], input_format="rot6d", output_format="axisangle"),
        dtype=np.float32,
    ).reshape(-1)
    if eef_rot.shape != (3,) or not np.isfinite(eef_rot).all():
        raise ValueError("decoded EEF axis-angle must be finite [3]")

    control = np.float32(1.0 if float(action[9]) >= 0.0 else -1.0)
    env_action = np.concatenate(
        [eef_xyz, eef_rot, action[19:20], base_motion, np.asarray([control], dtype=np.float32)]
    ).astype(np.float32)
    if env_action.shape != (12,) or not np.isfinite(env_action).all():
        raise ValueError("decoded RoboCasa env action must be finite [12]")
    return env_action


def completed_action20(
    predicted_action20: np.ndarray | list[float],
    pre_state16: np.ndarray,
    post_state16: np.ndarray,
) -> np.ndarray:
    """Build training-equivalent completed evidence for an executed step.

    Arm/control channels come from the executed command. Base channels are
    recomputed from observed pre/post state exactly like the training loader's
    _build_base_delta path.
    """
    predicted = np.asarray(predicted_action20, dtype=np.float32).reshape(-1)
    pre = np.asarray(pre_state16, dtype=np.float32).reshape(-1)
    post = np.asarray(post_state16, dtype=np.float32).reshape(-1)
    if predicted.shape != (20,) or pre.shape != (16,) or post.shape != (16,):
        raise ValueError("completed_action20 expects action[20], pre_state[16], post_state[16]")
    if not (np.isfinite(predicted).all() and np.isfinite(pre).all() and np.isfinite(post).all()):
        raise ValueError("completed_action20 inputs must be finite")

    pre_rot = _rotation_matrix(pre[3:7], "quat_xyzw")
    post_rot = _rotation_matrix(post[3:7], "quat_xyzw")
    d_world = (post[0:3] - pre[0:3]).reshape(3, 1)
    d_ego = (pre_rot.T @ d_world).reshape(3)
    r_rel = pre_rot.T @ post_rot
    base_rot6d = np.asarray(
        convert_rotation(r_rel, input_format="matrix", output_format="rot6d"),
        dtype=np.float32,
    ).reshape(6)

    control = np.float32(1.0 if float(predicted[9]) >= 0.0 else -1.0)
    result = np.concatenate(
        [d_ego.astype(np.float32), base_rot6d, np.asarray([control]), predicted[10:20]]
    ).astype(np.float32)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError("completed RoboCasa evidence must be finite [20]")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_task_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def _task_names(task_sets: list[str], explicit: list[str] | None, task_indices: list[int] | None) -> list[str]:
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    if explicit:
        tasks = list(explicit)
    else:
        tasks: list[str] = []
        for task_set in task_sets:
            if task_set not in TASK_SET_REGISTRY:
                raise KeyError(f"unknown RoboCasa task set {task_set!r}; available={sorted(TASK_SET_REGISTRY)}")
            tasks.extend(list(TASK_SET_REGISTRY[task_set]))
    tasks = list(dict.fromkeys(tasks))
    if task_indices is not None:
        bad = [index for index in task_indices if index < 0 or index >= len(tasks)]
        if bad:
            raise IndexError(f"task indices out of range for {len(tasks)} tasks: {bad}")
        tasks = [tasks[index] for index in task_indices]
    return tasks


def _reset_remote_memory(client: Any, memory: RoboCasaLocalMemoryClient) -> None:
    session_id = memory.end(0)
    if session_id is None:
        return
    reply = client.infer({"_local_memory_command": "reset", "session_id": session_id})
    if not isinstance(reply, dict) or reply.get("status") != "reset":
        raise ValueError(f"policy server did not confirm Local Memory reset: {reply!r}")


def evaluate_task(
    *,
    env_name: str,
    split: str,
    num_trials: int,
    host: str,
    port: int,
    seed: int,
    replan_steps: int,
    output_dir: Path,
    base_decode_mode: str,
    local_memory: bool,
    save_videos: bool,
    video_fps: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa  # noqa: F401
    from openpi_client import websocket_client_policy
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    if split not in {"pretrain", "target"}:
        raise ValueError("split must be pretrain or target")
    if num_trials <= 0 or replan_steps <= 0:
        raise ValueError("num_trials and replan_steps must be positive")

    horizon = int(get_task_horizon(env_name))
    env = gym.make(
        f"robocasa/{env_name}",
        split=split,
        seed=seed,
        camera_widths=256,
        camera_heights=256,
    )
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    task_root = output_dir / env_name
    task_root.mkdir(parents=True, exist_ok=True)

    successes = 0
    episodes: list[dict[str, Any]] = []
    try:
        for trial in range(num_trials):
            obs, reset_info = env.reset()
            prompt = str(obs["annotation.human.task_description"])
            memory = RoboCasaLocalMemoryClient(enabled=local_memory)
            memory.begin(0)
            success = False
            error: str | None = None
            step_count = 0
            query_count = 0
            action_records: list[dict[str, Any]] = []
            prediction_records: list[dict[str, Any]] = []
            video_frames: list[np.ndarray] = []

            try:
                if save_videos:
                    video_frames.append(_to_uint8(env.render()))

                while step_count < horizon and not success:
                    query_image = compose_left_wrist(obs)
                    query_state = state16_from_observation(obs)
                    request: dict[str, Any] = {
                        "observation/image": query_image,
                        "observation/state": query_state,
                        "prompt": prompt,
                    }
                    payload = memory.payload(0)
                    if payload is not None:
                        request["local_memory"] = payload

                    result = client.infer(request)
                    if not isinstance(result, dict) or "action" not in result:
                        raise ValueError(f"malformed RoboCasa policy response: {type(result)!r}")
                    chunk = np.asarray(result["action"], dtype=np.float32)
                    if chunk.ndim != 2 or chunk.shape[1] != 20 or not np.isfinite(chunk).all():
                        raise ValueError(f"RoboCasa policy action must be finite [T,20], got {chunk.shape}")
                    if len(chunk) < replan_steps:
                        raise ValueError(
                            f"replan_steps={replan_steps} exceeds returned action chunk length={len(chunk)}"
                        )
                    if local_memory:
                        status = result.get("local_memory")
                        if not isinstance(status, dict):
                            raise ValueError("Local-TTT response is missing local_memory acknowledgement")
                        memory.acknowledge(0, status)

                    prediction_records.append(
                        {
                            "query_index": query_count,
                            "step_before_query": step_count,
                            "action_chunk": chunk.tolist(),
                            "local_memory": result.get("local_memory"),
                        }
                    )
                    query_count += 1

                    for action20 in chunk[:replan_steps]:
                        if step_count >= horizon or success:
                            break
                        pre_image = compose_left_wrist(obs)
                        pre_state = state16_from_observation(obs)
                        env12 = canonical20_to_env12(
                            action20,
                            base_decode_mode=base_decode_mode,
                            fps=20.0,
                        )
                        next_obs, reward, terminated, truncated, info = env.step(convert_action(env12))
                        post_state = state16_from_observation(next_obs)
                        completed20 = completed_action20(action20, pre_state, post_state)
                        memory.record_executed(0, pre_image, completed20)

                        step_count += 1
                        success = bool(info.get("success", False))
                        action_records.append(
                            {
                                "step": step_count - 1,
                                "predicted_canonical20": np.asarray(action20).tolist(),
                                "completed_canonical20": completed20.tolist(),
                                "env_action12": env12.tolist(),
                                "success_after_step": success,
                                "reward": float(reward),
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                            }
                        )
                        obs = next_obs
                        if save_videos and (step_count % 2 == 0 or success or step_count >= horizon):
                            video_frames.append(_to_uint8(env.render()))

                if success:
                    successes += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    _reset_remote_memory(client, memory)
                except Exception as reset_exc:
                    reset_error = f"{type(reset_exc).__name__}: {reset_exc}"
                    error = reset_error if error is None else f"{error}; reset_error={reset_error}"

            episode_payload = {
                "env_name": env_name,
                "split": split,
                "trial": trial,
                "prompt": prompt,
                "success": success,
                "steps": step_count,
                "queries": query_count,
                "horizon": horizon,
                "base_decode_mode": base_decode_mode,
                "local_memory": local_memory,
                "error": error,
                "reset_info": reset_info,
                "actions": action_records,
                "predictions": prediction_records,
            }
            _write_json_atomic(task_root / "episodes" / f"episode_{trial:03d}.json", episode_payload)
            if save_videos and video_frames:
                suffix = "success" if success else "failure"
                video_path = task_root / "videos" / f"episode_{trial:03d}_{suffix}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_path, video_frames, fps=video_fps)

            episodes.append(
                {
                    "trial": trial,
                    "success": success,
                    "steps": step_count,
                    "queries": query_count,
                    "error": error,
                }
            )
            print(
                f"[robocasa-eval] task={env_name} trial={trial} "
                f"success={int(success)} steps={step_count}/{horizon} error={error}",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = {
        "env_name": env_name,
        "split": split,
        "successes": successes,
        "episodes": num_trials,
        "success_rate": successes / num_trials,
        "horizon": horizon,
        "base_decode_mode": base_decode_mode,
        "local_memory": local_memory,
        "episode_results": episodes,
    }
    _write_json_atomic(task_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--task-sets", nargs="+", default=["atomic_seen"])
    parser.add_argument("--task-names", nargs="*", default=None)
    parser.add_argument("--task-indices", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-decode-mode", choices=("velocity", "delta", "zero"), default="velocity")
    parser.add_argument("--local-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    tasks = _task_names(
        task_sets=list(args.task_sets),
        explicit=args.task_names,
        task_indices=_parse_task_indices(args.task_indices),
    )
    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            raise ValueError("--max-tasks must be positive")
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("no RoboCasa tasks selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for env_name in tasks:
        summaries.append(
            evaluate_task(
                env_name=env_name,
                split=args.split,
                num_trials=args.num_trials,
                host=args.host,
                port=args.port,
                seed=args.seed,
                replan_steps=args.replan_steps,
                output_dir=args.output_dir,
                base_decode_mode=args.base_decode_mode,
                local_memory=bool(args.local_memory),
                save_videos=bool(args.save_videos),
                video_fps=args.video_fps,
            )
        )

    successes = sum(int(item["successes"]) for item in summaries)
    episodes = sum(int(item["episodes"]) for item in summaries)
    final = {
        "split": args.split,
        "task_sets": list(args.task_sets),
        "task_count": len(summaries),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "num_trials_per_task": args.num_trials,
        "replan_steps": args.replan_steps,
        "base_decode_mode": args.base_decode_mode,
        "local_memory": bool(args.local_memory),
        "tasks": summaries,
    }
    _write_json_atomic(args.output_dir / "summary.json", final)
    print(json.dumps(_jsonable(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


# Episode-held-out inverse calibration from formal target-atomic base-active frames
# (87,013 pairs; train/test split by episode). Maps canonical
# [ego_dx, ego_dy, relative_yaw] -> raw RoboCasa base_motion[0:3].
_ROBOCASA_BASE_CALIBRATION_A = np.asarray(
    [
        [28.84, 0.396, 0.0438],
        [-0.104, 29.53, -0.396],
        [-0.0951, -5.776, 15.91],
    ],
    dtype=np.float32,
)
_ROBOCASA_BASE_CALIBRATION_B = np.asarray([0.039, -0.0076, -0.0014], dtype=np.float32)
_ROBOCASA_BASE_DECODER_VERSION = "target_atomic_calibrated_linear_v1_20260924"
_ROBOCASA_BASE_YAW_LEVER_ARM_M = 0.21
_ROBOCASA_BASE_SIDE_DEADZONE_CMD = 0.25
_ROBOCASA_RGB_CONTRACT = "training_matched_vertical_flip_v1"


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


def _annotate_frame_number(
    frame: Image.Image,
    frame_index: int,
    *,
    label: str = "",
    instruction: str = "",
    border_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Mirror LIBERO video annotation: instruction left, step/label right, optional 3px border."""
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    bar_h = 24
    draw.rectangle([(0, 0), (w, bar_h)], fill=(0, 0, 0))

    if instruction:
        instr = instruction.strip()
        if len(instr) > 80:
            instr = instr[:77] + "..."
        draw.text((4, 5), instr, fill=(255, 255, 255))

    frame_text = f"step {frame_index:04d}"
    if label:
        frame_text += f"  {label}"
    bbox = draw.textbbox((0, 0), frame_text)
    text_w = bbox[2] - bbox[0]
    draw.text((w - text_w - 4, 5), frame_text, fill=(255, 255, 0))

    if border_color is not None:
        for i in range(3):
            draw.rectangle([(i, i), (w - 1 - i, h - 1 - i)], outline=border_color)
    return annotated


def _save_mp4(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    """Mirror LIBERO MP4 writer: libx264/yuv420p first, OpenCV mp4v fallback."""
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = np.asarray(frames[0]).shape[:2]

    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(max(1, int(fps))),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            input=b"".join(np.asarray(frame).tobytes() for frame in frames),
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return

    import cv2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (w, h),
    )
    if not writer.isOpened():
        return
    try:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(arr)
    finally:
        writer.release()


def _rename_with_outcome(path: Path, success: bool) -> None:
    """Append _success or _fail to a media file or directory, matching LIBERO."""
    if not path.exists():
        return
    suffix = "success" if success else "fail"
    if path.is_dir():
        new_path = path.with_name(f"{path.name}_{suffix}")
    else:
        new_path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    path.rename(new_path)


def _prediction_frames_from_result(video: Any) -> list[Image.Image]:
    arr = np.asarray(video)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"predicted RoboCasa rollout must have shape [T,H,W,3], got {arr.shape}")
    return [Image.fromarray(_to_uint8(frame), mode="RGB") for frame in arr]


def _save_prediction_videos(
    prediction_videos: list[tuple[int, Image.Image, list[Image.Image], str]],
    output_dir: Path,
    fps: int,
) -> None:
    """Mirror LIBERO prediction videos: green input + red imagined frames, per query and combined."""
    if not prediction_videos:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: list[Image.Image] = []
    for step, input_frame, action_frames, task_description in prediction_videos:
        frames: list[Image.Image] = []
        output_size = input_frame.size
        frames.append(
            _annotate_frame_number(
                input_frame,
                step,
                label="input",
                instruction=task_description,
                border_color=(0, 255, 0),
            )
        )
        pred_frames = action_frames[1:] if len(action_frames) > 1 else action_frames
        n_pred = len(pred_frames)
        for index, pred_frame in enumerate(pred_frames, start=1):
            if pred_frame.size != output_size:
                pred_frame = pred_frame.resize(output_size, Image.Resampling.BILINEAR)
            frames.append(
                _annotate_frame_number(
                    pred_frame,
                    step,
                    label=f"pred {index:02d}/{n_pred:02d}",
                    instruction=task_description,
                    border_color=(255, 0, 0),
                )
            )
        _save_mp4(frames, output_dir / f"predict_step{step:06d}.mp4", fps)
        combined.extend(frames)
    _save_mp4(combined, output_dir / "predict_combined.mp4", fps)


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
    base_decode_mode: str = "calibrated",
    fps: float = 20.0,
) -> np.ndarray:
    """Decode canonical model action into RoboCasa's official flat env action.

    Canonical20:
      base ego translation3 + relative base rot6d + control1
      + EEF delta translation3 + EEF delta rot6d + gripper1.

    Env12:
      EEF xyz3 + EEF axis-angle3 + gripper1
      + base_motion4 + control_mode1.

    calibrated is the formal default: an episode-held-out linear inverse maps
    [ego_dx, ego_dy, relative_yaw] back to raw base_motion[0:3], clamps those
    normalized controller commands to [-1,1], and fixes base_motion[3] to 0
    because that channel is identically zero in the formal training set.

    Legacy diagnostic modes remain available:
    - velocity: [ego_dx, ego_dy, relative_yaw, base_dz] * fps
    - delta:    [ego_dx, ego_dy, relative_yaw, base_dz]
    - zero:     disable mobile-base motion
    """
    action = np.asarray(action20, dtype=np.float32).reshape(-1)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError("canonical RoboCasa action must be finite [20]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if base_decode_mode not in {"calibrated", "velocity", "delta", "zero"}:
        raise ValueError("base_decode_mode must be calibrated, velocity, delta, or zero")

    base_xyz = action[0:3]
    base_rot = _rotation_matrix(action[3:9], "rot6d")
    base_yaw = math.atan2(float(base_rot[1, 0]), float(base_rot[0, 0]))
    control = np.float32(1.0 if float(action[9]) >= 0.0 else -1.0)

    if base_decode_mode == "zero" or (base_decode_mode == "calibrated" and control < 0):
        base_motion = np.zeros(4, dtype=np.float32)
    elif base_decode_mode == "calibrated":
        canonical_base = np.asarray([base_xyz[0], base_xyz[1], base_yaw], dtype=np.float32)
        calibrated = _ROBOCASA_BASE_CALIBRATION_A @ canonical_base + _ROBOCASA_BASE_CALIBRATION_B
        calibrated = np.clip(calibrated, -1.0, 1.0).astype(np.float32)
        base_motion = np.asarray([calibrated[0], calibrated[1], calibrated[2], 0.0], dtype=np.float32)
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

    This function is training-parity evidence. Do not subtract yaw/reference-
    point coupling from completed ego-dy here: the formal training labels
    contain that coupling. Physics decontamination belongs only in diagnostics.
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


def _relative_yaw_from_rot6d(rot6d: np.ndarray | list[float]) -> float:
    matrix = _rotation_matrix(np.asarray(rot6d, dtype=np.float32), "rot6d")
    return float(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def _nonzero_sign_match(a: float, b: float, *, eps: float = 1e-6) -> bool | None:
    if abs(a) <= eps or abs(b) <= eps:
        return None
    return bool((a > 0.0) == (b > 0.0))


def base_step_diagnostics(
    predicted_action20: np.ndarray | list[float],
    completed_action20_value: np.ndarray | list[float],
    env_action12: np.ndarray | list[float],
) -> dict[str, Any]:
    """Evaluation-only base diagnostics; never used as Local-TTT evidence."""
    predicted = np.asarray(predicted_action20, dtype=np.float32).reshape(-1)
    completed = np.asarray(completed_action20_value, dtype=np.float32).reshape(-1)
    env12 = np.asarray(env_action12, dtype=np.float32).reshape(-1)
    if predicted.shape != (20,) or completed.shape != (20,) or env12.shape != (12,):
        raise ValueError("base diagnostics require predicted[20], completed[20], env_action[12]")
    if not (np.isfinite(predicted).all() and np.isfinite(completed).all() and np.isfinite(env12).all()):
        raise ValueError("base diagnostics require finite inputs")

    predicted_yaw = _relative_yaw_from_rot6d(predicted[3:9])
    completed_yaw = _relative_yaw_from_rot6d(completed[3:9])
    control_mode = float(env12[11])
    base_active = control_mode > 0.0
    decoded_base_motion = env12[7:11].astype(np.float32)
    yaw_coupling_dy = _ROBOCASA_BASE_YAW_LEVER_ARM_M * completed_yaw
    decontaminated_side_dy = float(completed[1]) - yaw_coupling_dy

    return {
        "base_decoder_version": _ROBOCASA_BASE_DECODER_VERSION,
        "control_mode": control_mode,
        "base_active": base_active,
        "predicted_ego_dx": float(predicted[0]),
        "predicted_ego_dy": float(predicted[1]),
        "predicted_ego_dz": float(predicted[2]),
        "predicted_relative_yaw": predicted_yaw,
        "decoded_base_motion": decoded_base_motion.tolist(),
        "completed_ego_dx": float(completed[0]),
        "completed_ego_dy": float(completed[1]),
        "completed_ego_dz": float(completed[2]),
        "completed_relative_yaw": completed_yaw,
        "yaw_coupling_side_dy": yaw_coupling_dy,
        "decontaminated_side_dy": decontaminated_side_dy,
        "side_deadzone_threshold": _ROBOCASA_BASE_SIDE_DEADZONE_CMD,
        "side_command_in_deadzone": bool(
            base_active and abs(float(decoded_base_motion[1])) < _ROBOCASA_BASE_SIDE_DEADZONE_CMD
        ),
        "predicted_dy_vs_completed_dy_sign_match": _nonzero_sign_match(
            float(predicted[1]), float(completed[1])
        ),
        "predicted_dy_vs_decontaminated_dy_sign_match": _nonzero_sign_match(
            float(predicted[1]), decontaminated_side_dy
        ),
        "decoded_bm1_vs_completed_dy_sign_match": _nonzero_sign_match(
            float(decoded_base_motion[1]), float(completed[1])
        ),
        "decoded_bm1_vs_decontaminated_dy_sign_match": _nonzero_sign_match(
            float(decoded_base_motion[1]), decontaminated_side_dy
        ),
    }


def _sign_summary(rows: list[dict[str, Any]], key: str) -> dict[str, int | float | None]:
    valid = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    matches = sum(int(value) for value in valid)
    return {
        "matches": matches,
        "count": len(valid),
        "rate": matches / len(valid) if valid else None,
    }


def summarize_base_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_rows = [row for row in rows if bool(row.get("base_active"))]
    arm_rows = [row for row in rows if not bool(row.get("base_active"))]
    deadzone_count = sum(int(bool(row.get("side_command_in_deadzone"))) for row in base_rows)

    decoded = (
        np.asarray([row["decoded_base_motion"][:3] for row in base_rows], dtype=np.float32)
        if base_rows
        else np.empty((0, 3), dtype=np.float32)
    )
    if decoded.size:
        decoded_stats: dict[str, Any] = {
            "mean": decoded.mean(axis=0).tolist(),
            "std": decoded.std(axis=0).tolist(),
            "min": decoded.min(axis=0).tolist(),
            "max": decoded.max(axis=0).tolist(),
        }
    else:
        decoded_stats = {"mean": None, "std": None, "min": None, "max": None}

    return {
        "base_decoder_version": _ROBOCASA_BASE_DECODER_VERSION,
        "rgb_contract": _ROBOCASA_RGB_CONTRACT,
        "yaw_lever_arm_m": _ROBOCASA_BASE_YAW_LEVER_ARM_M,
        "side_deadzone_threshold": _ROBOCASA_BASE_SIDE_DEADZONE_CMD,
        "steps_total": len(rows),
        "base_active_steps": len(base_rows),
        "arm_active_steps": len(arm_rows),
        "side_deadzone_steps": deadzone_count,
        "side_deadzone_fraction": deadzone_count / len(base_rows) if base_rows else None,
        "decoded_bm012_stats": decoded_stats,
        "predicted_dy_vs_completed_dy": _sign_summary(
            base_rows, "predicted_dy_vs_completed_dy_sign_match"
        ),
        "predicted_dy_vs_decontaminated_dy": _sign_summary(
            base_rows, "predicted_dy_vs_decontaminated_dy_sign_match"
        ),
        "decoded_bm1_vs_completed_dy": _sign_summary(
            base_rows, "decoded_bm1_vs_completed_dy_sign_match"
        ),
        "decoded_bm1_vs_decontaminated_dy": _sign_summary(
            base_rows, "decoded_bm1_vs_decontaminated_dy_sign_match"
        ),
        "arm_active_nonzero_base_steps": sum(
            int(any(abs(float(v)) > 1e-6 for v in row["decoded_base_motion"]))
            for row in arm_rows
        ),
        "bm3_nonzero_steps": sum(
            int(abs(float(row["decoded_base_motion"][3])) > 1e-6) for row in rows
        ),
    }


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
    save_pred_mp4: bool,
    video_fps: int,
) -> dict[str, Any]:
    import gymnasium as gym
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
    mp4_task_root = output_dir / "mp4" / f"task_{env_name}"
    mp4_pred_task_root = output_dir / "mp4_pred" / f"task_{env_name}"

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
            video_frames: list[Image.Image] = []
            prediction_videos: list[tuple[int, Image.Image, list[Image.Image], str]] = []
            base_diagnostics: list[dict[str, Any]] = []

            try:
                if save_videos:
                    initial_frame = Image.fromarray(compose_left_wrist(obs), mode="RGB")
                    video_frames.append(
                        _annotate_frame_number(initial_frame, 0, instruction=prompt)
                    )

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

                    predicted_video = result.get("video")
                    predicted_frames: list[Image.Image] = []
                    if predicted_video is not None:
                        predicted_frames = _prediction_frames_from_result(predicted_video)
                    if save_pred_mp4:
                        if not predicted_frames:
                            raise ValueError(
                                "save_pred_mp4 requested but policy server returned no predicted rollout video"
                            )
                        prediction_videos.append(
                            (
                                step_count,
                                Image.fromarray(query_image, mode="RGB"),
                                predicted_frames,
                                prompt,
                            )
                        )

                    prediction_records.append(
                        {
                            "query_index": query_count,
                            "step_before_query": step_count,
                            "action_chunk": chunk.tolist(),
                            "local_memory": result.get("local_memory"),
                            "video_frames": len(predicted_frames),
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
                        diagnostics = base_step_diagnostics(action20, completed20, env12)
                        base_diagnostics.append(diagnostics)
                        memory.record_executed(0, pre_image, completed20)

                        step_count += 1
                        success = bool(info.get("success", False))
                        action_records.append(
                            {
                                "step": step_count - 1,
                                "predicted_canonical20": np.asarray(action20).tolist(),
                                "completed_canonical20": completed20.tolist(),
                                "env_action12": env12.tolist(),
                                "base_diagnostics": diagnostics,
                                "success_after_step": success,
                                "reward": float(reward),
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                            }
                        )
                        obs = next_obs
                        if save_videos:
                            env_frame = Image.fromarray(compose_left_wrist(obs), mode="RGB")
                            video_frames.append(
                                _annotate_frame_number(env_frame, step_count, instruction=prompt)
                            )

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
                "base_decoder_version": _ROBOCASA_BASE_DECODER_VERSION,
                "rgb_contract": _ROBOCASA_RGB_CONTRACT,
                "base_diagnostics_summary": summarize_base_diagnostics(base_diagnostics),
                "error": error,
                "reset_info": reset_info,
                "actions": action_records,
                "predictions": prediction_records,
            }
            _write_json_atomic(task_root / "episodes" / f"episode_{trial:03d}.json", episode_payload)

            if save_videos and video_frames:
                mp4_path = mp4_task_root / f"episode_{trial:03d}.mp4"
                _save_mp4(video_frames, mp4_path, video_fps)
                _rename_with_outcome(mp4_path, success)

            if save_pred_mp4 and prediction_videos:
                pred_dir = mp4_pred_task_root / f"episode_{trial:03d}"
                _save_prediction_videos(prediction_videos, pred_dir, video_fps)
                _rename_with_outcome(pred_dir, success)

            episodes.append(
                {
                    "trial": trial,
                    "success": success,
                    "steps": step_count,
                    "queries": query_count,
                    "base_diagnostics_summary": summarize_base_diagnostics(base_diagnostics),
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
        "base_decoder_version": _ROBOCASA_BASE_DECODER_VERSION,
        "rgb_contract": _ROBOCASA_RGB_CONTRACT,
        "local_memory": local_memory,
        "save_pred_mp4": save_pred_mp4,
        "base_diagnostics": {
            "base_active_steps": sum(
                int(item["base_diagnostics_summary"]["base_active_steps"]) for item in episodes
            ),
            "arm_active_steps": sum(
                int(item["base_diagnostics_summary"]["arm_active_steps"]) for item in episodes
            ),
            "side_deadzone_steps": sum(
                int(item["base_diagnostics_summary"]["side_deadzone_steps"]) for item in episodes
            ),
            "bm3_nonzero_steps": sum(
                int(item["base_diagnostics_summary"]["bm3_nonzero_steps"]) for item in episodes
            ),
            "arm_active_nonzero_base_steps": sum(
                int(item["base_diagnostics_summary"]["arm_active_nonzero_base_steps"]) for item in episodes
            ),
        },
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
    parser.add_argument("--base-decode-mode", choices=("calibrated", "velocity", "delta", "zero"), default="calibrated")
    parser.add_argument("--local-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pred-mp4", action=argparse.BooleanOptionalAction, default=False)
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
                save_pred_mp4=bool(args.save_pred_mp4),
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
        "base_decoder_version": _ROBOCASA_BASE_DECODER_VERSION,
        "rgb_contract": _ROBOCASA_RGB_CONTRACT,
        "local_memory": bool(args.local_memory),
        "save_pred_mp4": bool(args.save_pred_mp4),
        "tasks": summaries,
    }
    _write_json_atomic(args.output_dir / "summary.json", final)
    print(json.dumps(_jsonable(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

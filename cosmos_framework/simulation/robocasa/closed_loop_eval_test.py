from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.simulation.robocasa import closed_loop_eval as robocasa_eval
from cosmos_framework.simulation.robocasa.closed_loop_eval import (
    _annotate_frame_number,
    _prediction_frames_from_result,
    _rename_with_outcome,
    _save_prediction_videos,
    base_step_diagnostics,
    canonical20_to_env12,
    completed_action20,
    compose_left_wrist,
    state16_from_observation,
    summarize_base_diagnostics,
)
from cosmos_framework.simulation.robocasa.local_memory_client import RoboCasaLocalMemoryClient


def _identity_rot6d() -> np.ndarray:
    return np.asarray(
        convert_rotation(np.eye(3, dtype=np.float32), input_format="matrix", output_format="rot6d"),
        dtype=np.float32,
    ).reshape(6)


def _state(base_pos=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray(
        [
            *base_pos,
            0.0,
            0.0,
            0.0,
            1.0,
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.02,
            -0.02,
        ],
        dtype=np.float32,
    )


def test_compose_left_wrist_and_state16() -> None:
    left = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.ones((256, 256, 3), dtype=np.uint8)
    obs = {
        "video.robot0_agentview_left": left,
        "video.robot0_eye_in_hand": wrist,
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "state.end_effector_position_relative": np.asarray([1, 2, 3], dtype=np.float32),
        "state.end_effector_rotation_relative": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "state.gripper_qpos": np.asarray([0.1, -0.1], dtype=np.float32),
    }
    image = compose_left_wrist(obs)
    state = state16_from_observation(obs)
    assert image.shape == (256, 512, 3)
    assert np.array_equal(image[:, :256], left)
    assert np.array_equal(image[:, 256:], wrist)
    assert state.shape == (16,)
    assert np.allclose(state[:7], [0, 0, 0, 0, 0, 0, 1])


def test_canonical20_to_env12_identity_rotations() -> None:
    rot6d = _identity_rot6d()
    action = np.zeros(20, dtype=np.float32)
    action[3:9] = rot6d
    action[9] = -1.0
    action[10:13] = [0.1, -0.2, 0.3]
    action[13:19] = rot6d
    action[19] = 0.7

    env = canonical20_to_env12(action, base_decode_mode="zero")
    assert env.shape == (12,)
    assert np.allclose(env[:3], [0.1, -0.2, 0.3])
    assert np.allclose(env[3:6], 0.0, atol=1e-6)
    assert np.isclose(env[6], 0.7)
    assert np.allclose(env[7:11], 0.0)
    assert np.isclose(env[11], -1.0)


def test_calibrated_base_decoder_matches_heldout_inverse_and_zeroes_bm3() -> None:
    yaw = 0.03
    base_rot6d = np.asarray(
        convert_rotation(
            np.asarray([0.0, 0.0, yaw], dtype=np.float32),
            input_format="axisangle",
            output_format="rot6d",
        ),
        dtype=np.float32,
    ).reshape(6)
    eef_rot6d = _identity_rot6d()

    action = np.zeros(20, dtype=np.float32)
    action[0:3] = [0.01, 0.02, 0.5]  # base z must be ignored by calibrated decoder
    action[3:9] = base_rot6d
    action[9] = 1.0
    action[13:19] = eef_rot6d

    env = canonical20_to_env12(action, base_decode_mode="calibrated")

    x = np.asarray([0.01, 0.02, yaw], dtype=np.float32)
    matrix = np.asarray(
        [
            [28.84, 0.396, 0.0438],
            [-0.104, 29.53, -0.396],
            [-0.0951, -5.776, 15.91],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.039, -0.0076, -0.0014], dtype=np.float32)
    expected = np.clip(matrix @ x + bias, -1.0, 1.0)

    np.testing.assert_allclose(env[7:10], expected, rtol=0, atol=1e-6)
    assert env[10] == 0.0
    assert env[11] == 1.0


def test_calibrated_base_decoder_keeps_arm_active_base_inactive() -> None:
    action = np.zeros(20, dtype=np.float32)
    action[0:3] = [0.02, -0.03, 0.4]
    action[3:9] = _identity_rot6d()
    action[9] = -1.0
    action[13:19] = _identity_rot6d()

    env = canonical20_to_env12(action, base_decode_mode="calibrated")

    np.testing.assert_allclose(env[7:11], 0.0, rtol=0, atol=0)
    assert env[11] == -1.0


def test_completed_action20_recomputes_observed_base_delta() -> None:
    rot6d = _identity_rot6d()
    predicted = np.zeros(20, dtype=np.float32)
    predicted[3:9] = rot6d
    predicted[9] = 1.0
    predicted[13:19] = rot6d
    predicted[19] = 0.2

    pre = _state((1.0, 2.0, 3.0))
    post = _state((1.1, 1.8, 3.05))
    completed = completed_action20(predicted, pre, post)

    assert completed.shape == (20,)
    assert np.allclose(completed[:3], [0.1, -0.2, 0.05], atol=1e-6)
    assert np.allclose(completed[3:9], rot6d, atol=1e-6)
    assert np.isclose(completed[9], 1.0)
    assert np.allclose(completed[10:], predicted[10:])


def test_base_step_diagnostics_decontaminates_yaw_only_for_metrics() -> None:
    predicted = np.zeros(20, dtype=np.float32)
    predicted[3:9] = _identity_rot6d()
    predicted[9] = 1.0
    predicted[13:19] = _identity_rot6d()
    predicted[1] = 0.01

    yaw = 0.02
    completed = predicted.copy()
    completed[1] = 0.0062  # includes 0.21 * 0.02 = 0.0042 yaw coupling + 0.002 true side
    completed[3:9] = np.asarray(
        convert_rotation(
            np.asarray([0.0, 0.0, yaw], dtype=np.float32),
            input_format="axisangle",
            output_format="rot6d",
        ),
        dtype=np.float32,
    ).reshape(6)

    env12 = np.zeros(12, dtype=np.float32)
    env12[7:11] = [0.1, 0.12, 0.2, 0.0]
    env12[11] = 1.0

    before = completed.copy()
    diag = base_step_diagnostics(predicted, completed, env12)

    assert np.array_equal(completed, before)
    assert diag["base_active"] is True
    assert np.isclose(diag["predicted_control_value"], 1.0)
    assert diag["control_mode_threshold"] == 0.0
    assert np.isclose(diag["completed_relative_yaw"], yaw, atol=1e-6)
    assert np.isclose(diag["yaw_coupling_side_dy"], 0.0042, atol=1e-6)
    assert np.isclose(diag["decontaminated_side_dy"], 0.0020, atol=1e-6)
    assert diag["side_command_in_deadzone"] is True
    assert diag["predicted_dy_vs_completed_dy_sign_match"] is True
    assert diag["predicted_dy_vs_decontaminated_dy_sign_match"] is True


def test_summarize_base_diagnostics_locks_contract_counts() -> None:
    rows = [
        {
            "base_active": True,
            "predicted_control_value": 0.2,
            "decoded_base_motion": [0.1, 0.12, 0.2, 0.0],
            "side_command_in_deadzone": True,
            "predicted_dy_vs_completed_dy_sign_match": False,
            "predicted_dy_vs_decontaminated_dy_sign_match": True,
            "decoded_bm1_vs_completed_dy_sign_match": False,
            "decoded_bm1_vs_decontaminated_dy_sign_match": True,
        },
        {
            "base_active": True,
            "predicted_control_value": 0.8,
            "decoded_base_motion": [-0.4, 0.4, -0.1, 0.0],
            "side_command_in_deadzone": False,
            "predicted_dy_vs_completed_dy_sign_match": True,
            "predicted_dy_vs_decontaminated_dy_sign_match": True,
            "decoded_bm1_vs_completed_dy_sign_match": True,
            "decoded_bm1_vs_decontaminated_dy_sign_match": True,
        },
        {
            "base_active": False,
            "predicted_control_value": -0.6,
            "decoded_base_motion": [0.0, 0.0, 0.0, 0.0],
            "side_command_in_deadzone": False,
            "predicted_dy_vs_completed_dy_sign_match": None,
            "predicted_dy_vs_decontaminated_dy_sign_match": None,
            "decoded_bm1_vs_completed_dy_sign_match": None,
            "decoded_bm1_vs_decontaminated_dy_sign_match": None,
        },
    ]

    summary = summarize_base_diagnostics(rows)

    assert summary["base_active_steps"] == 2
    assert summary["arm_active_steps"] == 1
    assert summary["side_deadzone_steps"] == 1
    assert summary["side_deadzone_fraction"] == 0.5
    assert summary["base_active_fraction"] == pytest.approx(2 / 3)
    assert summary["predicted_control_value_stats"]["positive_fraction"] == pytest.approx(2 / 3)
    assert summary["predicted_control_value_stats"]["min"] == pytest.approx(-0.6)
    assert summary["predicted_control_value_stats"]["max"] == pytest.approx(0.8)
    assert summary["bm3_nonzero_steps"] == 0
    assert summary["arm_active_nonzero_base_steps"] == 0
    assert summary["predicted_dy_vs_completed_dy"] == {
        "matches": 1,
        "count": 2,
        "rate": 0.5,
    }
    assert summary["predicted_dy_vs_decontaminated_dy"] == {
        "matches": 2,
        "count": 2,
        "rate": 1.0,
    }


def test_robocasa_local_memory_frontier_ack_and_reset() -> None:
    memory = RoboCasaLocalMemoryClient(enabled=True)
    memory.begin(0)
    initial = memory.payload(0)
    assert initial is not None
    assert initial["consumer_step"] == 0
    assert initial["reset"] is True
    assert initial["evidence_version"] == "causal_visual96_executed_action20_v1"
    assert initial["evidence_format"] == "robocasa_rgb_action20_v1"

    image = np.zeros((256, 512, 3), dtype=np.uint8)
    action = np.zeros(20, dtype=np.float32)
    memory.record_executed(0, image, action)
    payload = memory.payload(0)
    assert payload is not None
    assert payload["consumer_step"] == 1
    assert len(payload["evidence"]) == 1

    memory.acknowledge(
        0,
        {
            "session_id": payload["session_id"],
            "episode_id": payload["episode_id"],
            "consumer_step": payload["consumer_step"],
        },
    )
    after = memory.payload(0)
    assert after is not None
    assert after["reset"] is False
    assert after["consumer_step"] == 1
    assert after["evidence"] == []
    assert memory.end(0) == payload["session_id"]


def test_annotate_frame_number_matches_libero_overlay_contract() -> None:
    frame = Image.fromarray(np.full((64, 128, 3), 127, dtype=np.uint8), mode="RGB")
    annotated = _annotate_frame_number(
        frame,
        12,
        label="input",
        instruction="pick up the bowl",
        border_color=(0, 255, 0),
    )
    arr = np.asarray(annotated)
    assert annotated.size == frame.size
    assert np.array_equal(arr[0, 0], [0, 255, 0])
    # The top information bar is black away from text and the 3px border.
    assert np.array_equal(arr[20, 64], [0, 0, 0])


def test_prediction_video_writer_uses_green_input_red_predictions_and_libero_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, list[Image.Image], int]] = []

    def fake_save_mp4(frames: list[Image.Image], output_path, fps: int) -> None:
        calls.append((output_path.name, list(frames), fps))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")

    monkeypatch.setattr(robocasa_eval, "_save_mp4", fake_save_mp4)

    input_frame = Image.fromarray(np.zeros((64, 128, 3), dtype=np.uint8), mode="RGB")
    rollout = np.stack(
        [np.full((32, 64, 3), index, dtype=np.uint8) for index in range(17)],
        axis=0,
    )
    predicted = _prediction_frames_from_result(rollout)
    output_dir = tmp_path / "mp4_pred" / "task_TestTask" / "episode_000"
    _save_prediction_videos(
        [(7, input_frame, predicted, "test instruction")],
        output_dir,
        20,
    )

    assert [name for name, _, _ in calls] == [
        "predict_step000007.mp4",
        "predict_combined.mp4",
    ]
    per_query_frames = calls[0][1]
    assert len(per_query_frames) == 17
    assert np.array_equal(np.asarray(per_query_frames[0])[0, 0], [0, 255, 0])
    assert np.array_equal(np.asarray(per_query_frames[1])[0, 0], [255, 0, 0])
    assert per_query_frames[1].size == input_frame.size
    assert calls[0][2] == 20
    assert len(calls[1][1]) == 17


def test_rename_with_outcome_matches_libero_file_and_directory_suffixes(tmp_path) -> None:
    file_path = tmp_path / "episode_003.mp4"
    file_path.write_bytes(b"x")
    _rename_with_outcome(file_path, True)
    assert (tmp_path / "episode_003_success.mp4").is_file()

    directory = tmp_path / "episode_004"
    directory.mkdir()
    _rename_with_outcome(directory, False)
    assert (tmp_path / "episode_004_fail").is_dir()

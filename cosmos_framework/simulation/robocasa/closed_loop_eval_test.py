from __future__ import annotations

import numpy as np

from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.simulation.robocasa.closed_loop_eval import (
    canonical20_to_env12,
    completed_action20,
    compose_left_wrist,
    state16_from_observation,
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

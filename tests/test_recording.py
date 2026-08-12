from __future__ import annotations

import time

import h5py
import numpy as np

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.data.recording import (
    EpisodeRecorder,
    FramePayload,
    RenderedFrame,
    TeleopSnapshot,
    build_state_26,
)


def _teleop_snapshot() -> TeleopSnapshot:
    return TeleopSnapshot(
        raw=np.zeros(9, dtype=np.float64),
        raw_valid=True,
        aligned=True,
        raw_source_timestamp=1.0,
        raw_age_ms=2.0,
        user_command=np.zeros(8, dtype=np.float64),
        user_command_valid=True,
        user_command_source_timestamp=1.0,
        user_command_age_ms=2.0,
        control_orientation=True,
    )


def test_state_26_has_documented_shape_and_task_geometry() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        observation, _ = env.reset(
            seed=0,
            options={"randomize_object": False, "randomize_goal": False},
        )
        state = build_state_26(observation)
        assert state.shape == (26,)
        assert state.dtype == np.float32
        np.testing.assert_allclose(state[:7], observation["q_obs"])
        np.testing.assert_allclose(state[18:21], observation["object_pose"][:3, 3])
        np.testing.assert_allclose(state[21:24], observation["goal_pose"][:3, 3])
        np.testing.assert_allclose(
            state[25],
            np.linalg.norm(
                observation["object_pose"][:3, 3]
                - observation["goal_pose"][:3, 3]
            ),
        )
    finally:
        env.close()


def test_episode_recorder_writes_complete_synchronized_hdf5(tmp_path) -> None:
    env = PickPlaceEnv(enable_camera=False)
    recorder = EpisodeRecorder(tmp_path)
    try:
        observation, info = env.reset(
            seed=0,
            options={"randomize_object": False, "randomize_goal": False},
        )
        recorder.start("pick_place")
        for index in range(2):
            identity = recorder.reserve_step()
            assert identity is not None
            simulation_time = 0.05 * index
            observation = env.get_observation()
            observation["timestamp"][:] = simulation_time
            payload = FramePayload(
                identity=identity,
                simulation_timestamp=simulation_time,
                sample_monotonic_ns=time.monotonic_ns(),
                observation=observation,
                state_26=build_state_26(observation),
                policy_state_42=env.get_policy_observation(observation),
                teleop=_teleop_snapshot(),
                executed_action=np.r_[env.home_joint_positions, 0.08],
                mujoco_ctrl=np.r_[env.home_joint_positions, 0.04],
                ik_success=True,
                command_accepted=True,
                action_clipped=False,
                fallback_used=False,
                rejection_reason="",
                reward=0.0,
                task_success=False,
                stage=0,
                events=0,
            )
            recorder.submit_frame(
                RenderedFrame(
                    payload=payload,
                    rgb=np.full((12, 16, 3), index, dtype=np.uint8),
                    depth=np.full((12, 16), 0.5 + index, dtype=np.float32),
                    image_valid=True,
                    drop_reason="",
                    render_start_monotonic_ns=1_000_000 * index,
                    render_end_monotonic_ns=1_000_000 * index + 100_000,
                    camera_calibration={
                        "name": "front",
                        "width": 16,
                        "height": 12,
                        "fovy_degrees": 45.0,
                        "intrinsic_matrix": np.eye(3).tolist(),
                        "position_world": [1.25, 0.0, 0.8],
                        "rotation_camera_to_world": np.eye(3).tolist(),
                        "near": 0.012,
                        "far": 60.0,
                    },
                )
            )
        token = recorder.stop()
        result = recorder.finalize(token)
        assert result["valid"]
        assert result["written_frames"] == 2
        with h5py.File(result["path"], "r") as episode:
            assert episode["observations/state_26"].shape == (2, 26)
            assert episode["actions/executed"].shape == (2, 8)
            assert episode["actions/policy_output"].shape == (2, 8)
            assert not episode["actions/policy_output_valid"][:].any()
            assert episode["observations/images/front/rgb"].shape == (2, 12, 16, 3)
            assert episode["observations/images/front/depth"].shape == (2, 12, 16)
            np.testing.assert_allclose(episode["timestamps/simulation"][:], [0.0, 0.05])
    finally:
        if recorder.recording:
            token = recorder.stop()
            recorder.finalize(token, discard=True)
        recorder.close()
        env.close()


def test_episode_recorder_accepts_a_chinese_task_name(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    try:
        episode_id = recorder.start("抓取放置")
        assert episode_id.startswith("抓取放置_")
        token = recorder.stop()
        result = recorder.finalize(token, discard=True)
        assert result["discarded"]
    finally:
        recorder.close()

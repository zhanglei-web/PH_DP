from __future__ import annotations

import numpy as np

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.envs.pick_place_env import (
    DEFAULT_GOAL_XY,
    DEFAULT_OBJECT_XY,
    GOAL_XY_HIGH,
    GOAL_XY_LOW,
    MIN_OBJECT_GOAL_DISTANCE,
    OBJECT_XY_HIGH,
    OBJECT_XY_LOW,
)


def test_reset_and_step_interfaces() -> None:
    env = PickPlaceEnv(camera_width=160, camera_height=120)
    try:
        obs, info = env.reset(seed=0, options={"randomize_object": False})
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (8,)
        assert info["policy_obs"].shape == (42,)
        assert obs["gripper_joint_positions"].shape == (2,)
        assert obs["gripper_joint_velocities"].shape == (2,)

        action = np.concatenate((env.home_joint_positions, [0.08]))
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert env.observation_space.contains(next_obs)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert step_info["policy_obs"].shape == (42,)
        assert next_obs["timestamp"][0] > 0.0
    finally:
        env.close()


def test_camera_rgb_and_depth() -> None:
    env = PickPlaceEnv(camera_width=160, camera_height=120)
    try:
        env.reset(seed=0, options={"randomize_object": False})
        rgb = env.render_rgb("front")
        depth = env.render_depth("front")
        registered_rgb, registered_depth = env.render_rgbd("front")
        assert rgb.shape == (120, 160, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (120, 160)
        assert depth.dtype == np.float32
        assert np.isfinite(depth).all()
        assert float(depth.max()) > float(depth.min())
        assert registered_rgb.shape == rgb.shape
        assert registered_rgb.dtype == np.uint8
        assert registered_depth.shape == depth.shape
        assert registered_depth.dtype == np.float32
        assert np.isfinite(registered_depth).all()
    finally:
        env.close()


def test_reset_restores_object_and_goal_to_fixed_positions() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        env.reset(
            seed=0,
            options={"randomize_object": False, "randomize_goal": False},
        )
        object_qpos = env.data.qpos[
            env._object_qpos_address : env._object_qpos_address + 7
        ]
        object_qpos[:3] = [0.62, 0.24, 0.40]
        env.data.mocap_pos[env._goal_mocap_id, :2] = [0.44, 0.25]

        obs, info = env.reset(
            options={"randomize_object": False, "randomize_goal": False}
        )

        np.testing.assert_allclose(info["object_xy"], DEFAULT_OBJECT_XY)
        np.testing.assert_allclose(info["goal_xy"], DEFAULT_GOAL_XY)
        np.testing.assert_allclose(obs["object_pose"][:2, 3], DEFAULT_OBJECT_XY)
        np.testing.assert_allclose(obs["goal_pose"][:2, 3], DEFAULT_GOAL_XY)
    finally:
        env.close()


def test_randomized_reset_is_reproducible_and_stays_in_safe_bounds() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        options = {"randomize_object": True, "randomize_goal": True}
        _, first = env.reset(seed=23, options=options)
        _, repeated = env.reset(seed=23, options=options)
        np.testing.assert_allclose(first["object_xy"], repeated["object_xy"])
        np.testing.assert_allclose(first["goal_xy"], repeated["goal_xy"])

        samples = []
        for _ in range(32):
            obs, info = env.reset(options=options)
            object_xy = info["object_xy"]
            goal_xy = info["goal_xy"]
            assert np.all(object_xy >= OBJECT_XY_LOW)
            assert np.all(object_xy <= OBJECT_XY_HIGH)
            assert np.all(goal_xy >= GOAL_XY_LOW)
            assert np.all(goal_xy <= GOAL_XY_HIGH)
            assert np.linalg.norm(object_xy - goal_xy) >= MIN_OBJECT_GOAL_DISTANCE
            np.testing.assert_allclose(
                obs["object_pose"][:2, 3], object_xy, atol=1e-6
            )
            np.testing.assert_allclose(obs["goal_pose"][:2, 3], goal_xy)
            samples.append(np.concatenate((object_xy, goal_xy)))

        assert np.unique(np.round(samples, decimals=5), axis=0).shape[0] > 1
    finally:
        env.close()


def test_pinocchio_fk_and_ik_match_mujoco_gripper() -> None:
    env = PickPlaceEnv(camera_width=64, camera_height=48)
    try:
        env.reset(seed=0, options={"randomize_object": False})
        current_q = env.robot.get_joint_positions()
        fk_pose = env.forward_kinematics(current_q)
        mujoco_pose = env.robot.get_ee_pose()
        assert np.allclose(fk_pose, mujoco_pose, atol=1e-8)

        reachable_q = current_q + np.array(
            [0.10, 0.03, -0.08, 0.02, 0.08, -0.04, 0.03]
        )
        target_pose = env.forward_kinematics(reachable_q)
        result = env.ik_controller.inverse_kinematics(target_pose)
        assert result.converged
        reconstructed_pose = env.forward_kinematics(result.joint_positions)
        assert np.linalg.norm(reconstructed_pose[:3, 3] - target_pose[:3, 3]) < 1e-4
        assert result.orientation_error < 1e-3
    finally:
        env.close()

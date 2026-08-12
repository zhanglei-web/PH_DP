from __future__ import annotations

import numpy as np

from mujoco_shared_control import PickPlaceEnv


def main() -> None:
    env = PickPlaceEnv(render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=7)
        metadata = info["joint_metadata"]
        print("scene_loaded:", env.model.nbody, "bodies,", env.model.nu, "actuators")
        print("arm_joint_names:", metadata.arm_joint_names)
        print("gripper_joint_names:", metadata.gripper_joint_names)
        print("arm_actuator_names:", metadata.arm_actuator_names)
        print("gripper_actuator_name:", metadata.gripper_actuator_name)
        print("action_space:", env.action_space)
        print("observation_space:", env.observation_space)
        print("policy_obs_shape:", info["policy_obs"].shape)
        print("object_position:", obs["object_pose"][:3, 3])
        print("goal_position:", obs["goal_pose"][:3, 3])

        action = np.concatenate((env.home_joint_positions, [0.08]))
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        print("step_result:", reward, terminated, truncated)
        print("render_shape:", frame.shape, frame.dtype)
    finally:
        env.close()


if __name__ == "__main__":
    main()


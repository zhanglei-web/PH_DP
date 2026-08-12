from __future__ import annotations

import numpy as np

from mujoco_shared_control import PickPlaceEnv


def main() -> None:
    env = PickPlaceEnv()
    try:
        env.reset(seed=3, options={"randomize_object": False})
        home = env.home_joint_positions
        for step in range(120):
            phase = step / 119.0
            q_cmd = home.copy()
            q_cmd[0] += 0.25 * np.sin(np.pi * phase)
            q_cmd[2] -= 0.20 * np.sin(np.pi * phase)
            q_cmd[6] += 0.15 * np.sin(2.0 * np.pi * phase)
            gripper_cmd = 0.08 - 0.03 * np.sin(np.pi * phase)
            action = np.concatenate((q_cmd, [gripper_cmd]))
            obs, _, terminated, truncated, _ = env.step(action)

            if step % 20 == 0 or step == 119:
                print(f"step={step:03d} time={obs['timestamp'][0]:.3f}")
                print("  q_obs:", np.round(obs["q_obs"], 4))
                print("  dq_obs:", np.round(obs["dq_obs"], 4))
                print("  T_ee_obs:\n", np.round(obs["ee_pose"], 4))
                print("  g_obs:", np.round(obs["gripper"], 4))
                print("  T_object:\n", np.round(obs["object_pose"], 4))
                print("  T_goal:\n", np.round(obs["goal_pose"], 4))
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()


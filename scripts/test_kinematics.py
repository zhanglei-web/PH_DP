from __future__ import annotations

import numpy as np

from mujoco_shared_control import PickPlaceEnv


def main() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        env.reset(seed=0, options={"randomize_object": False})
        q_start = env.robot.get_joint_positions()
        T_start = env.forward_kinematics(q_start)
        print("FK at current arm configuration:\n", np.round(T_start, 6))

        # Create a reachable Cartesian goal from a known nearby joint posture.
        q_goal = q_start + np.array([0.10, 0.03, -0.08, 0.02, 0.08, -0.04, 0.03])
        q_goal = np.clip(
            q_goal, env.robot.arm_joint_limits[:, 0], env.robot.arm_joint_limits[:, 1]
        )
        T_goal = env.forward_kinematics(q_goal)
        result = env.ik_controller.inverse_kinematics(T_goal, initial_guess=q_start)
        print(
            f"IK converged={result.converged}, iterations={result.iterations}, "
            f"position_error={result.position_error:.6f} m, "
            f"orientation_error={result.orientation_error:.6f} rad"
        )
        if not result.converged:
            raise RuntimeError("Pinocchio IK failed for the reachable test target")

        for _ in range(300):
            env.step(np.concatenate((result.joint_positions, [0.08])))
        T_executed = env.get_observation()["ee_pose"]
        print("Executed position error (m):", np.linalg.norm(T_executed[:3, 3] - T_goal[:3, 3]))
        print("Executed pose:\n", np.round(T_executed, 6))
    finally:
        env.close()


if __name__ == "__main__":
    main()

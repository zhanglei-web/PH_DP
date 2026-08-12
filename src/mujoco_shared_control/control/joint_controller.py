from __future__ import annotations

import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from mujoco_shared_control.robots.franka import FrankaRobot


class JointPositionController:
    """Applies absolute arm-joint and gripper-opening targets."""

    def __init__(self, robot: FrankaRobot) -> None:
        self.robot = robot
        low = np.concatenate((robot.arm_joint_limits[:, 0], [0.0]))
        high = np.concatenate(
            (robot.arm_joint_limits[:, 1], [robot.max_gripper_opening])
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def apply(self, action: ArrayLike) -> NDArray[np.float64]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (self.robot.arm_dof + 1,):
            raise ValueError(
                f"Joint action must have shape ({self.robot.arm_dof + 1},), "
                f"got {command.shape}"
            )
        arm_command = self.robot.set_joint_position_target(command[: self.robot.arm_dof])
        self.robot.apply_arm_bias_compensation()
        gripper_command = self.robot.set_gripper_command(command[-1])
        return np.concatenate((arm_command, [gripper_command]))

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mujoco_shared_control.utils.pose import make_pose


ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR_NAME = "actuator8"
EE_SITE_NAME = "gripper"


def _require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model does not contain {object_type.name} '{name}'")
    return object_id


@dataclass(frozen=True)
class JointMetadata:
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_actuator_name: str
    arm_joint_limits: NDArray[np.float64]
    gripper_joint_limits: NDArray[np.float64]


class FrankaRobot:
    """Named, validated access to the Menagerie Franka model."""

    arm_dof = 7
    gripper_dof = 2
    max_gripper_opening = 0.08

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.arm_joint_ids = np.asarray(
            [_require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        )
        self.gripper_joint_ids = np.asarray(
            [_require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in GRIPPER_JOINT_NAMES]
        )
        self.arm_actuator_ids = np.asarray(
            [
                _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ]
        )
        self.gripper_actuator_id = _require_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self.ee_site_id = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

        self.arm_qpos_indices = model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_indices = model.jnt_dofadr[self.arm_joint_ids].copy()
        self.gripper_qpos_indices = model.jnt_qposadr[self.gripper_joint_ids].copy()
        self.gripper_dof_indices = model.jnt_dofadr[self.gripper_joint_ids].copy()

        self.arm_joint_limits = model.jnt_range[self.arm_joint_ids].copy()
        self.gripper_joint_limits = model.jnt_range[self.gripper_joint_ids].copy()

    @property
    def metadata(self) -> JointMetadata:
        return JointMetadata(
            arm_joint_names=ARM_JOINT_NAMES,
            gripper_joint_names=GRIPPER_JOINT_NAMES,
            arm_actuator_names=ARM_ACTUATOR_NAMES,
            gripper_actuator_name=GRIPPER_ACTUATOR_NAME,
            arm_joint_limits=self.arm_joint_limits.copy(),
            gripper_joint_limits=self.gripper_joint_limits.copy(),
        )

    def get_joint_positions(self) -> NDArray[np.float64]:
        return self.data.qpos[self.arm_qpos_indices].copy()

    def get_joint_velocities(self) -> NDArray[np.float64]:
        return self.data.qvel[self.arm_dof_indices].copy()

    def get_gripper_joint_positions(self) -> NDArray[np.float64]:
        positions = self.data.qpos[self.gripper_qpos_indices]
        return np.clip(
            positions, self.gripper_joint_limits[:, 0], self.gripper_joint_limits[:, 1]
        ).copy()

    def get_gripper_joint_velocities(self) -> NDArray[np.float64]:
        return self.data.qvel[self.gripper_dof_indices].copy()

    def get_gripper_opening(self) -> float:
        return float(np.sum(self.get_gripper_joint_positions()))

    def get_ee_pose(self) -> NDArray[np.float64]:
        return make_pose(
            self.data.site_xpos[self.ee_site_id],
            self.data.site_xmat[self.ee_site_id],
        )

    def set_joint_position_target(self, q_cmd: ArrayLike) -> NDArray[np.float64]:
        command = np.asarray(q_cmd, dtype=np.float64)
        if command.shape != (self.arm_dof,):
            raise ValueError(f"q_cmd must have shape ({self.arm_dof},), got {command.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("q_cmd must contain only finite values")
        clipped = np.clip(
            command, self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1]
        )
        self.data.ctrl[self.arm_actuator_ids] = clipped
        return clipped.copy()

    def apply_arm_bias_compensation(self) -> None:
        """Cancel gravity and velocity bias for the position-controlled arm joints."""
        self.data.qfrc_applied[self.arm_dof_indices] = self.data.qfrc_bias[
            self.arm_dof_indices
        ]

    def set_gripper_command(self, g_cmd: float | ArrayLike) -> float:
        """Command total finger opening in meters, from closed=0 to open=0.08."""
        command = np.asarray(g_cmd, dtype=np.float64)
        if command.size != 1 or not np.isfinite(command).all():
            raise ValueError("g_cmd must be one finite scalar total opening in meters")
        total_opening = float(np.clip(command.item(), 0.0, self.max_gripper_opening))
        self.data.ctrl[self.gripper_actuator_id] = total_opening / 2.0
        return total_opening

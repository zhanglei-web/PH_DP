from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pinocchio as pin

from mujoco_shared_control.robots.franka import FrankaRobot
from mujoco_shared_control.utils.pose import make_pose


def _panda_urdf_path() -> Path:
    package = distribution("example-robot-data")
    suffix = "panda_description/urdf/panda.urdf"
    for file in package.files or ():
        if str(file).endswith(suffix):
            return Path(package.locate_file(file))
    raise FileNotFoundError(f"Could not find {suffix} in example-robot-data")


@dataclass(frozen=True)
class IKResult:
    joint_positions: NDArray[np.float64]
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float


class PinocchioIKController:
    """Pinocchio FK and damped-least-squares IK for the MuJoCo Panda gripper site."""

    def __init__(
        self,
        robot: FrankaRobot,
        damping: float = 1e-4,
        step_size: float = 0.6,
        max_iterations: int = 200,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
    ) -> None:
        self.robot = robot
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        self.model = pin.buildModelFromUrdf(str(_panda_urdf_path()))
        self._tcp_frame_id = self.model.getFrameId("panda_hand_tcp")
        if self._tcp_frame_id >= len(self.model.frames):
            raise ValueError("Panda URDF does not contain panda_hand_tcp")

        # Align the URDF TCP frame with the named MuJoCo gripper site once.
        mujoco.mj_forward(robot.model, robot.data)
        provisional_data = self.model.createData()
        q = self._pinocchio_configuration(robot.get_joint_positions())
        pin.forwardKinematics(self.model, provisional_data, q)
        pin.updateFramePlacements(self.model, provisional_data)
        tcp = provisional_data.oMf[self._tcp_frame_id]
        mujoco_ee = self.robot.get_ee_pose()
        offset = tcp.inverse() * pin.SE3(mujoco_ee[:3, :3], mujoco_ee[:3, 3])
        tcp_frame = self.model.frames[self._tcp_frame_id]
        self.ee_frame_id = self.model.addFrame(
            pin.Frame(
                "mujoco_gripper",
                tcp_frame.parentJoint,
                self._tcp_frame_id,
                tcp_frame.placement * offset,
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.model.createData()

    def _pinocchio_configuration(self, q_arm: ArrayLike) -> NDArray[np.float64]:
        arm = np.asarray(q_arm, dtype=np.float64)
        if arm.shape != (self.robot.arm_dof,):
            raise ValueError(f"q_arm must have shape ({self.robot.arm_dof},), got {arm.shape}")
        fingers = self.robot.get_gripper_joint_positions()
        return np.concatenate((arm, fingers))

    def forward_kinematics(self, q_arm: ArrayLike) -> NDArray[np.float64]:
        """Return the world-frame 4x4 MuJoCo-gripper pose for seven arm joints."""
        q = self._pinocchio_configuration(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.ee_frame_id]
        return make_pose(placement.translation, placement.rotation)

    def inverse_kinematics(
        self, target_pose: ArrayLike, initial_guess: ArrayLike | None = None
    ) -> IKResult:
        """Solve a world-frame 4x4 target using local-frame DLS IK in Pinocchio."""
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {target.shape}")
        if not np.isfinite(target).all() or not np.allclose(target[3], [0, 0, 0, 1]):
            raise ValueError("target_pose must be a finite homogeneous transform")
        if not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("target_pose rotation must be orthonormal")

        seed = (
            self.robot.get_joint_positions()
            if initial_guess is None
            else np.asarray(initial_guess, dtype=np.float64)
        )
        q = self._pinocchio_configuration(seed)
        q[: self.robot.arm_dof] = np.clip(
            q[: self.robot.arm_dof],
            self.robot.arm_joint_limits[:, 0],
            self.robot.arm_joint_limits[:, 1],
        )
        target_se3 = pin.SE3(target[:3, :3], target[:3, 3])

        last_error = np.full(6, np.inf, dtype=np.float64)
        for iteration in range(1, self.max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_frame_id]
            error = pin.log6(current.inverse() * target_se3).vector
            last_error = np.asarray(error, dtype=np.float64)
            if (
                np.linalg.norm(last_error[:3]) <= self.position_tolerance
                and np.linalg.norm(last_error[3:]) <= self.orientation_tolerance
            ):
                return IKResult(
                    q[: self.robot.arm_dof].copy(),
                    True,
                    iteration,
                    float(np.linalg.norm(last_error[:3])),
                    float(np.linalg.norm(last_error[3:])),
                )

            jacobian = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )[:, : self.robot.arm_dof]
            system = jacobian @ jacobian.T + self.damping * np.eye(6)
            velocity = jacobian.T @ np.linalg.solve(system, last_error)
            q[: self.robot.arm_dof] += self.step_size * velocity
            q[: self.robot.arm_dof] = np.clip(
                q[: self.robot.arm_dof],
                self.robot.arm_joint_limits[:, 0],
                self.robot.arm_joint_limits[:, 1],
            )

        return IKResult(
            q[: self.robot.arm_dof].copy(),
            False,
            self.max_iterations,
            float(np.linalg.norm(last_error[:3])),
            float(np.linalg.norm(last_error[3:])),
        )

    def solve(self, target_pose: ArrayLike) -> NDArray[np.float64]:
        result = self.inverse_kinematics(target_pose)
        if not result.converged:
            raise ValueError(
                "Pinocchio IK did not converge: "
                f"position_error={result.position_error:.6f} m, "
                f"orientation_error={result.orientation_error:.6f} rad"
            )
        return result.joint_positions


# Preserve the existing public import while switching its implementation to Pinocchio.
IKController = PinocchioIKController

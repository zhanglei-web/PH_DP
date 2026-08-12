"""Pose conversion and relative-pose mapping for VR teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mujoco_shared_control.utils.pose import (
    make_pose,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


@dataclass(frozen=True)
class ControllerPose:
    """A controller pose represented in its tracking world's coordinates."""

    position: NDArray[np.float64]
    rotation: NDArray[np.float64]


def controller_pose_from_xrt(values: ArrayLike) -> ControllerPose:
    """Parse XRT's ``[x, y, z, qx, qy, qz, qw]`` controller pose."""
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("XRT controller pose must contain 7 finite values")

    quaternion_xyzw = pose[3:]
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1e-8:
        raise ValueError("XRT controller pose has a zero quaternion")
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]] / norm
    return ControllerPose(
        position=pose[:3].copy(),
        rotation=quaternion_to_matrix(quaternion_wxyz),
    )


class RelativePoseMapper:
    """Map controller motion after alignment onto an end-effector target pose."""

    def __init__(
        self,
        translation_scale: float = 1.0,
        vr_to_world_axes: ArrayLike | None = None,
        control_orientation: bool = True,
    ) -> None:
        if not np.isfinite(translation_scale) or translation_scale <= 0.0:
            raise ValueError("translation_scale must be a positive finite number")
        axes = (
            np.eye(3, dtype=np.float64)
            if vr_to_world_axes is None
            else np.asarray(vr_to_world_axes, dtype=np.float64)
        )
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("vr_to_world_axes must be a finite 3x3 matrix")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
            raise ValueError("vr_to_world_axes must be an orthonormal matrix")

        self._translation_scale = float(translation_scale)
        self._axes = axes.copy()
        self._control_orientation = bool(control_orientation)
        self._controller_initial: ControllerPose | None = None
        self._ee_initial: NDArray[np.float64] | None = None

    @property
    def aligned(self) -> bool:
        return self._controller_initial is not None

    def reset(self) -> None:
        """Discard the current alignment reference."""
        self._controller_initial = None
        self._ee_initial = None

    def align(self, controller_pose: ControllerPose, ee_pose: ArrayLike) -> None:
        """Capture the controller and end-effector poses as a shared reference."""
        target = np.asarray(ee_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("end-effector pose must be a finite 4x4 matrix")
        self._controller_initial = controller_pose
        self._ee_initial = target.copy()

    def target(self, controller_pose: ControllerPose) -> NDArray[np.float64]:
        """Return the world-frame end-effector target for a controller pose."""
        if self._controller_initial is None or self._ee_initial is None:
            raise RuntimeError("RelativePoseMapper must be aligned before use")

        controller_initial = self._controller_initial
        delta_position = self._axes @ (
            controller_pose.position - controller_initial.position
        )
        delta_rotation_vr = controller_pose.rotation @ controller_initial.rotation.T
        delta_rotation_world = self._axes @ delta_rotation_vr @ self._axes.T
        target_rotation = self._ee_initial[:3, :3]
        if self._control_orientation:
            target_rotation = delta_rotation_world @ target_rotation

        return make_pose(
            self._ee_initial[:3, 3] + self._translation_scale * delta_position,
            target_rotation,
        )


class PoseTargetFilter:
    """Smooth pose targets while enforcing Cartesian velocity limits."""

    def __init__(
        self,
        update_period: float,
        position_time_constant: float = 0.08,
        orientation_time_constant: float = 0.10,
        max_linear_speed: float = 0.5,
        max_angular_speed: float = 2.0,
    ) -> None:
        values = {
            "update_period": update_period,
            "position_time_constant": position_time_constant,
            "orientation_time_constant": orientation_time_constant,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("pose filter parameters must be finite")
        if update_period <= 0.0:
            raise ValueError("update_period must be positive")
        if position_time_constant < 0.0 or orientation_time_constant < 0.0:
            raise ValueError("pose filter time constants must be non-negative")
        if max_linear_speed <= 0.0 or max_angular_speed <= 0.0:
            raise ValueError("pose filter speed limits must be positive")

        self._dt = float(update_period)
        self._position_alpha = self._smoothing_alpha(position_time_constant)
        self._orientation_alpha = self._smoothing_alpha(orientation_time_constant)
        self._max_linear_step = float(max_linear_speed) * self._dt
        self._max_angular_step = float(max_angular_speed) * self._dt
        self._state: NDArray[np.float64] | None = None

    def _smoothing_alpha(self, time_constant: float) -> float:
        if time_constant == 0.0:
            return 1.0
        return float(1.0 - np.exp(-self._dt / time_constant))

    @staticmethod
    def _validated_pose(pose: ArrayLike) -> NDArray[np.float64]:
        target = np.asarray(pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("pose filter target must be a finite 4x4 matrix")
        if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("pose filter target must be a homogeneous transform")
        return target

    @staticmethod
    def _slerp(
        source_wxyz: NDArray[np.float64],
        target_wxyz: NDArray[np.float64],
        fraction: float,
    ) -> NDArray[np.float64]:
        source = source_wxyz / np.linalg.norm(source_wxyz)
        target = target_wxyz / np.linalg.norm(target_wxyz)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot < 0.0:
            target = -target
            dot = -dot
        if dot > 1.0 - 1e-8:
            result = source + fraction * (target - source)
            return result / np.linalg.norm(result)

        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        source_weight = np.sin((1.0 - fraction) * theta) / sin_theta
        target_weight = np.sin(fraction * theta) / sin_theta
        return source_weight * source + target_weight * target

    def reset(self, pose: ArrayLike | None = None) -> None:
        """Clear the filter or seed it with the current end-effector pose."""
        self._state = (
            None if pose is None else self._validated_pose(pose).copy()
        )

    def update(self, pose: ArrayLike) -> NDArray[np.float64]:
        """Return a smoothed and velocity-limited target pose."""
        target = self._validated_pose(pose)
        if self._state is None:
            self._state = target.copy()
            return self._state.copy()

        position_step = self._position_alpha * (
            target[:3, 3] - self._state[:3, 3]
        )
        position_step_norm = float(np.linalg.norm(position_step))
        if position_step_norm > self._max_linear_step:
            position_step *= self._max_linear_step / position_step_norm

        source_quaternion = matrix_to_quaternion(self._state[:3, :3])
        target_quaternion = matrix_to_quaternion(target[:3, :3])
        quaternion_dot = abs(float(np.dot(source_quaternion, target_quaternion)))
        rotation_angle = 2.0 * float(
            np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        )
        rotation_fraction = self._orientation_alpha
        if rotation_angle > 1e-10:
            rotation_fraction = min(
                rotation_fraction, self._max_angular_step / rotation_angle
            )
        filtered_quaternion = self._slerp(
            source_quaternion, target_quaternion, rotation_fraction
        )

        self._state = make_pose(
            self._state[:3, 3] + position_step,
            quaternion_to_matrix(filtered_quaternion),
        )
        return self._state.copy()

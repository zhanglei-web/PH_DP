"""Convert the canonical Cartesian delta command to an executable joint target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from mujoco_shared_control.control.ik_controller import IKController, IKResult
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.utils.pose import make_pose


@dataclass(frozen=True)
class AdaptedCommand:
    requested: NDArray[np.float64]
    clipped: NDArray[np.float64]
    normalized: NDArray[np.float64]
    cartesian_target: NDArray[np.float64]
    joint_target: NDArray[np.float64]
    ik_result: IKResult | None
    accepted: bool
    action_clipped: bool
    fallback_used: bool
    rejection_reason: str


class ExpertCommandAdapter:
    """Integrate world-frame pose deltas against the previous control target."""

    def __init__(self, ik_controller: IKController, action_spec: ExpertActionSpec) -> None:
        self.ik_controller = ik_controller
        self.action_spec = action_spec
        self._target: NDArray[np.float64] | None = None
        self._joint_target: NDArray[np.float64] | None = None

    @property
    def target(self) -> NDArray[np.float64]:
        if self._target is None:
            raise RuntimeError("adapter must be reset before use")
        return self._target.copy()

    def reset(
        self, ee_pose: NDArray[np.floating], joint_position: NDArray[np.floating]
    ) -> None:
        pose = np.asarray(ee_pose, dtype=np.float64)
        joints = np.asarray(joint_position, dtype=np.float64)
        if pose.shape != (4, 4) or joints.shape != (7,):
            raise ValueError("reset requires ee_pose (4,4) and joint_position (7,)")
        self._target = pose.copy()
        self._joint_target = joints.copy()

    def adapt(self, command: NDArray[np.floating]) -> AdaptedCommand:
        if self._target is None or self._joint_target is None:
            raise RuntimeError("adapter must be reset before use")
        requested = np.asarray(command, dtype=np.float64)
        if requested.shape != (7,) or not np.isfinite(requested).all():
            return self._fallback(requested, "non_finite_or_wrong_shape")
        clipped = requested.copy()
        translation_norm = float(np.linalg.norm(clipped[:3]))
        if translation_norm > self.action_spec.max_translation_step_m:
            clipped[:3] *= self.action_spec.max_translation_step_m / translation_norm
        rotation_norm = float(np.linalg.norm(clipped[3:6]))
        if rotation_norm > self.action_spec.max_rotation_step_rad:
            clipped[3:6] *= self.action_spec.max_rotation_step_rad / rotation_norm
        clipped[6] = np.clip(
            clipped[6], self.action_spec.gripper_min_m, self.action_spec.gripper_max_m
        )
        target_rotation = Rotation.from_matrix(self._target[:3, :3])
        delta_world = Rotation.from_rotvec(clipped[3:6])
        proposed = make_pose(
            self._target[:3, 3] + clipped[:3],
            (delta_world * target_rotation).as_matrix(),
        )
        result = self.ik_controller.inverse_kinematics(
            proposed, initial_guess=self._joint_target
        )
        if not result.converged:
            return self._fallback(requested, "ik_nonconvergence", clipped, proposed, result)
        self._target = proposed
        self._joint_target = result.joint_positions.copy()
        executed = np.concatenate((self._joint_target, [clipped[6]]))
        return AdaptedCommand(
            requested=requested.copy(), clipped=clipped,
            normalized=self.action_spec.normalize(clipped),
            cartesian_target=proposed.copy(), joint_target=executed,
            ik_result=result, accepted=True,
            action_clipped=not np.allclose(requested, clipped),
            fallback_used=False, rejection_reason="",
        )

    def _fallback(
        self, requested: NDArray[np.float64], reason: str,
        clipped: NDArray[np.float64] | None = None,
        proposed: NDArray[np.float64] | None = None,
        result: IKResult | None = None,
    ) -> AdaptedCommand:
        safe_requested = np.asarray(requested, dtype=np.float64)
        if safe_requested.shape != (7,):
            safe_requested = np.full(7, np.nan, dtype=np.float64)
        safe_clipped = (
            np.full(7, np.nan, dtype=np.float64) if clipped is None else clipped.copy()
        )
        gripper = self.action_spec.gripper_max_m
        if np.isfinite(safe_clipped[6]):
            gripper = float(np.clip(safe_clipped[6], 0.0, 0.08))
        joint_target = np.concatenate((self._joint_target.copy(), [gripper]))
        normalized = np.full(7, np.nan, dtype=np.float64)
        if np.isfinite(safe_clipped).all():
            normalized = self.action_spec.normalize(safe_clipped)
        return AdaptedCommand(
            requested=safe_requested, clipped=safe_clipped, normalized=normalized,
            cartesian_target=self._target.copy() if proposed is None else proposed.copy(),
            joint_target=joint_target, ik_result=result, accepted=False,
            action_clipped=False, fallback_used=True, rejection_reason=reason,
        )

"""ROS- and environment-independent expert policy boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


EXPERT_ACTION_DIM = 7


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class ExpertActionSpec:
    max_translation_step_m: float = 0.025
    max_rotation_step_rad: float = 0.10
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.08

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.max_translation_step_m, self.max_rotation_step_rad,
             self.gripper_min_m, self.gripper_max_m], dtype=np.float64
        )
        if not np.isfinite(values).all() or self.max_translation_step_m <= 0.0:
            raise ValueError("action limits must be finite and positive")
        if self.max_rotation_step_rad <= 0.0 or self.gripper_min_m >= self.gripper_max_m:
            raise ValueError("invalid rotation or gripper limits")

    @property
    def scale(self) -> NDArray[np.float64]:
        return np.array(
            [self.max_translation_step_m] * 3
            + [self.max_rotation_step_rad] * 3
            + [self.gripper_max_m - self.gripper_min_m],
            dtype=np.float64,
        )

    def normalize(self, command: NDArray[np.floating]) -> NDArray[np.float64]:
        physical = _finite_array(command, (EXPERT_ACTION_DIM,), "command")
        result = physical.copy()
        result[:6] /= self.scale[:6]
        result[6] = 2.0 * (
            (physical[6] - self.gripper_min_m)
            / (self.gripper_max_m - self.gripper_min_m)
        ) - 1.0
        return np.clip(result, -1.0, 1.0)

    def denormalize(self, action: NDArray[np.floating]) -> NDArray[np.float64]:
        normalized = np.clip(
            _finite_array(action, (EXPERT_ACTION_DIM,), "normalized action"), -1.0, 1.0
        )
        result = normalized.copy()
        result[:6] *= self.scale[:6]
        result[6] = self.gripper_min_m + 0.5 * (normalized[6] + 1.0) * (
            self.gripper_max_m - self.gripper_min_m
        )
        return result


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    task_name: str
    run_id: str
    worker_id: int
    worker_episode_index: int
    environment_seed: int
    policy_seed: int
    perturbation_seed: int
    reset_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertObservation:
    episode_id: str
    worker_id: int
    step_index: int
    simulation_time: float
    state_26: NDArray[np.float32]
    policy_state: NDArray[np.float32]
    joint_position: NDArray[np.float64]
    joint_velocity: NDArray[np.float64]
    ee_pose: NDArray[np.float64]
    gripper_opening: float
    object_pose: NDArray[np.float64]
    object_linear_velocity: NDArray[np.float64]
    object_angular_velocity: NDArray[np.float64]
    object_grasped: bool
    goal_pose: NDArray[np.float64]
    previous_command: NDArray[np.float64] | None = None
    previous_executed_action: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ExpertCommand:
    delta_pose_gripper: NDArray[np.float64]
    valid: bool = True
    control_active: bool = True
    confidence: float = 1.0
    stage: int = 0
    policy_name: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        command = np.asarray(self.delta_pose_gripper, dtype=np.float64)
        if command.shape != (EXPERT_ACTION_DIM,):
            raise ValueError("expert command must have shape (7,)")
        if self.valid and not np.isfinite(command).all():
            raise ValueError("valid expert command must contain finite values")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "delta_pose_gripper", np.ascontiguousarray(command))


@runtime_checkable
class ExpertPolicy(Protocol):
    policy_name: str
    action_spec: ExpertActionSpec

    def reset(self, context: EpisodeContext) -> None: ...
    def predict(self, observation: ExpertObservation) -> ExpertCommand: ...
    def close(self) -> None: ...

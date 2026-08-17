"""One geometric milestone interface shared by Offline and closed-loop AWAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig
from mujoco_shared_control.tasks.pick_place import PickPlaceTask


MILESTONE_NAMES = ("grasp", "lift", "transport", "release", "retreat")
_RULE = RuleExpertConfig()
_ACTION_SPEC = ExpertActionSpec()
_HYBRID_GRIPPER_THRESHOLD = 0.375


class GeometricTaskPhase(IntEnum):
    """Coarse ground-truth phase derived only from latched milestones."""

    APPROACH = 0
    GRASP_LIFT = 1
    TRANSPORT = 2
    PLACE_RELEASE = 3
    RETREAT = 4
    COMPLETE = 5


def phase_from_milestones(milestones: np.ndarray) -> GeometricTaskPhase:
    """Map the cumulative five-bit geometric task state to its coarse phase."""
    value = np.asarray(milestones, bool)
    if value.shape != (5,) or np.any(value[1:] & ~value[:-1]):
        raise ValueError("phase requires one legal cumulative milestone vector")
    if value[4]:
        return GeometricTaskPhase.COMPLETE
    if value[3]:
        return GeometricTaskPhase.RETREAT
    if value[2]:
        return GeometricTaskPhase.PLACE_RELEASE
    if value[1]:
        return GeometricTaskPhase.TRANSPORT
    if value[0]:
        return GeometricTaskPhase.GRASP_LIFT
    return GeometricTaskPhase.APPROACH


@dataclass(frozen=True)
class MilestoneConfig:
    lift_height_m: float = 0.10
    transport_xy_tolerance_m: float = PickPlaceTask().success_tolerance
    goal_tolerance_m: float = PickPlaceTask().success_tolerance
    gripper_open_threshold_m: float = float(
        _ACTION_SPEC.denormalize(np.r_[np.zeros(6), _HYBRID_GRIPPER_THRESHOLD])[6]
    )
    retreat_height_m: float = _RULE.retreat_height_m
    retreat_tolerance_m: float = _RULE.position_tolerance_m


@dataclass(frozen=True)
class MilestoneUpdate:
    previous: np.ndarray
    current: np.ndarray
    rising: np.ndarray
    conditions: dict[str, bool]


def _state(value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, np.float32)
    if state.shape != (43,) or not np.isfinite(state).all():
        raise ValueError("MilestoneTracker requires a finite 43-D physical+grasp state")
    return state


class MilestoneTracker:
    """Cumulative environment-state milestones with one transition-time contract.

    `current` belongs to s_t. Calling `update(s_t_plus_1)` returns and latches the
    milestones belonging to s_t_plus_1. No Rule Expert stage/history is used.
    """

    def __init__(self, config: MilestoneConfig = MilestoneConfig()) -> None:
        self.config = config
        self.initial_object_z: float | None = None
        self._milestones = np.zeros(5, dtype=bool)

    @property
    def current(self) -> np.ndarray:
        if self.initial_object_z is None:
            raise RuntimeError("MilestoneTracker must be reset before use")
        return self._milestones.copy()

    @staticmethod
    def object_position(state: np.ndarray) -> np.ndarray:
        return _state(state)[22:25]

    @staticmethod
    def goal_position(state: np.ndarray) -> np.ndarray:
        return _state(state)[29:32]

    @staticmethod
    def ee_position(state: np.ndarray) -> np.ndarray:
        return _state(state)[14:17]

    def goal_contained(self, state: np.ndarray) -> bool:
        state = _state(state)
        return bool(
            np.linalg.norm(state[22:25] - state[29:32])
            < self.config.goal_tolerance_m
        )

    def gripper_open(self, state: np.ndarray) -> bool:
        return bool(_state(state)[21] >= self.config.gripper_open_threshold_m)

    def retreat_target(self, state: np.ndarray) -> np.ndarray:
        goal = self.goal_position(state).astype(np.float64)
        return goal + np.array([0.0, 0.0, self.config.retreat_height_m])

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = _state(initial_state)
        self.initial_object_z = float(state[24])
        self._milestones.fill(False)
        return self.current

    def update(self, next_state: np.ndarray) -> MilestoneUpdate:
        state = _state(next_state)
        if self.initial_object_z is None:
            raise RuntimeError("MilestoneTracker must be reset before update")
        previous = self._milestones.copy()
        grasped = bool(state[42])
        object_position = state[22:25]
        goal_position = state[29:32]
        goal_contained = self.goal_contained(state)
        gripper_open = self.gripper_open(state)
        self._milestones[0] |= grasped
        self._milestones[1] |= bool(
            self._milestones[0]
            and object_position[2] - self.initial_object_z >= self.config.lift_height_m
        )
        self._milestones[2] |= bool(
            self._milestones[1]
            and np.linalg.norm(object_position[:2] - goal_position[:2])
            < self.config.transport_xy_tolerance_m
        )
        self._milestones[3] |= bool(
            self._milestones[2] and goal_contained and not grasped and gripper_open
        )
        retreat_reached = bool(
            np.linalg.norm(state[14:17] - self.retreat_target(state))
            <= self.config.retreat_tolerance_m
        )
        self._milestones[4] |= bool(
            self._milestones[3] and goal_contained and retreat_reached
        )
        if np.any(self._milestones[1:] & ~self._milestones[:-1]):
            raise RuntimeError("MilestoneTracker produced an illegal milestone order")
        return MilestoneUpdate(
            previous, self._milestones.copy(), self._milestones & ~previous,
            {
                "object_grasped": grasped,
                "goal_contained": goal_contained,
                "gripper_open": gripper_open,
                "retreat_reached": retreat_reached,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        if self.initial_object_z is None:
            raise RuntimeError("MilestoneTracker must be reset before serialization")
        return {
            "config": asdict(self.config),
            "initial_object_z": self.initial_object_z,
            "milestones": self._milestones.astype(np.uint8),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if MilestoneConfig(**state["config"]) != self.config:
            raise ValueError("MilestoneTracker config mismatch")
        milestones = np.asarray(state["milestones"], bool)
        if milestones.shape != (5,) or np.any(milestones[1:] & ~milestones[:-1]):
            raise ValueError("invalid serialized milestone state")
        self.initial_object_z = float(state["initial_object_z"])
        self._milestones = milestones.copy()

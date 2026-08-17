"""Frozen four-phase reward and deterministic task protocol for online SAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Mapping

import numpy as np


SAC_REWARD_VERSION = "sac_reward_v1"
SAC_DISCOUNT_GAMMA = 0.995


class SACPhase(IntEnum):
    """The four frozen task phases; milestones remain internal to a phase."""

    PRE_GRASP = 1
    GRASP = 2
    TRANSPORT = 3
    PLACE_AND_RETREAT = 4


@dataclass(frozen=True)
class SACRewardV1Config:
    epsilon: float = 1e-6
    grasp_offset_z_m: float = 0.012
    above_goal_height_m: float = 0.16
    retreat_height_m: float = 0.16
    position_tolerance_m: float = 0.008
    goal_tolerance_m: float = 0.055
    stable_grasp_steps: int = 8
    stable_release_steps: int = 4
    p1_progress_weight: float = 2.0
    stable_grasp_bonus: float = 2.0
    p3_progress_weight: float = 3.0
    p4_place_progress_weight: float = 2.0
    place_bonus: float = 3.0
    retreat_progress_weight: float = 1.0
    success_bonus: float = 10.0
    failure_penalty: float = -5.0


@dataclass(frozen=True)
class SACRewardComponents:
    p1_progress: float = 0.0
    grasp_event: float = 0.0
    p3_progress: float = 0.0
    p4_place_progress: float = 0.0
    place_event: float = 0.0
    retreat_progress: float = 0.0
    success_terminal: float = 0.0
    failure_terminal: float = 0.0
    illegal_drop: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def as_dict(self) -> dict[str, float]:
        result = {name: float(value) for name, value in asdict(self).items()}
        result["reward_total"] = self.total
        return result


@dataclass(frozen=True)
class SACRewardStep:
    reward: float
    components: SACRewardComponents
    terminated: bool
    truncated: bool
    termination_reason: str
    phase: SACPhase
    next_phase: SACPhase
    stable_grasp: bool
    successful_release: bool


def _position(observation: Mapping[str, Any], name: str) -> np.ndarray:
    pose = np.asarray(observation[name], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 pose")
    return pose[:3, 3]


class SACRewardV1:
    """Stateful, one-shot event reward used by the SAC task protocol.

    Phase transitions and safety failures are ground-truth task signals. This
    class owns reward state only; it never predicts phase from a learned model.
    """

    version = SAC_REWARD_VERSION

    def __init__(self, config: SACRewardV1Config = SACRewardV1Config()) -> None:
        self.config = config
        self.reset()

    @staticmethod
    def above_goal(goal_position: np.ndarray, height_m: float = 0.16) -> np.ndarray:
        return np.asarray(goal_position, dtype=np.float64) + np.array(
            [0.0, 0.0, height_m], dtype=np.float64
        )

    @staticmethod
    def retreat_target(goal_position: np.ndarray, height_m: float = 0.16) -> np.ndarray:
        return np.asarray(goal_position, dtype=np.float64) + np.array(
            [0.0, 0.0, height_m], dtype=np.float64
        )

    def reset(self) -> None:
        self._initial_distances: dict[str, float] = {}
        self.stable_grasp_rewarded = False
        self.place_rewarded = False
        self.success_rewarded = False
        self.had_stable_grasp = False
        self.successful_release = False
        self.terminal = False

    def _progress(
        self, key: str, distance_before: float, distance_after: float, weight: float
    ) -> float:
        if key not in self._initial_distances:
            self._initial_distances[key] = float(distance_before)
        denominator = self._initial_distances[key] + self.config.epsilon
        return float(weight * (distance_before - distance_after) / denominator)

    def step(
        self,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        phase: SACPhase,
        *,
        next_phase: SACPhase | None = None,
        stable_grasp_event: bool = False,
        successful_release_event: bool = False,
        full_success: bool = False,
        true_failure: bool = False,
        failure_reason: str = "explicit_failure",
        time_limit: bool = False,
        apply_phase_progress: bool = True,
        force_illegal_drop: bool = False,
    ) -> SACRewardStep:
        if self.terminal:
            raise RuntimeError("sac_reward_v1 cannot accumulate after a terminal step")
        next_phase = phase if next_phase is None else next_phase
        ee = _position(observation, "ee_pose")
        next_ee = _position(next_observation, "ee_pose")
        obj = _position(observation, "object_pose")
        next_obj = _position(next_observation, "object_pose")
        goal = _position(observation, "goal_pose")
        next_goal = _position(next_observation, "goal_pose")
        grasped = bool(observation["object_grasped"])
        next_grasped = bool(next_observation["object_grasped"])

        values = {name: 0.0 for name in SACRewardComponents.__dataclass_fields__}
        if apply_phase_progress and phase == SACPhase.PRE_GRASP:
            grasp = obj + np.array([0.0, 0.0, self.config.grasp_offset_z_m])
            next_grasp = next_obj + np.array([0.0, 0.0, self.config.grasp_offset_z_m])
            values["p1_progress"] = self._progress(
                "p1", np.linalg.norm(ee - grasp), np.linalg.norm(next_ee - next_grasp),
                self.config.p1_progress_weight,
            )
        elif apply_phase_progress and phase == SACPhase.TRANSPORT:
            values["p3_progress"] = self._progress(
                "p3",
                np.linalg.norm(ee - self.above_goal(goal, self.config.above_goal_height_m)),
                np.linalg.norm(next_ee - self.above_goal(next_goal, self.config.above_goal_height_m)),
                self.config.p3_progress_weight,
            )
        elif apply_phase_progress and phase == SACPhase.PLACE_AND_RETREAT:
            if self.successful_release:
                values["retreat_progress"] = self._progress(
                    "retreat",
                    np.linalg.norm(ee - self.retreat_target(goal, self.config.retreat_height_m)),
                    np.linalg.norm(next_ee - self.retreat_target(next_goal, self.config.retreat_height_m)),
                    self.config.retreat_progress_weight,
                )
            else:
                values["p4_place_progress"] = self._progress(
                    "p4_place", np.linalg.norm(obj - goal), np.linalg.norm(next_obj - next_goal),
                    self.config.p4_place_progress_weight,
                )

        if stable_grasp_event and not self.stable_grasp_rewarded:
            values["grasp_event"] = self.config.stable_grasp_bonus
            self.stable_grasp_rewarded = True
            self.had_stable_grasp = True

        inside_goal = bool(np.linalg.norm(next_obj - next_goal) < self.config.goal_tolerance_m)
        legal_release_edge = bool(
            phase == SACPhase.PLACE_AND_RETREAT and grasped and not next_grasped and inside_goal
        )
        illegal_drop = force_illegal_drop or bool(
            self.had_stable_grasp and grasped and not next_grasped and not legal_release_edge
        )

        if successful_release_event and not self.place_rewarded and not illegal_drop:
            values["place_event"] = self.config.place_bonus
            self.place_rewarded = True
            self.successful_release = True

        terminated = False
        reason = ""
        if illegal_drop:
            values["illegal_drop"] = self.config.failure_penalty
            terminated, reason = True, "illegal_drop"
        elif true_failure:
            values["failure_terminal"] = self.config.failure_penalty
            terminated, reason = True, failure_reason
        elif full_success:
            if not (self.had_stable_grasp and self.successful_release):
                raise ValueError("full_success requires stable grasp and successful release")
            if not self.success_rewarded:
                values["success_terminal"] = self.config.success_bonus
                self.success_rewarded = True
            terminated, reason = True, "task_success"

        truncated = bool(time_limit and not terminated)
        if truncated:
            reason = "time_limit"
        self.terminal = terminated or truncated
        components = SACRewardComponents(**values)
        return SACRewardStep(
            components.total, components, terminated, truncated, reason,
            phase, next_phase, self.had_stable_grasp, self.successful_release,
        )


class SACPickPlaceProtocol:
    """Ground-truth deterministic phase/milestone source for SAC Expert training."""

    def __init__(self, config: SACRewardV1Config = SACRewardV1Config()) -> None:
        self.config = config
        self.reward = SACRewardV1(config)
        self.reset()

    def reset(self) -> None:
        self.phase = SACPhase.PRE_GRASP
        self.reward.reset()
        self._stable_grasp_steps = 0
        self._stable_release_steps = 0
        self._release_started = False

    def step(
        self,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        *,
        true_failure: bool = False,
        failure_reason: str = "explicit_failure",
        time_limit: bool = False,
    ) -> SACRewardStep:
        phase = self.phase
        next_phase = phase
        stable_event = False
        release_event = False
        full_success = False
        next_ee = _position(next_observation, "ee_pose")
        next_obj = _position(next_observation, "object_pose")
        next_goal = _position(next_observation, "goal_pose")
        next_grasped = bool(next_observation["object_grasped"])

        if phase == SACPhase.PRE_GRASP:
            grasp = next_obj + np.array([0.0, 0.0, self.config.grasp_offset_z_m])
            if np.linalg.norm(next_ee - grasp) <= self.config.position_tolerance_m:
                next_phase = SACPhase.GRASP
        elif phase == SACPhase.GRASP:
            self._stable_grasp_steps = self._stable_grasp_steps + 1 if next_grasped else 0
            if self._stable_grasp_steps >= self.config.stable_grasp_steps:
                stable_event = True
                next_phase = SACPhase.TRANSPORT
        elif phase == SACPhase.TRANSPORT:
            above = SACRewardV1.above_goal(next_goal, self.config.above_goal_height_m)
            if np.linalg.norm(next_ee - above) <= self.config.position_tolerance_m:
                next_phase = SACPhase.PLACE_AND_RETREAT
        else:
            previous_grasped = bool(observation["object_grasped"])
            released_inside = bool(
                not next_grasped
                and np.linalg.norm(next_obj - next_goal) < self.config.goal_tolerance_m
            )
            if previous_grasped and not next_grasped and released_inside:
                self._release_started = True
            self._stable_release_steps = self._stable_release_steps + 1 if released_inside else 0
            if (
                not self.reward.successful_release
                and self._stable_release_steps >= self.config.stable_release_steps
            ):
                release_event = True
            release_complete = self.reward.successful_release or release_event
            retreat = SACRewardV1.retreat_target(next_goal, self.config.retreat_height_m)
            full_success = bool(
                release_complete
                and np.linalg.norm(next_ee - retreat) <= self.config.position_tolerance_m
            )
            if self._release_started and not release_complete and not released_inside:
                true_failure = True
                failure_reason = "unstable_release"

        result = self.reward.step(
            observation, next_observation, phase, next_phase=next_phase,
            stable_grasp_event=stable_event,
            successful_release_event=release_event,
            full_success=full_success,
            true_failure=true_failure,
            failure_reason=failure_reason,
            time_limit=time_limit,
        )
        self.phase = next_phase
        return result

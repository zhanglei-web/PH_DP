"""Candidate place-and-release SAC reward; frozen v1 remains untouched."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mujoco_shared_control.tasks.sac_reward import (
    SAC_DISCOUNT_GAMMA,
    SACPhase,
    SACRewardStep,
    SACRewardV1,
    SACRewardV1Config,
    _position,
)


SAC_REWARD_V2_CANDIDATE = "sac_reward_v2_candidate"


@dataclass(frozen=True)
class SACRewardV2Config(SACRewardV1Config):
    """V1 thresholds/weights, with release and retreat rewards disabled."""

    place_bonus: float = 0.0
    retreat_progress_weight: float = 0.0


class SACRewardV2(SACRewardV1):
    """P1--P3 unchanged; P4 terminates at stable in-goal release."""

    version = SAC_REWARD_V2_CANDIDATE

    def __init__(self, config: SACRewardV2Config = SACRewardV2Config()) -> None:
        super().__init__(config)

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
        if full_success and not successful_release_event:
            raise ValueError("v2 success must be the stable release event")
        # In V2 the stable release event is itself full task success. The inherited
        # implementation preserves P1/P2/P3/P4-place and failure semantics exactly;
        # zero V2 place/retreat weights eliminate the superseded rewards.
        return super().step(
            observation, next_observation, phase, next_phase=next_phase,
            stable_grasp_event=stable_grasp_event,
            successful_release_event=successful_release_event,
            full_success=successful_release_event,
            true_failure=true_failure, failure_reason=failure_reason,
            time_limit=time_limit, apply_phase_progress=apply_phase_progress,
            force_illegal_drop=force_illegal_drop,
        )


class SACPickPlaceProtocolV2:
    """Four phases ending at stable in-goal release; retreat is outside RL."""

    version = SAC_REWARD_V2_CANDIDATE

    def __init__(self, config: SACRewardV2Config = SACRewardV2Config()) -> None:
        self.config = config
        self.reward = SACRewardV2(config)
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
        release_success = False
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
            release_success = self._stable_release_steps >= self.config.stable_release_steps
            if self._release_started and not release_success and not released_inside:
                true_failure = True
                failure_reason = "unstable_release"

        result = self.reward.step(
            observation, next_observation, phase, next_phase=next_phase,
            stable_grasp_event=stable_event,
            successful_release_event=release_success,
            true_failure=true_failure, failure_reason=failure_reason,
            time_limit=time_limit,
        )
        self.phase = next_phase
        return result


assert SAC_DISCOUNT_GAMMA == 0.995

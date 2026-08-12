"""Deterministic closed-loop finite-state expert for Franka pick-and-place."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np

from mujoco_shared_control.experts.interfaces import (
    EpisodeContext,
    ExpertActionSpec,
    ExpertCommand,
    ExpertObservation,
)


class RuleExpertStage(IntEnum):
    PRE_GRASP = 0
    DESCEND = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    TRANSPORT = 4
    DESCEND_TO_GOAL = 5
    OPEN_GRIPPER = 6
    RETREAT = 7
    COMPLETE = 8
    FAILED = 9
    # Collector-owned observation state. The rule policy never transitions here.
    SETTLING = 10


@dataclass(frozen=True)
class RuleExpertConfig:
    hover_height_m: float = 0.12
    grasp_offset_z_m: float = 0.012
    lift_height_m: float = 0.16
    place_height_m: float = 0.025
    retreat_height_m: float = 0.16
    position_gain: float = 0.35
    motion_step_m: float = 0.010
    position_tolerance_m: float = 0.008
    settle_steps: int = 8
    close_gripper_m: float = 0.030
    open_gripper_m: float = 0.08
    stage_timeout_steps: int = 180

    def state_dict(self) -> dict:
        return asdict(self)


class RulePickPlaceExpert:
    policy_name = "rule_pick_place_v1"

    def __init__(
        self, config: RuleExpertConfig = RuleExpertConfig(),
        action_spec: ExpertActionSpec = ExpertActionSpec(),
    ) -> None:
        self.config = config
        self.action_spec = action_spec
        self._stage = RuleExpertStage.PRE_GRASP
        self._stage_step = 0
        self._settle = 0

    @property
    def stage(self) -> RuleExpertStage:
        return self._stage

    def reset(self, context: EpisodeContext) -> None:
        del context
        self._stage = RuleExpertStage.PRE_GRASP
        self._stage_step = 0
        self._settle = 0

    def close(self) -> None:
        return None

    def _transition(self, stage: RuleExpertStage) -> None:
        self._stage = stage
        self._stage_step = 0
        self._settle = 0

    def _move(
        self, observation: ExpertObservation, target: np.ndarray, gripper: float
    ) -> ExpertCommand:
        error = np.asarray(target, dtype=np.float64) - observation.ee_pose[:3, 3]
        delta = self.config.position_gain * error
        norm = float(np.linalg.norm(delta))
        motion_limit = min(
            self.config.motion_step_m, self.action_spec.max_translation_step_m
        )
        if norm > motion_limit:
            delta *= motion_limit / norm
        command = np.concatenate((delta, np.zeros(3), [gripper]))
        return ExpertCommand(
            command, stage=int(self._stage), policy_name=self.policy_name,
            diagnostics={"position_error_m": float(np.linalg.norm(error))},
        )

    def _at(self, observation: ExpertObservation, target: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(observation.ee_pose[:3, 3] - target)
            <= self.config.position_tolerance_m
        )

    def predict(self, observation: ExpertObservation) -> ExpertCommand:
        self._stage_step += 1
        if self._stage_step > self.config.stage_timeout_steps:
            self._transition(RuleExpertStage.FAILED)
        obj = observation.object_pose[:3, 3]
        goal = observation.goal_pose[:3, 3]
        hover = obj + np.array([0.0, 0.0, self.config.hover_height_m])
        grasp = obj + np.array([0.0, 0.0, self.config.grasp_offset_z_m])
        lifted = obj.copy()
        # Use a fixed world height.  Adding lift_height to the live object z would
        # make the target recede upward as the grasped object moves.
        lifted[2] = goal[2] + self.config.lift_height_m
        above_goal = goal + np.array([0.0, 0.0, self.config.lift_height_m])
        place = goal + np.array([0.0, 0.0, self.config.place_height_m])
        retreat = goal + np.array([0.0, 0.0, self.config.retreat_height_m])

        if self._stage == RuleExpertStage.PRE_GRASP:
            if self._at(observation, hover):
                self._transition(RuleExpertStage.DESCEND)
            return self._move(observation, hover, self.config.open_gripper_m)
        if self._stage == RuleExpertStage.DESCEND:
            if self._at(observation, grasp):
                self._transition(RuleExpertStage.CLOSE_GRIPPER)
            return self._move(observation, grasp, self.config.open_gripper_m)
        if self._stage == RuleExpertStage.CLOSE_GRIPPER:
            if observation.object_grasped:
                self._settle += 1
                if self._settle >= self.config.settle_steps:
                    self._transition(RuleExpertStage.LIFT)
            else:
                self._settle = 0
            if self._stage_step >= self.config.settle_steps * 4:
                self._transition(RuleExpertStage.FAILED)
            return self._move(observation, grasp, self.config.close_gripper_m)
        if self._stage == RuleExpertStage.LIFT:
            if not observation.object_grasped:
                self._transition(RuleExpertStage.FAILED)
            elif self._at(observation, lifted):
                self._transition(RuleExpertStage.TRANSPORT)
            return self._move(observation, lifted, self.config.close_gripper_m)
        if self._stage == RuleExpertStage.TRANSPORT:
            if not observation.object_grasped:
                self._transition(RuleExpertStage.FAILED)
            elif self._at(observation, above_goal):
                self._transition(RuleExpertStage.DESCEND_TO_GOAL)
            return self._move(observation, above_goal, self.config.close_gripper_m)
        if self._stage == RuleExpertStage.DESCEND_TO_GOAL:
            if not observation.object_grasped:
                self._transition(RuleExpertStage.FAILED)
            elif self._at(observation, place):
                self._transition(RuleExpertStage.OPEN_GRIPPER)
            return self._move(observation, place, self.config.close_gripper_m)
        if self._stage == RuleExpertStage.OPEN_GRIPPER:
            if observation.object_grasped:
                self._settle = 0
            else:
                self._settle += 1
                if self._settle >= self.config.settle_steps:
                    self._transition(RuleExpertStage.RETREAT)
            return self._move(observation, place, self.config.open_gripper_m)
        if self._stage == RuleExpertStage.RETREAT:
            if self._at(observation, retreat):
                self._transition(RuleExpertStage.COMPLETE)
            return self._move(observation, retreat, self.config.open_gripper_m)
        active = self._stage not in (RuleExpertStage.COMPLETE, RuleExpertStage.FAILED)
        return ExpertCommand(
            np.r_[np.zeros(6), self.config.open_gripper_m],
            valid=active, control_active=active, confidence=1.0,
            stage=int(self._stage), policy_name=self.policy_name,
            diagnostics={"terminal_stage": self._stage.name},
        )

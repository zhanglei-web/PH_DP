"""State-reactive Rule Expert pilot for stage/recovery data collection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mujoco_shared_control.experts.interfaces import ExpertActionSpec, ExpertCommand, ExpertObservation
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig


class ActivePhase(IntEnum):
    APPROACH = 0
    GRASP_LIFT = 1
    TRANSPORT = 2
    PLACE_RELEASE = 3
    RETREAT = 4


@dataclass(frozen=True)
class RecoveryPilotConfig:
    hover_height_m: float = 0.12
    grasp_offset_z_m: float = 0.012
    safe_lift_m: float = 0.10
    transport_height_m: float = 0.16
    place_height_m: float = 0.025
    retreat_height_m: float = 0.16
    position_tolerance_m: float = 0.008
    grasp_enter_threshold_m: float = 0.008
    grasp_exit_threshold_m: float = 0.012
    grasp_failure_debounce_steps: int = 3
    transport_xy_tolerance_m: float = 0.080
    goal_tolerance_m: float = 0.055
    open_gripper_m: float = 0.08
    close_gripper_m: float = 0.030
    position_gain: float = 0.35
    nominal_motion_step_m: float = 0.010


class RuleBasedRecoveryPilot:
    """Replan from current geometry every step; no cumulative phase latch."""

    policy_name = "rule_based_recovery_pilot_v1"

    def __init__(
        self, config: RecoveryPilotConfig = RecoveryPilotConfig(),
        action_spec: ExpertActionSpec = ExpertActionSpec(),
    ) -> None:
        self.config = config
        self.action_spec = action_spec
        self.initial_object_z = 0.0
        self.approach_offset = np.zeros(2)
        self.motion_step = config.nominal_motion_step_m
        self.active_phase = ActivePhase.APPROACH
        self.close_attempt_completed = False
        self.grasp_failure_frames = 0
        self.forced_grasp_failure = False
        self.forced_reapproach_steps = 0

    def reset(self, initial_object_z: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.initial_object_z = float(initial_object_z)
        self.approach_offset = np.clip(rng.normal(0.0, 0.0015, 2), -0.003, 0.003)
        self.motion_step = self.config.nominal_motion_step_m * rng.uniform(0.9, 1.1)
        self.active_phase = ActivePhase.APPROACH
        self.close_attempt_completed = False
        self.grasp_failure_frames = 0
        self.forced_grasp_failure = False
        self.forced_reapproach_steps = 0

    def confirm_grasp_failure(self) -> None:
        """Request one semantic GRASP_LIFT -> APPROACH regression next step."""
        self.forced_grasp_failure = True

    def confirm_external_failure(self) -> None:
        """Latch one observable APPROACH frame after a transport/place drop."""
        self.active_phase = ActivePhase.APPROACH
        self.close_attempt_completed = False
        self.grasp_failure_frames = 0
        self.forced_reapproach_steps = max(self.forced_reapproach_steps, 1)

    def _move(
        self, observation: ExpertObservation, target: np.ndarray, gripper: float,
        phase: ActivePhase,
    ) -> tuple[ExpertCommand, ActivePhase]:
        error = np.asarray(target, np.float64) - observation.ee_pose[:3, 3]
        delta = self.config.position_gain * error
        norm = float(np.linalg.norm(delta))
        limit = min(self.motion_step, self.action_spec.max_translation_step_m)
        if norm > limit:
            delta *= limit / norm
        command = ExpertCommand(
            np.r_[delta, np.zeros(3), gripper], stage=int(phase),
            policy_name=self.policy_name,
            diagnostics={"active_phase": phase.name, "position_error_m": float(np.linalg.norm(error))},
        )
        return command, phase

    def predict(self, observation: ExpertObservation) -> tuple[ExpertCommand, ActivePhase]:
        cfg = self.config
        ee = observation.ee_pose[:3, 3]
        obj = observation.object_pose[:3, 3]
        goal = observation.goal_pose[:3, 3]
        grasped = bool(observation.object_grasped)
        opening = float(observation.gripper_opening)
        goal_distance = float(np.linalg.norm(obj - goal))

        released_in_goal = bool(not grasped and goal_distance < cfg.goal_tolerance_m and opening >= 0.055)
        if released_in_goal:
            self.active_phase = ActivePhase.RETREAT
            return self._move(
                observation, goal + np.array([0.0, 0.0, cfg.retreat_height_m]),
                cfg.open_gripper_m, ActivePhase.RETREAT,
            )

        if not grasped:
            hover = obj + np.r_[self.approach_offset, cfg.hover_height_m]
            grasp = obj + np.array([0.0, 0.0, cfg.grasp_offset_z_m])
            xy_error = float(np.linalg.norm(ee[:2] - obj[:2]))
            grasp_error = float(np.linalg.norm(ee - grasp))
            if self.forced_reapproach_steps > 0:
                self.forced_reapproach_steps -= 1
                self.active_phase = ActivePhase.APPROACH
                return self._move(observation, hover, cfg.open_gripper_m, ActivePhase.APPROACH)
            # A transport/place drop is a semantic regression and immediately
            # replans from the live object pose. During grasping, however, keep
            # the phase latched until a completed close attempt has failed for
            # a short dwell; this prevents 0/1 chatter at the grasp boundary.
            if self.active_phase not in (ActivePhase.APPROACH, ActivePhase.GRASP_LIFT):
                self.active_phase = ActivePhase.APPROACH
                self.close_attempt_completed = False
                self.grasp_failure_frames = 0
                return self._move(observation, hover, cfg.open_gripper_m, ActivePhase.APPROACH)
            if self.forced_grasp_failure:
                self.forced_grasp_failure = False
                self.active_phase = ActivePhase.APPROACH
                self.close_attempt_completed = False
                self.grasp_failure_frames = 0
                return self._move(observation, hover, cfg.open_gripper_m, ActivePhase.APPROACH)
            if self.active_phase == ActivePhase.GRASP_LIFT:
                self.close_attempt_completed |= opening <= cfg.close_gripper_m + 0.010
                failed_state = self.close_attempt_completed and (
                    grasp_error >= cfg.grasp_exit_threshold_m
                    or opening <= cfg.close_gripper_m + 0.010
                )
                self.grasp_failure_frames = self.grasp_failure_frames + 1 if failed_state else 0
                if self.grasp_failure_frames < cfg.grasp_failure_debounce_steps:
                    return self._move(observation, grasp, cfg.close_gripper_m, ActivePhase.GRASP_LIFT)
                self.active_phase = ActivePhase.APPROACH
                self.close_attempt_completed = False
                self.grasp_failure_frames = 0
            if opening < 0.055:
                return self._move(observation, hover, cfg.open_gripper_m, ActivePhase.APPROACH)
            if grasp_error > cfg.grasp_enter_threshold_m:
                target = hover if xy_error > 0.010 else grasp
                return self._move(observation, target, cfg.open_gripper_m, ActivePhase.APPROACH)
            self.active_phase = ActivePhase.GRASP_LIFT
            self.close_attempt_completed = False
            self.grasp_failure_frames = 0
            return self._move(observation, grasp, cfg.close_gripper_m, ActivePhase.GRASP_LIFT)

        # Goal proximity takes precedence over absolute height: descending for
        # placement must never be misclassified as a new lift.
        if np.linalg.norm(obj[:2] - goal[:2]) < cfg.transport_xy_tolerance_m:
            self.active_phase = ActivePhase.PLACE_RELEASE
            target = goal + np.array([0.0, 0.0, cfg.place_height_m])
            gripper = cfg.open_gripper_m if np.linalg.norm(ee - target) <= cfg.position_tolerance_m else cfg.close_gripper_m
            return self._move(observation, target, gripper, ActivePhase.PLACE_RELEASE)
        if obj[2] - self.initial_object_z < cfg.safe_lift_m:
            self.active_phase = ActivePhase.GRASP_LIFT
            target = np.array([ee[0], ee[1], goal[2] + cfg.transport_height_m])
            return self._move(observation, target, cfg.close_gripper_m, ActivePhase.GRASP_LIFT)
        target = goal + np.array([0.0, 0.0, cfg.transport_height_m])
        self.active_phase = ActivePhase.TRANSPORT
        return self._move(observation, target, cfg.close_gripper_m, ActivePhase.TRANSPORT)


def rule_config() -> RuleExpertConfig:
    """Expose the source defaults used for shared retreat/gripper semantics."""
    return RuleExpertConfig()

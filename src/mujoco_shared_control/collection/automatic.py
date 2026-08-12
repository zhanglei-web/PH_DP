"""Single-worker, no-render expert collection loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math

import numpy as np

from mujoco_shared_control.collection.recording import (
    AutoEpisodeRecorder,
    code_version,
    config_hash,
)
from mujoco_shared_control.collection.types import (
    AutoTransition,
    CollectionVariant,
    EpisodeOutcome,
    TerminationReason,
)
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.data.recording import FrameEvent, Stage, build_state_26
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import EpisodeContext, ExpertObservation
from mujoco_shared_control.experts.rule_pick_place import RuleExpertStage, RulePickPlaceExpert


@dataclass(frozen=True)
class CollectionConfig:
    dataset_root: str = "datasets/pick_box/expert_rule"
    control_timestep_s: float = 0.05
    max_steps: int = 500
    max_consecutive_ik_failures: int = 5
    success_settle_steps: int = 4
    settling_duration_s: float = 1.0
    randomize_arm: bool = True
    arm_joint_noise_scale: float = 1.0
    randomize_object: bool = True
    randomize_goal: bool = True
    perturbation_probability: float = 0.20
    perturbation_gamma: float = 0.35
    config_version: str = "rule_collection_v1"

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.success_settle_steps < 1:
            raise ValueError("step limits must be positive")
        if not 0.0 <= self.perturbation_probability <= 1.0:
            raise ValueError("perturbation_probability must be in [0, 1]")
        if not 0.0 <= self.perturbation_gamma <= 1.0:
            raise ValueError("perturbation_gamma must be in [0, 1]")
        if not np.isfinite(self.settling_duration_s) or self.settling_duration_s <= 0.0:
            raise ValueError("settling_duration_s must be finite and positive")

    @property
    def settling_steps(self) -> int:
        return int(math.ceil(self.settling_duration_s / self.control_timestep_s - 1e-12))


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    outcome: EpisodeOutcome
    termination_reason: TerminationReason
    transitions: int
    path: str
    valid: bool
    environment_seed: int
    variant: CollectionVariant


def _coarse_stage(expert_stage: RuleExpertStage) -> int:
    if expert_stage in (RuleExpertStage.PRE_GRASP, RuleExpertStage.DESCEND):
        return int(Stage.APPROACH)
    if expert_stage == RuleExpertStage.CLOSE_GRIPPER:
        return int(Stage.GRASP)
    if expert_stage in (RuleExpertStage.LIFT, RuleExpertStage.TRANSPORT):
        return int(Stage.TRANSPORT)
    if expert_stage in (RuleExpertStage.DESCEND_TO_GOAL, RuleExpertStage.OPEN_GRIPPER,
                        RuleExpertStage.RETREAT):
        return int(Stage.PLACE)
    return int(Stage.COMPLETE)


def _expert_observation(episode_id: str, worker_id: int, step: int,
                        obs: dict[str, Any], policy_state: np.ndarray,
                        previous_command: np.ndarray | None,
                        previous_action: np.ndarray | None) -> ExpertObservation:
    return ExpertObservation(
        episode_id=episode_id, worker_id=worker_id, step_index=step,
        simulation_time=float(obs["timestamp"][0]), state_26=build_state_26(obs),
        policy_state=policy_state, joint_position=obs["q_obs"],
        joint_velocity=obs["dq_obs"], ee_pose=obs["ee_pose"],
        gripper_opening=float(obs["gripper"][0]), object_pose=obs["object_pose"],
        object_linear_velocity=obs["object_linear_velocity"],
        object_angular_velocity=obs["object_angular_velocity"],
        object_grasped=bool(obs["object_grasped"]), goal_pose=obs["goal_pose"],
        previous_command=previous_command, previous_executed_action=previous_action,
    )


class AutomaticCollector:
    def __init__(self, config: CollectionConfig, *, worker_id: int = 0,
                 run_id: str | None = None, project_root: str | Path | None = None) -> None:
        self.config = config
        self.worker_id = worker_id
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.project_root = Path(project_root or Path(__file__).resolve().parents[4])
        self.env = PickPlaceEnv(
            render_mode=None, control_timestep=config.control_timestep_s,
            max_episode_steps=config.max_steps, enable_camera=False,
        )
        self.expert = RulePickPlaceExpert()
        self.adapter = ExpertCommandAdapter(self.env.ik_controller, self.expert.action_spec)

    def close(self) -> None:
        self.expert.close()
        self.env.close()

    def _perturb(self, command: np.ndarray, rng: np.random.Generator,
                 variant: CollectionVariant) -> tuple[np.ndarray, bool, float]:
        if variant == CollectionVariant.NOMINAL or rng.random() >= self.config.perturbation_probability:
            return command.copy(), False, 0.0
        random_action = self.expert.action_spec.denormalize(rng.uniform(-1.0, 1.0, 7))
        mixed = (1.0 - self.config.perturbation_gamma) * command + \
            self.config.perturbation_gamma * random_action
        return mixed, True, float(np.linalg.norm(mixed[:6] - command[:6]))

    def collect_episode(self, *, worker_episode_index: int, environment_seed: int,
                        variant: CollectionVariant = CollectionVariant.NOMINAL,
                        policy_seed: int | None = None,
                        perturbation_seed: int | None = None) -> EpisodeResult:
        policy_seed = environment_seed if policy_seed is None else policy_seed
        perturbation_seed = environment_seed + 1_000_003 if perturbation_seed is None else perturbation_seed
        episode_id = f"pick_box_{self.run_id}_w{self.worker_id:03d}_{worker_episode_index:06d}"
        reset_options = {
            "randomize_arm": self.config.randomize_arm,
            "arm_joint_noise_scale": self.config.arm_joint_noise_scale,
            "randomize_object": self.config.randomize_object,
            "randomize_goal": self.config.randomize_goal,
        }
        obs, info = self.env.reset(seed=environment_seed, options=reset_options)
        reset_parameters = {
            **reset_options,
            "arm_joint_position": info["arm_joint_position"].tolist(),
            "object_xy": info["object_xy"].tolist(), "goal_xy": info["goal_xy"].tolist(),
        }
        context = EpisodeContext(
            episode_id, "pick_box", self.run_id, self.worker_id, worker_episode_index,
            environment_seed, policy_seed, perturbation_seed, reset_parameters,
        )
        self.expert.reset(context)
        self.adapter.reset(obs["ee_pose"], obs["q_obs"])
        policy_rng = np.random.default_rng(policy_seed)
        del policy_rng  # Reserved for stochastic experts without changing this interface.
        perturbation_rng = np.random.default_rng(perturbation_seed)
        metadata = {
            "episode_id": episode_id, "task_name": "pick_box", "run_id": self.run_id,
            "worker_id": self.worker_id, "worker_episode_index": worker_episode_index,
            "environment_seed": environment_seed, "policy_seed": policy_seed,
            "perturbation_seed": perturbation_seed, "expert_type": self.expert.policy_name,
            "expert_code_version": code_version(self.project_root),
            "config_version": self.config.config_version,
            "config_hash": config_hash({"collection": asdict(self.config),
                                        "expert": self.expert.config.state_dict()}),
            "collection_variant": variant.value,
            "gamma": self.config.perturbation_gamma if variant == CollectionVariant.PERTURBED else 0.0,
            "gamma_definition": "a_policy=(1-gamma)*a_expert+gamma*a_random on selected perturbation steps",
            "reset_parameters_json": reset_parameters,
        }
        recorder = AutoEpisodeRecorder(self.config.dataset_root, metadata)
        previous_command = None
        previous_action = None
        previous_grasped = bool(obs["object_grasped"])
        previous_success = False
        consecutive_ik_failures = 0
        success_settle = 0
        entered_settling = False
        settling_step = -1
        expert_failed_step = -1
        settling_cause = ""
        milestones = np.zeros(5, dtype=np.uint8)  # grasp, lift, transport, release, retreat
        initial_object_z = float(obs["object_pose"][2, 3])
        any_perturbation = False
        outcome = EpisodeOutcome.FAILURE
        reason = TerminationReason.TIME_LIMIT
        try:
            for step in range(self.config.max_steps):
                was_settling = entered_settling
                policy_state = self.env.get_policy_observation(obs)
                expert_obs = _expert_observation(episode_id, self.worker_id, step, obs,
                                                 policy_state, previous_command, previous_action)
                if entered_settling:
                    stage_before = RuleExpertStage.SETTLING
                    command = self.expert.predict(expert_obs)
                    safe_command = np.r_[np.zeros(6), self.expert.action_spec.gripper_max_m]
                    perturbed, perturbation_active, magnitude = safe_command, False, 0.0
                else:
                    stage_before = self.expert.stage
                    command = self.expert.predict(expert_obs)
                    perturbed, perturbation_active, magnitude = self._perturb(
                        command.delta_pose_gripper, perturbation_rng, variant
                    )
                any_perturbation |= perturbation_active
                adapted = self.adapter.adapt(perturbed)
                consecutive_ik_failures = 0 if adapted.accepted else consecutive_ik_failures + 1
                next_obs, reward, env_success, _env_truncated, _step_info = self.env.step(adapted.joint_target)
                next_policy_state = self.env.get_policy_observation(next_obs)
                grasped = bool(next_obs["object_grasped"])
                milestones[0] |= int(grasped)
                milestones[1] |= int(milestones[0] and
                                     next_obs["object_pose"][2, 3] - initial_object_z >= 0.10)
                milestones[2] |= int(milestones[1] and self.expert.stage in (
                    RuleExpertStage.TRANSPORT, RuleExpertStage.DESCEND_TO_GOAL,
                    RuleExpertStage.OPEN_GRIPPER, RuleExpertStage.RETREAT,
                    RuleExpertStage.COMPLETE))
                milestones[3] |= int(milestones[0] and not grasped and
                                     (was_settling or self.expert.stage in (
                                         RuleExpertStage.RETREAT,
                                         RuleExpertStage.COMPLETE)))
                if entered_settling:
                    no_contact = not bool(next_obs["contact"]["count"][0])
                    # On expert failure the arm holds its current pose.  A pose is
                    # safe once it has no object contact; requiring an arbitrary
                    # distance would reject valid delayed drops despite no
                    # re-contact or active object manipulation.
                    milestones[4] |= int(no_contact)
                else:
                    milestones[4] |= int(self.expert.stage == RuleExpertStage.COMPLETE)
                released_inside = bool(env_success and not grasped)
                success_settle = success_settle + 1 if released_inside else 0
                complete_success = bool(
                    milestones.all() and success_settle >= self.config.success_settle_steps
                )
                expert_stage_after = self.expert.stage
                stage_after = (RuleExpertStage.SETTLING if entered_settling
                               else expert_stage_after)
                terminated = False
                truncated = False
                termination_reason = ""
                if not entered_settling and expert_stage_after == RuleExpertStage.FAILED:
                    entered_settling = True
                    settling_step = 0
                    expert_failed_step = step
                    settling_cause = "expert_failed"
                    stage_after = RuleExpertStage.SETTLING
                elif not entered_settling and expert_stage_after == RuleExpertStage.COMPLETE:
                    if complete_success:
                        terminated = True
                    else:
                        entered_settling = True
                        settling_step = 0
                        settling_cause = "post_retreat_stability"
                        stage_after = RuleExpertStage.SETTLING
                elif was_settling and complete_success:
                    terminated = True
                if terminated:
                    outcome = (EpisodeOutcome.SUCCESS if variant == CollectionVariant.NOMINAL
                               else EpisodeOutcome.RECOVERED)
                    reason = (TerminationReason.DELAYED_RECOVERY if expert_failed_step >= 0
                              else TerminationReason.TASK_SUCCESS)
                    termination_reason = reason.value
                elif consecutive_ik_failures >= self.config.max_consecutive_ik_failures:
                    terminated, reason, termination_reason = True, TerminationReason.IK_FAILURE_LIMIT, TerminationReason.IK_FAILURE_LIMIT.value
                elif was_settling and settling_step + 1 >= self.config.settling_steps:
                    terminated, reason, termination_reason = True, TerminationReason.SETTLING_TIMEOUT, TerminationReason.SETTLING_TIMEOUT.value
                elif step + 1 >= self.config.max_steps:
                    truncated, reason, termination_reason = True, TerminationReason.TIME_LIMIT, TerminationReason.TIME_LIMIT.value
                events = FrameEvent.NONE
                if grasped and not previous_grasped:
                    events |= FrameEvent.GRASP_ACQUIRED
                if not grasped and previous_grasped:
                    events |= FrameEvent.GRASP_LOST
                if env_success and not previous_success:
                    events |= FrameEvent.ENTERED_GOAL
                if complete_success:
                    events |= FrameEvent.TASK_SUCCESS
                transition = AutoTransition(
                    step, obs, next_obs, expert_obs.state_26, build_state_26(next_obs),
                    policy_state, next_policy_state, float(obs["timestamp"][0]),
                    float(next_obs["timestamp"][0]), command.delta_pose_gripper, perturbed,
                    adapted.clipped, adapted.normalized, adapted.cartesian_target,
                    adapted.joint_target, np.asarray(self.env.data.ctrl).copy(), command.valid,
                    adapted.accepted, adapted.action_clipped, adapted.fallback_used,
                    adapted.rejection_reason, reward, terminated, truncated, complete_success,
                    termination_reason, int(stage_before), int(stage_after),
                    _coarse_stage(stage_before), _coarse_stage(stage_after), int(events),
                    perturbation_active, "random_action_mix" if perturbation_active else "none",
                    magnitude, entered_settling, settling_step, expert_failed_step,
                    milestones.copy(),
                )
                recorder.append(transition)
                obs = next_obs
                previous_command = perturbed.copy()
                previous_action = adapted.joint_target.copy()
                previous_grasped, previous_success = grasped, bool(env_success)
                if terminated or truncated:
                    break
                if was_settling:
                    settling_step += 1
            report = recorder.finalize(outcome, reason.value, {
                "entered_settling": entered_settling,
                "expert_failed_step": expert_failed_step,
                "settling_cause": settling_cause,
                "settling_steps_limit": self.config.settling_steps,
                "settling_duration_s": self.config.settling_duration_s,
                "task_milestones_final": ",".join(map(str, milestones.tolist())),
            })
        except BaseException:
            recorder.abort()
            raise
        return EpisodeResult(episode_id, outcome, reason, step + 1, report["path"],
                             bool(report["valid"]), environment_seed, variant)

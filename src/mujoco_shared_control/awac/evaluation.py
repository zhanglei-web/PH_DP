"""Deterministic MuJoCo evaluation shared by AWAC and the BC baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mujoco_shared_control.actor_bc.evaluate import ActorPredictor
from mujoco_shared_control.awac.offline import AWACGaussianActor, OfflineAWACConfig
from mujoco_shared_control.awac.milestones import MilestoneConfig, MilestoneTracker
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


MILESTONES = ("grasp_success", "lift_success", "transport_success", "place_success", "retreat_success")


class DeterministicPredictor(Protocol):
    action_spec: ExpertActionSpec

    def normalized_action(
        self, policy_state: np.ndarray, object_grasped: bool | None = None,
        task_milestones: np.ndarray | None = None,
    ) -> np.ndarray: ...


class AWACCheckpointPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        self.model = AWACGaussianActor(OfflineAWACConfig(**payload["training_config"]))
        self.model.load_state_dict(payload["actor"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mean = np.asarray(payload["observation_mean"], np.float32)
        self.std = np.asarray(payload["observation_std"], np.float32)
        self.action_spec = ExpertActionSpec()

    def normalized_action(
        self, policy_state: np.ndarray, object_grasped: bool | None = None,
        task_milestones: np.ndarray | None = None,
    ) -> np.ndarray:
        del object_grasped, task_milestones
        normalized = (np.asarray(policy_state, np.float32) - self.mean) / self.std
        with torch.inference_mode():
            action = self.model.deterministic_action(
                torch.from_numpy(normalized).to(self.device).unsqueeze(0)
            ).squeeze(0).cpu().numpy()
        return np.asarray(action, np.float64)


class BCPredictorAdapter:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.predictor = ActorPredictor(checkpoint_path, device_name="cpu")
        self.action_spec = self.predictor.action_spec

    def normalized_action(
        self, policy_state: np.ndarray, object_grasped: bool | None = None,
        task_milestones: np.ndarray | None = None,
    ) -> np.ndarray:
        del object_grasped, task_milestones
        return np.clip(self.predictor.predict_unclipped(policy_state), -1.0, 1.0)


def evaluate_episode(
    predictor: DeterministicPredictor,
    seed: int,
    reward_config: AWACRewardV1Config = AWACRewardV1Config(),
) -> dict[str, Any]:
    config = CollectionConfig()
    env = PickPlaceEnv(
        render_mode=None, control_timestep=config.control_timestep_s,
        max_episode_steps=config.max_steps, enable_camera=False,
    )
    adapter = ExpertCommandAdapter(env.ik_controller, predictor.action_spec)
    try:
        observation, _ = env.reset(seed=seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        initial_state = np.r_[
            env.get_policy_observation(observation),
            np.float32(bool(observation["object_grasped"])),
        ].astype(np.float32)
        tracker = MilestoneTracker(MilestoneConfig(
            retreat_height_m=reward_config.retreat_height_m,
        ))
        reward_protocol = AWACRewardV1Online(initial_state, reward_config, tracker=tracker)
        milestones = tracker.current
        consecutive_ik_failures = 0
        ik_fallback_count = 0
        action_clipping_count = 0
        illegal_drop = False
        termination_reason = "time_limit"
        episode_return = 0.0
        release_step: int | None = None
        retreat_step: int | None = None
        for step in range(config.max_steps):
            state = env.get_policy_observation(observation)
            action = predictor.normalized_action(
                state, bool(observation["object_grasped"]), milestones.copy()
            )
            if action.shape != (7,) or not np.isfinite(action).all():
                termination_reason = "non_finite_actor_output"
                episode_return += reward_config.failure_penalty
                break
            if np.any((action < -1.0) | (action > 1.0)):
                raise RuntimeError("deterministic policy produced action outside [-1,1]")
            adapted = adapter.adapt(predictor.action_spec.denormalize(action))
            ik_fallback_count += int(adapted.fallback_used)
            action_clipping_count += int(adapted.action_clipped)
            consecutive_ik_failures = 0 if adapted.accepted else consecutive_ik_failures + 1
            next_observation, _env_reward, inside_goal, _env_truncated, _info = env.step(adapted.joint_target)
            next_state = env.get_policy_observation(next_observation)
            state_43 = np.r_[state, np.float32(bool(observation["object_grasped"]))].astype(np.float32)
            next_state_43 = np.r_[next_state, np.float32(bool(next_observation["object_grasped"]))].astype(np.float32)
            reward_step = reward_protocol.step(
                state_43, next_state_43,
                ik_failure=consecutive_ik_failures >= config.max_consecutive_ik_failures,
                time_limit=step + 1 >= config.max_steps,
            )
            milestones = tracker.current
            if release_step is None and bool(milestones[3]):
                release_step = step + 1
            if retreat_step is None and bool(milestones[4]):
                retreat_step = step + 1
            illegal_drop = reward_step.termination_reason == "illegal_drop"
            termination_reason = reward_step.termination_reason or "time_limit"
            episode_return += reward_step.reward
            observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                break
        return {
            "seed": seed,
            "task_success": termination_reason == "task_success",
            **{name: bool(milestones[index]) for index, name in enumerate(MILESTONES)},
            "illegal_drop": illegal_drop,
            "ik_failure": termination_reason == "ik_failure_limit",
            "timeout": termination_reason == "timeout",
            "termination_reason": termination_reason,
            "episode_return": float(episode_return),
            "episode_length": step + 1,
            "release_step": release_step,
            "retreat_step": retreat_step,
            "release_to_retreat_steps": (
                retreat_step - release_step
                if release_step is not None and retreat_step is not None else None
            ),
            "ik_fallback_count": ik_fallback_count,
            "action_clipping_count": action_clipping_count,
        }
    finally:
        env.close()


def evaluate_policy(
    predictor: DeterministicPredictor,
    seeds: list[int],
    reward_config: AWACRewardV1Config = AWACRewardV1Config(),
) -> dict[str, Any]:
    rows = []
    for index, seed in enumerate(seeds):
        rows.append(evaluate_episode(predictor, seed, reward_config))
        if (index + 1) % 10 == 0:
            print(f"closed-loop {index + 1}/{len(seeds)}", flush=True)
    returns = np.asarray([row["episode_return"] for row in rows], np.float64)
    lengths = np.asarray([row["episode_length"] for row in rows], np.float64)
    return {
        "episodes": len(rows), "seeds": [min(seeds), max(seeds)],
        "task_success": int(sum(row["task_success"] for row in rows)),
        "task_success_rate": float(np.mean([row["task_success"] for row in rows])),
        **{name: {"count": int(sum(row[name] for row in rows)), "rate": float(np.mean([row[name] for row in rows]))}
           for name in MILESTONES},
        "illegal_drop": {"count": int(sum(row["illegal_drop"] for row in rows)), "rate": float(np.mean([row["illegal_drop"] for row in rows]))},
        "ik_failure": {"count": int(sum(row["ik_failure"] for row in rows)), "rate": float(np.mean([row["ik_failure"] for row in rows]))},
        "timeout": {"count": int(sum(row["timeout"] for row in rows)), "rate": float(np.mean([row["timeout"] for row in rows]))},
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "average_episode_return": float(returns.mean()),
        "episode_return": {"mean": float(returns.mean()), "std": float(returns.std()), "min": float(returns.min()), "max": float(returns.max())},
        "episode_length": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        "ik_fallback_count": int(sum(row["ik_fallback_count"] for row in rows)),
        "action_clipping_count": int(sum(row["action_clipping_count"] for row in rows)),
        "rows": rows,
    }

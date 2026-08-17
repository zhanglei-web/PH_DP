"""Deterministic, isolated evaluation for online SAC v1."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.agent import SACCore


MILESTONES = ("grasped", "lifted", "transported", "released", "retreated")


def evaluate_sac(
    core: SACCore, seeds: list[int], *, reward_version: str = "sac_reward_v1",
) -> dict[str, Any]:
    """Uses independent environments and performs no replay writes or updates."""
    actor_training = core.actor.training
    core.actor.eval()
    rows = []
    evaluation_log_stds: list[np.ndarray] = []
    config = CollectionConfig()
    try:
        for seed in seeds:
            env = PickPlaceEnv(
                enable_camera=False, reward_version=reward_version,
                control_timestep=config.control_timestep_s, max_episode_steps=config.max_steps,
            )
            adapter = ExpertCommandAdapter(env.ik_controller, ExpertActionSpec(**core.action_spec))
            try:
                obs, info = env.reset(seed=seed, options={
                    "randomize_arm": config.randomize_arm,
                    "arm_joint_noise_scale": config.arm_joint_noise_scale,
                    "randomize_object": config.randomize_object,
                    "randomize_goal": config.randomize_goal,
                })
                adapter.reset(obs["ee_pose"], obs["q_obs"])
                initial_z = float(obs["object_pose"][2, 3])
                milestones = np.zeros(5, dtype=bool)
                episode_return = discounted_return = 0.0
                log_probs: list[float] = []
                consecutive_ik = 0
                reason = "time_limit"
                for step in range(config.max_steps):
                    state = info["policy_obs"]
                    action = core.select_action(state, deterministic=True)
                    normalized = core.normalize_observation(torch.as_tensor(state)).unsqueeze(0)
                    with torch.no_grad():
                        mean, log_std, std = core.actor.distribution_stats(normalized)
                        # Report log density at the deterministic pre-transform mean
                        # using the Actor's own transform/Jacobian convention.
                        normal = torch.distributions.Normal(mean, std)
                        if hasattr(core.actor, "_squashed_log_prob"):
                            log_probability = core.actor._squashed_log_prob(normal, mean)
                        else:
                            from mujoco_shared_control.sac.constrained_actor import constrained_transform
                            _mean_action, log_det = constrained_transform(mean)
                            log_probability = normal.log_prob(mean).sum(-1, keepdim=True) - log_det
                        log_probs.append(float(log_probability))
                        evaluation_log_stds.append(log_std.squeeze(0).cpu().numpy())
                    adapted = adapter.adapt(ExpertActionSpec(**core.action_spec).denormalize(action))
                    consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
                    safety = consecutive_ik >= config.max_consecutive_ik_failures
                    next_obs, reward, terminated, truncated, next_info = env.step(
                        adapted.joint_target, true_failure=safety,
                        failure_reason="ik_failure_limit",
                    )
                    episode_return += reward
                    discounted_return += (core.config.gamma ** step) * reward
                    grasped = bool(next_obs["object_grasped"])
                    obj = next_obs["object_pose"][:3, 3]
                    goal = next_obs["goal_pose"][:3, 3]
                    ee = next_obs["ee_pose"][:3, 3]
                    milestones[0] |= grasped
                    milestones[1] |= bool(milestones[0] and grasped and obj[2] - initial_z >= .10)
                    milestones[2] |= bool(milestones[1] and grasped and np.linalg.norm(obj[:2]-goal[:2]) < .055)
                    milestones[3] |= bool(next_info.get("successful_release", False))
                    if reward_version == "sac_reward_v2_candidate":
                        milestones[4] |= bool(next_info.get("success", False))
                    else:
                        milestones[4] |= bool(milestones[3] and np.linalg.norm(ee-(goal+[0,0,.16])) <= .008)
                    obs, info = next_obs, next_info
                    if terminated or truncated:
                        reason = next_info.get("termination_reason", "time_limit" if truncated else "other_failure")
                        break
                rows.append({
                    "seed": seed, "success": reason == "task_success",
                    "termination_reason": reason, "episode_return": episode_return,
                    "discounted_return": discounted_return, "episode_length": step + 1,
                    **{name: bool(milestones[i]) for i, name in enumerate(MILESTONES)},
                    "mean_log_prob": float(np.mean(log_probs)),
                })
            finally:
                env.close()
    finally:
        core.actor.train(actor_training)
    lengths = np.asarray([row["episode_length"] for row in rows])
    returns = np.asarray([row["episode_return"] for row in rows])
    discounted = np.asarray([row["discounted_return"] for row in rows])
    all_log_std = (
        np.concatenate(evaluation_log_stds)
        if evaluation_log_stds else np.asarray([np.nan])
    )
    return {
        "seeds": [min(seeds), max(seeds)], "episodes": len(rows),
        "reward_version": reward_version,
        "success": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "milestone_rates": {name: float(np.mean([row[name] for row in rows])) for name in MILESTONES},
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "episode_return": {"mean": float(returns.mean()), "median": float(np.median(returns)),
                           "min": float(returns.min()), "max": float(returns.max())},
        "discounted_return": {"mean": float(discounted.mean()), "min": float(discounted.min()), "max": float(discounted.max())},
        "episode_length": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        "mean_log_prob": float(np.mean([row["mean_log_prob"] for row in rows])),
        "log_std": {"mean": float(all_log_std.mean()), "min": float(all_log_std.min()),
                    "max": float(all_log_std.max())},
        "alpha": float(core.alpha.detach()), "rows": rows,
    }

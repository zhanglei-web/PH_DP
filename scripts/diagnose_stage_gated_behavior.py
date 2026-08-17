#!/usr/bin/env python3
"""Evaluate stage-gated Online behavior candidates without any learning."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor
from mujoco_shared_control.awac.online import low_noise_behavior_action
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig


CHECKPOINT = Path(
    "outputs/awac_training/awac_v3_geometric_milestone_state_offline25k_20260814T160000Z/checkpoint_best.pt"
)
PAIRED_REPORT = Path(
    "outputs/awac_diagnostics/online_retreat_paired_ab_20260814T190000Z/paired_ab_diagnosis.json"
)
OLD_BEHAVIOR_REPORT = Path(
    "outputs/awac_online/online_awac_v3_geometric_hybrid_20260814T180000Z/final_report.json"
)
OUTPUT = Path("outputs/awac_diagnostics/stage_gated_behavior_20260814T200000Z")
SEEDS = tuple(range(800_000, 800_005))
STD_SCALE = .25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenActor:
    def __init__(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["step"] != 20_000 or payload["format_version"] != "offline_awac_v3_geometric_milestone_state":
            raise RuntimeError("stage-gated diagnosis requires clean Offline 20k")
        if payload.get("online_transition_count", 0) != 0:
            raise RuntimeError("stage-gated diagnosis refuses an Online checkpoint")
        self.payload = payload
        self.actor = HybridActor(HybridAWACConfig(**payload["training_config"])).eval()
        self.actor.load_state_dict(payload["actor"])
        self.mean = torch.as_tensor(payload["observation_mean"], dtype=torch.float32)
        self.std = torch.as_tensor(payload["observation_std"], dtype=torch.float32)

    @torch.no_grad()
    def action(self, state: np.ndarray, scale: float) -> tuple[np.ndarray, float, float, float, float]:
        normalized = (torch.from_numpy(np.asarray(state, np.float32)) - self.mean) / self.std
        continuous, gripper, policy_std, effective_std, probability = low_noise_behavior_action(
            self.actor, normalized.unsqueeze(0), exploration_std_scale=scale)
        return (
            continuous[0].numpy().astype(np.float64), float(gripper.item()),
            float(probability.item()), float(policy_std.mean()), float(effective_std.mean()),
        )


def run_episode(seed: int, candidate: str, frozen: FrozenActor) -> dict[str, Any]:
    config = CollectionConfig(); reward_config = AWACRewardV1Config()
    spec = ExpertActionSpec(); rule = RuleExpertConfig()
    open_action = float(spec.normalize(np.r_[np.zeros(6), rule.open_gripper_m])[6])
    close_action = float(spec.normalize(np.r_[np.zeros(6), rule.close_gripper_m])[6])
    env = PickPlaceEnv(
        render_mode=None, control_timestep=config.control_timestep_s,
        max_episode_steps=config.max_steps, enable_camera=False,
    )
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    try:
        observation, _ = env.reset(seed=seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        initial43 = np.r_[env.get_policy_observation(observation),
                          np.float32(bool(observation["object_grasped"]))].astype(np.float32)
        reward_protocol = AWACRewardV1Online(initial43, reward_config)
        episode_return = 0.0; consecutive_ik = 0; clipping = fallback = 0
        transport_step = release_step = retreat_step = None
        release_rows: list[dict[str, Any]] = []
        stochastic_steps = deterministic_steps = 0
        for step_index in range(config.max_steps):
            state43 = np.r_[env.get_policy_observation(observation),
                            np.float32(bool(observation["object_grasped"]))].astype(np.float32)
            milestones = reward_protocol.tracker.current
            state48 = np.r_[state43, milestones.astype(np.float32)].astype(np.float32)
            if candidate == "transport_gated":
                scale = STD_SCALE if not milestones[2] else 0.0
            elif candidate == "lift_gated":
                scale = STD_SCALE if not milestones[1] else 0.0
            else:
                raise ValueError(candidate)
            stochastic_steps += int(scale > 0); deterministic_steps += int(scale == 0)
            continuous, binary_gripper, close_probability, policy_std, effective_std = frozen.action(
                state48, scale)
            if milestones[3] and not milestones[4]:
                target = reward_protocol.tracker.retreat_target(state43)
                release_rows.append({
                    "episode_step": step_index + 1,
                    "distance_to_retreat_target_m": float(np.linalg.norm(state43[14:17] - target)),
                    "continuous_action": continuous.astype(float).tolist(),
                    "dx": float(continuous[0]), "dy": float(continuous[1]),
                    "dz": float(continuous[2]), "drx": float(continuous[3]),
                    "dry": float(continuous[4]), "drz": float(continuous[5]),
                    "gripper_close_probability": close_probability,
                    "policy_std_mean": policy_std, "effective_std_mean": effective_std,
                })
            normalized_gripper = close_action if binary_gripper else open_action
            adapted = adapter.adapt(spec.denormalize(np.r_[continuous, normalized_gripper]))
            clipping += int(adapted.action_clipped); fallback += int(adapted.fallback_used)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, *_ = env.step(adapted.joint_target)
            next43 = np.r_[env.get_policy_observation(next_observation),
                           np.float32(bool(next_observation["object_grasped"]))].astype(np.float32)
            reward_step = reward_protocol.step(
                state43, next43,
                ik_failure=consecutive_ik >= config.max_consecutive_ik_failures,
                time_limit=step_index + 1 >= config.max_steps,
            )
            episode_return += reward_step.reward
            if transport_step is None and reward_step.milestones[2]: transport_step = step_index + 1
            if release_step is None and reward_step.milestones[3]: release_step = step_index + 1
            if retreat_step is None and reward_step.milestones[4]: retreat_step = step_index + 1
            observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                break
        if release_rows:
            distance = np.asarray([row["distance_to_retreat_target_m"] for row in release_rows])
            minimum_index = int(np.argmin(distance)); minimum = release_rows[minimum_index]
            first_twenty = release_rows[:20]
            first_negative = None
            for previous, current in zip(release_rows, release_rows[1:]):
                if previous["dz"] > 0 and current["dz"] < 0:
                    first_negative = {
                        "episode_step": current["episode_step"],
                        "distance_to_retreat_target_m": current["distance_to_retreat_target_m"],
                        "continuous_action": current["continuous_action"],
                    }
                    break
            release_diagnostic = {
                "states": len(release_rows),
                "minimum_distance_m": float(distance[minimum_index]),
                "action_at_minimum_distance": minimum["continuous_action"],
                "close_probability_at_minimum_distance": minimum["gripper_close_probability"],
                "first_20_dz_mean": float(np.mean([row["dz"] for row in first_twenty])),
                "first_20_dz_positive_ratio": float(np.mean([row["dz"] > 0 for row in first_twenty])),
                "effective_std_max": float(max(row["effective_std_mean"] for row in release_rows)),
                "first_positive_to_negative": first_negative,
            }
        else:
            release_diagnostic = None
        return {
            "seed": seed, "candidate": candidate,
            "task_success": reward_step.task_success,
            "termination_reason": reward_step.termination_reason,
            "episode_length": step_index + 1, "episode_return": episode_return,
            "milestones": reward_step.milestones.astype(int).tolist(),
            "transport_step": transport_step, "release_step": release_step,
            "retreat_step": retreat_step,
            "release_to_retreat_steps": (
                retreat_step - release_step
                if release_step is not None and retreat_step is not None else None),
            "stochastic_steps": stochastic_steps,
            "deterministic_steps": deterministic_steps,
            "action_clipping_count": clipping, "ik_fallback_count": fallback,
            "release_diagnostic": release_diagnostic,
        }
    finally:
        env.close()


def run_candidate(name: str, frozen: FrozenActor) -> dict[str, Any]:
    # Each candidate starts from the exact clean checkpoint RNG state.
    torch.set_rng_state(frozen.payload["rng_state"]["torch"])
    rows = [run_episode(seed, name, frozen) for seed in SEEDS]
    successes = int(sum(row["task_success"] for row in rows))
    long_timeout = int(sum(
        row["termination_reason"] == "timeout"
        and row["release_step"] is not None and row["retreat_step"] is None
        for row in rows
    ))
    release_to_retreat = [row["release_to_retreat_steps"] for row in rows
                          if row["release_to_retreat_steps"] is not None]
    return {
        "candidate": name, "episodes": 5, "success": successes,
        "success_rate": successes / 5,
        "grasp": int(sum(row["milestones"][0] for row in rows)),
        "lift": int(sum(row["milestones"][1] for row in rows)),
        "transport": int(sum(row["milestones"][2] for row in rows)),
        "release": int(sum(row["milestones"][3] for row in rows)),
        "retreat": int(sum(row["milestones"][4] for row in rows)),
        "timeout": int(sum(row["termination_reason"] == "timeout" for row in rows)),
        "illegal_drop": int(sum(row["termination_reason"] == "illegal_drop" for row in rows)),
        "ik_failure": int(sum(row["termination_reason"] == "ik_failure_limit" for row in rows)),
        "long_post_release_timeout": long_timeout,
        "average_return": float(np.mean([row["episode_return"] for row in rows])),
        "release_to_retreat_mean_steps": (
            float(np.mean(release_to_retreat)) if release_to_retreat else None),
        "stochastic_steps": int(sum(row["stochastic_steps"] for row in rows)),
        "deterministic_steps": int(sum(row["deterministic_steps"] for row in rows)),
        "passed": successes >= 4 and long_timeout == 0,
        "rows": rows,
    }


def main() -> None:
    output = OUTPUT.resolve(); output.mkdir(parents=True, exist_ok=False)
    checkpoint = CHECKPOINT.resolve(); frozen = FrozenActor(checkpoint)
    if asdict(AWACRewardV1Config()) != frozen.payload["reward_config"]:
        raise RuntimeError("Reward V1 changed")
    candidate_a = run_candidate("transport_gated", frozen)
    candidate_b = None if candidate_a["passed"] else run_candidate("lift_gated", frozen)
    if candidate_a["passed"]:
        selected = "transport_gated"
    elif candidate_b is not None and candidate_b["passed"]:
        selected = "lift_gated"
    else:
        selected = None
    paired = json.loads(PAIRED_REPORT.read_text())
    old = json.loads(OLD_BEHAVIOR_REPORT.read_text())
    report = {
        "status": "complete_no_training", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "checkpoint_step": frozen.payload["step"],
        "models_or_optimizers_updated": False, "online_awac_updates": 0,
        "seeds": list(SEEDS), "std_scale": STD_SCALE,
        "full_deterministic_reference": {
            "success": paired["full_deterministic_success"],
            "release": 5, "retreat": 5, "timeout": 0,
            "release_to_retreat_mean_steps": float(np.mean([
                pair["full_deterministic"]["release_to_retreat_steps"]
                for pair in paired["pairs"]])),
        },
        "old_release_gated_reference": {
            **old["diagnostics"]["sanity_rollout"],
            "release_to_retreat_mean_steps": old["diagnostics"]["release_to_retreat"]["average_steps"],
        },
        "candidate_a": candidate_a, "candidate_b": candidate_b,
        "selected_candidate": selected,
        "eligible_to_start_online_5k": selected is not None,
    }
    (output / "stage_gated_behavior_diagnosis.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "candidate_a": {k: candidate_a[k] for k in (
            "success", "release", "retreat", "timeout", "long_post_release_timeout",
            "release_to_retreat_mean_steps", "passed")},
        "candidate_b": ({k: candidate_b[k] for k in (
            "success", "release", "retreat", "timeout", "long_post_release_timeout",
            "release_to_retreat_mean_steps", "passed")} if candidate_b else None),
        "selected_candidate": selected,
    }, indent=2))


if __name__ == "__main__":
    main()

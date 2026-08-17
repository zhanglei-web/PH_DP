#!/usr/bin/env python3
"""Collect an immutable deterministic Step-0 Actor dataset under Reward v2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_adaptation import (
    ActorTransitionArrays, monte_carlo_returns, module_checksum,
)


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
SEED_START = 800_000
EPISODES = 1_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(seed: int) -> str:
    offset = seed - SEED_START
    if 0 <= offset < 800: return "train"
    if offset < 900: return "validation"
    if offset < 1000: return "test"
    raise ValueError(seed)


def outcome_for(reason: str) -> str:
    if reason == "task_success": return "success"
    if reason == "illegal_drop": return "illegal_drop"
    if reason == "time_limit": return "timeout"
    return "other_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    root = Path("outputs/critic_adaptation") / f"critic_only_online_adaptation_v1_{args.run_id}"
    data_dir = root / "online_actor_dataset"
    data_dir.mkdir(parents=True, exist_ok=False)

    actor, _critic, _target, payload = load_aligned_v2(ALIGNED)
    actor.eval(); actor.requires_grad_(False)
    actor_before = module_checksum(actor)
    mean = payload["observation_mean"]; std = payload["observation_std"]
    config = CollectionConfig(); spec = ExpertActionSpec(**payload["action_spec"])
    accum = {split: {name: [] for name in ActorTransitionArrays.__dataclass_fields__}
             for split in ("train", "validation", "test")}
    episodes = []
    phase_counts = Counter(); outcome_counts = Counter(); projection_count = 0
    fallback_count = 0; policy_deployed_error = []

    for number, seed in enumerate(range(SEED_START, SEED_START + EPISODES), 1):
        split = split_for(seed)
        env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v2_candidate",
                           control_timestep=config.control_timestep_s,
                           max_episode_steps=config.max_steps)
        adapter = ExpertCommandAdapter(env.ik_controller, spec)
        rows = []
        milestones = np.zeros(4, dtype=bool); reason = "time_limit"
        try:
            obs, info = env.reset(seed=seed, options={
                "randomize_arm": config.randomize_arm,
                "arm_joint_noise_scale": config.arm_joint_noise_scale,
                "randomize_object": config.randomize_object,
                "randomize_goal": config.randomize_goal,
            })
            adapter.reset(obs["ee_pose"], obs["q_obs"]); consecutive_ik = 0
            initial_z = float(obs["object_pose"][2, 3])
            for step in range(config.max_steps):
                state = np.asarray(info["policy_obs"], np.float32)
                with torch.no_grad():
                    normalized = (torch.from_numpy(state) - mean) / std
                    action = actor.deterministic_action(normalized.unsqueeze(0)).squeeze(0).numpy()
                if (np.linalg.norm(action[:3]) > 1 + 1e-6
                        or np.linalg.norm(action[3:6]) > 1 + 1e-6
                        or abs(action[6]) > 1 + 1e-6):
                    raise RuntimeError("inadmissible deterministic Actor action")
                adapted = adapter.adapt(spec.denormalize(action))
                projection_count += int(adapted.action_clipped)
                fallback_count += int(adapted.fallback_used)
                if not adapted.fallback_used:
                    policy_deployed_error.append(float(np.max(np.abs(action-adapted.normalized))))
                consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
                safety = consecutive_ik >= config.max_consecutive_ik_failures
                next_obs, reward, terminated, truncated, next_info = env.step(
                    adapted.joint_target, true_failure=safety,
                    failure_reason="ik_failure_limit",
                )
                grasped = bool(next_obs["object_grasped"])
                obj = next_obs["object_pose"][:3, 3]; goal = next_obs["goal_pose"][:3, 3]
                milestones[0] |= grasped
                milestones[1] |= bool(milestones[0] and grasped and obj[2]-initial_z >= .10)
                milestones[2] |= bool(milestones[1] and grasped and np.linalg.norm(obj[:2]-goal[:2]) < .055)
                milestones[3] |= bool(next_info.get("success", False))
                rows.append({
                    "observation": state, "action": action.astype(np.float32),
                    "reward": np.asarray([reward], np.float32),
                    "next_observation": np.asarray(next_info["policy_obs"], np.float32),
                    "terminated": np.asarray([terminated], bool),
                    "truncated": np.asarray([truncated], bool),
                    "phase": str(next_info["phase_name"]), "step": step,
                })
                phase_counts[str(next_info["phase_name"])] += 1
                obs, info = next_obs, next_info
                if terminated or truncated:
                    reason = str(next_info.get("termination_reason",
                                               "time_limit" if truncated else "other_failure"))
                    break
        finally:
            env.close()
        outcome = outcome_for(reason); outcome_counts[outcome] += 1
        rewards = np.stack([row["reward"] for row in rows])
        terminated = np.stack([row["terminated"] for row in rows])
        truncated = np.stack([row["truncated"] for row in rows])
        returns = monte_carlo_returns(rewards, terminated, truncated)
        episode_id = f"frozen_actor_v2_{seed}"
        values = {
            "observation": np.stack([row["observation"] for row in rows]),
            "action": np.stack([row["action"] for row in rows]),
            "reward": rewards, "next_observation": np.stack([row["next_observation"] for row in rows]),
            "terminated": terminated, "truncated": truncated, "mc_return": returns,
            "phase": np.asarray([row["phase"] for row in rows], dtype="U24"),
            "episode_id": np.full(len(rows), episode_id, dtype="U32"),
            "seed": np.full(len(rows), seed, np.int32),
            "step": np.asarray([row["step"] for row in rows], np.int32),
            "outcome": np.full(len(rows), outcome, dtype="U16"),
        }
        for name, value in values.items(): accum[split][name].append(value)
        episodes.append({"episode_id": episode_id, "seed": seed, "split": split,
                         "outcome": outcome, "termination_reason": reason,
                         "transitions": len(rows), "return": float(rewards.sum()),
                         "discounted_return": float(returns[0]),
                         "grasp": bool(milestones[0]), "lift": bool(milestones[1]),
                         "transport": bool(milestones[2]),
                         "release_stable_success": bool(milestones[3])})
        if number % 100 == 0:
            print(f"episodes={number}/{EPISODES} outcomes={dict(outcome_counts)}", flush=True)

    arrays = {split: ActorTransitionArrays(**{
        name: np.concatenate(parts) for name, parts in fields.items()
    }) for split, fields in accum.items()}
    for split, array in arrays.items(): array.save(data_dir / f"{split}.npz")
    if actor_before != module_checksum(actor): raise RuntimeError("frozen Actor changed")
    ids = {split: set(np.unique(array.episode_id)) for split, array in arrays.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise RuntimeError("online episode split leakage")
    manifest = {
        "format_version": "frozen_actor_v2_deterministic_dataset_v1",
        "actor_source": str(ALIGNED.resolve()), "actor_file_sha256": sha(ALIGNED),
        "actor_checksum": actor_before, "reward_version": "sac_reward_v2_candidate",
        "policy": "deterministic constrained mean; no sampling/noise",
        "seed_range": [SEED_START, SEED_START+EPISODES-1],
        "sealed_final_test_range": [500000, 500099],
        "split": {"train": [800000,800799], "validation": [800800,800899],
                  "test": [800900,800999]},
        "episodes": {split: len(ids[split]) for split in ids},
        "transitions": {split: len(array) for split, array in arrays.items()},
        "adapter_projection_count": projection_count, "fallback_count": fallback_count,
        "normal_policy_deployed_max_abs_error": max(policy_deployed_error, default=0.0),
    }
    phase_stats = {"transition_counts": dict(phase_counts),
                   "split_transition_counts": {split: dict(Counter(array.phase))
                                                for split,array in arrays.items()}}
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (data_dir / "episode_outcomes.json").write_text(json.dumps({
        "counts": dict(outcome_counts), "milestones": {
            key: int(sum(row[key] for row in episodes)) for key in
            ("grasp","lift","transport","release_stable_success")}, "episodes": episodes}, indent=2)+"\n")
    (data_dir / "phase_statistics.json").write_text(json.dumps(phase_stats, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()

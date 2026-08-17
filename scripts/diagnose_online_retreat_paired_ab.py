#!/usr/bin/env python3
"""Paired deterministic-vs-milestone-conditioned retreat diagnosis; no training."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig


CHECKPOINT = Path(
    "outputs/awac_training/awac_v3_geometric_milestone_state_offline25k_20260814T160000Z/checkpoint_best.pt"
)
TRAIN_DATASET = Path("outputs/awac_dataset/awac_v3_geometric_milestone_state/train.npz")
BEHAVIOR_RUN = Path("outputs/awac_online/online_awac_v3_geometric_hybrid_20260814T180000Z")
OUTPUT = Path("outputs/awac_diagnostics/online_retreat_paired_ab_20260814T190000Z")
SEEDS = tuple(range(800_000, 800_005))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_detail(
    state: np.ndarray, continuous: np.ndarray, close_probability: float,
    tracker: Any,
) -> dict[str, Any]:
    state = np.asarray(state, np.float32)
    target = tracker.retreat_target(state[:43])
    return {
        "observation_48": state.astype(float).tolist(),
        "policy_state_42": state[:42].astype(float).tolist(),
        "object_grasped": bool(state[42] > .5),
        "milestones": state[43:48].astype(int).tolist(),
        "joint_positions": state[:7].astype(float).tolist(),
        "ee_pose_xyz_wxyz": state[14:21].astype(float).tolist(),
        "gripper_opening": float(state[21]),
        "object_pose_xyz_wxyz": state[22:29].astype(float).tolist(),
        "goal_pose_xyz_wxyz": state[29:36].astype(float).tolist(),
        "retreat_target_xyz": target.astype(float).tolist(),
        "ee_to_retreat_target_m": float(np.linalg.norm(state[14:17] - target)),
        "actor_deterministic_continuous": np.asarray(continuous).astype(float).tolist(),
        "actor_dz": float(continuous[2]),
        "gripper_close_probability": float(close_probability),
    }


class FrozenActor:
    def __init__(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["step"] != 20_000 or payload["format_version"] != "offline_awac_v3_geometric_milestone_state":
            raise RuntimeError("paired diagnosis requires clean geometric Offline 20k")
        self.payload = payload
        self.config = HybridAWACConfig(**payload["training_config"])
        self.actor = HybridActor(self.config).eval()
        self.actor.load_state_dict(payload["actor"])
        self.mean = np.asarray(payload["observation_mean"], np.float32)
        self.std = np.asarray(payload["observation_std"], np.float32)

    @torch.no_grad()
    def action(self, state: np.ndarray) -> tuple[np.ndarray, float, float]:
        normalized = torch.from_numpy((np.asarray(state, np.float32) - self.mean) / self.std).unsqueeze(0)
        continuous, gripper, probability = self.actor.deterministic_action(normalized)
        return (
            continuous[0].numpy().astype(np.float64),
            float(gripper.item()), float(probability.item()),
        )


def deterministic_episode(seed: int, frozen: FrozenActor) -> dict[str, Any]:
    config = CollectionConfig()
    reward_config = AWACRewardV1Config()
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
        release_state = None; release_step = None; retreat_step = None
        episode_return = 0.0; consecutive_ik = 0; fallback = clipping = 0
        min_retreat_distance = float("inf")
        for step_index in range(config.max_steps):
            state43 = np.r_[env.get_policy_observation(observation),
                            np.float32(bool(observation["object_grasped"]))].astype(np.float32)
            state48 = np.r_[state43, reward_protocol.tracker.current.astype(np.float32)].astype(np.float32)
            continuous, binary_gripper, close_probability = frozen.action(state48)
            normalized_gripper = close_action if binary_gripper else open_action
            adapted = adapter.adapt(spec.denormalize(np.r_[continuous, normalized_gripper]))
            fallback += int(adapted.fallback_used); clipping += int(adapted.action_clipped)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, *_ = env.step(adapted.joint_target)
            next43 = np.r_[env.get_policy_observation(next_observation),
                           np.float32(bool(next_observation["object_grasped"]))].astype(np.float32)
            reward_step = reward_protocol.step(
                state43, next43,
                ik_failure=consecutive_ik >= config.max_consecutive_ik_failures,
                time_limit=step_index + 1 >= config.max_steps,
            )
            next48 = np.r_[next43, reward_protocol.tracker.current.astype(np.float32)].astype(np.float32)
            episode_return += reward_step.reward
            if release_state is None and bool(next48[46]):
                release_step = step_index + 1
                release_continuous, _release_binary, release_probability = frozen.action(next48)
                release_state = state_detail(
                    next48, release_continuous, release_probability,
                    reward_protocol.tracker,
                )
            if bool(next48[46]) and not bool(next48[47]):
                min_retreat_distance = min(
                    min_retreat_distance,
                    float(np.linalg.norm(
                        next43[14:17] - reward_protocol.tracker.retreat_target(next43))),
                )
            if retreat_step is None and bool(next48[47]):
                retreat_step = step_index + 1
            observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                return {
                    "seed": seed, "task_success": reward_step.task_success,
                    "termination_reason": reward_step.termination_reason,
                    "episode_length": step_index + 1, "episode_return": episode_return,
                    "milestones": reward_step.milestones.astype(int).tolist(),
                    "release_step": release_step, "retreat_step": retreat_step,
                    "release_to_retreat_steps": (
                        retreat_step - release_step
                        if release_step is not None and retreat_step is not None else None),
                    "minimum_retreat_distance_after_release_m": (
                        min_retreat_distance if np.isfinite(min_retreat_distance) else None),
                    "release_state": release_state,
                    "ik_fallback_count": fallback, "action_clipping_count": clipping,
                }
        raise RuntimeError("episode loop ended without Reward V1 terminal")
    finally:
        env.close()


def behavior_rows(frozen: FrozenActor) -> dict[int, dict[str, Any]]:
    log_path = BEHAVIOR_RUN / "online_replay/online_transitions_online_sanity_05ep.npz"
    episodes = {
        int(row["environment_seed"]): row
        for row in map(json.loads, (BEHAVIOR_RUN / "online_episodes.jsonl").read_text().splitlines())
    }
    output: dict[int, dict[str, Any]] = {}
    with np.load(log_path, allow_pickle=False) as data:
        for seed in SEEDS:
            episode_id = f"online_{seed - 800_000:06d}"
            indices = np.flatnonzero(data["episode_id"] == episode_id)
            release_indices = indices[
                (data["obs"][indices, 46] > .5) & (data["obs"][indices, 47] <= .5)]
            if not len(release_indices):
                raise RuntimeError(f"behavior episode has no release state: {seed}")
            index = int(release_indices[0]); state = np.asarray(data["obs"][index], np.float32)
            continuous, _gripper, probability = frozen.action(state)
            # A temporary tracker is only used for the frozen retreat target helper.
            from mujoco_shared_control.awac.milestones import MilestoneTracker
            tracker = MilestoneTracker(); tracker.reset(state[:43])
            detail = state_detail(state, continuous, probability, tracker)
            episode = episodes[seed]
            output[seed] = {
                "seed": seed, "task_success": episode["task_success"],
                "termination_reason": episode["termination_reason"],
                "episode_length": episode["episode_length"],
                "episode_return": episode["episode_return"],
                "milestones": episode["milestones"],
                "release_step": episode["release_step"],
                "retreat_step": episode["retreat_step"],
                "release_to_retreat_steps": episode["release_to_retreat_steps"],
                "release_state": detail,
            }
    return output


def add_nearest_neighbors(
    rows: dict[int, dict[str, Any]], frozen: FrozenActor,
    offline_states: np.ndarray, offline_normalized: np.ndarray,
) -> None:
    for row in rows.values():
        state = np.asarray(row["release_state"]["observation_48"], np.float32)
        normalized = (state - frozen.mean) / frozen.std
        distances = np.linalg.norm(offline_normalized - normalized, axis=1)
        index = int(np.argmin(distances))
        nearest = offline_states[index]
        nearest_continuous, _gripper, nearest_probability = frozen.action(nearest)
        row["release_state_support"] = {
            "normalized_euclidean_nearest_distance": float(distances[index]),
            "normalized_rms_nearest_distance": float(distances[index] / np.sqrt(48)),
            "nearest_offline_state_index_within_release_subset": index,
            "nearest_offline_ee_pose_xyz_wxyz": nearest[14:21].astype(float).tolist(),
            "nearest_offline_object_pose_xyz_wxyz": nearest[22:29].astype(float).tolist(),
            "nearest_offline_goal_pose_xyz_wxyz": nearest[29:36].astype(float).tolist(),
            "nearest_offline_actor_continuous": nearest_continuous.astype(float).tolist(),
            "nearest_offline_actor_dz": float(nearest_continuous[2]),
            "nearest_offline_gripper_close_probability": nearest_probability,
        }


def summarize_support(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows.values() if row["task_success"]]
    failed = [row for row in rows.values() if not row["task_success"]]
    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        if not values:
            return {"episodes": 0, "distance_mean": None, "distance_min": None,
                    "distance_max": None, "actor_dz_mean": None}
        distances = [row["release_state_support"]["normalized_euclidean_nearest_distance"] for row in values]
        dz = [row["release_state"]["actor_dz"] for row in values]
        return {"episodes": len(values), "distance_mean": float(np.mean(distances)),
                "distance_min": float(np.min(distances)), "distance_max": float(np.max(distances)),
                "actor_dz_mean": float(np.mean(dz))}
    return {"successful": group(successful), "failed": group(failed)}


def main() -> None:
    output = OUTPUT.resolve(); output.mkdir(parents=True, exist_ok=False)
    checkpoint = CHECKPOINT.resolve(); dataset = TRAIN_DATASET.resolve()
    frozen = FrozenActor(checkpoint)
    if frozen.config.observation_dim != 48:
        raise RuntimeError("paired diagnosis requires 48-D Actor")
    if asdict(AWACRewardV1Config()) != frozen.payload["reward_config"]:
        raise RuntimeError("Reward V1 mismatch")
    deterministic = {seed: deterministic_episode(seed, frozen) for seed in SEEDS}
    behavior = behavior_rows(frozen)
    with np.load(dataset, allow_pickle=False) as data:
        states = np.asarray(data["obs"], np.float32)
        mask = (states[:, 46] > .5) & (states[:, 47] <= .5)
        offline_release = states[mask]
    offline_normalized = (offline_release - frozen.mean) / frozen.std
    add_nearest_neighbors(deterministic, frozen, offline_release, offline_normalized)
    add_nearest_neighbors(behavior, frozen, offline_release, offline_normalized)
    pairs = [{
        "seed": seed,
        "full_deterministic": deterministic[seed],
        "low_noise_before_release_deterministic_after_release": behavior[seed],
    } for seed in SEEDS]
    report = {
        "status": "complete_no_training",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": frozen.payload["step"],
        "actor_critic_optimizer_updated": False,
        "seeds": list(SEEDS),
        "retreat_condition": {
            "target": "goal_xyz + [0,0,0.16]", "distance_tolerance_m": .008,
        },
        "offline_release_state_count": int(len(offline_release)),
        "pairs": pairs,
        "support_summary": {
            "full_deterministic": summarize_support(deterministic),
            "low_noise_behavior": summarize_support(behavior),
        },
        "full_deterministic_success": int(sum(row["task_success"] for row in deterministic.values())),
        "low_noise_behavior_success": int(sum(row["task_success"] for row in behavior.values())),
        "online_awac_updates": 0,
    }
    (output / "paired_ab_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "full_deterministic_success": report["full_deterministic_success"],
        "low_noise_behavior_success": report["low_noise_behavior_success"],
        "paired": [{
            "seed": pair["seed"],
            "deterministic": pair["full_deterministic"]["termination_reason"],
            "behavior": pair["low_noise_before_release_deterministic_after_release"]["termination_reason"],
            "deterministic_nn": pair["full_deterministic"]["release_state_support"]["normalized_euclidean_nearest_distance"],
            "behavior_nn": pair["low_noise_before_release_deterministic_after_release"]["release_state_support"]["normalized_euclidean_nearest_distance"],
        } for pair in pairs],
        "support_summary": report["support_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()

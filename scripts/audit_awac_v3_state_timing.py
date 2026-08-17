#!/usr/bin/env python3
"""Read-only timing and evaluator-state audit for the stopped AWAC-v3 BC."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv


MILESTONES = ("grasp", "lift", "transport", "release", "retreat")


def combo(value: np.ndarray) -> str:
    return "[" + ",".join(str(int(item)) for item in value) + "]"


def combo_report(values: np.ndarray) -> dict[str, Any]:
    counts = Counter(combo(row) for row in values.astype(np.uint8))
    total = len(values)
    illegal = sum(
        count for key, count in counts.items()
        if any(int(key[1 + 2 * index]) and "0" in key[1:1 + 2 * index] for index in range(1, 5))
    )
    return {
        "total_states": total,
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "fractions": {key: value / total for key, value in sorted(counts.items())},
        "illegal_order_count": illegal,
    }


def geometric_update(
    milestones: np.ndarray, state43: np.ndarray, next_state43: np.ndarray,
    initial_object_z: float, config: AWACRewardV1Config,
) -> np.ndarray:
    result = milestones.copy()
    was_grasped = bool(state43[42]); grasped = bool(next_state43[42])
    obj = next_state43[22:25]; goal = next_state43[29:32]; ee = next_state43[14:17]
    inside_goal = bool(np.linalg.norm(obj - goal) < 0.055)
    result[0] |= grasped
    result[1] |= bool(result[0] and grasped and obj[2] - initial_object_z >= 0.10)
    result[2] |= bool(result[1] and grasped and np.linalg.norm(obj[:2] - goal[:2]) < 0.055)
    valid_release = bool(result[2] and was_grasped and not grasped and inside_goal)
    result[3] |= valid_release
    result[4] |= bool(
        result[3]
        and np.linalg.norm(ee - (goal + np.array([0.0, 0.0, config.retreat_height_m]))) <= 0.008
    )
    return result


def evaluator_milestones_for_dataset(data: dict[str, np.ndarray]) -> np.ndarray:
    generated = np.zeros((len(data["obs"]), 5), np.uint8)
    ids = data["episode_id"]
    for episode_id in np.unique(ids):
        indices = np.flatnonzero(ids == episode_id)
        indices = indices[np.argsort(data["step_index"][indices])]
        milestones = np.zeros(5, bool)
        initial_z = float(data["obs"][indices[0], 24])
        for index in indices:
            generated[index] = milestones
            milestones = geometric_update(
                milestones, data["obs"][index, :43], data["next_obs"][index, :43],
                initial_z, AWACRewardV1Config(),
            )
    return generated


@torch.no_grad()
def action_metrics(predictor: HybridCheckpointPredictor, states: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    normalized = (states.astype(np.float32) - predictor.mean) / predictor.std
    continuous, gripper, probability = predictor.model.deterministic_action(torch.from_numpy(normalized))
    predicted = continuous.numpy(); target = data["continuous_action"]
    predicted_gripper = gripper.numpy().reshape(-1); target_gripper = data["gripper_action"]
    return {
        "continuous_mse": float(np.mean((predicted - target) ** 2)),
        "continuous_per_dimension_mse": np.mean((predicted - target) ** 2, axis=0).astype(float).tolist(),
        "gripper_accuracy": float(np.mean(predicted_gripper == target_gripper)),
        "predicted_close_ratio": float(predicted_gripper.mean()),
        "target_close_ratio": float(target_gripper.mean()),
        "mean_close_probability": float(probability.mean()),
    }


def closed_loop_combinations(checkpoint: Path, seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictor = HybridCheckpointPredictor(checkpoint)
    config = CollectionConfig(); all_states = []; outcomes = []
    for seed in seeds:
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
            state43 = np.r_[
                env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"])),
            ].astype(np.float32)
            protocol = AWACRewardV1Online(state43)
            milestones = np.zeros(5, bool); consecutive = 0; reason = "timeout"
            for step in range(config.max_steps):
                all_states.append(milestones.astype(np.uint8).copy())
                policy_state = env.get_policy_observation(observation)
                action = predictor.normalized_action(
                    policy_state, bool(observation["object_grasped"]), milestones,
                )
                adapted = adapter.adapt(predictor.action_spec.denormalize(action))
                consecutive = 0 if adapted.accepted else consecutive + 1
                next_observation, *_ = env.step(adapted.joint_target)
                next43 = np.r_[
                    env.get_policy_observation(next_observation),
                    np.float32(bool(next_observation["object_grasped"])),
                ].astype(np.float32)
                reward_step = protocol.step(
                    state43, next43,
                    ik_failure=consecutive >= config.max_consecutive_ik_failures,
                    time_limit=step + 1 >= config.max_steps,
                )
                milestones = reward_step.milestones
                reason = reward_step.termination_reason or "timeout"
                observation = next_observation; state43 = next43
                if reward_step.terminated or reward_step.truncated:
                    break
            outcomes.append({"seed": seed, "steps": step + 1, "reason": reason})
        finally:
            env.close()
    return combo_report(np.asarray(all_states, np.uint8)), outcomes


def timing_samples(
    manifest_path: Path, dataset_path: Path, sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    eligible = [item for item in manifest["episodes"] if item["category"] != "delayed_recovery"]
    rng = np.random.default_rng(20260814)
    sampled = [eligible[index] for index in rng.choice(len(eligible), size=sample_count, replace=False)]
    with np.load(dataset_path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    row_lookup = {
        (str(data["episode_id"][index]), int(data["step_index"][index])): index
        for index in range(len(data["reward"]))
    }
    episodes = []; exact = Counter()
    for item in sampled:
        with h5py.File(root / item["path"], "r") as episode:
            recorded = np.asarray(episode["labels/task_milestones"], np.uint8)
            current = np.vstack((np.zeros((1, 5), np.uint8), recorded[:-1]))
            rises = np.argwhere((recorded == 1) & (current == 0))
            event_windows = []
            for trigger_step, milestone_index in rises:
                rows = []
                for step in range(max(0, trigger_step - 2), min(len(recorded), trigger_step + 3)):
                    retained = row_lookup.get((item["episode_id"], step))
                    rows.append({
                        "step_index": step,
                        "expert_stage": int(episode["labels/expert_stage"][step]),
                        "next_expert_stage": int(episode["labels/next_expert_stage"][step]),
                        "task_milestones_t": current[step].astype(int).tolist(),
                        "action_t_normalized": np.asarray(episode["actions/normalized"][step]).astype(float).tolist(),
                        "next_task_milestones": recorded[step].astype(int).tolist(),
                        "object_grasped_t": bool(episode["observations/object_grasped"][step]),
                        "object_grasped_t_plus_1": bool(episode["next_observations/object_grasped"][step]),
                        "hdf5_reward": float(episode["labels/reward"][step]),
                        "awac_reward_v1": float(data["reward"][retained]) if retained is not None else None,
                        "events": int(episode["labels/events"][step]),
                        "retained_in_awac": retained is not None,
                    })
                event_windows.append({
                    "milestone": MILESTONES[int(milestone_index)],
                    "trigger_transition": int(trigger_step), "rows": rows,
                })
                exact[f"{MILESTONES[int(milestone_index)]}_edges"] += 1
                exact["trigger_obs_is_zero"] += int(current[trigger_step, milestone_index] == 0)
                exact["trigger_next_is_one"] += int(recorded[trigger_step, milestone_index] == 1)
                if trigger_step + 1 < len(recorded):
                    exact["following_obs_is_one"] += int(current[trigger_step + 1, milestone_index] == 1)
            episodes.append({
                "episode_id": item["episode_id"], "category": item["category"],
                "split": item["split"], "event_windows": event_windows,
            })
    return episodes, {"sampled_episodes": sample_count, **dict(exact)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/awac_training/awac_v3_milestone_state_20260814T170000Z"))
    args = parser.parse_args()
    run = args.run.resolve()
    dataset_dir = Path("outputs/awac_dataset/awac_v3_milestone_state").resolve()
    manifest = Path("manifests/rule_expert_v1_formal.json").resolve()
    checkpoint = run / "checkpoints/hybrid_bc_best.pt"
    old_run = Path("outputs/awac_training/awac_v2_hybrid_20260814T110000Z")
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as loaded:
        train = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(dataset_dir / "validation.npz", allow_pickle=False) as loaded:
        validation = {name: np.asarray(loaded[name]) for name in loaded.files}

    recorded_current = validation["obs"][:, 43:48].astype(np.uint8)
    recorded_next = validation["next_obs"][:, 43:48].astype(np.uint8)
    evaluator_current = evaluator_milestones_for_dataset(validation)
    mismatch = recorded_current != evaluator_current
    predictor = HybridCheckpointPredictor(checkpoint)
    recorded_metrics = action_metrics(predictor, validation["obs"], validation)
    evaluator_states = validation["obs"].copy(); evaluator_states[:, 43:48] = evaluator_current
    evaluator_metrics = action_metrics(predictor, evaluator_states, validation)
    action_shift = []
    for states in (validation["obs"], evaluator_states):
        normalized = (states.astype(np.float32) - predictor.mean) / predictor.std
        with torch.no_grad():
            action_shift.append(predictor.model.deterministic_action(torch.from_numpy(normalized))[0].numpy())

    samples, sample_check = timing_samples(manifest, dataset_dir / "validation.npz", 20)
    train_combos = combo_report(train["obs"][:, 43:48])
    validation_combos = combo_report(recorded_current)
    evaluator_validation_combos = combo_report(evaluator_current)
    rollout_combos, rollout_outcomes = closed_loop_combinations(checkpoint, list(range(300_000, 300_100)))
    train_keys = set(train_combos["counts"])
    rollout_ood = {
        key: count for key, count in rollout_combos["counts"].items() if key not in train_keys
    }
    metrics43 = json.loads((old_run / "hybrid_bc_metrics.json").read_text())
    metrics48 = json.loads((run / "hybrid_bc_metrics.json").read_text())
    report = {
        "status": "audit_complete_training_remains_stopped",
        "recording_schema": {
            "pre_action_milestone_field": None,
            "post_action_field": "labels/task_milestones",
            "explicit_next_milestone_field": None,
            "source_code_semantics": "AutoCollector updates cumulative milestones after env.step(next_obs), then records them on the transition",
        },
        "dataset_timing": {
            "obs_t": "policy_state_42[t] + object_grasped[t] + shifted previous recorded milestone (milestone_t)",
            "next_obs_t": "next_policy_state_42[t] + next_object_grasped[t] + labels/task_milestones[t] (milestone_t+1)",
            "obs_uses_next_milestone": False,
            "next_obs_uses_post_action_milestone": True,
            "all_obs_to_next_monotonic": bool(np.all(recorded_current <= recorded_next)),
        },
        "twenty_episode_timing_samples": samples,
        "timing_sample_checks": sample_check,
        "logic_comparison": {
            "recording": {
                "grasp": "post-step object_grasped; cumulative OR",
                "lift": "post-step object z - initial z >= 0.10; requires grasp milestone; cumulative OR",
                "transport": "requires lift and Rule Expert internal stage in TRANSPORT/DESCEND_TO_GOAL/OPEN_GRIPPER/RETREAT/COMPLETE",
                "release": "requires grasp, not grasped, and was_settling or Rule Expert stage RETREAT/COMPLETE",
                "retreat": "Rule Expert stage COMPLETE; during settling uses no_contact",
            },
            "evaluator": {
                "grasp": "post-step object_grasped; cumulative OR",
                "lift": "post-step object z - initial z >= 0.10; requires grasp milestone; cumulative OR",
                "transport": "requires lift+grasp and object-goal XY distance < 0.055",
                "release": "requires transport, grasped->not grasped, and object-goal 3D distance < 0.055",
                "retreat": "requires release and EE distance to goal+[0,0,0.16] <= 0.008",
            },
            "definitions_identical": False,
        },
        "milestone_state_mismatch_on_validation_physical_trajectory": {
            "transitions": len(mismatch),
            "any_dimension_mismatch_count": int(np.any(mismatch, axis=1).sum()),
            "any_dimension_mismatch_rate": float(np.any(mismatch, axis=1).mean()),
            "per_dimension_mismatch_count": {
                name: int(mismatch[:, index].sum()) for index, name in enumerate(MILESTONES)
            },
            "recorded_combinations": validation_combos,
            "evaluator_generated_combinations": evaluator_validation_combos,
        },
        "combination_distribution": {
            "train_recorded_current": train_combos,
            "validation_recorded_current": validation_combos,
            "bc_closed_loop_evaluator_current": rollout_combos,
            "closed_loop_exact_combinations_absent_from_train": rollout_ood,
            "closed_loop_illegal_order_count": rollout_combos["illegal_order_count"],
            "closed_loop_outcomes": dict(Counter(row["reason"] for row in rollout_outcomes)),
        },
        "offline_bc_metrics": {
            "old_43d": metrics43,
            "new_48d": metrics48,
        },
        "teacher_state_replay": {
            "recorded_48d_state": recorded_metrics,
            "same_physical_state_with_evaluator_generated_milestones": evaluator_metrics,
            "predicted_continuous_action_shift_mse": float(np.mean((action_shift[0] - action_shift[1]) ** 2)),
            "interpretation": "Only milestone bits were replaced; policy_state42 and object_grasped were identical.",
        },
        "root_cause": (
            "The 48-D dataset timing is correctly shifted, but recorded milestones are Rule-Expert-history labels. "
            "The closed-loop evaluator supplies geometrically reconstructed RewardV1 milestones with different "
            "transport/release/retreat definitions. Conditioning BC on these non-equivalent bits creates a state-interface "
            "distribution/semantic mismatch that was invisible to the old 43-D actor."
        ),
        "training_restarted": False,
    }
    (run / "state_timing_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    concise = {
        "dataset_timing": report["dataset_timing"],
        "definitions_identical": False,
        "validation_mismatch": report["milestone_state_mismatch_on_validation_physical_trajectory"],
        "closed_loop_ood_combinations": rollout_ood,
        "teacher_state_replay": report["teacher_state_replay"],
        "root_cause": report["root_cause"],
    }
    (run / "state_timing_audit.txt").write_text(json.dumps(concise, indent=2) + "\n")
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()

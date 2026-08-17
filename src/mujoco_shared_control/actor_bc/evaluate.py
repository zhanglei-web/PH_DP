"""Offline sanity and closed-loop MuJoCo evaluation for Actor BC v1."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mujoco_shared_control.actor_bc.model import ActorBC
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


MILESTONE_NAMES = ("grasped", "lifted", "transported", "released", "retreated")


class ActorPredictor:
    def __init__(self, checkpoint_path: str | Path, device_name: str = "auto") -> None:
        self.path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model = ActorBC()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available()
            else "cpu" if device_name == "auto" else device_name
        )
        self.model.to(self.device).eval()
        self.mean = np.asarray(checkpoint["observation_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["observation_std"], dtype=np.float32)
        if self.mean.shape != (42,) or self.std.shape != (42,):
            raise ValueError("checkpoint observation normalization must be 42-D")
        self.action_spec = ExpertActionSpec(**checkpoint["action_spec"])
        self.checkpoint = checkpoint

    def predict_unclipped(self, policy_state: np.ndarray) -> np.ndarray:
        normalized_state = (np.asarray(policy_state, np.float32) - self.mean) / self.std
        tensor = torch.from_numpy(normalized_state).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            return self.model(tensor).squeeze(0).cpu().numpy().astype(np.float64)

    def predict(self, policy_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.predict_unclipped(policy_state)
        return raw, self.action_spec.denormalize(np.clip(raw, -1.0, 1.0))


def _validation_arrays(dataset: ManifestActorDataset) -> tuple[np.ndarray, np.ndarray]:
    states, actions = [], []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            states.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
            actions.append(np.asarray(episode["actions/normalized"][:], np.float32))
    return np.concatenate(states), np.concatenate(actions)


def offline_sanity(
    predictor: ActorPredictor,
    manifest_path: str | Path,
    sample_count: int = 1_000,
    seed: int = 20260812,
) -> dict[str, Any]:
    dataset = ManifestActorDataset(manifest_path, "validation")
    states, targets = _validation_arrays(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(states), size=sample_count, replace=False)
    normalized = (states[indices] - predictor.mean) / predictor.std
    with torch.inference_mode():
        prediction = predictor.model(
            torch.from_numpy(normalized).to(predictor.device)
        ).cpu().numpy()
    clipped = np.clip(prediction, -1.0, 1.0)
    physical = np.stack([predictor.action_spec.denormalize(row) for row in clipped])
    target = targets[indices]
    rotation = np.abs(prediction[:, 3:6])
    gripper_width = physical[:, 6]
    return {
        "sample_count": sample_count,
        "sample_seed": seed,
        "nan_or_inf_elements": int(np.size(prediction) - np.isfinite(prediction).sum()),
        "predictions_with_any_preclip_exceedance": int(np.any(np.abs(prediction) > 1.0, axis=1).sum()),
        "preclip_exceedance_fraction": float(np.any(np.abs(prediction) > 1.0, axis=1).mean()),
        "preclip_exceedance_element_fraction": float((np.abs(prediction) > 1.0).mean()),
        "postclip_legal": bool(np.isfinite(physical).all() and
                               np.all(np.abs(clipped) <= 1.0) and
                               np.all((physical[:, 6] >= 0.0) & (physical[:, 6] <= 0.08))),
        "rotation_prediction_absolute_mean": float(rotation.mean()),
        "rotation_prediction_absolute_max": float(rotation.max()),
        "gripper_normalized": {
            "min": float(prediction[:, 6].min()), "max": float(prediction[:, 6].max()),
            "mean": float(prediction[:, 6].mean()), "std": float(prediction[:, 6].std()),
            "predicted_open": int((prediction[:, 6] >= 0.375).sum()),
            "predicted_closed": int((prediction[:, 6] < 0.375).sum()),
        },
        "gripper_width_m": {
            "min": float(gripper_width.min()), "max": float(gripper_width.max()),
            "mean": float(gripper_width.mean()), "std": float(gripper_width.std()),
        },
        "sample_total_mse": float(np.mean((prediction - target) ** 2)),
    }


def evaluate_episode(predictor: ActorPredictor, seed: int) -> dict[str, Any]:
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
        initial_z = float(observation["object_pose"][2, 3])
        milestones = np.zeros(5, dtype=bool)
        first_steps = np.full(5, -1, dtype=np.int64)
        stable_steps = 0
        consecutive_ik_failures = 0
        ik_fallback_count = 0
        action_clipping_count = 0
        adapter_identity_transitions = 0
        adapter_translation_projections = 0
        adapter_rotation_projections = 0
        adapter_gripper_clips = 0
        adapter_action_differences: list[float] = []
        network_clip_steps = 0
        drop_count = 0
        wrong_gripper_switches = 0
        previous_grasped = bool(observation["object_grasped"])
        previous_gripper_class: bool | None = None
        expected_switch_index = 0
        expected_switches = ((True, False), (False, True))  # open->closed, closed->open
        termination_reason = "time_limit"
        success = False
        for step in range(config.max_steps):
            prediction, command = predictor.predict(env.get_policy_observation(observation))
            if not np.isfinite(prediction).all():
                termination_reason = "non_finite_actor_output"
                break
            network_clip_steps += int(np.any(np.abs(prediction) > 1.0))
            gripper_open = bool(prediction[6] >= 0.375)
            if previous_gripper_class is not None and gripper_open != previous_gripper_class:
                switch = (previous_gripper_class, gripper_open)
                if expected_switch_index < len(expected_switches) and switch == expected_switches[expected_switch_index]:
                    expected_switch_index += 1
                else:
                    wrong_gripper_switches += 1
            previous_gripper_class = gripper_open
            adapted = adapter.adapt(command)
            ik_fallback_count += int(adapted.fallback_used)
            action_clipping_count += int(adapted.action_clipped)
            if adapted.accepted:
                adapter_identity_transitions += 1
                difference = np.asarray(adapted.normalized) - np.asarray(prediction)
                adapter_action_differences.append(float(np.linalg.norm(difference)))
                adapter_translation_projections += int(
                    not np.allclose(adapted.requested[:3], adapted.clipped[:3])
                )
                adapter_rotation_projections += int(
                    not np.allclose(adapted.requested[3:6], adapted.clipped[3:6])
                )
                adapter_gripper_clips += int(
                    not np.isclose(adapted.requested[6], adapted.clipped[6])
                )
            consecutive_ik_failures = 0 if adapted.accepted else consecutive_ik_failures + 1
            next_observation, _reward, inside_goal, _truncated, _info = env.step(adapted.joint_target)
            grasped = bool(next_observation["object_grasped"])
            object_position = next_observation["object_pose"][:3, 3]
            goal_position = next_observation["goal_pose"][:3, 3]
            ee_position = next_observation["ee_pose"][:3, 3]
            if grasped:
                milestones[0] = True
            if milestones[0] and grasped and object_position[2] - initial_z >= 0.10:
                milestones[1] = True
            # Observable equivalent of reaching the above-goal transport target.
            if milestones[1] and grasped and np.linalg.norm(object_position[:2] - goal_position[:2]) < 0.055:
                milestones[2] = True
            valid_release = bool(milestones[2] and previous_grasped and not grasped and inside_goal)
            if valid_release:
                milestones[3] = True
            if previous_grasped and not grasped and not valid_release:
                drop_count += 1
            retreat_target = goal_position + np.array([0.0, 0.0, 0.16])
            if milestones[3] and np.linalg.norm(ee_position - retreat_target) <= 0.008:
                milestones[4] = True
            for index in np.flatnonzero(milestones & (first_steps < 0)):
                first_steps[index] = step
            released_inside = bool(milestones[3] and inside_goal and not grasped)
            stable_steps = stable_steps + 1 if released_inside else 0
            if milestones.all() and stable_steps >= config.success_settle_steps:
                success = True
                termination_reason = "task_success"
            elif consecutive_ik_failures >= config.max_consecutive_ik_failures:
                termination_reason = "ik_failure_limit"
            observation = next_observation
            previous_grasped = grasped
            if success or termination_reason == "ik_failure_limit":
                break
        episode_length = step + 1
        if not success and termination_reason == "time_limit" and episode_length < config.max_steps:
            termination_reason = "non_finite_actor_output"
        last_index = np.flatnonzero(milestones)
        last_milestone = MILESTONE_NAMES[int(last_index[-1])] if len(last_index) else "none"
        result: dict[str, Any] = {
            "seed": seed, "success": success, "termination_reason": termination_reason,
            "episode_length": episode_length,
            **{name: bool(milestones[index]) for index, name in enumerate(MILESTONE_NAMES)},
            "first_milestone_step": {
                name: int(first_steps[index]) for index, name in enumerate(MILESTONE_NAMES)
            },
            "ik_fallback_count": ik_fallback_count,
            "action_clipping_count": action_clipping_count,
            "adapter_identity_transitions": adapter_identity_transitions,
            "adapter_translation_projections": adapter_translation_projections,
            "adapter_rotation_projections": adapter_rotation_projections,
            "adapter_gripper_clips": adapter_gripper_clips,
            "adapter_action_difference_sum": float(sum(adapter_action_differences)),
            "adapter_action_difference_max": float(max(adapter_action_differences, default=0.0)),
            "network_output_clip_steps": network_clip_steps,
            "last_milestone": last_milestone,
            "wrong_gripper_switch_count": wrong_gripper_switches,
            "drop_count": drop_count,
        }
        return result
    finally:
        env.close()


def closed_loop_evaluation(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    start_seed: int = 300_000,
    episodes: int = 100,
) -> dict[str, Any]:
    predictor = ActorPredictor(checkpoint_path)
    sanity = offline_sanity(predictor, manifest_path)
    rows = []
    for seed in range(start_seed, start_seed + episodes):
        row = evaluate_episode(predictor, seed)
        rows.append(row)
        print(
            f"seed={seed} success={int(row['success'])} "
            f"reason={row['termination_reason']} length={row['episode_length']} "
            f"last={row['last_milestone']}", flush=True,
        )
    lengths = np.asarray([row["episode_length"] for row in rows])
    summary = {
        "episodes": episodes,
        "seed_start": start_seed,
        "seed_end": start_seed + episodes - 1,
        "success": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "milestone_rates": {
            name: float(np.mean([row[name] for row in rows])) for name in MILESTONE_NAMES
        },
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "last_milestone_counts": dict(Counter(row["last_milestone"] for row in rows if not row["success"])),
        "ik_fallback_count": sum(row["ik_fallback_count"] for row in rows),
        "action_clipping_count": sum(row["action_clipping_count"] for row in rows),
        "network_output_clip_steps": sum(row["network_output_clip_steps"] for row in rows),
        "wrong_gripper_switch_count": sum(row["wrong_gripper_switch_count"] for row in rows),
        "drop_count": sum(row["drop_count"] for row in rows),
        "episode_length": {
            "mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())
        },
        "milestone_first_reached_counts": {
            name: sum(row["first_milestone_step"][name] >= 0 for row in rows)
            for name in MILESTONE_NAMES
        },
    }
    report = {
        "format_version": "actor_bc_v1_evaluation",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_content_sha": predictor.checkpoint["manifest_content_sha"],
        "milestone_definitions": {
            "grasped": "two-finger object_grasped heuristic became true",
            "lifted": "grasped and object z rose at least 0.10 m from reset",
            "transported": "lifted, grasped, and object-goal XY distance < 0.055 m",
            "released": "after transported, grasped true->false inside goal tolerance",
            "retreated": "after release, EE within 0.008 m of goal + [0,0,0.16] m",
            "success": "all milestones and released inside goal for 4 consecutive steps",
        },
        "gripper_classification_definition": "normalized >= 0.375 is open; midpoint of -0.25 and 1.0",
        "offline_sanity": sanity,
        "summary": summary,
        "episodes": rows,
    }
    output = Path(output_path).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/rule_expert_v1_formal.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-seed", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    output = args.output or args.checkpoint.resolve().parent / "evaluation.json"
    closed_loop_evaluation(args.checkpoint, args.manifest, output, args.start_seed, args.episodes)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect and minimally audit 30 successful deterministic Final Expert episodes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.milestones import (
    GeometricTaskPhase, MilestoneTracker, phase_from_milestones,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv


CHECKPOINT = Path(
    "outputs/awac_online/online_awac_v3_geometric_hybrid_20260814T210000Z/"
    "checkpoints/online_step_20000.pt"
)
EXPECTED_SHA256 = "2aceee20f15c21ba3a4e544f12e4ce8558bc89667fa35d9707b73fbb09abf56b"
SEED_START = 1_100_000
TARGET_SUCCESSES = 30
MAX_ATTEMPTS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def actual_action(adapted, predictor: HybridCheckpointPredictor) -> np.ndarray:
    if adapted.accepted:
        return np.asarray(adapted.normalized, np.float32)
    gripper = float(predictor.action_spec.normalize(
        np.r_[np.zeros(6), adapted.joint_target[7]]
    )[6])
    return np.r_[np.zeros(6, np.float32), np.float32(gripper)].astype(np.float32)


def collect_episode(
    predictor: HybridCheckpointPredictor, seed: int, episode_id: str,
    reward_config: AWACRewardV1Config,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = CollectionConfig()
    env = PickPlaceEnv(
        render_mode=None, control_timestep=config.control_timestep_s,
        max_episode_steps=config.max_steps, enable_camera=False,
    )
    adapter = ExpertCommandAdapter(env.ik_controller, predictor.action_spec)
    rows: list[dict[str, Any]] = []
    try:
        observation, reset_info = env.reset(seed=seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        initial_object_pose = np.asarray(observation["object_pose"], np.float32).copy()
        goal_pose = np.asarray(observation["goal_pose"], np.float32).copy()
        initial_43 = np.r_[
            env.get_policy_observation(observation),
            np.float32(bool(observation["object_grasped"])),
        ].astype(np.float32)
        reward_protocol = AWACRewardV1Online(initial_43, reward_config)
        consecutive_ik = 0
        terminal_reason = "timeout"
        task_success = False
        for step in range(config.max_steps):
            diffusion_43 = np.r_[
                env.get_policy_observation(observation),
                np.float32(bool(observation["object_grasped"])),
            ].astype(np.float32)
            milestone_t = reward_protocol.tracker.current.astype(np.uint8)
            state_48 = np.r_[diffusion_43, milestone_t.astype(np.float32)].astype(np.float32)
            phase = phase_from_milestones(milestone_t)
            requested = np.asarray(predictor.normalized_action(
                diffusion_43[:42], bool(diffusion_43[42]), milestone_t,
            ), np.float32)
            adapted = adapter.adapt(predictor.action_spec.denormalize(requested))
            executed = actual_action(adapted, predictor)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, _env_reward, _env_terminated, _env_truncated, _env_info = env.step(
                adapted.joint_target
            )
            next_diffusion_43 = np.r_[
                env.get_policy_observation(next_observation),
                np.float32(bool(next_observation["object_grasped"])),
            ].astype(np.float32)
            reward_step = reward_protocol.step(
                diffusion_43, next_diffusion_43,
                ik_failure=consecutive_ik >= config.max_consecutive_ik_failures,
                time_limit=step + 1 >= config.max_steps,
            )
            next_milestone = reward_protocol.tracker.current.astype(np.uint8)
            next_phase = phase_from_milestones(next_milestone)
            next_state_48 = np.r_[
                next_diffusion_43, next_milestone.astype(np.float32),
            ].astype(np.float32)
            execution_verified = bool(
                executed.shape == (7,) and np.isfinite(executed).all()
                and (
                    np.allclose(executed, adapted.normalized, atol=1e-7)
                    if adapted.accepted else np.allclose(executed[:6], 0.0)
                )
            )
            rows.append({
                "diffusion_observation_43": diffusion_43,
                "state_48": state_48,
                "executed_action_7": executed,
                "next_diffusion_observation_43": next_diffusion_43,
                "next_state_48": next_state_48,
                "milestone_t": milestone_t,
                "next_milestone_t1": next_milestone,
                "phase_t": np.int8(int(phase)),
                "phase_name": phase.name,
                "next_phase_t1": np.int8(int(next_phase)),
                "next_phase_name": next_phase.name,
                "step_index": np.int32(step),
                "requested_action_7": requested,
                "adapter_accepted": bool(adapted.accepted),
                "action_clipped": bool(adapted.action_clipped),
                "fallback_used": bool(adapted.fallback_used),
                "execution_verified": execution_verified,
            })
            observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                terminal_reason = reward_step.termination_reason
                task_success = bool(reward_step.task_success)
                break
        length = len(rows)
        arrays = {
            name: np.asarray([row[name] for row in rows])
            for name in rows[0]
        }
        arrays.update({
            "episode_id": np.full(length, episode_id, dtype="U64"),
            "seed": np.full(length, seed, dtype=np.int64),
            "success": np.full(length, task_success, dtype=bool),
            "termination_reason": np.full(length, terminal_reason, dtype="U32"),
            "initial_object_pose": initial_object_pose,
            "goal_pose": goal_pose,
        })
        metadata = {
            "episode_id": episode_id, "seed": seed,
            "success": task_success, "termination_reason": terminal_reason,
            "transitions": length,
            "initial_object_pose": initial_object_pose.astype(float).tolist(),
            "goal_pose": goal_pose.astype(float).tolist(),
            "reset_object_xy": np.asarray(reset_info["object_xy"]).astype(float).tolist(),
            "reset_goal_xy": np.asarray(reset_info["goal_xy"]).astype(float).tolist(),
            "action_clipping_count": int(arrays["action_clipped"].sum()),
            "fallback_count": int(arrays["fallback_used"].sum()),
        }
        return arrays, metadata
    finally:
        env.close()


def numeric_nonfinite(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    nan = inf = 0
    for value in arrays.values():
        if np.issubdtype(value.dtype, np.number):
            nan += int(np.isnan(value).sum())
            inf += int(np.isinf(value).sum())
    return nan, inf


def compressed_phases(names: np.ndarray) -> list[str]:
    result: list[str] = []
    for name in names.astype(str):
        if not result or result[-1] != name:
            result.append(name)
    return result


def audit(success_paths: list[Path], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    loaded: list[dict[str, np.ndarray]] = []
    for path in success_paths:
        with np.load(path, allow_pickle=False) as source:
            loaded.append({name: source[name].copy() for name in source.files})
    shape_failures = []
    nan_count = inf_count = 0
    regression_count = illegal_combination_count = 0
    phase_ok = True
    for path, arrays in zip(success_paths, loaded):
        length = len(arrays["step_index"])
        expected = {
            "diffusion_observation_43": (length, 43),
            "state_48": (length, 48),
            "executed_action_7": (length, 7),
            "next_diffusion_observation_43": (length, 43),
            "milestone_t": (length, 5),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                shape_failures.append({"episode": path.stem, "field": name,
                                       "actual": list(arrays[name].shape), "expected": list(shape)})
        nan, inf = numeric_nonfinite(arrays); nan_count += nan; inf_count += inf
        sequence = np.vstack([arrays["milestone_t"], arrays["next_milestone_t1"][-1:]]).astype(int)
        regression_count += int((np.diff(sequence, axis=0) < 0).sum())
        illegal_combination_count += int(np.any(sequence[:, 1:] > sequence[:, :-1], axis=1).sum())
        phases = arrays["phase_t"].astype(int)
        expected_phases = np.asarray([
            int(phase_from_milestones(value)) for value in arrays["milestone_t"]
        ])
        expanded_phases = np.r_[phases, arrays["next_phase_t1"][-1].astype(int)]
        phase_ok &= bool(
            np.array_equal(phases, expected_phases)
            and np.all(np.diff(expanded_phases) >= 0)
            and expanded_phases[0] == int(GeometricTaskPhase.APPROACH)
            and expanded_phases[-1] == int(GeometricTaskPhase.COMPLETE)
        )

    rng = np.random.default_rng(2_026_081_600)
    sample_indices = sorted(rng.choice(len(loaded), size=5, replace=False).tolist())
    alignment_details = []
    alignment_ok = True
    phase_samples = []
    for index in sample_indices:
        arrays = loaded[index]
        chain_43 = bool(np.allclose(
            arrays["next_diffusion_observation_43"][:-1],
            arrays["diffusion_observation_43"][1:], atol=1e-7,
        ))
        chain_48 = bool(np.allclose(
            arrays["next_state_48"][:-1], arrays["state_48"][1:], atol=1e-7,
        ))
        execution = bool(arrays["execution_verified"].all())
        steps = bool(np.array_equal(arrays["step_index"], np.arange(len(arrays["step_index"]))))
        passed = chain_43 and chain_48 and execution and steps
        alignment_ok &= passed
        alignment_details.append({
            "episode_id": str(arrays["episode_id"][0]), "pass": passed,
            "obs_chain_43": chain_43, "state_chain_48": chain_48,
            "executed_action_verified": execution, "step_sequence": steps,
        })
        phase_samples.append({
            "episode_id": str(arrays["episode_id"][0]),
            "sequence": compressed_phases(np.concatenate((
                arrays["phase_name"], arrays["next_phase_name"][-1:],
            ))),
        })

    success_meta = [row for row in episodes if row["success"]]
    object_positions = np.asarray([row["initial_object_pose"] for row in success_meta], np.float64)[:, :3, 3]
    goal_positions = np.asarray([row["goal_pose"] for row in success_meta], np.float64)[:, :3, 3]
    def diversity(value: np.ndarray) -> dict[str, list[float]]:
        return {
            "min": value.min(axis=0).astype(float).tolist(),
            "max": value.max(axis=0).astype(float).tolist(),
            "std": value.std(axis=0).astype(float).tolist(),
        }
    reset_ok = bool(
        np.all(object_positions[:, :2].std(axis=0) > 1e-6)
        and np.all(goal_positions[:, :2].std(axis=0) > 1e-6)
    )
    return {
        "success_count": {"pass": len(success_paths) == TARGET_SUCCESSES,
                          "successful_episodes": len(success_paths)},
        "shapes": {"pass": not shape_failures, "failures": shape_failures,
                   "diffusion_observation": [43], "state": [48], "action": [7]},
        "nonfinite": {"pass": nan_count == 0 and inf_count == 0,
                      "nan_count": nan_count, "inf_count": inf_count},
        "alignment": {"pass": alignment_ok, "sampled_episodes": alignment_details},
        "milestone_integrity": {
            "pass": regression_count == 0 and illegal_combination_count == 0,
            "one_to_zero_transitions": regression_count,
            "illegal_combinations": illegal_combination_count,
        },
        "phase_sequence": {"pass": phase_ok, "sampled_sequences": phase_samples},
        "reset_diversity": {"pass": reset_ok,
                            "object_initial_xyz": diversity(object_positions),
                            "goal_xyz": diversity(goal_positions)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    checkpoint = CHECKPOINT.resolve()
    before_hash = sha256(checkpoint)
    if before_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Final Expert checkpoint SHA-256 mismatch: {before_hash}")
    run = Path(
        "outputs/learned_expert_collection/"
        "final_online_awac20k_sanity30_20260816T120000Z"
    ).resolve()
    success_dir = run / "success"; failure_dir = run / "failure_diagnostics"
    reward_config = AWACRewardV1Config()
    episodes: list[dict[str, Any]] = []
    success_paths: list[Path] = []
    if args.audit_existing:
        paths = sorted(success_dir.glob("*.npz")) + sorted(failure_dir.glob("*.npz"))
        if not paths:
            raise RuntimeError("no existing sanity episodes to audit")
        for path in paths:
            with np.load(path, allow_pickle=False) as source:
                metadata = {
                    "episode_id": str(source["episode_id"][0]),
                    "seed": int(source["seed"][0]),
                    "success": bool(source["success"][0]),
                    "termination_reason": str(source["termination_reason"][0]),
                    "transitions": int(len(source["step_index"])),
                    "initial_object_pose": source["initial_object_pose"].astype(float).tolist(),
                    "goal_pose": source["goal_pose"].astype(float).tolist(),
                    "action_clipping_count": int(source["action_clipped"].sum()),
                    "fallback_count": int(source["fallback_used"].sum()),
                    "path": str(path),
                }
            episodes.append(metadata)
            if metadata["success"]:
                success_paths.append(path)
    else:
        success_dir.mkdir(parents=True, exist_ok=False); failure_dir.mkdir()
        predictor = HybridCheckpointPredictor(checkpoint)
        for attempt in range(MAX_ATTEMPTS):
            if len(success_paths) == TARGET_SUCCESSES:
                break
            seed = SEED_START + attempt
            episode_id = f"final_expert_sanity_{seed}"
            arrays, metadata = collect_episode(predictor, seed, episode_id, reward_config)
            directory = success_dir if metadata["success"] else failure_dir
            path = directory / f"{episode_id}.npz"
            atomic_npz(path, arrays)
            metadata["path"] = str(path)
            episodes.append(metadata)
            if metadata["success"]:
                success_paths.append(path)
            print(json.dumps({
                "attempted": len(episodes), "successful": len(success_paths),
                "seed": seed, "outcome": metadata["termination_reason"],
            }), flush=True)
    audit_result = audit(success_paths, episodes)
    failure_rows = [row for row in episodes if not row["success"]]
    breakdown = Counter(row["termination_reason"] for row in failure_rows)
    failure_breakdown = {
        "illegal_drop": int(breakdown.get("illegal_drop", 0)),
        "ik_failure": int(breakdown.get("ik_failure_limit", 0)),
        "timeout": int(breakdown.get("timeout", 0)),
        "other": int(len(failure_rows) - sum(
            breakdown.get(name, 0) for name in ("illegal_drop", "ik_failure_limit", "timeout")
        )),
    }
    checks = [
        audit_result["success_count"]["pass"], audit_result["shapes"]["pass"],
        audit_result["nonfinite"]["pass"], audit_result["alignment"]["pass"],
        audit_result["milestone_integrity"]["pass"], audit_result["phase_sequence"]["pass"],
        audit_result["reset_diversity"]["pass"],
    ]
    after_hash = sha256(checkpoint)
    report = {
        "status": "PASS" if all(checks) and after_hash == before_hash else "FAIL",
        "checkpoint": str(checkpoint), "checkpoint_sha256_before": before_hash,
        "checkpoint_sha256_after": after_hash, "checkpoint_unchanged": after_hash == before_hash,
        "policy": "deterministic Actor mean + deterministic gripper; exploration=0",
        "gradient_updates": 0, "optimizer_updates": 0, "replay_appends": 0,
        "seed_start": SEED_START,
        "attempted_episodes": len(episodes), "successful_episodes": len(success_paths),
        "failed_episodes": len(failure_rows),
        "successful_transitions": int(sum(row["transitions"] for row in episodes if row["success"])),
        "failure_breakdown": failure_breakdown,
        "audit": audit_result,
        "episodes": episodes,
        "reward_config": asdict(reward_config),
        "schema": {
            "diffusion_observation_43": "policy_state_42 + object_grasped",
            "state_48": "diffusion_observation_43 + geometric milestones[5]",
            "executed_action_7": "adapter-executed normalized [dx,dy,dz,drx,dry,drz,gripper]",
            "phase_t": [phase.name for phase in GeometricTaskPhase],
        },
    }
    atomic_json(run / "sanity_report.json", report)
    atomic_json(run / "episode_manifest.json", episodes)
    print(json.dumps({
        "run": str(run), "status": report["status"],
        "attempted": len(episodes), "successful": len(success_paths),
        "failed": len(failure_rows), "successful_transitions": report["successful_transitions"],
        "failure_breakdown": failure_breakdown, "audit": audit_result,
    }, indent=2))


if __name__ == "__main__":
    main()

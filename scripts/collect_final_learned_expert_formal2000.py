#!/usr/bin/env python3
"""Collect, split, and minimally audit 2000 frozen Final Expert successes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from collect_final_learned_expert_sanity import (
    CHECKPOINT, EXPECTED_SHA256, atomic_json, atomic_npz, collect_episode,
    compressed_phases, numeric_nonfinite, sha256,
)
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.milestones import GeometricTaskPhase, phase_from_milestones
from mujoco_shared_control.awac.reward import AWACRewardV1Config


TARGET_SUCCESSES = 2_000
SEED_START = 1_200_000
MAX_ATTEMPTS = 3_000
SPLIT_SEED = 2_026_081_601
ALIGNMENT_SEED = 2_026_081_602
RUN = Path(
    "outputs/learned_expert_collection/"
    "final_online_awac20k_formal2000_20260816T130000Z"
)


def metadata_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        return {
            "episode_id": str(source["episode_id"][0]),
            "seed": int(source["seed"][0]),
            "success": bool(source["success"][0]),
            "termination_reason": str(source["termination_reason"][0]),
            "transitions": int(len(source["step_index"])),
            "initial_object_pose": source["initial_object_pose"].astype(float).tolist(),
            "goal_pose": source["goal_pose"].astype(float).tolist(),
            "action_clipping_count": int(source["action_clipped"].sum()),
            "fallback_count": int(source["fallback_used"].sum()),
            "path": str(path.resolve()),
        }


def existing_episodes(success_dir: Path, failure_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(success_dir.glob("*.npz")) + sorted(failure_dir.glob("*.npz"))
    return sorted((metadata_from_npz(path) for path in paths), key=lambda row: row["seed"])


def write_collection_state(run: Path, episodes: list[dict[str, Any]], complete: bool) -> None:
    atomic_json(run / "episode_manifest.json", episodes)
    atomic_json(run / "collection_progress.json", {
        "complete": complete,
        "attempted_episodes": len(episodes),
        "successful_episodes": int(sum(row["success"] for row in episodes)),
        "failed_episodes": int(sum(not row["success"] for row in episodes)),
        "last_seed": max((row["seed"] for row in episodes), default=None),
    })


def alignment_check(paths: list[Path]) -> dict[str, Any]:
    generator = np.random.default_rng(ALIGNMENT_SEED)
    chosen = sorted(generator.choice(len(paths), size=20, replace=False).tolist())
    details = []
    for index in chosen:
        with np.load(paths[index], allow_pickle=False) as source:
            chain_43 = bool(np.allclose(
                source["next_diffusion_observation_43"][:-1],
                source["diffusion_observation_43"][1:], atol=1e-7,
            ))
            chain_48 = bool(np.allclose(
                source["next_state_48"][:-1], source["state_48"][1:], atol=1e-7,
            ))
            execution = bool(source["execution_verified"].all())
            steps = bool(np.array_equal(
                source["step_index"], np.arange(len(source["step_index"])),
            ))
            details.append({
                "episode_id": str(source["episode_id"][0]),
                "pass": chain_43 and chain_48 and execution and steps,
                "obs_chain_43": chain_43, "state_chain_48": chain_48,
                "executed_action_verified": execution, "step_sequence": steps,
            })
    return {"pass": all(row["pass"] for row in details), "sampled_episodes": details}


def statistics_xyz(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
    }


def integrity_audit(success_paths: list[Path]) -> tuple[dict[str, Any], dict[str, int], list[int]]:
    shape_failures: list[dict[str, Any]] = []
    nan_count = inf_count = milestone_regressions = illegal_milestones = 0
    phase_regressions = phase_mapping_errors = 0
    phase_counts: Counter[str] = Counter()
    lengths: list[int] = []
    object_positions: list[np.ndarray] = []
    goal_positions: list[np.ndarray] = []
    required = {
        "diffusion_observation_43": 43,
        "state_48": 48,
        "executed_action_7": 7,
        "next_diffusion_observation_43": 43,
        "milestone_t": 5,
        "phase_t": None,
    }
    for path in success_paths:
        with np.load(path, allow_pickle=False) as source:
            length = int(len(source["step_index"])); lengths.append(length)
            for name, width in required.items():
                if name not in source.files:
                    shape_failures.append({"episode": path.stem, "field": name, "error": "missing"})
                elif width is not None and source[name].shape != (length, width):
                    shape_failures.append({
                        "episode": path.stem, "field": name,
                        "actual": list(source[name].shape), "expected": [length, width],
                    })
            arrays = {name: source[name] for name in source.files}
            nan, inf = numeric_nonfinite(arrays); nan_count += nan; inf_count += inf
            milestones = np.vstack((source["milestone_t"], source["next_milestone_t1"][-1:])).astype(int)
            milestone_regressions += int((np.diff(milestones, axis=0) < 0).sum())
            illegal_milestones += int(np.any(milestones[:, 1:] > milestones[:, :-1], axis=1).sum())
            phases = source["phase_t"].astype(int)
            expected = np.asarray([int(phase_from_milestones(value)) for value in source["milestone_t"]])
            phase_mapping_errors += int(np.sum(phases != expected))
            expanded = np.r_[phases, int(source["next_phase_t1"][-1])]
            phase_regressions += int((np.diff(expanded) < 0).sum())
            phase_counts.update(source["phase_name"].astype(str).tolist())
            object_positions.append(source["initial_object_pose"][:3, 3].astype(np.float64))
            goal_positions.append(source["goal_pose"][:3, 3].astype(np.float64))
    objects = np.asarray(object_positions); goals = np.asarray(goal_positions)
    diversity_pass = bool(
        np.all(objects[:, :2].std(axis=0) > 1e-6)
        and np.all(goals[:, :2].std(axis=0) > 1e-6)
    )
    total_frames = int(sum(phase_counts.values()))
    report = {
        "successful_episode_count": {"pass": len(success_paths) == TARGET_SUCCESSES,
                                     "count": len(success_paths)},
        "shape": {"pass": not shape_failures, "failures": shape_failures,
                  "diffusion_observation": [43], "state_48": [48], "executed_action": [7]},
        "nonfinite": {"pass": nan_count == 0 and inf_count == 0,
                      "nan_count": nan_count, "inf_count": inf_count},
        "milestone": {"pass": milestone_regressions == 0 and illegal_milestones == 0,
                      "one_to_zero_transitions": milestone_regressions,
                      "illegal_combinations": illegal_milestones},
        "phase": {"pass": phase_regressions == 0 and phase_mapping_errors == 0,
                  "illegal_regressions": phase_regressions,
                  "mapping_errors": phase_mapping_errors},
        "reset_diversity": {"pass": diversity_pass,
                            "object_initial_xyz": statistics_xyz(objects),
                            "goal_xyz": statistics_xyz(goals)},
        "alignment": alignment_check(success_paths),
        "phase_distribution": {
            name: {"frame_count": int(phase_counts.get(name, 0)),
                   "percentage": 100.0 * phase_counts.get(name, 0) / max(total_frames, 1)}
            for name in (phase.name for phase in GeometricTaskPhase)
        },
    }
    return report, dict(phase_counts), lengths


def make_split(success_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = np.asarray([row["episode_id"] for row in success_rows])
    generator = np.random.default_rng(SPLIT_SEED)
    order = generator.permutation(len(ids))
    assignments = {
        "train": order[:1600], "validation": order[1600:1800], "test": order[1800:2000],
    }
    by_id = {row["episode_id"]: row for row in success_rows}
    manifest = {
        "split_unit": "episode", "shuffle_seed": SPLIT_SEED,
        "shared_for": ["global_diffusion", "phase_classifier", "phase_conditioned_diffusion"],
        "splits": {},
    }
    stats = {}
    for split, indices in assignments.items():
        episode_ids = ids[indices].tolist()
        manifest["splits"][split] = episode_ids
        stats[split] = {
            "episodes": len(episode_ids),
            "transitions": int(sum(by_id[episode_id]["transitions"] for episode_id in episode_ids)),
        }
    return manifest, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checkpoint = CHECKPOINT.resolve(); checkpoint_hash_before = sha256(checkpoint)
    if checkpoint_hash_before != EXPECTED_SHA256:
        raise RuntimeError(f"Final Expert checkpoint hash mismatch: {checkpoint_hash_before}")
    run = RUN.resolve(); success_dir = run / "success"; failure_dir = run / "failure"
    if args.resume:
        if not success_dir.is_dir() or not failure_dir.is_dir():
            raise RuntimeError("formal collection output does not exist for resume")
    else:
        success_dir.mkdir(parents=True, exist_ok=False); failure_dir.mkdir()
        atomic_json(run / "collection_config.json", {
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash_before,
            "target_successful_episodes": TARGET_SUCCESSES,
            "seed_start": SEED_START, "max_attempts": MAX_ATTEMPTS,
            "policy": "deterministic actor mean + deterministic OPEN/CLOSE",
            "gradient_updates": 0, "optimizer_updates": 0,
            "replay_appends": 0, "exploration": 0,
            "schema_source": "Sanity30 PASS schema",
        })
    episodes = existing_episodes(success_dir, failure_dir)
    success_count = int(sum(row["success"] for row in episodes))
    next_seed = max((row["seed"] for row in episodes), default=SEED_START - 1) + 1
    predictor = HybridCheckpointPredictor(checkpoint)
    reward_config = AWACRewardV1Config()
    while success_count < TARGET_SUCCESSES and len(episodes) < MAX_ATTEMPTS:
        seed = next_seed; next_seed += 1
        episode_id = f"final_expert_formal_{seed}"
        arrays, metadata = collect_episode(predictor, seed, episode_id, reward_config)
        directory = success_dir if metadata["success"] else failure_dir
        path = directory / f"{episode_id}.npz"
        atomic_npz(path, arrays); metadata["path"] = str(path.resolve())
        episodes.append(metadata)
        success_count += int(metadata["success"])
        if len(episodes) % 25 == 0 or not metadata["success"] or success_count == TARGET_SUCCESSES:
            write_collection_state(run, episodes, success_count == TARGET_SUCCESSES)
            print(json.dumps({
                "attempted": len(episodes), "successful": success_count,
                "failed": len(episodes) - success_count, "last_seed": seed,
                "last_outcome": metadata["termination_reason"],
            }), flush=True)
    write_collection_state(run, episodes, success_count == TARGET_SUCCESSES)
    success_rows = [row for row in episodes if row["success"]]
    failure_rows = [row for row in episodes if not row["success"]]
    success_paths = [Path(row["path"]) for row in success_rows]
    integrity, phase_counts, lengths = integrity_audit(success_paths)
    split_manifest, split_stats = make_split(success_rows)
    atomic_json(run / "split_manifest.json", split_manifest)
    failure_counter = Counter(row["termination_reason"] for row in failure_rows)
    failure_breakdown = {
        "illegal_drop": int(failure_counter.get("illegal_drop", 0)),
        "ik_failure": int(failure_counter.get("ik_failure_limit", 0)),
        "timeout": int(failure_counter.get("timeout", 0)),
        "other": int(len(failure_rows) - sum(
            failure_counter.get(name, 0)
            for name in ("illegal_drop", "ik_failure_limit", "timeout")
        )),
    }
    checks = (
        integrity["successful_episode_count"]["pass"], integrity["shape"]["pass"],
        integrity["nonfinite"]["pass"], integrity["milestone"]["pass"],
        integrity["phase"]["pass"], integrity["reset_diversity"]["pass"],
        integrity["alignment"]["pass"],
    )
    checkpoint_hash_after = sha256(checkpoint)
    integrity.update({
        "status": "PASS" if all(checks) else "FAIL",
        "checkpoint_unchanged": checkpoint_hash_after == checkpoint_hash_before,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
    })
    atomic_json(run / "integrity_report.json", integrity)
    length_array = np.asarray(lengths, np.float64)
    report = {
        "status": "PASS" if all(checks) and checkpoint_hash_after == checkpoint_hash_before else "FAIL",
        "attempted_episodes": len(episodes), "successful_episodes": len(success_rows),
        "failed_episodes": len(failure_rows),
        "total_successful_transitions": int(length_array.sum()),
        "episode_length": {"mean": float(length_array.mean()), "std": float(length_array.std()),
                           "min": int(length_array.min()), "max": int(length_array.max())},
        "split_statistics": split_stats,
        "phase_distribution": integrity["phase_distribution"],
        "failure_breakdown": failure_breakdown,
        "nan_count": integrity["nonfinite"]["nan_count"],
        "inf_count": integrity["nonfinite"]["inf_count"],
        "alignment": "PASS" if integrity["alignment"]["pass"] else "FAIL",
        "milestone": "PASS" if integrity["milestone"]["pass"] else "FAIL",
        "phase": "PASS" if integrity["phase"]["pass"] else "FAIL",
        "reset_diversity": "PASS" if integrity["reset_diversity"]["pass"] else "FAIL",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash_after,
        "gradient_updates": 0, "optimizer_updates": 0,
        "replay_appends": 0, "exploration": 0,
        "reward_config": asdict(reward_config),
    }
    atomic_json(run / "collection_report.json", report)
    print(json.dumps({"run": str(run), **report}, indent=2))


if __name__ == "__main__":
    main()

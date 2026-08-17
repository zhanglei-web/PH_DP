#!/usr/bin/env python3
"""Build AWAC-v3 from formal HDF5 with the shared geometric tracker."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np

from mujoco_shared_control.awac.milestones import (
    MILESTONE_NAMES, MilestoneConfig, MilestoneTracker,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Config, _distance_for_stage
from mujoco_shared_control.collection.manifest import sha256_file


GRIPPER_THRESHOLD = 0.375
VALID_COMBINATIONS = {"00000", "10000", "11000", "11100", "11110", "11111"}


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def state43(episode: h5py.File, prefix: str) -> np.ndarray:
    return np.concatenate((
        np.asarray(episode[f"{prefix}/policy_state_42"], np.float32),
        np.asarray(episode[f"{prefix}/object_grasped"], np.float32)[:, None],
    ), axis=1)


def reconstruct(obs43: np.ndarray, next43: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tracker = MilestoneTracker()
    tracker.reset(obs43[0])
    current = np.empty((len(obs43), 5), np.uint8)
    following = np.empty((len(obs43), 5), np.uint8)
    for index, next_state in enumerate(next43):
        current[index] = tracker.current
        following[index] = tracker.update(next_state).current
    return current, following


def reconstruct_reward(
    obs43: np.ndarray, next43: np.ndarray, stages: np.ndarray,
    current: np.ndarray, following: np.ndarray,
    terminated: np.ndarray, truncated: np.ndarray, success: np.ndarray,
    config: AWACRewardV1Config,
) -> np.ndarray:
    reward = np.full(len(obs43), config.step_penalty, np.float64)
    for index, stage in enumerate(stages.astype(int)):
        before = _distance_for_stage(obs43[index], stage, config)
        after = _distance_for_stage(next43[index], stage, config)
        if before is not None and after is not None:
            reward[index] += float(np.clip(
                config.progress_scale * (before - after),
                -config.progress_clip, config.progress_clip,
            ))
    bonuses = np.asarray([
        config.grasp_bonus, config.lift_bonus, config.transport_bonus,
        config.release_bonus, config.retreat_bonus,
    ])
    reward += ((current == 0) & (following == 1)) @ bonuses
    boundary = np.asarray(terminated, bool) | np.asarray(truncated, bool)
    reward[boundary & np.asarray(success, bool)] += config.success_bonus
    reward[boundary & ~np.asarray(success, bool)] += config.failure_penalty
    return reward.astype(np.float32)


def combo_rows(values: np.ndarray) -> list[str]:
    return ["".join(str(int(bit)) for bit in row) for row in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(
        "outputs/awac_dataset/awac_v2_hybrid_formal_rule"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "manifests/rule_expert_v1_formal.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/awac_dataset/awac_v3_geometric_milestone_state"))
    args = parser.parse_args()
    source, manifest_path, output = (
        args.source.resolve(), args.manifest.resolve(), args.output.resolve())
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    by_id = {item["episode_id"]: item for item in manifest["episodes"]}
    source_report = json.loads((source / "report.json").read_text())
    if (source_report["episode_count"], source_report["transition_count"]) != (1234, 150406):
        raise RuntimeError("AWAC-v3 requires the frozen 1234-episode corpus")
    if source_report["episodes_by_category"]["delayed_recovery"] != 0:
        raise RuntimeError("delayed recovery exists in source dataset")

    reward_config = AWACRewardV1Config()
    milestone_config = MilestoneConfig()
    split_reports: dict[str, dict] = {}
    category_ids: dict[str, set[str]] = {}
    total = 0
    for split in ("train", "validation"):
        with np.load(source / f"{split}.npz", allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]).copy() for name in data.files}
            ids = np.asarray(data["episode_id"])
            steps = np.asarray(data["step_index"], np.int64)
            current = np.empty((len(steps), 5), np.uint8)
            following = np.empty((len(steps), 5), np.uint8)
            independent_mismatch = np.zeros(5, np.int64)
            legacy_mismatch = np.zeros(5, np.int64)
            raw_latch_violations = 0
            combinations: Counter[str] = Counter()

            for raw_id in np.unique(ids):
                episode_id = str(raw_id)
                item = by_id[episode_id]
                if item["category"] == "delayed_recovery":
                    raise RuntimeError("delayed recovery leaked into AWAC-v3")
                path = (root / item["path"]).resolve()
                if not path.is_relative_to(root) or sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"formal HDF5 integrity failure: {episode_id}")
                indices = np.flatnonzero(ids == raw_id)
                source_steps = steps[indices]
                with h5py.File(path, "r") as episode:
                    obs_full = state43(episode, "observations")
                    next_full = state43(episode, "next_observations")
                    legacy = np.asarray(episode["labels/task_milestones"], np.uint8)
                if len(obs_full) != item["transitions"]:
                    raise RuntimeError(f"transition count mismatch: {episode_id}")
                raw_current, raw_following = reconstruct(obs_full, next_full)
                # A second independent pass is the evaluator-equivalent state replay test.
                check_current, check_following = reconstruct(obs_full, next_full)
                independent_mismatch += np.count_nonzero(
                    raw_current != check_current, axis=0)
                independent_mismatch += np.count_nonzero(
                    raw_following != check_following, axis=0)
                raw_latch_violations += int(np.count_nonzero(
                    np.diff(raw_following.astype(np.int8), axis=0) < 0))
                current[indices] = raw_current[source_steps]
                following[indices] = raw_following[source_steps]
                # Preserve and explicitly quantify the legacy semantic difference.
                legacy_mismatch += np.count_nonzero(
                    legacy[source_steps] != raw_following[source_steps], axis=0)
                combinations.update(combo_rows(raw_current))
                combinations.update(combo_rows(raw_following))
                category_ids.setdefault(item["category"], set()).add(episode_id)

            illegal = {key: value for key, value in combinations.items()
                       if key not in VALID_COMBINATIONS}
            if independent_mismatch.sum() or raw_latch_violations or illegal:
                raise RuntimeError(
                    f"milestone consistency failure in {split}: mismatch="
                    f"{independent_mismatch.tolist()} latch={raw_latch_violations} illegal={illegal}")
            obs43 = np.asarray(data["obs"], np.float32)
            next43 = np.asarray(data["next_obs"], np.float32)
            arrays["legacy_rule_milestones"] = np.asarray(data["task_milestones"], np.uint8)
            arrays["awac_milestones_t"] = current
            arrays["awac_milestones_t1"] = following
            arrays["obs"] = np.concatenate((obs43, current.astype(np.float32)), axis=1)
            arrays["next_obs"] = np.concatenate((next43, following.astype(np.float32)), axis=1)
            arrays["reward"] = reconstruct_reward(
                obs43, next43, np.asarray(data["expert_stage"]), current, following,
                np.asarray(data["terminated"]), np.asarray(data["truncated"]),
                np.asarray(data["task_success"]), reward_config,
            )
            # Keep the historical field explicitly legacy; policy state uses awac_* only.
            if arrays["obs"].shape != (len(steps), 48) or not np.isfinite(arrays["obs"]).all():
                raise RuntimeError("invalid 48-D observation")
            if arrays["next_obs"].shape != (len(steps), 48) or not np.isfinite(arrays["next_obs"]).all():
                raise RuntimeError("invalid next 48-D observation")
            gripper = np.asarray(data["gripper_action"], np.float32)
            normalized = np.asarray(data["normalized_gripper_metadata"], np.float32)
            if not np.array_equal(gripper, (normalized < GRIPPER_THRESHOLD).astype(np.float32)):
                raise RuntimeError("hybrid gripper mapping changed")
            atomic_npz(output / f"{split}.npz", arrays)
            split_reports[split] = {
                "episodes": int(len(np.unique(ids))), "transitions": int(len(steps)),
                "npz_sha256": sha256_file(output / f"{split}.npz"),
                "offline_reconstruction_vs_tracker_mismatch": {
                    name: int(independent_mismatch[index])
                    for index, name in enumerate(MILESTONE_NAMES)},
                "legacy_rule_vs_geometric_next_mismatch": {
                    name: int(legacy_mismatch[index])
                    for index, name in enumerate(MILESTONE_NAMES)},
                "latch_violations": raw_latch_violations,
                "illegal_combinations": illegal,
                "combination_frequency_raw_current_and_next": dict(sorted(combinations.items())),
                "milestones": {
                    name: {
                        "obs_true": int(current[:, index].sum()),
                        "next_obs_true": int(following[:, index].sum()),
                        "rising_edges": int(np.sum((current[:, index] == 0) & (following[:, index] == 1))),
                    } for index, name in enumerate(MILESTONE_NAMES)},
                "reward": {
                    "mean": float(arrays["reward"].mean()),
                    "std": float(arrays["reward"].std()),
                    "min": float(arrays["reward"].min()),
                    "max": float(arrays["reward"].max()),
                    "nonfinite": int((~np.isfinite(arrays["reward"])).sum()),
                },
            }
            total += len(steps)

    report = {
        "dataset_version": "awac_v3_geometric_milestone_state_48d_v1",
        "source_awac_v2": str(source),
        "source_awac_v2_report_sha256": sha256_file(source / "report.json"),
        "formal_manifest": str(manifest_path),
        "formal_manifest_sha256": sha256_file(manifest_path),
        "episode_count": sum(value["episodes"] for value in split_reports.values()),
        "transition_count": total, "splits": split_reports,
        "episodes_by_category": {name: len(category_ids.get(name, set())) for name in (
            "nominal_success", "normal_recovered", "delayed_recovery", "failure")},
        "observation": {"shape": [48], "definition":
            "policy_state_42 + object_grasped + awac_milestones_t[5]"},
        "next_observation": {"shape": [48], "definition":
            "next_policy_state_42 + next_object_grasped + awac_milestones_t1[5]"},
        "milestone_tracker": {
            "implementation": "mujoco_shared_control.awac.milestones.MilestoneTracker",
            "config": asdict(milestone_config), "order": list(MILESTONE_NAMES),
            "timing": "current before action; update(next_state) after action",
            "legacy_hdf5_path_preserved": "labels/task_milestones",
            "legacy_used_by_policy_or_reward": False,
        },
        "reward": {"version": "awac_reward_v1", "config": asdict(reward_config),
                   "milestone_events": "awac_milestones_t 0->awac_milestones_t1 1"},
        "continuous_action": source_report["continuous_action"],
        "gripper_action": source_report["gripper_action"],
        "delayed_recovery_included": False,
        "source_hdf5_modified": False, "formal_manifest_modified": False,
    }
    expected_categories = {"nominal_success": 1000, "normal_recovered": 78,
                           "delayed_recovery": 0, "failure": 156}
    if (report["episode_count"], total, report["episodes_by_category"]) != (
            1234, 150406, expected_categories):
        raise RuntimeError("frozen corpus invariant failed")
    if split_reports["train"]["transitions"] != 135237 or split_reports["validation"]["transitions"] != 15169:
        raise RuntimeError("frozen split invariant failed")
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

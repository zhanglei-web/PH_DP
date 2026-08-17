#!/usr/bin/env python3
"""Derive AWAC-v2 43-D state and hybrid actions from frozen AWAC-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from mujoco_shared_control.collection.manifest import sha256_file


GRIPPER_OPEN_THRESHOLD = 0.375


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/awac_dataset/awac_v1_formal_rule"))
    parser.add_argument("--formal-manifest", type=Path, default=Path("manifests/rule_expert_v1_formal.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/awac_dataset/awac_v2_hybrid_formal_rule"))
    args = parser.parse_args()
    source = args.source.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    formal_path = args.formal_manifest.resolve(); formal = json.loads(formal_path.read_text())
    root = (formal_path.parent / formal["dataset_root"]).resolve()
    entries = {item["episode_id"]: item for item in formal["episodes"]}
    split_report = {}
    category_episodes: dict[str, set[str]] = {}
    stage_counts: dict[str, dict[str, int]] = {}
    total = 0
    for split in ("train", "validation"):
        with np.load(source / f"{split}.npz", allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]).copy() for name in data.files if name != "action"}
            episode_ids = np.asarray(data["episode_id"])
            steps = np.asarray(data["step_index"], np.int64)
            grasped = np.empty(len(steps), np.float32)
            next_grasped = np.empty(len(steps), np.float32)
            for episode_id in np.unique(episode_ids):
                key = str(episode_id); item = entries[key]
                if item["category"] == "delayed_recovery":
                    raise RuntimeError("frozen AWAC-v1 unexpectedly contains delayed recovery")
                path = (root / item["path"]).resolve()
                if sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"HDF5 checksum mismatch: {key}")
                indices = np.flatnonzero(episode_ids == episode_id)
                source_steps = steps[indices]
                with h5py.File(path, "r") as episode:
                    grasped[indices] = episode["observations/object_grasped"][source_steps].astype(np.float32)
                    next_grasped[indices] = episode["next_observations/object_grasped"][source_steps].astype(np.float32)
                category_episodes.setdefault(str(item["category"]), set()).add(key)
            full_action = np.asarray(data["action"], np.float32)
            arrays["obs"] = np.concatenate((np.asarray(data["obs"], np.float32), grasped[:, None]), axis=1)
            arrays["next_obs"] = np.concatenate((np.asarray(data["next_obs"], np.float32), next_grasped[:, None]), axis=1)
            arrays["continuous_action"] = full_action[:, :6].copy()
            arrays["gripper_action"] = (full_action[:, 6] < GRIPPER_OPEN_THRESHOLD).astype(np.float32)
            arrays["normalized_gripper_metadata"] = full_action[:, 6].copy()
            if arrays["obs"].shape != (len(steps), 43) or arrays["continuous_action"].shape != (len(steps), 6):
                raise RuntimeError("hybrid dataset dimensions are invalid")
            if not np.all(np.isin(arrays["gripper_action"], [0.0, 1.0])):
                raise RuntimeError("hybrid gripper labels are not binary")
            _atomic_npz(output / f"{split}.npz", arrays)
            stages = np.asarray(data["expert_stage"], int)
            stage_report = {}
            for stage in sorted(np.unique(stages)):
                mask = stages == stage
                stage_report[str(stage)] = {
                    "transitions": int(mask.sum()),
                    "close_count": int(arrays["gripper_action"][mask].sum()),
                    "close_ratio": float(arrays["gripper_action"][mask].mean()),
                }
            stage_counts[split] = stage_report
            split_report[split] = {
                "episodes": int(len(np.unique(episode_ids))), "transitions": len(steps),
                "close_count": int(arrays["gripper_action"].sum()),
                "close_ratio": float(arrays["gripper_action"].mean()),
                "sha256": sha256_file(output / f"{split}.npz"),
            }
            total += len(steps)
    report = {
        "dataset_version": "awac_v2_hybrid_43d_v1",
        "source_awac_v1": str(source),
        "source_awac_v1_report_sha256": sha256_file(source / "report.json"),
        "formal_manifest": str(formal_path),
        "formal_manifest_sha256": sha256_file(formal_path),
        "episode_count": sum(value["episodes"] for value in split_report.values()),
        "transition_count": total,
        "splits": split_report,
        "episodes_by_category": {name: len(category_episodes.get(name, set())) for name in (
            "nominal_success", "normal_recovered", "delayed_recovery", "failure"
        )},
        "observation": {"shape": [43], "definition": "policy_state_42 + observations/object_grasped"},
        "next_observation": {"shape": [43], "definition": "next_policy_state_42 + next_observations/object_grasped"},
        "continuous_action": {"shape": [6], "source": "actions/normalized[:6]", "range": [-1, 1]},
        "gripper_action": {
            "shape": [1], "source": "actions/normalized[6]", "labels": {"0": "OPEN", "1": "CLOSE"},
            "rule": "CLOSE iff normalized gripper < 0.375",
            "threshold_provenance": "existing Actor BC midpoint of Rule Expert CLOSE=-0.25 and OPEN=+1.0",
        },
        "stage_label_statistics": stage_counts,
        "reward_version": "awac_reward_v1_unchanged",
        "delayed_recovery_included": False,
    }
    if report["episode_count"] != 1234 or total != 150406 or report["episodes_by_category"]["delayed_recovery"]:
        raise RuntimeError("AWAC-v2 dataset does not preserve the frozen 1234/150406 corpus")
    _atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

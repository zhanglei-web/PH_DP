#!/usr/bin/env python3
"""Append 18k successes and freeze a reference-based 20k Expert Dataset V2."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collect_final_learned_expert_formal2000 import (
    alignment_check, metadata_from_npz, statistics_xyz,
)
from collect_final_learned_expert_sanity import (
    CHECKPOINT, EXPECTED_SHA256, atomic_json, atomic_npz, collect_episode,
    numeric_nonfinite, sha256,
)
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.milestones import GeometricTaskPhase, phase_from_milestones
from mujoco_shared_control.awac.reward import AWACRewardV1Config


V1 = Path("outputs/learned_expert_collection/final_online_awac20k_formal2000_20260816T130000Z")
RUN = Path("outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z")
NEW_TARGET = 18_000
TOTAL_TARGET = 20_000
SEED_START = 1_300_000
MAX_NEW_ATTEMPTS = 22_000
SPLIT_SEED = 2_026_081_603
WORKERS = 12

_PREDICTOR: HybridCheckpointPredictor | None = None


def _init_worker(checkpoint: str) -> None:
    global _PREDICTOR
    torch.set_num_threads(1)
    _PREDICTOR = HybridCheckpointPredictor(checkpoint)


def _collect_worker(args: tuple[int, str, str]) -> dict[str, Any]:
    seed, success_dir, failure_dir = args
    assert _PREDICTOR is not None
    episode_id = f"final_expert_v2_new_{seed}"
    arrays, metadata = collect_episode(
        _PREDICTOR, seed, episode_id, AWACRewardV1Config(),
    )
    directory = Path(success_dir) if metadata["success"] else Path(failure_dir)
    path = directory / f"{episode_id}.npz"
    atomic_npz(path, arrays)
    metadata["path"] = str(path.resolve())
    metadata["source_dataset"] = "v2_append_18000"
    return metadata


def _existing(directory: Path, success: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.npz")):
        row = metadata_from_npz(path)
        row["source_dataset"] = "v2_append_18000"
        if row["success"] != success:
            raise ValueError(f"wrong outcome directory: {path}")
        rows.append(row)
    return rows


def _audit(paths: list[Path], checkpoint_hash: str) -> dict[str, Any]:
    shape_failures = []
    nan_count = inf_count = milestone_regressions = illegal_milestones = 0
    phase_regressions = phase_mapping_errors = 0
    objects, goals = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            length = len(source["step_index"])
            for name, width in {
                "diffusion_observation_43": 43, "state_48": 48,
                "executed_action_7": 7, "next_diffusion_observation_43": 43,
                "milestone_t": 5,
            }.items():
                if name not in source or source[name].shape != (length, width):
                    shape_failures.append({"episode": path.stem, "field": name})
            nan, inf = numeric_nonfinite({name: source[name] for name in source.files})
            nan_count += nan; inf_count += inf
            milestones = np.vstack((source["milestone_t"], source["next_milestone_t1"][-1:])).astype(int)
            milestone_regressions += int((np.diff(milestones, axis=0) < 0).sum())
            illegal_milestones += int(np.any(milestones[:, 1:] > milestones[:, :-1], axis=1).sum())
            phases = source["phase_t"].astype(int)
            expected = np.asarray([int(phase_from_milestones(value)) for value in source["milestone_t"]])
            phase_mapping_errors += int((phases != expected).sum())
            phase_regressions += int((np.diff(np.r_[phases, int(source["next_phase_t1"][-1])]) < 0).sum())
            objects.append(source["initial_object_pose"][:3, 3])
            goals.append(source["goal_pose"][:3, 3])
    objects_array = np.asarray(objects); goals_array = np.asarray(goals)
    alignment = alignment_check(paths)
    return {
        "status": "PASS" if (
            len(paths) == TOTAL_TARGET and not shape_failures and nan_count == 0 and inf_count == 0
            and milestone_regressions == 0 and illegal_milestones == 0
            and phase_regressions == 0 and phase_mapping_errors == 0 and alignment["pass"]
            and sha256(CHECKPOINT.resolve()) == checkpoint_hash
        ) else "FAIL",
        "successful_episodes": len(paths),
        "shape": {"pass": not shape_failures, "failures": shape_failures},
        "nonfinite": {"nan_count": nan_count, "inf_count": inf_count},
        "milestone": {"pass": milestone_regressions == 0 and illegal_milestones == 0,
                      "one_to_zero": milestone_regressions, "illegal": illegal_milestones},
        "phase": {"pass": phase_regressions == 0 and phase_mapping_errors == 0,
                  "regressions": phase_regressions, "mapping_errors": phase_mapping_errors},
        "alignment": alignment,
        "reset_diversity": {
            "pass": bool(np.all(objects_array[:, :2].std(0) > 1e-6) and np.all(goals_array[:, :2].std(0) > 1e-6)),
            "object_initial_xyz": statistics_xyz(objects_array), "goal_xyz": statistics_xyz(goals_array),
        },
        "checkpoint_sha256": sha256(CHECKPOINT.resolve()),
        "checkpoint_unchanged": sha256(CHECKPOINT.resolve()) == checkpoint_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    checkpoint = CHECKPOINT.resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Final Expert checkpoint hash mismatch: {checkpoint_hash}")
    v1_report = json.loads((V1 / "collection_report.json").read_text())
    v1_rows = json.loads((V1 / "episode_manifest.json").read_text())
    old_success = [dict(row, source_dataset="formal2000_v1") for row in v1_rows if row["success"]]
    if len(old_success) != 2000 or v1_report["total_successful_transitions"] != 254350:
        raise RuntimeError("frozen formal2000 no longer matches its contract")

    run = RUN.resolve(); success_dir = run / "new_success"; failure_dir = run / "new_failure"
    if not args.resume:
        success_dir.mkdir(parents=True, exist_ok=False); failure_dir.mkdir()
    elif not success_dir.is_dir() or not failure_dir.is_dir():
        raise RuntimeError("V2 append directory is not resumable")
    successes = _existing(success_dir, True)
    failures = _existing(failure_dir, False)
    next_seed = max([row["seed"] for row in successes + failures], default=SEED_START - 1) + 1
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(checkpoint),)) as pool:
        while len(successes) < NEW_TARGET:
            remaining = NEW_TARGET - len(successes)
            batch_size = 1 if remaining <= args.workers else args.workers
            if len(successes) + len(failures) + batch_size > MAX_NEW_ATTEMPTS:
                raise RuntimeError("maximum append attempts reached")
            task_args = [
                (seed, str(success_dir), str(failure_dir))
                for seed in range(next_seed, next_seed + batch_size)
            ]
            next_seed += batch_size
            rows = pool.map(_collect_worker, task_args)
            successes.extend(row for row in rows if row["success"])
            failures.extend(row for row in rows if not row["success"])
            if (len(successes) + len(failures)) % 120 < args.workers or batch_size == 1:
                atomic_json(run / "collection_progress.json", {
                    "new_attempted": len(successes) + len(failures),
                    "new_successful": len(successes), "new_failed": len(failures),
                    "target": NEW_TARGET, "last_seed": next_seed - 1,
                })
                print(json.dumps({"attempted": len(successes) + len(failures),
                                  "successful": len(successes), "failed": len(failures)}), flush=True)

    combined = old_success + sorted(successes, key=lambda row: row["seed"])
    if len(combined) != TOTAL_TARGET:
        raise RuntimeError("combined V2 episode count is not 20,000")
    atomic_json(run / "episode_manifest.json", combined)
    atomic_json(run / "new_failure_manifest.json", sorted(failures, key=lambda row: row["seed"]))

    ids = np.asarray([row["episode_id"] for row in combined])
    order = np.random.default_rng(SPLIT_SEED).permutation(TOTAL_TARGET)
    assignments = {"train": order[:16000], "validation": order[16000:18000], "test": order[18000:]}
    by_id = {row["episode_id"]: row for row in combined}
    split_manifest = {
        "dataset_version": "unified_expert_dataset_v2_20000", "split_unit": "episode",
        "shuffle_seed": SPLIT_SEED,
        "shared_for": ["global_diffusion", "phase_classifier", "phase_conditioned_diffusion"],
        "splits": {name: ids[index].tolist() for name, index in assignments.items()},
    }
    split_stats = {
        name: {"episodes": len(index), "transitions": int(sum(by_id[value]["transitions"] for value in ids[index]))}
        for name, index in assignments.items()
    }
    sets = [set(split_manifest["splits"][name]) for name in ("train", "validation", "test")]
    leakage = bool(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    atomic_json(run / "split_manifest.json", split_manifest)

    paths = [Path(row["path"]).resolve() for row in combined]
    integrity = _audit(paths, checkpoint_hash)
    integrity["split_leakage"] = {"pass": not leakage, "intersection_count": 0 if not leakage else -1}
    if leakage:
        integrity["status"] = "FAIL"
    atomic_json(run / "integrity_report.json", integrity)
    transitions = int(sum(row["transitions"] for row in combined))
    failure_counts = Counter(row["termination_reason"] for row in failures)
    report = {
        "status": integrity["status"], "dataset_version": "Unified Expert Dataset V2",
        "v1_reference_root": str(V1.resolve()), "v1_successful_episodes": 2000,
        "v1_successful_transitions": 254350,
        "new_attempted_episodes": len(successes) + len(failures),
        "new_successful_episodes": len(successes), "new_failed_episodes": len(failures),
        "total_successful_episodes": len(combined), "total_successful_transitions": transitions,
        "split_statistics": split_stats, "split_leakage": leakage,
        "failure_breakdown": dict(failure_counts),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "gradient_updates": 0, "optimizer_updates": 0, "replay_appends": 0,
        "exploration": 0, "reward_config": asdict(AWACRewardV1Config()),
    }
    atomic_json(run / "collection_report.json", report)
    print(json.dumps({"run": str(run), **report}, indent=2), flush=True)
    if integrity["status"] != "PASS":
        raise RuntimeError("Unified Expert Dataset V2 integrity failed")


if __name__ == "__main__":
    main()

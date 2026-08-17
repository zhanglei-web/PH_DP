#!/usr/bin/env python3
"""Run the bounded three-round Learned Expert DAgger v1 protocol."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mujoco_shared_control.actor_bc.dagger import (
    DAggerConfig, collect_round, evaluate_actor, load_d0, load_round,
    module_checksum, train_round,
)
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2


ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt"
MANIFEST = ROOT / "manifests/rule_expert_v1_formal.json"
PRIMARY = list(range(300_000, 300_100))
SECONDARY = list(range(420_000, 420_100))
FINAL_SEALED = list(range(500_000, 500_100))
ROUND_SEEDS = {
    1: list(range(1_000_000, 1_001_000)),
    2: list(range(1_010_000, 1_011_000)),
    3: list(range(1_020_000, 1_021_000)),
}
CHECKPOINT_STEPS = (0, 1000, 2000, 5000, 10000)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_states(manifest: Path) -> np.ndarray:
    data = ManifestActorDataset(manifest, "validation")
    chunks = []
    for entry in data.entries:
        with h5py.File(entry.path, "r") as handle:
            chunks.append(np.asarray(handle["observations/policy_state_42"], np.float32))
    return np.concatenate(chunks)


@torch.no_grad()
def drift(actor, initial, states, mean, std) -> dict[str, Any]:
    current, reference = [], []
    for start in range(0, len(states), 1024):
        x = torch.as_tensor((states[start:start + 1024] - mean) / std)
        current.append(actor.deterministic_action(x).cpu())
        reference.append(initial.deterministic_action(x).cpu())
    error = (torch.cat(current) - torch.cat(reference)).abs()
    return {"normalized_mae": float(error.mean()), "xyz_mae": float(error[:, :3].mean()),
            "rotation_mae": float(error[:, 3:6].mean()), "gripper_mae": float(error[:, 6].mean()),
            "max_absolute_difference": float(error.max()), "per_dimension_mae": error.mean(0).tolist()}


def compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    m = evaluation["milestone_rates"]; reasons = evaluation["termination_reason_counts"]
    return {"success": evaluation["success"], "success_rate": evaluation["success_rate"],
            "grasp": m["grasped"], "lift": m["lifted"], "transport": m["transported"],
            "release": m["released"], "release_stable": m["retreated"],
            "illegal_drop": reasons.get("illegal_drop", 0),
            "ik_failure": reasons.get("ik_failure_limit", 0),
            "timeout": reasons.get("time_limit", 0), "termination_reason_counts": reasons,
            "rows": evaluation["rows"]}


def paired(initial: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    a = {row["seed"]: row["success"] for row in initial["rows"]}
    b = {row["seed"]: row["success"] for row in candidate["rows"]}
    return {"initial_success": sum(a.values()), "candidate_success": sum(b.values()),
            "new_successes": sum(not a[s] and b[s] for s in a),
            "lost_successes": sum(a[s] and not b[s] for s in a)}


def load_actor(path: Path):
    actor, _critic, _target, payload = load_aligned_v2(ALIGNED)
    if path != ALIGNED:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    return actor, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("dagger_v1_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-round", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--episodes-per-round", type=int, default=1000)
    args = parser.parse_args()
    run = ROOT / "outputs/dagger" / args.run_id; run.mkdir(parents=True, exist_ok=False)
    config = DAggerConfig(episodes_per_round=args.episodes_per_round)
    actor, payload = load_actor(ALIGNED); initial_actor = deepcopy(actor).eval().requires_grad_(False)
    mean = np.asarray(payload["observation_mean"], np.float32); std = np.asarray(payload["observation_std"], np.float32)
    d0_state, d0_action, d0_manifest = load_d0(MANIFEST)
    val_states = validation_states(MANIFEST)
    dump(run / "config.json", {**asdict(config), "round_seed_ranges": {k: [min(v), max(v)] for k, v in ROUND_SEEDS.items()},
                                "primary_dev_seeds": [min(PRIMARY), max(PRIMARY)],
                                "secondary_dev_seeds": [min(SECONDARY), max(SECONDARY)],
                                "final_test_seeds_sealed": [min(FINAL_SEALED), max(FINAL_SEALED)],
                                "round_mixture": "50% D0; remaining 50% equal across D1..Dk"})
    dump(run / "base_actor_source.json", {"path": str(ALIGNED), "sha256": sha(ALIGNED),
                                           "actor_checksum": module_checksum(actor)})
    dump(run / "rule_expert_source.json", {"policy": "rule_pick_place_v1",
                                            "frozen_commit": "d5ce43ff70af25491c545ec513d56e9f988c4f6b"})
    dump(run / "reward_v2_source.json", {"reward_version": "sac_reward_v2_candidate"})
    dump(run / "D0_manifest.json", d0_manifest)
    baseline_primary = compact(evaluate_actor(actor, mean, std, PRIMARY))
    baseline_secondary = compact(evaluate_actor(actor, mean, std, SECONDARY))
    dump(run / "baseline_evaluation.json", {"primary": baseline_primary, "secondary": baseline_secondary})
    policies = {"BC0": {"primary": baseline_primary, "secondary": baseline_secondary,
                         "drift": drift(actor, initial_actor, val_states, mean, std)}}
    round_datasets = []; current_checkpoint = ALIGNED
    status = None
    for round_index in range(1, args.max_round + 1):
        round_dir = run / f"round{round_index}"; round_dir.mkdir()
        actor, _ = load_actor(current_checkpoint)
        rollout = collect_round(actor, mean, std, ROUND_SEEDS[round_index][:config.episodes_per_round],
                                round_dir, config, round_index)
        detail = rollout.pop("episodes_detail")
        dump(round_dir / "rollout_manifest.json", {"round": round_index, "seeds": rollout["seeds"],
                                                    "episodes": rollout["episodes"], "actor_checksum": rollout["actor_checksum"]})
        dump(round_dir / "learner_outcomes.json", {**rollout, "episodes": detail})
        dump(round_dir / "oracle_labels_manifest.json", {"file": "oracle_labels.npz",
                                                          "sha256": sha(round_dir / "oracle_labels.npz"),
                                                          "queried": rollout["queried_transitions"],
                                                          "valid": rollout["oracle_query_success"]})
        dump(round_dir / f"dataset_D{round_index}_manifest.json", {
            "file": "oracle_labels.npz", "subsampled_transitions": rollout["subsampled_transitions"],
            "temporal_stride": config.temporal_stride, "phase_distribution": rollout["phase_distribution_subsampled"]})
        dump(round_dir / "action_correction_stats.json", rollout["action_correction_stats"])
        round_datasets.append(load_round(round_dir / "oracle_labels.npz"))
        train_actor = deepcopy(actor)
        training = train_round(train_actor, mean, std, (d0_state, d0_action), round_datasets,
                               round_dir / "checkpoints", config)
        dump(round_dir / "training_metrics.json", training)
        evaluations = {}; drifts = {}
        for step in CHECKPOINT_STEPS:
            path = round_dir / "checkpoints" / f"actor_step_{step:05d}.pt"
            candidate, _ = load_actor(path)
            evaluations[str(step)] = compact(evaluate_actor(candidate, mean, std, PRIMARY))
            drifts[str(step)] = drift(candidate, initial_actor, val_states, mean, std)
            dump(round_dir / "primary_eval.json", evaluations)
        trained_steps = CHECKPOINT_STEPS[1:]
        best_step = max(trained_steps, key=lambda step: (
            evaluations[str(step)]["success"], -evaluations[str(step)]["illegal_drop"],
            -drifts[str(step)]["normalized_mae"]))
        best_path = round_dir / "checkpoints" / f"actor_step_{best_step:05d}.pt"
        best_actor, _ = load_actor(best_path)
        secondary_initial = compact(evaluate_actor(actor, mean, std, SECONDARY))
        secondary_best = compact(evaluate_actor(best_actor, mean, std, SECONDARY))
        secondary = {"initial": secondary_initial, "best": secondary_best,
                     "paired": paired(secondary_initial, secondary_best)}
        dump(round_dir / "secondary_eval.json", secondary)
        dump(round_dir / "actor_drift.json", drifts)
        policies[f"BC{round_index}"] = {"best_step": best_step,
            "primary": evaluations[str(best_step)], "secondary": secondary_best,
            "drift": drifts[str(best_step)], "checkpoint": str(best_path)}
        current_checkpoint = best_path
        previous_primary = policies[f"BC{round_index-1}"]["primary"]["success"]
        previous_secondary = policies[f"BC{round_index-1}"]["secondary"]["success"]
        current_primary = policies[f"BC{round_index}"]["primary"]["success"]
        current_secondary = policies[f"BC{round_index}"]["secondary"]["success"]
        # A paired 20-point loss in both 100-seed pools is an unambiguous catastrophic regression.
        if current_primary <= previous_primary - 20 and current_secondary <= previous_secondary - 20:
            status = "DAGGER_V1_REGRESSED"; break
        if current_primary >= 80 and current_secondary >= 80:
            status = "DAGGER_V1_LEARNED_EXPERT_STRONG" if min(current_primary, current_secondary) >= 90 else "DAGGER_V1_LEARNED_EXPERT_USABLE"
            break
    if status is None:
        final = policies[f"BC{len(policies)-1}"]
        stable = min(final["primary"]["success"], final["secondary"]["success"])
        status = ("DAGGER_V1_LEARNED_EXPERT_STRONG" if stable >= 90 else
                  "DAGGER_V1_LEARNED_EXPERT_USABLE" if stable >= 80 else
                  "DAGGER_V1_PARTIAL" if stable >= 70 else "DAGGER_V1_INSUFFICIENT")
    summary = {"status": status, "policies": policies, "rounds_completed": len(policies)-1,
               "final_test_used": False}
    dump(run / "summary.json", summary)
    lines = ["# DAgger Learned Expert v1", "", f"Status: `{status}`", "",
             "Pure deterministic learner rollout; frozen Rule Expert labels only; no SAC/Critic updates.", "",
             "| Policy | Primary success | Secondary success | Grasp | Lift | Transport | Release/stable | Illegal drop | IK failure |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in policies.items():
        p, s = row["primary"], row["secondary"]
        lines.append(f"| {name} | {p['success']}/100 | {s['success']}/100 | {p['grasp']:.0%} | {p['lift']:.0%} | {p['transport']:.0%} | {p['release_stable']:.0%} | {p['illegal_drop']} | {p['ik_failure']} |")
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"run": str(run), "status": status,
                      "success": {k: [v["primary"]["success"], v["secondary"]["success"]] for k, v in policies.items()}}, indent=2))


if __name__ == "__main__": main()

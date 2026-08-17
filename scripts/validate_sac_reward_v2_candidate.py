#!/usr/bin/env python3
"""Read-only formal-corpus validation of sac_reward_v2_candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mujoco_shared_control.tasks.sac_reward import SACPhase
from mujoco_shared_control.tasks.sac_reward_v2 import SACRewardV2


GAMMA = .995
PHASE_BY_STAGE = {0: "P1", 1: "P1", 2: "P2", 3: "P3", 4: "P3",
                  5: "P4", 6: "P4", 7: "P4"}
ENUM_PHASE = {"P1": SACPhase.PRE_GRASP, "P2": SACPhase.GRASP,
              "P3": SACPhase.TRANSPORT, "P4": SACPhase.PLACE_AND_RETREAT}
COMPONENTS = ("p1_progress", "grasp_event", "p3_progress", "p4_place_progress",
              "place_event", "retreat_progress", "success_terminal",
              "failure_terminal", "illegal_drop")
CATEGORIES = ("nominal_success", "normal_recovered", "delayed_recovery", "failure")


def stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, np.float64)
    return {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()), "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def observation(ee: np.ndarray, obj: np.ndarray, goal: np.ndarray, grasped: bool) -> dict[str, Any]:
    def pose(position: np.ndarray) -> np.ndarray:
        value = np.eye(4); value[:3, 3] = position; return value
    return {"ee_pose": pose(ee), "object_pose": pose(obj), "goal_pose": pose(goal),
            "object_grasped": bool(grasped)}


def first_stable_release(inside: np.ndarray, released: np.ndarray, limit: int) -> int | None:
    run = 0
    for index in range(limit):
        run = run + 1 if inside[index] and released[index] else 0
        if run >= 4: return index
    return None


def episode(path: Path, item: dict[str, Any], v1: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with h5py.File(path, "r") as handle:
        ee = handle["observations/ee_pose_xyz_wxyz"][:, :3]
        next_ee = handle["next_observations/ee_pose_xyz_wxyz"][:, :3]
        obj = handle["observations/object_pose_xyz_wxyz"][:, :3]
        next_obj = handle["next_observations/object_pose_xyz_wxyz"][:, :3]
        goal = handle["observations/goal_pose_xyz_wxyz"][:, :3]
        next_goal = handle["next_observations/goal_pose_xyz_wxyz"][:, :3]
        grasped = handle["observations/object_grasped"][:].astype(bool)
        next_grasped = handle["next_observations/object_grasped"][:].astype(bool)
        stage = handle["labels/expert_stage"][:].astype(int)
        next_stage = handle["labels/next_expert_stage"][:].astype(int)
        reason = str(handle.attrs["termination_reason"])
    length = len(stage)
    stable_edges = np.flatnonzero((stage == 2) & (next_stage == 3))
    stable_step = int(stable_edges[0]) if len(stable_edges) else None
    inside = np.linalg.norm(next_obj - next_goal, axis=1) < .055
    release_edge = grasped & ~next_grasped
    illegal = []
    if stable_step is not None:
        for index in np.flatnonzero(release_edge):
            if index >= stable_step and not (stage[index] in (5, 6, 7) and inside[index]):
                illegal.append(int(index))
    drop_step = illegal[0] if illegal else None
    failure_edges = np.flatnonzero(next_stage == 10)
    explicit_step = int(failure_edges[0]) if len(failure_edges) else None
    failure_step = drop_step if drop_step is not None else explicit_step
    pre_failure_limit = failure_step + 1 if failure_step is not None else length
    success_step = first_stable_release(inside, ~next_grasped, pre_failure_limit)
    cutoff = success_step + 1 if success_step is not None else pre_failure_limit

    reward = SACRewardV2(); rows: list[dict[str, Any]] = []
    for index in range(cutoff):
        phase_name = PHASE_BY_STAGE.get(int(stage[index]), "OTHER")
        phase = ENUM_PHASE.get(phase_name, SACPhase.PLACE_AND_RETREAT)
        is_drop = drop_step == index and failure_step == drop_step
        is_explicit = explicit_step == index and failure_step == explicit_step and not is_drop
        is_final_failure = (index == cutoff - 1 and item["category"] == "failure"
                            and reason != "time_limit" and not is_drop and not is_explicit)
        result = reward.step(
            observation(ee[index], obj[index], goal[index], grasped[index]),
            observation(next_ee[index], next_obj[index], next_goal[index], next_grasped[index]),
            phase, stable_grasp_event=index == stable_step,
            successful_release_event=index == success_step,
            true_failure=is_explicit or is_final_failure,
            force_illegal_drop=is_drop,
            time_limit=index == cutoff - 1 and reason == "time_limit",
            apply_phase_progress=phase_name != "OTHER",
        )
        row = {"episode_id": item["episode_id"], "category": item["category"],
               "step": index, "phase": phase_name, "expert_stage": int(stage[index]),
               **result.components.as_dict(), "terminated": result.terminated,
               "truncated": result.truncated, "termination_reason": result.termination_reason}
        rows.append(row)
    rewards = np.asarray([row["reward_total"] for row in rows])
    g0 = float(np.dot(rewards, GAMMA ** np.arange(len(rewards))))
    returns = np.empty(len(rewards)); running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + GAMMA * running; returns[index] = running
    for row, value in zip(rows, returns, strict=True): row["mc_return"] = float(value)
    ep = {"episode_id": item["episode_id"], "category": item["category"],
          "seed": item["environment_seed"], "original_transitions": length,
          "v1_transitions": int(v1["sac_length"]), "v2_transitions": cutoff,
          "t_success_v2": -1 if success_step is None else success_step,
          "stable_grasp_step": -1 if stable_step is None else stable_step,
          "drop_step": -1 if drop_step is None else drop_step,
          "formal_reason": reason, "v2_reason": rows[-1]["termination_reason"],
          "v2_return": float(rewards.sum()), "v2_g0": g0,
          "v1_return": float(v1["return"]), "v1_g0": float(v1["g0"]),
          "removed_after_success": length - cutoff,
          **{f"sum_{key}": float(sum(row[key] for row in rows)) for key in COMPONENTS}}
    return ep, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    manifest_path = project / "manifests/rule_expert_v1_formal.json"
    manifest = json.loads(manifest_path.read_text())
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    v1_run = project / "outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z"
    with (v1_run / "episode_returns.csv").open() as stream:
        v1_eps = {row["episode_id"]: row for row in csv.DictReader(stream)}
    with (v1_run / "reward_components.csv").open() as stream:
        v1_rows = {(row["episode_id"], int(row["step"])): row for row in csv.DictReader(stream)}
    run = project / "outputs/sac_reward" / f"sac_reward_v2_candidate_{args.run_id}"
    run.mkdir(parents=True, exist_ok=False)
    episodes, transitions = [], []
    for number, item in enumerate(manifest["episodes"], 1):
        ep, rows = episode(root / item["path"], item, v1_eps[item["episode_id"]])
        episodes.append(ep); transitions.extend(rows)
        if number % 200 == 0: print(f"processed={number}/1300", flush=True)

    compatibility = {}
    incompatible = []
    for category in CATEGORIES:
        selected = [ep for ep in episodes if ep["category"] == category]
        found = [ep for ep in selected if ep["t_success_v2"] >= 0]
        missing = [ep["episode_id"] for ep in selected if ep["t_success_v2"] < 0]
        compatibility[category] = {"episodes": len(selected), "v2_success_found": len(found),
                                   "v2_success_missing": len(missing), "missing_episode_ids": missing}
        incompatible.extend(missing if category == "nominal_success" else [])

    # Exact unchanged component regression on every row before V2's earlier terminal.
    unchanged = {name: 0.0 for name in ("p1_progress", "grasp_event", "p3_progress",
                                        "p4_place_progress", "failure_terminal", "illegal_drop")}
    for row in transitions:
        old = v1_rows[(row["episode_id"], row["step"])]
        for name in unchanged:
            unchanged[name] = max(unchanged[name], abs(row[name] - float(old[name])))

    returns = {}
    for category in CATEGORIES:
        selected = [ep for ep in episodes if ep["category"] == category]
        returns[category] = {"episodes": len(selected), "v2_start_return": stats([e["v2_g0"] for e in selected]),
                             "v1_start_return": stats([e["v1_g0"] for e in selected]),
                             "v2_undiscounted_return": stats([e["v2_return"] for e in selected])}
    phase_stats = {phase: {"transition_count": len(values),
                           "reward": stats([row["reward_total"] for row in values])}
                   for phase in ("P1", "P2", "P3", "P4")
                   if (values := [row for row in transitions if row["phase"] == phase])}

    v1_p4 = sum(1 for row in v1_rows.values() if row["phase"] == "P4")
    v1_substage = Counter()
    v2_substage = Counter()
    for item in manifest["episodes"]:
        with h5py.File(root / item["path"], "r") as handle:
            stages = handle["labels/expert_stage"][:].astype(int)
        v1_length = int(v1_eps[item["episode_id"]]["sac_length"])
        v2_length = next(ep["v2_transitions"] for ep in episodes if ep["episode_id"] == item["episode_id"])
        for value in stages[:v1_length]:
            if value == 5: v1_substage["place"] += 1
            elif value == 6: v1_substage["release_stabilize"] += 1
            elif value == 7: v1_substage["retreat"] += 1
        for value in stages[:v2_length]:
            if value == 5: v2_substage["place"] += 1
            elif value == 6: v2_substage["release_stabilize"] += 1
            elif value == 7: v2_substage["retreat"] += 1

    success_rows = [row for row in transitions if row["success_terminal"] == 10]
    around_success = []
    by_episode = defaultdict(list)
    for row in transitions: by_episode[row["episode_id"]].append(row)
    for terminal in success_rows[:20]:
        rows = by_episode[terminal["episode_id"]]
        around_success.append({"episode_id": terminal["episode_id"],
                               "timeline": rows[max(0, terminal["step"] - 5):terminal["step"] + 1]})
    all_reward = np.asarray([row["reward_total"] for row in transitions])
    definition = {"reward_version": "sac_reward_v2_candidate", "gamma": GAMMA,
                  "p1_p2_p3": "identical to sac_reward_v1", "p4_place_weight": 2.0,
                  "stable_release_steps": 4, "goal_tolerance_m": .055,
                  "success_bonus": 10.0, "place_bonus": 0.0, "retreat_reward": 0.0,
                  "failure_penalty": -5.0}
    derived = {"format_version": "sac_reward_v2_candidate_semantic_manifest",
               "source_manifest": str(manifest_path.resolve()),
               "source_manifest_content_sha": manifest["content_sha256"],
               "reward_version": "sac_reward_v2_candidate", "gamma": GAMMA,
               "total_episodes": len(episodes), "total_transitions": len(transitions),
               "episodes": episodes}
    p4 = {"v1_p4_transition_count": v1_p4,
          "v2_p4_transition_count": phase_stats["P4"]["transition_count"],
          "v1_substages": dict(v1_substage), "v2_substages": dict(v2_substage),
          "v2_retreat_transition_count": v2_substage["retreat"]}
    event_audit = {"success_count": len(success_rows),
                   "max_success_events_per_episode": max(sum(r["success_terminal"] == 10 for r in rows)
                                                         for rows in by_episode.values()),
                   "place_bonus_nonzero_count": sum(row["place_event"] != 0 for row in transitions),
                   "retreat_reward_nonzero_count": sum(row["retreat_progress"] != 0 for row in transitions),
                   "post_success_transition_count": 0,
                   "reward_finite": bool(np.isfinite(all_reward).all()),
                   "success_neighborhoods": around_success}
    comparison = {"unchanged_component_max_abs_difference": unchanged,
                  "v1_vs_v2_returns": returns, "p4": p4,
                  "v1_source": str(v1_run.resolve())}
    config = {"source_formal_run": "formal_rule_v1_20260812T050822Z",
              "source_episodes": 1300, "writes_original_hdf5": False,
              "trains_networks": False, **definition}
    artifacts = {"config.json": config, "reward_v2_definition.json": definition,
                 "dataset_manifest.json": derived, "episode_compatibility.json": compatibility,
                 "return_statistics.json": returns, "phase_statistics.json": phase_stats,
                 "p4_substage_statistics.json": p4, "reward_event_audit.json": event_audit,
                 "v1_vs_v2_comparison.json": comparison}
    for name, value in artifacts.items():
        (run / name).write_text(json.dumps(value, indent=2) + "\n")
    with (run / "episode_returns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=episodes[0].keys())
        writer.writeheader(); writer.writerows(episodes)
    with (run / "reward_components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=transitions[0].keys())
        writer.writeheader(); writer.writerows(transitions)
    summary = {"run": str(run.resolve()), "episodes": len(episodes),
               "v2_transitions": len(transitions), "compatibility": compatibility,
               "unchanged_max_abs_difference": unchanged, "p4": p4,
               "events": {key: value for key, value in event_audit.items()
                          if key != "success_neighborhoods"}, "return_statistics": returns,
               "nominal_incompatible": incompatible}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

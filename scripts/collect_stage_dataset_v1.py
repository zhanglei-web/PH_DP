#!/usr/bin/env python3
"""Collect sanity and formal recovery-capable Stage Dataset V1."""

from __future__ import annotations

import argparse
from collections import Counter
from enum import IntEnum
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig, _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.recovery_pilot import ActivePhase, RuleBasedRecoveryPilot


class StageEvent(IntEnum):
    NONE = 0
    GRASP_FAILURE = 1
    DROP = 2
    PLACE_FAILURE = 3
    SUCCESS = 4


TRAJECTORY_TYPES = ("NORMAL", "GRASP_RECOVERY", "TRANSPORT_DROP", "PLACE_RECOVERY")
FORMAL_TARGETS = {"NORMAL": 1000, "GRASP_RECOVERY": 300, "TRANSPORT_DROP": 400, "PLACE_RECOVERY": 300}
SPLIT_TARGETS = {
    "train": {"NORMAL": 800, "GRASP_RECOVERY": 240, "TRANSPORT_DROP": 320, "PLACE_RECOVERY": 240},
    "validation": {"NORMAL": 100, "GRASP_RECOVERY": 30, "TRANSPORT_DROP": 40, "PLACE_RECOVERY": 30},
    "test": {"NORMAL": 100, "GRASP_RECOVERY": 30, "TRANSPORT_DROP": 40, "PLACE_RECOVERY": 30},
}
RUN = Path("outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z")
CONTROL_DT = CollectionConfig().control_timestep_s
MAX_STEPS = 700
WORKERS = 4

_ENV: PickPlaceEnv | None = None
_PILOT: RuleBasedRecoveryPilot | None = None
_ADAPTER: ExpertCommandAdapter | None = None


def _init_worker() -> None:
    global _ENV, _PILOT, _ADAPTER
    torch.set_num_threads(1)
    _ENV = PickPlaceEnv(render_mode=None, control_timestep=CONTROL_DT, max_episode_steps=MAX_STEPS, enable_camera=False)
    _PILOT = RuleBasedRecoveryPilot()
    _ADAPTER = ExpertCommandAdapter(_ENV.ik_controller, _PILOT.action_spec)


def _state43(env: PickPlaceEnv, observation: dict[str, Any]) -> np.ndarray:
    return np.r_[env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"]))].astype(np.float32)


def _features(observation: dict[str, Any], raw_action: np.ndarray) -> np.ndarray:
    ee = observation["ee_pose"][:3, 3]
    obj = observation["object_pose"][:3, 3]
    goal = observation["goal_pose"][:3, 3]
    ee_obj = obj - ee
    obj_goal = goal - obj
    value = np.r_[
        raw_action, ee_obj, np.linalg.norm(ee_obj), obj_goal,
        np.linalg.norm(obj_goal), np.linalg.norm(goal - ee),
        float(observation["gripper"][0]), float(bool(observation["object_grasped"])), obj[2],
    ].astype(np.float32)
    if value.shape != (19,) or not np.isfinite(value).all():
        raise ValueError("stage feature must be finite 19D")
    return value


def _executed_action(adapted, pilot: RuleBasedRecoveryPilot) -> np.ndarray:
    if adapted.accepted:
        return np.asarray(adapted.normalized, np.float32)
    gripper = pilot.action_spec.normalize(np.r_[np.zeros(6), adapted.joint_target[7]])[6]
    return np.r_[np.zeros(6), gripper].astype(np.float32)


def _write_h5(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".inprogress.h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as file:
        for name, value in arrays.items():
            file.create_dataset(name, data=value, compression="gzip", compression_opts=1)
        for name, value in metadata.items():
            file.attrs[name] = value
        file.attrs["feature_names_json"] = json.dumps([
            *(f"raw_pilot_action_{i}" for i in range(7)),
            "ee_to_object_x", "ee_to_object_y", "ee_to_object_z", "ee_object_distance",
            "object_to_goal_x", "object_to_goal_y", "object_to_goal_z", "object_goal_distance",
            "ee_goal_distance", "gripper_opening", "object_grasped", "object_height",
        ])
        file.attrs["event_names_json"] = json.dumps([event.name for event in StageEvent])
        file.attrs["phase_names_json"] = json.dumps([phase.name for phase in ActivePhase])
    temporary.replace(path)


def collect_one(trajectory_type: str, seed: int, path: Path) -> dict[str, Any]:
    assert _ENV is not None and _PILOT is not None and _ADAPTER is not None
    env, pilot, adapter = _ENV, _PILOT, _ADAPTER
    observation, reset_info = env.reset(seed=seed, options={
        "randomize_arm": True, "arm_joint_noise_scale": 1.0,
        "randomize_object": True, "randomize_goal": True,
    })
    adapter.reset(observation["ee_pose"], observation["q_obs"])
    pilot.reset(float(observation["object_pose"][2, 3]), seed + 17)
    rng = np.random.default_rng(seed + 101)
    drop_bucket = int(rng.integers(0, 3))
    drop_progress_threshold = (0.25, 0.50, 0.75)[drop_bucket]
    transport_start_object_goal_distance: float | None = None
    injection_steps = 0
    injection_active = False
    injected = False
    previous_phase: ActivePhase | None = None
    stable_success = 0
    failure_step = -1
    previous_command = previous_action = None
    rows: list[dict[str, Any]] = []
    previous_grasped = bool(observation["object_grasped"])
    place_direction = np.array([1.0, 0.0])
    final_success = False
    forced_grasp_failure_completed = False
    for step in range(MAX_STEPS):
        state = _state43(env, observation)
        expert_observation = _expert_observation(
            f"stage_{trajectory_type.lower()}_{seed}", 0, step, observation,
            state[:42], previous_command, previous_action,
        )
        command, phase = pilot.predict(expert_observation)
        # The externally forced off-goal release remains an active placement
        # attempt until the object is actually released; the following state is
        # then re-evaluated by the pilot as APPROACH.
        if trajectory_type == "PLACE_RECOVERY" and injection_active:
            phase = ActivePhase.PLACE_RELEASE
        raw_physical = command.delta_pose_gripper.copy()
        raw_normalized = pilot.action_spec.normalize(raw_physical).astype(np.float32)
        executed_physical = raw_physical.copy()
        event = StageEvent.NONE

        if trajectory_type == "GRASP_RECOVERY" and not injected and phase == ActivePhase.GRASP_LIFT:
            injection_active = True; injection_steps = 3; injected = True; failure_step = step
            event = StageEvent.GRASP_FAILURE
        if trajectory_type == "TRANSPORT_DROP" and not injected:
            current_distance = float(np.linalg.norm(
                observation["object_pose"][:2, 3] - observation["goal_pose"][:2, 3]
            ))
            if phase == ActivePhase.TRANSPORT:
                if transport_start_object_goal_distance is None:
                    transport_start_object_goal_distance = max(current_distance, 1e-8)
                progress = 1.0 - current_distance / transport_start_object_goal_distance
                if progress >= drop_progress_threshold:
                    injection_active = True; injected = True; failure_step = step
            elif previous_phase == ActivePhase.TRANSPORT and phase == ActivePhase.PLACE_RELEASE:
                # Guarantee the designated drop before the pilot leaves transport.
                phase = ActivePhase.TRANSPORT
                injection_active = True; injected = True; failure_step = step
        place_target = observation["goal_pose"][:3, 3] + np.array([0.0, 0.0, pilot.config.place_height_m])
        if (
            trajectory_type == "PLACE_RECOVERY" and not injected
            and phase == ActivePhase.PLACE_RELEASE
            and np.linalg.norm(observation["ee_pose"][:3, 3] - place_target) <= 0.012
        ):
            delta = observation["object_pose"][:2, 3] - observation["goal_pose"][:2, 3]
            place_direction = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-6 else np.array([1.0, 0.0])
            injection_active = True; injection_steps = 6; injected = True; failure_step = step

        if injection_active and trajectory_type == "GRASP_RECOVERY":
            direction = np.array([1.0, -1.0]); direction /= np.linalg.norm(direction)
            executed_physical = np.r_[direction * 0.014, 0.0, np.zeros(3), pilot.config.open_gripper_m]
            injection_steps -= 1
            if injection_steps <= 0:
                injection_active = False
                forced_grasp_failure_completed = True
        elif injection_active and trajectory_type == "TRANSPORT_DROP":
            executed_physical = np.r_[np.zeros(6), pilot.config.open_gripper_m]
        elif injection_active and trajectory_type == "PLACE_RECOVERY":
            if injection_steps > 0:
                gripper = pilot.config.open_gripper_m if injection_steps <= 1 else pilot.config.close_gripper_m
                executed_physical = np.r_[place_direction * 0.018, 0.0, np.zeros(3), gripper]
                injection_steps -= 1
            else:
                executed_physical = np.r_[np.zeros(6), pilot.config.open_gripper_m]

        adapted = adapter.adapt(executed_physical)
        executed_normalized = _executed_action(adapted, pilot)
        next_observation, _, _, _, _ = env.step(adapted.joint_target)
        if forced_grasp_failure_completed:
            pilot.confirm_grasp_failure()
            forced_grasp_failure_completed = False
        next_grasped = bool(next_observation["object_grasped"])
        if trajectory_type == "TRANSPORT_DROP" and injection_active and previous_grasped and not next_grasped:
            event = StageEvent.DROP; injection_active = False; pilot.confirm_external_failure()
        if trajectory_type == "PLACE_RECOVERY" and injection_active and previous_grasped and not next_grasped:
            event = StageEvent.PLACE_FAILURE; injection_active = False; pilot.confirm_external_failure()
        next_state = _state43(env, next_observation)
        released = bool(
            not next_grasped
            and np.linalg.norm(next_observation["object_pose"][:3, 3] - next_observation["goal_pose"][:3, 3]) < pilot.config.goal_tolerance_m
            and float(next_observation["gripper"][0]) >= 0.055
        )
        retreat = bool(
            released and np.linalg.norm(
                next_observation["ee_pose"][:3, 3]
                - (next_observation["goal_pose"][:3, 3] + np.array([0.0, 0.0, pilot.config.retreat_height_m]))
            ) <= pilot.config.position_tolerance_m
        )
        stable_success = stable_success + 1 if retreat else 0
        final_success = stable_success >= 4
        if final_success:
            event = StageEvent.SUCCESS
        rows.append({
            "stage_features": _features(observation, raw_normalized),
            "raw_pilot_action": raw_normalized,
            "executed_action": executed_normalized,
            "active_phase": np.int8(phase), "event": np.int8(event),
            "full_physical_state": state, "next_full_physical_state": next_state,
            "step_index": np.int32(step), "adapter_accepted": bool(adapted.accepted),
            "action_clipped": bool(adapted.action_clipped), "fallback_used": bool(adapted.fallback_used),
        })
        previous_command = raw_physical.copy(); previous_action = executed_normalized.copy()
        previous_grasped = next_grasped; previous_phase = phase; observation = next_observation
        if final_success:
            break

    arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    phases = arrays["active_phase"].astype(int)
    required_regression = {"GRASP_RECOVERY": (1, 0), "TRANSPORT_DROP": (2, 0), "PLACE_RECOVERY": (3, 0)}.get(trajectory_type)
    regression_found = True if required_regression is None else bool(
        np.any((phases[:-1] == required_regression[0]) & (phases[1:] == required_regression[1]))
    )
    valid = bool(final_success and regression_found and np.isfinite(arrays["stage_features"]).all())
    metadata = {
        "episode_id": f"stage_{trajectory_type.lower()}_{seed}", "trajectory_type": trajectory_type,
        "seed": seed, "failure_step": failure_step, "final_success": final_success,
        "episode_length": len(rows), "valid": valid, "reset_count": 1,
        "control_dt": CONTROL_DT, "drop_timing_bucket": drop_bucket if trajectory_type == "TRANSPORT_DROP" else -1,
        "drop_progress_threshold": drop_progress_threshold if trajectory_type == "TRANSPORT_DROP" else -1.0,
        "transport_start_object_goal_distance": (
            transport_start_object_goal_distance
            if transport_start_object_goal_distance is not None else -1.0
        ),
        "object_initial_x": float(reset_info["object_xy"][0]), "object_initial_y": float(reset_info["object_xy"][1]),
        "goal_x": float(reset_info["goal_xy"][0]), "goal_y": float(reset_info["goal_xy"][1]),
    }
    _write_h5(path, arrays, metadata)
    return {**metadata, "path": str(path.resolve()), "regression_found": regression_found}


def _worker(task: tuple[str, int, str]) -> dict[str, Any]:
    trajectory_type, seed, path = task
    return collect_one(trajectory_type, seed, Path(path))


def compressed(value: np.ndarray) -> list[int]:
    return [int(item) for index, item in enumerate(value) if index == 0 or item != value[index - 1]]


def sanity_audit(paths: list[Path]) -> dict[str, Any]:
    failures = []
    for path in paths:
        with h5py.File(path, "r") as file:
            features = file["stage_features"][:]; phase = file["active_phase"][:].astype(int)
            event = file["event"][:].astype(int); state = file["full_physical_state"][:]
            next_state = file["next_full_physical_state"][:]
            trajectory_type = str(file.attrs["trajectory_type"])
            reasons = []
            if not np.isfinite(features).all() or not np.isfinite(state).all() or not np.isfinite(next_state).all(): reasons.append("nonfinite")
            if not np.allclose(next_state[:-1], state[1:], atol=1e-7): reasons.append("alignment")
            if np.any((phase < 0) | (phase > 4)): reasons.append("phase_range")
            if int(file.attrs["reset_count"]) != 1: reasons.append("reset_count")
            sequence = compressed(phase)
            if trajectory_type == "NORMAL" and not all(value in sequence for value in [0, 1, 2, 3, 4]): reasons.append("normal_sequence")
            one_to_zero = int(np.sum((phase[:-1] == 1) & (phase[1:] == 0)))
            if trajectory_type == "NORMAL" and one_to_zero > 1: reasons.append("grasp_phase_chatter")
            if trajectory_type == "GRASP_RECOVERY" and not 1 <= one_to_zero <= 2: reasons.append("grasp_regression_count")
            required = {"GRASP_RECOVERY": (1, 0), "TRANSPORT_DROP": (2, 0), "PLACE_RECOVERY": (3, 0)}.get(trajectory_type)
            if required and not np.any((phase[:-1] == required[0]) & (phase[1:] == required[1])): reasons.append("missing_regression")
            if trajectory_type == "TRANSPORT_DROP":
                drops = np.flatnonzero(event == int(StageEvent.DROP))
                if len(drops) != 1: reasons.append("drop_event")
                elif np.linalg.norm(next_state[min(drops[0] + 10, len(state)-1), 22:25] - next_state[drops[0], 22:25]) < 0.005: reasons.append("drop_pose_static")
            if not bool(file.attrs["final_success"]): reasons.append("not_success")
            if reasons: failures.append({"path": str(path), "reasons": reasons, "sequence": sequence})
    return {"status": "PASS" if not failures else "FAIL", "episodes": len(paths), "failures": failures}


def formal_audit(rows: list[dict[str, Any]], split_manifest: dict[str, Any]) -> dict[str, Any]:
    nan = inf = alignment_failures = 0
    phase_counts = np.zeros(5, np.int64); regressions = Counter()
    for row in rows:
        with h5py.File(row["path"], "r") as file:
            for name in ("stage_features", "raw_pilot_action", "executed_action", "full_physical_state", "next_full_physical_state"):
                value = file[name][:]; nan += int(np.isnan(value).sum()); inf += int(np.isinf(value).sum())
            state = file["full_physical_state"][:]; nxt = file["next_full_physical_state"][:]
            alignment_failures += int(not np.allclose(nxt[:-1], state[1:], atol=1e-7))
            phase = file["active_phase"][:].astype(int)
            phase_counts += np.bincount(phase, minlength=5)
            for old in (1, 2, 3): regressions[f"{old}->0"] += int(np.sum((phase[:-1] == old) & (phase[1:] == 0)))
    split_sets = [set(split_manifest["splits"][name]) for name in ("train", "validation", "test")]
    leakage = bool(split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
    return {
        "status": "PASS" if nan == 0 and inf == 0 and alignment_failures == 0 and not leakage else "FAIL",
        "nan": nan, "inf": inf, "alignment": "PASS" if alignment_failures == 0 else "FAIL",
        "alignment_failures": alignment_failures, "split_leakage": leakage,
        "phase_frame_distribution": {ActivePhase(i).name: int(phase_counts[i]) for i in range(5)},
        "regression_counts": dict(regressions),
    }


def run_collection(run: Path = RUN, workers: int = WORKERS) -> Path:
    run = run.resolve(); sanity_dir = run / "sanity"; episode_dir = run / "episodes"; failure_dir = run / "failure_diagnostics"
    sanity_dir.mkdir(parents=True, exist_ok=False); episode_dir.mkdir(); failure_dir.mkdir()
    ctx = mp.get_context("spawn")
    sanity_tasks = []
    seed = 3_000_000
    for trajectory_type in TRAJECTORY_TYPES:
        for index in range(5):
            path = sanity_dir / trajectory_type / f"sanity_{trajectory_type.lower()}_{seed}.h5"
            sanity_tasks.append((trajectory_type, seed, str(path))); seed += 1
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        sanity_rows = pool.map(_worker, sanity_tasks)
    sanity_report = sanity_audit([Path(row["path"]) for row in sanity_rows])
    (run / "sanity_report.json").write_text(json.dumps(sanity_report, indent=2) + "\n")
    print(json.dumps({"sanity": sanity_report["status"], "failures": sanity_report["failures"]}), flush=True)
    if sanity_report["status"] != "PASS":
        raise RuntimeError("Stage Dataset sanity failed")

    successful: list[dict[str, Any]] = []; failed: list[dict[str, Any]] = []
    next_seed = 3_100_000
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for trajectory_type in TRAJECTORY_TYPES:
            target = FORMAL_TARGETS[trajectory_type]; accepted = 0
            while accepted < target:
                remaining = target - accepted
                batch = min(workers, remaining)
                tasks = []
                for current_seed in range(next_seed, next_seed + batch):
                    path = episode_dir / trajectory_type / f"stage_{trajectory_type.lower()}_{current_seed}.h5"
                    tasks.append((trajectory_type, current_seed, str(path)))
                next_seed += batch
                for row in pool.map(_worker, tasks):
                    if row["valid"]:
                        successful.append(row); accepted += 1
                    else:
                        destination = failure_dir / Path(row["path"]).name
                        Path(row["path"]).replace(destination); row["path"] = str(destination.resolve()); failed.append(row)
                if accepted % 100 < workers or accepted == target:
                    print(json.dumps({"type": trajectory_type, "successful": accepted, "target": target, "failed": len(failed)}), flush=True)

    by_type = {name: [row for row in successful if row["trajectory_type"] == name] for name in TRAJECTORY_TYPES}
    split_manifest = {"split_unit": "episode", "splits": {name: [] for name in ("train", "validation", "test")}}
    for trajectory_type in TRAJECTORY_TYPES:
        rows = by_type[trajectory_type]
        rng = np.random.default_rng(3_200_000 + TRAJECTORY_TYPES.index(trajectory_type))
        order = rng.permutation(len(rows)); start = 0
        for split in ("train", "validation", "test"):
            count = SPLIT_TARGETS[split][trajectory_type]
            chosen = [rows[int(i)] for i in order[start:start + count]]; start += count
            split_manifest["splits"][split].extend(row["episode_id"] for row in chosen)
    by_id = {row["episode_id"]: row for row in successful}
    split_manifest["episode_paths"] = {episode_id: by_id[episode_id]["path"] for episode_id in by_id}
    split_manifest["counts_by_type"] = SPLIT_TARGETS
    (run / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    report = formal_audit(successful, split_manifest)
    report.update({
        "episodes": len(successful), "transitions": int(sum(row["episode_length"] for row in successful)),
        "trajectory_counts": {name: len(by_type[name]) for name in TRAJECTORY_TYPES},
        "split_counts": {name: len(split_manifest["splits"][name]) for name in ("train", "validation", "test")},
        "failed_attempts": len(failed),
    })
    (run / "episode_manifest.json").write_text(json.dumps(successful, indent=2) + "\n")
    (run / "failure_manifest.json").write_text(json.dumps(failed, indent=2) + "\n")
    (run / "integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise RuntimeError("formal Stage Dataset integrity failed")
    print(json.dumps({"dataset": str(run), **report}, indent=2), flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(); run_collection(workers=args.workers)

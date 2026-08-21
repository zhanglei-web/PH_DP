#!/usr/bin/env python3
"""E2 NORMAL validation sweep for RuleBasedRecoveryPilot + Global gamma.

This is the pre-failure gate for E2.  It does not inspect or use any existing
failure-recovery outcomes when selecting gamma_RP.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

try:
    from evaluate_experiment1_global_effectiveness import (
        DEFAULT_CHECKPOINT,
        GlobalSharedController,
        _sha256,
    )
except ModuleNotFoundError:
    from scripts.evaluate_experiment1_global_effectiveness import (
        DEFAULT_CHECKPOINT,
        GlobalSharedController,
        _sha256,
    )
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.evaluation.experiment_recorder import EpisodeTraceRecorder, load_trace
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/e2_rp_normal_gamma_sweep"
PILOT_SOURCE = PROJECT_ROOT / "src/mujoco_shared_control/experts/recovery_pilot.py"
PILOT_SHA256 = "30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58"
CONTROL_DT = 0.05
MAX_STEPS = 700
MAX_IK_FAILURES = 5
NUM_DIFFUSION_STEPS = 50
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9, 1.0)
IDENTITY_SEEDS = tuple(range(4_290_000, 4_290_010))
SMOKE_SEEDS = tuple(range(4_300_000, 4_300_002))
VALIDATION_SEEDS = tuple(range(4_400_000, 4_400_050))
DIFFUSION_SEED_BASE = 7_200_000


def state43(env: PickPlaceEnv, obs: dict[str, Any]) -> np.ndarray:
    state = np.r_[env.get_policy_observation(obs), np.float32(bool(obs["object_grasped"]))].astype(np.float32)
    if state.shape != (43,) or not np.isfinite(state).all():
        raise ValueError("E2 NORMAL Global state must be finite 43D")
    return state


def gamma_key(gamma: float) -> str:
    return f"{gamma:.3f}".rstrip("0").rstrip(".") or "0"


def effective_step(gamma: float) -> int:
    return int((NUM_DIFFUSION_STEPS - 1) * float(gamma))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_pilot_action(pilot: RuleBasedRecoveryPilot, physical_action_7: np.ndarray) -> np.ndarray:
    action = pilot.action_spec.normalize(np.asarray(physical_action_7, np.float64)).astype(np.float32)
    post = GlobalActionPostprocessor.from_expert_spec(pilot.action_spec)
    return post(action).astype(np.float32)


def run_episode(
    *,
    method: str,
    environment_seed: int,
    pilot_seed: int,
    diffusion_seed: int,
    gamma: float,
    controller: GlobalSharedController | None,
    postprocessor: GlobalActionPostprocessor,
    trace_path: Path,
) -> dict[str, Any]:
    if method not in {"noassist", "global"}:
        raise ValueError("method must be noassist or global")
    if method == "global" and controller is None:
        raise ValueError("global method requires a controller")
    env = PickPlaceEnv(render_mode=None, control_timestep=CONTROL_DT, max_episode_steps=MAX_STEPS, enable_camera=False)
    pilot = RuleBasedRecoveryPilot()
    adapter = ExpertCommandAdapter(env.ik_controller, pilot.action_spec)
    episode_id = f"e2_rp_normal_{method}_gamma_{gamma_key(gamma)}_seed_{environment_seed}"
    recorder = EpisodeTraceRecorder(episode_id)
    try:
        obs, reset = env.reset(
            seed=int(environment_seed),
            options={
                "randomize_arm": True,
                "arm_joint_noise_scale": 1.0,
                "randomize_object": True,
                "randomize_goal": True,
            },
        )
        adapter.reset(obs["ee_pose"], obs["q_obs"])
        pilot.reset(float(obs["object_pose"][2, 3]), int(pilot_seed))
        if method == "global":
            controller.reset_sampling(int(diffusion_seed))  # type: ignore[union-attr]
        reward = AWACRewardV1Online(state43(env, obs))
        previous_command = None
        previous_action = None
        consecutive_ik = 0
        policy_clip_steps = 0
        adapter_rejection_count = 0
        fallback_count = 0
        inference_ms_values: list[float] = []
        reward_step = None
        final_step = 0
        for step in range(MAX_STEPS):
            state = state43(env, obs)
            expert_obs = _expert_observation(
                episode_id, 0, step, obs, state[:42], previous_command, previous_action
            )
            command, phase = pilot.predict(expert_obs)
            raw_physical = command.delta_pose_gripper.copy()
            raw_action = canonical_pilot_action(pilot, raw_physical)
            if method == "noassist":
                assisted_action = raw_action.copy()
                inference_ms = 0.0
            else:
                started = time.perf_counter()
                assisted_action = controller.assist(state, raw_action, float(gamma))  # type: ignore[union-attr]
                inference_ms = (time.perf_counter() - started) * 1000.0
            clipped_action = np.clip(assisted_action, -1.0, 1.0)
            postprocessed_action = postprocessor(clipped_action)
            policy_clip = bool(not np.array_equal(assisted_action, clipped_action))
            policy_clip_steps += int(policy_clip)
            assisted_physical = pilot.action_spec.denormalize(postprocessed_action)
            adapted = adapter.adapt(assisted_physical)
            adapter_rejection_count += int(not adapted.accepted)
            fallback_count += int(adapted.fallback_used)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            executed_action = np.asarray(adapted.normalized, np.float32)
            next_obs, _, _, _, _ = env.step(adapted.joint_target)
            next_state = state43(env, next_obs)
            milestone_before = reward.tracker.current.copy()
            reward_step = reward.step(
                state,
                next_state,
                ik_failure=consecutive_ik >= MAX_IK_FAILURES,
                time_limit=step + 1 >= MAX_STEPS,
            )
            correction = np.asarray(assisted_action - raw_action, np.float64)
            raw_motion = float(np.linalg.norm(raw_action[:6]))
            assisted_motion = float(np.linalg.norm(assisted_action[:6]))
            cosine = (
                float(np.dot(raw_action[:6], assisted_action[:6]) / (raw_motion * assisted_motion))
                if raw_motion > 1e-12 and assisted_motion > 1e-12
                else 1.0
            )
            ee = obs["ee_pose"][:3, 3].astype(np.float64)
            obj = obs["object_pose"][:3, 3].astype(np.float64)
            goal = obs["goal_pose"][:3, 3].astype(np.float64)
            recorder.append_step(
                step_index=step,
                simulation_time=float(obs["timestamp"][0]),
                state_43=state,
                clean_pilot_action_7=raw_action,
                raw_pilot_action_7=raw_action,
                assisted_action_7=assisted_action,
                clipped_assisted_action_7=clipped_action,
                postprocessed_action_7=postprocessed_action,
                executed_action_7=executed_action,
                milestone_t=milestone_before,
                active_stage=int(phase),
                object_grasped=bool(obs["object_grasped"]),
                reward=reward_step.reward,
                adapter_accepted=adapted.accepted,
                action_clipped=adapted.action_clipped or policy_clip,
                fallback_used=adapted.fallback_used,
                diffusion_inference_ms=inference_ms,
                ee_position=ee,
                object_position=obj,
                goal_position=goal,
                ee_object_distance=float(np.linalg.norm(obj - ee)),
                object_goal_distance=float(np.linalg.norm(obj - goal)),
                ee_goal_distance=float(np.linalg.norm(goal - ee)),
                gripper_opening=float(obs["gripper"][0]),
                translation_correction_norm_normalized=float(np.linalg.norm(correction[:3])),
                rotation_correction_norm_normalized=float(np.linalg.norm(correction[3:6])),
                translation_correction_m=float(np.linalg.norm(correction[:3] * pilot.action_spec.scale[:3])),
                rotation_correction_rad=float(np.linalg.norm(correction[3:6] * pilot.action_spec.scale[3:6])),
                motion_cosine_similarity=cosine,
                gripper_changed_by_assist=bool(postprocessed_action[6] != raw_action[6]),
            )
            previous_command = raw_physical.copy()
            previous_action = executed_action.copy()
            obs = next_obs
            final_step = step
            inference_ms_values.append(inference_ms)
            if reward_step.terminated or reward_step.truncated:
                break
        if reward_step is None:
            raise RuntimeError("episode produced no transition")
        arrays = recorder.arrays()
        recorder.save(trace_path)
        total_steps = len(arrays["step_index"])
        return {
            "episode_id": episode_id,
            "environment_seed": int(environment_seed),
            "pilot_seed": int(pilot_seed),
            "diffusion_seed": int(diffusion_seed),
            "method": method,
            "gamma": float(gamma),
            "effective_diffusion_step": effective_step(gamma),
            "success": bool(reward_step.task_success),
            "termination_reason": reward_step.termination_reason,
            "episode_steps": total_steps,
            "completion_time_s": float(total_steps * CONTROL_DT),
            "grasp": bool(reward_step.milestones[0]),
            "lift": bool(reward_step.milestones[1]),
            "transport": bool(reward_step.milestones[2]),
            "place": bool(reward_step.milestones[3]),
            "retreat": bool(reward_step.milestones[4]),
            "illegal_drop": reward_step.termination_reason == "illegal_drop",
            "ik_failure": reward_step.termination_reason == "ik_failure_limit",
            "timeout": reward_step.termination_reason == "timeout",
            "policy_clip_steps": int(policy_clip_steps),
            "adapter_rejection_count": int(adapter_rejection_count),
            "fallback_count": int(fallback_count),
            "mean_translation_correction_m": float(np.mean(arrays["translation_correction_m"])),
            "mean_rotation_correction_rad": float(np.mean(arrays["rotation_correction_rad"])),
            "mean_motion_cosine_similarity": float(np.mean(arrays["motion_cosine_similarity"])),
            "mean_inference_ms": float(np.mean(inference_ms_values)),
            "initial_object_xy": json.dumps(np.asarray(reset["object_xy"], dtype=float).tolist(), separators=(",", ":")),
            "initial_goal_xy": json.dumps(np.asarray(reset["goal_xy"], dtype=float).tolist(), separators=(",", ":")),
            "trace_path": str(trace_path.resolve()),
            "trace_length": total_steps,
            "nan_count": int(sum(np.isnan(value).sum() for value in arrays.values() if value.dtype.kind in "fc")),
            "inf_count": int(sum(np.isinf(value).sum() for value in arrays.values() if value.dtype.kind in "fc")),
            "last_step": final_step,
        }
    finally:
        env.close()


def summarize_gamma(gamma: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = np.asarray([row["episode_steps"] for row in rows], dtype=np.float64)
    total_steps = max(1, int(steps.sum()))
    return {
        "gamma": float(gamma),
        "effective_diffusion_step": effective_step(gamma),
        "N": len(rows),
        "success": float(np.mean([row["success"] for row in rows])),
        "grasp": float(np.mean([row["grasp"] for row in rows])),
        "lift": float(np.mean([row["lift"] for row in rows])),
        "transport": float(np.mean([row["transport"] for row in rows])),
        "place": float(np.mean([row["place"] for row in rows])),
        "retreat": float(np.mean([row["retreat"] for row in rows])),
        "illegal_drop": float(np.mean([row["illegal_drop"] for row in rows])),
        "ik_failure": float(np.mean([row["ik_failure"] for row in rows])),
        "timeout": float(np.mean([row["timeout"] for row in rows])),
        "mean_steps": float(steps.mean()),
        "median_steps": float(np.median(steps)),
        "translation_correction_mm": 1000.0 * float(np.mean([row["mean_translation_correction_m"] for row in rows])),
        "rotation_correction_rad": float(np.mean([row["mean_rotation_correction_rad"] for row in rows])),
        "motion_cosine": float(np.mean([row["mean_motion_cosine_similarity"] for row in rows])),
        "policy_clip_fraction": float(sum(row["policy_clip_steps"] for row in rows) / total_steps),
        "adapter_rejection_rate": float(sum(row["adapter_rejection_count"] for row in rows) / total_steps),
        "fallback_rate": float(sum(row["fallback_count"] for row in rows) / total_steps),
        "nan_count": int(sum(row["nan_count"] for row in rows)),
        "inf_count": int(sum(row["inf_count"] for row in rows)),
    }


def selection_status(row: dict[str, Any]) -> bool:
    return (
        row["gamma"] > 0.0
        and row["success"] >= 0.95
        and row["illegal_drop"] <= 0.05
        and row["ik_failure"] <= 0.05
        and row["timeout"] <= 0.05
        and row["nan_count"] == 0
        and row["inf_count"] == 0
        and (row["translation_correction_mm"] > 0.0 or row["rotation_correction_rad"] > 0.0)
    )


def select_gamma(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in summaries if selection_status(row)]
    if not eligible:
        return {
            "status": "FAIL",
            "selected_gamma": None,
            "selected_row": None,
            "eligible_gammas": [],
            "selection_rule": "NORMAL only: success>=95%, IK/drop/timeout<=5%, finite outputs, non-zero correction; within 2 pp of best success choose largest gamma.",
            "failure_results_used": False,
        }
    best_success = max(row["success"] for row in eligible)
    near_best = [row for row in eligible if best_success - row["success"] <= 0.02 + 1e-12]
    selected = max(near_best, key=lambda row: row["gamma"])
    return {
        "status": "PASS",
        "selected_gamma": selected["gamma"],
        "selected_row": selected,
        "eligible_gammas": [row["gamma"] for row in eligible],
        "best_success": best_success,
        "selection_rule": "NORMAL only: success>=95%, IK/drop/timeout<=5%, finite outputs, non-zero correction; within 2 pp of best success choose largest gamma.",
        "failure_results_used": False,
    }


def validate_rows(rows: list[dict[str, Any]], expected: int, gamma: float, post: GlobalActionPostprocessor) -> dict[str, Any]:
    failures: list[str] = []
    if len(rows) != expected:
        failures.append(f"rollout count {len(rows)} != {expected}")
    for row in rows:
        if float(row["gamma"]) != float(gamma):
            failures.append(f"gamma mismatch {row['episode_id']}")
        trace = load_trace(row["trace_path"])
        if trace["state_43"].shape[1:] != (43,) or trace["raw_pilot_action_7"].shape[1:] != (7,):
            failures.append(f"contract mismatch {row['episode_id']}")
        if any(not np.isfinite(value).all() for value in trace.values() if value.dtype.kind in "fc"):
            failures.append(f"nonfinite trace {row['episode_id']}")
        if not np.array_equal(trace["clean_pilot_action_7"], trace["raw_pilot_action_7"]):
            failures.append(f"raw/clean mismatch {row['episode_id']}")
        for field in ("raw_pilot_action_7", "postprocessed_action_7", "executed_action_7"):
            if not np.isin(trace[field][:, 6], [post.normalized_close, post.normalized_open]).all():
                failures.append(f"noncanonical gripper {field} {row['episode_id']}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "gamma": float(gamma),
        "expected_rollouts": int(expected),
        "nan_count": int(sum(row["nan_count"] for row in rows)),
        "inf_count": int(sum(row["inf_count"] for row in rows)),
        "failures": failures[:100],
    }


def identity_audit(root: Path, controller: GlobalSharedController, post: GlobalActionPostprocessor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for seed in IDENTITY_SEEDS:
        no = run_episode(
            method="noassist",
            environment_seed=seed,
            pilot_seed=seed + 17,
            diffusion_seed=DIFFUSION_SEED_BASE + seed,
            gamma=0.0,
            controller=None,
            postprocessor=post,
            trace_path=root / "identity_traces" / f"noassist_seed_{seed}.npz",
        )
        gl = run_episode(
            method="global",
            environment_seed=seed,
            pilot_seed=seed + 17,
            diffusion_seed=DIFFUSION_SEED_BASE + seed,
            gamma=0.0,
            controller=controller,
            postprocessor=post,
            trace_path=root / "identity_traces" / f"global_gamma_0_seed_{seed}.npz",
        )
        no_trace = load_trace(no["trace_path"])
        gl_trace = load_trace(gl["trace_path"])
        state_equal = np.array_equal(no_trace["state_43"], gl_trace["state_43"])
        raw_equal = np.array_equal(no_trace["raw_pilot_action_7"], gl_trace["raw_pilot_action_7"])
        executed_equal = np.array_equal(no_trace["executed_action_7"], gl_trace["executed_action_7"])
        termination_equal = (
            no["termination_reason"] == gl["termination_reason"]
            and no["success"] == gl["success"]
            and no["episode_steps"] == gl["episode_steps"]
        )
        row = {
            "seed": seed,
            "state_trajectory_exact": state_equal,
            "raw_action_exact": raw_equal,
            "executed_action_exact": executed_equal,
            "termination_exact": termination_equal,
            "noassist_termination": no["termination_reason"],
            "global_termination": gl["termination_reason"],
            "steps": no["episode_steps"],
        }
        rows.append(row)
        if not all((state_equal, raw_equal, executed_equal, termination_equal)):
            failures.append(str(row))
    write_csv(root / "gamma0_identity_audit.csv", rows)
    return {"status": "PASS" if not failures else "FAIL", "N": len(rows), "failures": failures[:10]}


def run_gamma_group(
    root: Path,
    group: str,
    seeds: tuple[int, ...],
    gamma: float,
    gamma_index: int,
    controller: GlobalSharedController,
    post: GlobalActionPostprocessor,
) -> list[dict[str, Any]]:
    trace_root = root / "traces" / group / f"gamma_{gamma_key(gamma)}"
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, 1):
        rows.append(
            run_episode(
                method="global",
                environment_seed=seed,
                pilot_seed=seed + 17,
                diffusion_seed=DIFFUSION_SEED_BASE + seed + gamma_index * 100_000,
                gamma=gamma,
                controller=controller,
                postprocessor=post,
                trace_path=trace_root / f"normal_gamma_{gamma_key(gamma)}_seed_{seed}.npz",
            )
        )
        if index % 10 == 0 or index == len(seeds):
            print(f"{group} gamma={gamma:.3f}: {index}/{len(seeds)} complete", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 RuleBasedRecoveryPilot NORMAL x gamma evaluator")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if _sha256(PILOT_SOURCE) != PILOT_SHA256:
        raise SystemExit("STOP frozen RuleBasedRecoveryPilot source hash mismatch")
    checkpoint = args.checkpoint.expanduser().resolve()
    stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = OUTPUT_ROOT / f"run_{stamp}"
    root.mkdir(parents=True)
    controller = GlobalSharedController(checkpoint, args.device)
    post = GlobalActionPostprocessor.from_expert_spec()
    metadata = {
        "experiment": "E2 RuleBasedRecoveryPilot NORMAL gamma validation",
        "purpose": "select gamma_RP from NORMAL validation only before formal failure comparison",
        "recovery_pilot_source": str(PILOT_SOURCE.resolve()),
        "recovery_pilot_sha256": _sha256(PILOT_SOURCE),
        "global_checkpoint": str(checkpoint),
        "global_checkpoint_sha256": _sha256(checkpoint),
        "gammas": list(GAMMAS),
        "identity_seeds": list(IDENTITY_SEEDS),
        "smoke_seeds": list(SMOKE_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "diffusion_seed_mapping": "7200000 + environment_seed + gamma_index * 100000",
        "global_input": "state43 only (policy_state_42 + object_grasped)",
        "no_milestone_leak": True,
        "no_active_stage_leak": True,
        "no_failure_label_input": True,
        "no_tcn_input": True,
        "failure_results_used_for_selection": False,
        "control_dt": CONTROL_DT,
        "max_steps": MAX_STEPS,
        "gripper_modes": post.report(),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    identity = identity_audit(root, controller, post)
    (root / "gamma0_identity_audit.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    if identity["status"] != "PASS":
        raise SystemExit("STOP gamma=0 identity audit failed")

    all_smoke: list[dict[str, Any]] = []
    smoke_audits: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(GAMMAS):
        rows = run_gamma_group(root, "smoke", SMOKE_SEEDS, gamma, gamma_index, controller, post)
        all_smoke.extend(rows)
        smoke_audits.append(validate_rows(rows, len(SMOKE_SEEDS), gamma, post))
    write_csv(root / "smoke_episode_summary.csv", all_smoke)
    smoke_summary = [summarize_gamma(gamma, [r for r in all_smoke if r["gamma"] == gamma]) for gamma in GAMMAS]
    write_csv(root / "smoke_gamma_summary.csv", smoke_summary)
    smoke = {
        "status": "PASS" if all(audit["status"] == "PASS" for audit in smoke_audits) else "FAIL",
        "audits": smoke_audits,
    }
    (root / "smoke_report.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    if smoke["status"] != "PASS":
        raise SystemExit("STOP smoke failed")

    all_rows: list[dict[str, Any]] = []
    validation_audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(GAMMAS):
        rows = run_gamma_group(root, "formal", VALIDATION_SEEDS, gamma, gamma_index, controller, post)
        all_rows.extend(rows)
        audit = validate_rows(rows, len(VALIDATION_SEEDS), gamma, post)
        validation_audits.append(audit)
        if audit["status"] != "PASS":
            (root / "validation_audit.json").write_text(json.dumps(validation_audits, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(f"STOP structural audit failure gamma={gamma}: {audit['failures'][:1]}")
        summaries.append(summarize_gamma(gamma, rows))
    write_csv(root / "normal_validation_episode_summary.csv", all_rows)
    write_csv(root / "normal_gamma_validation_summary.csv", summaries)
    (root / "normal_gamma_validation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (root / "validation_audit.json").write_text(json.dumps(validation_audits, indent=2) + "\n", encoding="utf-8")
    selection = select_gamma(summaries)
    (root / "gamma_rp_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    final_audit = {
        "status": "PASS" if selection["status"] == "PASS" and all(a["status"] == "PASS" for a in validation_audits) else "FAIL",
        "identity": identity["status"],
        "smoke": smoke["status"],
        "formal_rollouts": len(all_rows),
        "expected_formal_rollouts": len(GAMMAS) * len(VALIDATION_SEEDS),
        "selection_uses_normal_validation_only": True,
        "failure_results_used_for_selection": False,
        "selected_gamma_RP": selection["selected_gamma"],
    }
    (root / "audit.json").write_text(json.dumps(final_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(root), "audit": final_audit["status"], "selected_gamma_RP": selection["selected_gamma"]}, indent=2))


if __name__ == "__main__":
    main()

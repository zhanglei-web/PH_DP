#!/usr/bin/env python3
"""Leakage-free, AWAC-2.5k shared-control Experiment 2.

Pipeline (no training):
  1. freeze three mutually disjoint Normal/Place-Recovery case sets;
  2. independently select the TCN-V2 checkpoint on 25+25 cases;
  3. independently select one gamma per assistance model on 25+25 cases;
  4. print and persist the provenance gate;
  5. run the paired final 50+50 holdout.

RuleBasedRecoveryPilot is used only inside the existing snapshot-generation
backend called by ``prepare_case_manifest``.  Formal rollouts use the frozen
AWAC checkpoint as their sole user-action source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

import validate_recovery_stage_checkpoints as generation
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import (
    StageEmbeddingDiffusion,
    StageEmbeddingDiffusionConfig,
)
from mujoco_shared_control.stage.tcn import StageTCNV1
from train_recovery_causal_tcn_v1 import feature as tcn_feature


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/e2_formal_v2_awac2500_20260821"
OLD_INVALID = ROOT / "outputs/final_stage_ambiguity_experiments_20260820/experiment2_e2_shared_autonomy"
OLD_SELECTION_MANIFEST = ROOT / "outputs/recovery_stage_dp_validation_80k_120k/validation_case_manifest.json"

PILOT = ROOT / "outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_LOCAL20K_FORMAL/checkpoints/checkpoint_step_02500.pt"
PILOT_SHA = "a53db3ad811fa58ed159233c197e505982a3c5f6ce925bc2a27e77b7dd140d9e"
GLOBAL_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_global_120k_20260820"
ORACLE_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820"
TCN_DP_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_tcn_v2_120k_20260820"
CAUSAL_TCN_DIR = ROOT / "outputs/recovery_stage_dp_training/causal_tcn_recovery_v1_20260820"
GLOBAL_STEP, ORACLE_STEP = 110_000, 90_000
TCN_CANDIDATES = (80_000, 90_000, 100_000, 110_000, 120_000)
GAMMAS = tuple(round(i / 10, 1) for i in range(11))
METHODS = ("NoAssist", "Global", "Oracle-V2", "TCN-V2")
DT, MAX_STEPS, IK_LIMIT = generation.DT, generation.MAX, generation.IKMAX

SPLIT_SPECS = {
    "tcn_selection": {"n": 25, "seed_base": 10_500_000, "prefix": "TCNSEL"},
    "gamma_calibration": {"n": 25, "seed_base": 11_500_000, "prefix": "GAMMACAL"},
    "final": {"n": 50, "seed_base": 12_500_000, "prefix": "E2FINAL"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state43(env: PickPlaceEnv, observation: dict[str, Any]) -> np.ndarray:
    value = np.r_[env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"]))].astype("f4")
    if value.shape != (43,) or not np.isfinite(value).all():
        raise ValueError("physical state must be finite 43-D")
    return value


def snapshot_identity(case: dict[str, Any]) -> str:
    explicit = case.get("snapshot_sha256") or case.get("snapshot_file_sha256") or case.get("snapshot_hash")
    if explicit:
        return str(explicit)
    path = case.get("snapshot_path")
    return sha256(Path(path)) if path and Path(path).is_file() else ""


def identity(case: dict[str, Any]) -> dict[str, str]:
    seed = str(case["environment_seed"])
    return {
        "case_id": str(case["case_id"]),
        "seed": seed,
        "snapshot": snapshot_identity(case),
        "source_episode": str(case.get("source_episode") or case.get("source_episode_id") or f"env:{seed}"),
    }


def pair_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, list[str]]:
    a, b = [identity(case) for case in left], [identity(case) for case in right]
    return {
        key: sorted({row[key] for row in a if row[key]} & {row[key] for row in b if row[key]})
        for key in ("case_id", "seed", "snapshot", "source_episode")
    }


def prepare_case_manifest(name: str) -> dict[str, Any]:
    spec = SPLIT_SPECS[name]
    directory = OUT / "case_splits" / name
    manifest = directory / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())
    # The backend may instantiate RuleBasedRecoveryPilot only while creating
    # valid physical failure snapshots.  No object from it enters rollout().
    generated = generation.make_cases(directory / "generation_backend", int(spec["n"]), seed_base=int(spec["seed_base"]))
    cases: list[dict[str, Any]] = []
    for source in generated:
        if source["kind"] not in ("NORMAL", "PLACE_RECOVERY"):
            continue
        case = dict(source)
        case["case_id"] = f"{spec['prefix']}_{source['case_id']}"
        case["source_episode"] = f"env:{case['environment_seed']}"
        case["snapshot_sha256"] = snapshot_identity(case)
        cases.append(case)
    counts = Counter(case["kind"] for case in cases)
    expected = Counter({"NORMAL": int(spec["n"]), "PLACE_RECOVERY": int(spec["n"])})
    if counts != expected:
        raise RuntimeError(f"{name} composition mismatch: {counts} != {expected}")
    payload = {
        "version": "e2-formal-v2-split-1.0",
        "name": name,
        "frozen": True,
        "seed_base": spec["seed_base"],
        "counts": dict(counts),
        "case_generation_only_rule_based_pilot": True,
        "formal_user_surrogate": "AWAC step 2500",
        "cases": cases,
    }
    write_json(manifest, payload)
    return payload


def prepare_and_audit_splits() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    splits = {name: prepare_case_manifest(name) for name in SPLIT_SPECS}
    old_cases = json.loads(OLD_SELECTION_MANIFEST.read_text())["cases"]
    groups = {"global_oracle_selection": old_cases, **{name: value["cases"] for name, value in splits.items()}}
    required_pairs = (
        ("global_oracle_selection", "tcn_selection"),
        ("global_oracle_selection", "gamma_calibration"),
        ("global_oracle_selection", "final"),
        ("tcn_selection", "gamma_calibration"),
        ("tcn_selection", "final"),
        ("gamma_calibration", "final"),
    )
    comparisons = {}
    for left, right in required_pairs:
        overlaps = pair_overlap(groups[left], groups[right])
        comparisons[f"{left}_vs_{right}"] = {"overlaps": overlaps, "overlap_count": sum(len(v) for v in overlaps.values())}
    passed = all(item["overlap_count"] == 0 for item in comparisons.values())
    report = {
        "ALL_E2_DATA_SPLITS_DISJOINT": "YES" if passed else "NO",
        "identity_dimensions": ["case_id", "seed", "snapshot", "source_episode"],
        "comparisons": comparisons,
        "old_global_oracle_selection_manifest": str(OLD_SELECTION_MANIFEST.resolve()),
        "old_global_oracle_selection_manifest_sha256": sha256(OLD_SELECTION_MANIFEST),
    }
    write_json(OUT / "all_split_overlap_audit.json", report)
    if not passed:
        raise RuntimeError("E2 split overlap gate failed")
    return splits, report


@dataclass
class GeometryStageOracle:
    """Stage-only state machine; it emits no action and is not a user pilot."""

    initial_object_z: float = 0.0
    active_phase: int = 0
    close_attempt_completed: bool = False
    grasp_failure_frames: int = 0
    forced_grasp_failure: bool = False
    forced_reapproach_steps: int = 0

    def reset(self, initial_object_z: float) -> None:
        self.initial_object_z = float(initial_object_z)
        self.active_phase = 0
        self.close_attempt_completed = False
        self.grasp_failure_frames = 0
        self.forced_grasp_failure = False
        self.forced_reapproach_steps = 0

    def restore_generation_state(self, state: dict[str, Any]) -> None:
        for key in ("initial_object_z", "active_phase", "close_attempt_completed", "grasp_failure_frames", "forced_grasp_failure", "forced_reapproach_steps"):
            if key in state:
                setattr(self, key, int(state[key]) if key in ("active_phase", "grasp_failure_frames", "forced_reapproach_steps") else state[key])

    def stage(self, observation: dict[str, Any]) -> int:
        ee = np.asarray(observation["ee_pose"][:3, 3], float)
        obj = np.asarray(observation["object_pose"][:3, 3], float)
        goal = np.asarray(observation["goal_pose"][:3, 3], float)
        grasped = bool(observation["object_grasped"])
        opening = float(observation["gripper"][0])
        if not grasped and np.linalg.norm(obj - goal) < 0.055 and opening >= 0.055:
            self.active_phase = 4
            return 4
        if not grasped:
            grasp = obj + np.array([0.0, 0.0, 0.012])
            grasp_error = float(np.linalg.norm(ee - grasp))
            if self.forced_reapproach_steps > 0:
                self.forced_reapproach_steps -= 1
                self.active_phase = 0
                return 0
            if self.active_phase not in (0, 1) or self.forced_grasp_failure:
                self.forced_grasp_failure = False
                self.active_phase = 0
                self.close_attempt_completed = False
                self.grasp_failure_frames = 0
                return 0
            if self.active_phase == 1:
                self.close_attempt_completed |= opening <= 0.040
                failed = self.close_attempt_completed and (grasp_error >= 0.012 or opening <= 0.040)
                self.grasp_failure_frames = self.grasp_failure_frames + 1 if failed else 0
                if self.grasp_failure_frames < 3:
                    return 1
                self.active_phase = 0
                self.close_attempt_completed = False
                self.grasp_failure_frames = 0
            if opening < 0.055 or grasp_error > 0.008:
                return 0
            self.active_phase = 1
            return 1
        if np.linalg.norm(obj[:2] - goal[:2]) < 0.080:
            self.active_phase = 3
        elif obj[2] - self.initial_object_z < 0.10:
            self.active_phase = 1
        else:
            self.active_phase = 2
        return self.active_phase


def restore_snapshot(env: PickPlaceEnv, adapter: ExpertCommandAdapter, reward: AWACRewardV1Online,
                     stage_oracle: GeometryStageOracle, snapshot: dict[str, Any]) -> tuple[dict[str, Any], int]:
    mujoco.mj_setState(env.model, env.data, snapshot["integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    env.data.ctrl[:] = snapshot["ctrl"]
    env.data.mocap_pos[:] = snapshot["mocap_pos"]
    env.data.mocap_quat[:] = snapshot["mocap_quat"]
    env.data.userdata[:] = snapshot["userdata"]
    mujoco.mj_forward(env.model, env.data)
    env._episode_steps = snapshot["episode_steps"]
    env._previous_observation = snapshot["previous_observation"]
    env.sac_task = snapshot["sac_task"]
    adapter._target = snapshot["adapter_target"].copy()
    adapter._joint_target = snapshot["adapter_joint_target"].copy()
    reward.load_state_dict(snapshot["reward"])
    stage_oracle.restore_generation_state(snapshot.get("pilot", {}))
    return snapshot["obs"], int(snapshot["consecutive_ik"])


class CausalTCN:
    def __init__(self, device: torch.device) -> None:
        checkpoint = CAUSAL_TCN_DIR / "checkpoints/best_validation_macro_f1.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = StageTCNV1().to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.model.requires_grad_(False)
        self.device = device
        with np.load(CAUSAL_TCN_DIR / "normalization_stats.npz", allow_pickle=False) as values:
            self.mean, self.std = values["mean"].astype("f4"), values["std"].astype("f4")

    def initial(self, state: np.ndarray) -> deque[np.ndarray]:
        first = ((tcn_feature(state, np.zeros(7, "f4")) - self.mean) / self.std).astype("f4")
        return deque([first.copy() for _ in range(20)], maxlen=20)

    @torch.inference_mode()
    def predict(self, history: deque[np.ndarray]) -> tuple[int, np.ndarray]:
        posterior = self.model.posterior(torch.as_tensor(np.asarray(history, "f4"), device=self.device)[None])[0].cpu().numpy()
        return int(posterior.argmax()), posterior.astype("f4")


def normalization_path(method: str, directory: Path) -> Path:
    direct = directory / "normalization_stats.npz"
    if direct.is_file():
        return direct
    reference = directory / "normalization_reference.json"
    if method == "TCN-V2" and reference.is_file():
        path = Path(json.loads(reference.read_text())["ORACLE_V2_NORMALIZATION"])
        if path.is_file():
            return path
    raise FileNotFoundError(direct)


class AssistanceController:
    def __init__(self, method: str, checkpoint: Path, device: torch.device) -> None:
        self.method, self.checkpoint, self.device = method, checkpoint, device
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.step = int(payload["step"])
        config = payload["diffusion_config"]
        if method == "Global":
            cfg = DiffusionConfig(**config)
            if cfg.observation_dim != 43:
                raise ValueError("Global must consume physical43 only")
            self.model = RSS2023Diffusion(cfg)
            directory = GLOBAL_DIR
        else:
            cfg = StageEmbeddingDiffusionConfig(**{key: config[key] for key in StageEmbeddingDiffusionConfig.__dataclass_fields__ if key in config})
            self.model = StageEmbeddingDiffusion(cfg)
            directory = ORACLE_DIR if method == "Oracle-V2" else TCN_DP_DIR
        self.model.load_state_dict(payload["model"])
        self.model.to(device).eval().requires_grad_(False)
        with np.load(normalization_path(method, directory), allow_pickle=False) as stats:
            self.pm, self.ps = stats["physical_mean"].astype("f4"), stats["physical_std"].astype("f4")
            self.am, self.astd = stats["action_mean"].astype("f4"), stats["action_std"].astype("f4")
        self.generator: torch.Generator | None = None

    def reset(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def assist(self, physical: np.ndarray, pilot_action: np.ndarray, gamma: float, stage: int | None) -> np.ndarray:
        if gamma == 0.0:
            return np.asarray(pilot_action, "f4").copy()
        if self.generator is None:
            raise RuntimeError("controller sampling generator was not reset")
        p = (np.asarray(physical, "f4") - self.pm) / self.ps
        human = (np.asarray(pilot_action, "f4") - self.am) / self.astd
        if self.method == "Global":
            observation = p
        else:
            if stage not in range(5):
                raise ValueError(f"{self.method} needs one valid stage")
            observation = np.r_[p, np.eye(5, dtype="f4")[int(stage)]]
        output = self.model.assist(
            torch.as_tensor(observation, device=self.device)[None],
            torch.as_tensor(human, device=self.device)[None], float(gamma), generator=self.generator,
        )[0].cpu().numpy()
        result = output * self.astd + self.am
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite assisted action")
        return result.astype("f4")


class RolloutLimitReached(RuntimeError):
    pass


class ResumableRunner:
    def __init__(self, output: Path, maximum_new: int | None) -> None:
        self.output, self.maximum_new, self.new = output, maximum_new, 0
        self.cached: dict[str, dict[str, Any]] = {}
        events = output / "rollout_events.jsonl"
        if not events.is_file():
            return
        for number, line in enumerate(events.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                trace = Path(row["trace_path"])
                if trace.is_file() and isinstance(json.loads(trace.read_text()), list):
                    self.cached[str(trace.resolve())] = row
            except (json.JSONDecodeError, KeyError, OSError, TypeError):
                print({"ignored_incomplete_event_line": number}, flush=True)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        trace = Path(kwargs["trace_path"])
        key = str(trace.resolve())
        if key in self.cached:
            row = self.cached[key]
            expected = (kwargs["case"]["case_id"], kwargs["method"], float(kwargs["gamma"]), kwargs.get("checkpoint_step"))
            actual = (row["case_id"], row["method"], float(row["gamma"]), row.get("checkpoint_step"))
            if actual != expected:
                raise RuntimeError(f"cached rollout provenance mismatch: {actual} != {expected}")
            return row
        if self.maximum_new is not None and self.new >= self.maximum_new:
            raise RolloutLimitReached
        row = rollout(**kwargs)
        self.cached[key] = row
        self.new += 1
        if self.new % 10 == 0:
            torch.cuda.empty_cache()
        print({"new_rollout": self.new, "phase": kwargs["phase"], "method": kwargs["method"],
               "checkpoint": kwargs.get("checkpoint_step"), "gamma": kwargs["gamma"], "case": kwargs["case"]["case_id"]}, flush=True)
        return row


def rollout(*, case: dict[str, Any], phase: str, method: str, gamma: float,
            pilot: HybridCheckpointPredictor, controller: AssistanceController | None,
            causal_tcn: CausalTCN, checkpoint_step: int | None, trace_path: Path) -> dict[str, Any]:
    env = PickPlaceEnv(render_mode=None, control_timestep=DT, max_episode_steps=MAX_STEPS, enable_camera=False)
    spec = ExpertActionSpec()
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    postprocess = GlobalActionPostprocessor.from_expert_spec(spec)
    stage_oracle = GeometryStageOracle()
    try:
        initial, _ = env.reset(seed=int(case["environment_seed"]), options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
        adapter.reset(initial["ee_pose"], initial["q_obs"])
        reward = AWACRewardV1Online(state43(env, initial))
        stage_oracle.reset(float(initial["object_pose"][2, 3]))
        observation, consecutive = initial, 0
        if case["kind"] == "PLACE_RECOVERY":
            snapshot = pickle.loads(Path(case["snapshot_path"]).read_bytes())
            observation, consecutive = restore_snapshot(env, adapter, reward, stage_oracle, snapshot)
        if controller is not None:
            controller.reset(int(case["sampling_seed"]) + int(round(gamma * 1000)))
        history = causal_tcn.initial(state43(env, observation))
        reason, rows = "timeout", []
        milestones = reward.tracker.current.copy()
        for step in range(MAX_STEPS):
            physical = state43(env, observation)
            gt_stage = int(stage_oracle.stage(observation))
            predicted_stage, posterior = causal_tcn.predict(history)
            # Sole formal user-action source.  No RuleBasedRecoveryPilot action
            # is constructed or available in this function.
            pilot_action = pilot.normalized_action(
                physical[:42], bool(physical[42]), current_active_stage=gt_stage,
            ).astype("f4")
            assist_stage = gt_stage if method == "Oracle-V2" else predicted_stage if method == "TCN-V2" else None
            started = time.perf_counter()
            assisted = pilot_action.copy() if method == "NoAssist" else controller.assist(physical, pilot_action, gamma, assist_stage)  # type: ignore[union-attr]
            inference_ms = 0.0 if method == "NoAssist" else (time.perf_counter() - started) * 1000
            bounded = np.clip(assisted, -1.0, 1.0)
            canonical = postprocess(bounded)
            adapted = adapter.adapt(spec.denormalize(canonical))
            next_observation, *_ = env.step(adapted.joint_target)
            next_physical = state43(env, next_observation)
            consecutive = 0 if adapted.accepted else consecutive + 1
            outcome = reward.step(physical, next_physical, ik_failure=consecutive >= IK_LIMIT, time_limit=step + 1 >= MAX_STEPS)
            milestones = reward.tracker.current.copy()
            executed = np.asarray(adapted.normalized, "f4")
            rows.append({
                "step": step, "gt_stage_audit_only": gt_stage, "predicted_stage": predicted_stage,
                "posterior": posterior.tolist(), "awac2500_pilot_action": pilot_action.tolist(),
                "assisted_action": assisted.tolist(), "executed_action": executed.tolist(),
                "adapter_accepted": bool(adapted.accepted), "fallback_used": bool(adapted.fallback_used),
                "action_clipped": bool(adapted.action_clipped or not np.array_equal(assisted, bounded)),
                "inference_ms": inference_ms, "termination": outcome.termination_reason,
            })
            history.append(((tcn_feature(next_physical, executed) - causal_tcn.mean) / causal_tcn.std).astype("f4"))
            observation = next_observation
            if outcome.terminated or outcome.truncated:
                reason = outcome.termination_reason
                break
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows) + "\n")
        temporary.replace(trace_path)
        result = {
            "phase": phase, "case_id": case["case_id"], "kind": case["kind"], "method": method,
            "gamma": gamma, "checkpoint_step": checkpoint_step, "success": reason == "task_success",
            "illegal_drop": reason == "illegal_drop", "ik_failure": reason == "ik_failure_limit",
            "timeout": reason == "timeout", "termination": reason, "steps": step + 1,
            "grasp": bool(milestones[0]), "lift": bool(milestones[1]), "transport": bool(milestones[2]),
            "place": bool(milestones[3]), "retreat": bool(milestones[4]),
            "tcn_gt_agreement_audit_only": float(np.mean([row["gt_stage_audit_only"] == row["predicted_stage"] for row in rows])),
            "pilot_checkpoint": str(PILOT.resolve()), "pilot_sha256": PILOT_SHA,
            "rule_based_pilot_used_as_user_surrogate": False, "trace_path": str(trace_path.resolve()),
        }
        with (OUT / "rollout_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
        return result
    finally:
        env.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero rows")
    return {
        "N": len(rows),
        **{key: float(np.mean([bool(row[key]) for row in rows])) for key in ("success", "illegal_drop", "ik_failure", "timeout", "grasp", "lift", "transport", "place", "retreat")},
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
    }


def select_tcn_checkpoint(runner: ResumableRunner, cases: list[dict[str, Any]], pilot: HybridCheckpointPredictor,
                          causal_tcn: CausalTCN, device: torch.device) -> tuple[dict[str, Any], Path]:
    output = OUT / "tcn_checkpoint_selection"
    report_path = OUT / "tcn_v2_checkpoint_selection.json"
    candidate_reports = []
    all_rows: list[dict[str, Any]] = []
    for checkpoint_step in TCN_CANDIDATES:
        checkpoint = TCN_DP_DIR / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
        controller = AssistanceController("TCN-V2", checkpoint, device)
        rows = [runner.run(case=case, phase="tcn_checkpoint_selection", method="TCN-V2", gamma=1.0,
                           pilot=pilot, controller=controller, causal_tcn=causal_tcn,
                           checkpoint_step=checkpoint_step,
                           trace_path=output / f"step_{checkpoint_step:06d}" / f"{case['case_id']}.json") for case in cases]
        all_rows.extend(rows)
        place = summarize([row for row in rows if row["kind"] == "PLACE_RECOVERY"])
        overall = summarize(rows)
        candidate_reports.append({"step": checkpoint_step, "checkpoint": str(checkpoint.resolve()),
                                  "checkpoint_sha256": sha256(checkpoint), "fixed_selection_gamma": 1.0,
                                  "place_recovery": place, "overall": overall})
    chosen = max(candidate_reports, key=lambda item: (
        item["place_recovery"]["success"], item["overall"]["success"],
        -item["overall"]["timeout"], -item["step"],
    ))
    payload = {
        "status": "FROZEN", "pilot_checkpoint": str(PILOT.resolve()), "pilot_sha256": PILOT_SHA,
        "stage_source": "causal TCN predicted hard one-hot; GT audit only",
        "selection_case_manifest": str((OUT / "case_splits/tcn_selection/manifest.json").resolve()),
        "selection_case_manifest_sha256": sha256(OUT / "case_splits/tcn_selection/manifest.json"),
        "selection_rule": ["Place Recovery success", "Overall success", "lower overall timeout", "earlier checkpoint"],
        "fixed_selection_gamma": 1.0,
        "fixed_selection_gamma_rationale": "pre-outcome full-assistance checkpoint comparison before separate gamma calibration",
        "candidates": candidate_reports, "TCN_V2_SELECTED_CHECKPOINT": chosen,
    }
    write_json(report_path, payload)
    write_csv(output / "rollouts.csv", all_rows)
    return payload, Path(chosen["checkpoint"])


def select_gamma(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    candidates = [summarize([row for row in rows if row["gamma"] == gamma]) | {"gamma": gamma} for gamma in GAMMAS]
    eligible = [row for row in candidates if row["illegal_drop"] <= 0.05 and row["ik_failure"] <= 0.05 and row["timeout"] <= 0.05]
    chosen = max(eligible if eligible else candidates, key=lambda row: (row["success"], row["gamma"]))
    return {"method": method, "selection_rule": "safety <=5% each; maximize success; largest gamma tie-break",
            "candidates": candidates, "selected_gamma": chosen["gamma"], "selected": chosen}


def calibrate_gammas(runner: ResumableRunner, cases: list[dict[str, Any]], pilot: HybridCheckpointPredictor,
                     causal_tcn: CausalTCN, controllers: dict[str, AssistanceController]) -> dict[str, Any]:
    selections = {}
    for method in METHODS[1:]:
        rows = [runner.run(case=case, phase="gamma_calibration", method=method, gamma=gamma,
                           pilot=pilot, controller=controllers[method], causal_tcn=causal_tcn,
                           checkpoint_step=controllers[method].step,
                           trace_path=OUT / "gamma_calibration" / method / f"gamma_{gamma:.1f}" / f"{case['case_id']}.json")
                for gamma in GAMMAS for case in cases]
        write_csv(OUT / "gamma_calibration" / method / "rollouts.csv", rows)
        selections[method] = select_gamma(rows, method)
    payload = {
        "status": "FROZEN", "pilot_checkpoint": str(PILOT.resolve()), "pilot_sha256": PILOT_SHA,
        "calibration_case_manifest": str((OUT / "case_splits/gamma_calibration/manifest.json").resolve()),
        "calibration_case_manifest_sha256": sha256(OUT / "case_splits/gamma_calibration/manifest.json"),
        "selections": selections,
    }
    write_json(OUT / "gamma_selection_frozen.json", payload)
    return payload


def paired_statistics(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {method: {row["case_id"]: row for row in final_rows if row["method"] == method} for method in METHODS}
    baseline, report = methods["NoAssist"], {}
    for method in METHODS[1:]:
        report[method] = {}
        for group in ("ALL", "NORMAL", "PLACE_RECOVERY"):
            ids = sorted(case_id for case_id in baseline.keys() & methods[method].keys()
                         if group == "ALL" or baseline[case_id]["kind"] == group)
            metrics = {}
            for number, metric in enumerate(("success", "illegal_drop", "ik_failure", "timeout")):
                a = np.asarray([baseline[i][metric] for i in ids], bool)
                b = np.asarray([methods[method][i][metric] for i in ids], bool)
                difference = b.astype(float) - a.astype(float)
                rng = np.random.default_rng(20260821 + number)
                bootstrap = np.asarray([difference[rng.integers(len(difference), size=len(difference))].mean() for _ in range(10_000)])
                plus, minus = int((~a & b).sum()), int((a & ~b).sum())
                discordant = plus + minus
                p_value = 1.0 if not discordant else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(plus, minus) + 1)) / 2**discordant)
                metrics[metric] = {"difference_method_minus_NoAssist": float(difference.mean()),
                                   "paired_bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                                   "mcnemar_exact": {"method_only": plus, "noassist_only": minus, "p_value": p_value}}
            report[method][group] = {"N": len(ids), "metrics": metrics}
    return report


def static_preflight() -> dict[str, Any]:
    checkpoints = {
        "Pilot": (PILOT, 2500, PILOT_SHA),
        "Global": (GLOBAL_DIR / "checkpoints/step_110000.pt", GLOBAL_STEP, None),
        "Oracle-V2": (ORACLE_DIR / "checkpoints/step_090000.pt", ORACLE_STEP, None),
        "Causal-TCN": (CAUSAL_TCN_DIR / "checkpoints/best_validation_macro_f1.pt", None, None),
    }
    report = {"status": "PASS", "artifacts": {}}
    for name, (path, expected_step, expected_sha) in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        if expected_sha and digest != expected_sha:
            raise RuntimeError(f"{name} SHA mismatch")
        step = None
        if expected_step is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            step = int(payload["step"])
            if step != expected_step:
                raise RuntimeError(f"{name} step mismatch: {step} != {expected_step}")
        report["artifacts"][name] = {"path": str(path.resolve()), "step": step, "sha256": digest}
    for step in TCN_CANDIDATES:
        path = TCN_DP_DIR / "checkpoints" / f"step_{step:06d}.pt"
        if not path.is_file() or int(torch.load(path, map_location="cpu", weights_only=False)["step"]) != step:
            raise RuntimeError(f"invalid TCN candidate {path}")
    marker = OLD_INVALID / "INVALID_PROTOCOL_DO_NOT_REPORT.json"
    if json.loads(marker.read_text()).get("INVALID_PROTOCOL_DO_NOT_REPORT") != "YES":
        raise RuntimeError("old invalid E2 is not marked")
    report["old_e2"] = {"path": str(OLD_INVALID.resolve()), "INVALID_PROTOCOL_DO_NOT_REPORT": "YES"}
    report["no_model_training"] = True
    return report


def provenance_gate(split_audit: dict[str, Any], tcn_selection: dict[str, Any], gamma_selection: dict[str, Any],
                    split_manifests: dict[str, dict[str, Any]], tcn_checkpoint: Path) -> dict[str, Any]:
    pilot_payload = torch.load(PILOT, map_location="cpu", weights_only=False)
    gate = {
        "Pilot": "AWAC Step2500", "Pilot_checkpoint": str(PILOT.resolve()), "Pilot_SHA256": sha256(PILOT),
        "Pilot_gradient_step": int(pilot_payload["step"]),
        "Global": "110k", "Global_checkpoint": str((GLOBAL_DIR / "checkpoints/step_110000.pt").resolve()),
        "Global_SHA256": sha256(GLOBAL_DIR / "checkpoints/step_110000.pt"),
        "Oracle": "90k", "Oracle_checkpoint": str((ORACLE_DIR / "checkpoints/step_090000.pt").resolve()),
        "Oracle_SHA256": sha256(ORACLE_DIR / "checkpoints/step_090000.pt"),
        "TCN-V2": f"selected checkpoint {tcn_selection['TCN_V2_SELECTED_CHECKPOINT']['step']}",
        "TCN_V2_checkpoint": str(tcn_checkpoint.resolve()), "TCN_V2_SHA256": sha256(tcn_checkpoint),
        "Causal_TCN_checkpoint": str((CAUSAL_TCN_DIR / "checkpoints/best_validation_macro_f1.pt").resolve()),
        "Causal_TCN_SHA256": sha256(CAUSAL_TCN_DIR / "checkpoints/best_validation_macro_f1.pt"),
        "Checkpoint_selection_manifest": str((OUT / "tcn_v2_checkpoint_selection.json").resolve()),
        "Checkpoint_selection_manifest_SHA256": sha256(OUT / "tcn_v2_checkpoint_selection.json"),
        "Gamma_calibration_manifest": str((OUT / "case_splits/gamma_calibration/manifest.json").resolve()),
        "Gamma_calibration_manifest_SHA256": sha256(OUT / "case_splits/gamma_calibration/manifest.json"),
        "Gamma_selection_result_SHA256": sha256(OUT / "gamma_selection_frozen.json"),
        "Final_E2_manifest": str((OUT / "case_splits/final/manifest.json").resolve()),
        "Final_E2_manifest_SHA256": sha256(OUT / "case_splits/final/manifest.json"),
        "ALL_E2_DATA_SPLITS_DISJOINT": split_audit["ALL_E2_DATA_SPLITS_DISJOINT"],
        "RULE_BASED_PILOT_USED_IN_FORMAL_E2": "NO",
        "RULE_BASED_PILOT_USE": "case/snapshot generation only",
        "all_methods_user_action_source": "same frozen AWAC2.5k policy checkpoint",
        "gamma": {method: value["selected_gamma"] for method, value in gamma_selection["selections"].items()},
        "final_counts": split_manifests["final"]["counts"],
        "FINAL_VS_GLOBAL_ORACLE_SELECTION_OVERLAP": split_audit["comparisons"]["global_oracle_selection_vs_final"]["overlap_count"],
        "FINAL_VS_TCN_SELECTION_OVERLAP": split_audit["comparisons"]["tcn_selection_vs_final"]["overlap_count"],
        "FINAL_VS_GAMMA_CALIBRATION_OVERLAP": split_audit["comparisons"]["gamma_calibration_vs_final"]["overlap_count"],
    }
    required = (
        gate["Pilot_SHA256"] == PILOT_SHA,
        gate["Pilot_gradient_step"] == 2500,
        gate["ALL_E2_DATA_SPLITS_DISJOINT"] == "YES",
        gate["RULE_BASED_PILOT_USED_IN_FORMAL_E2"] == "NO",
        gate["FINAL_VS_GLOBAL_ORACLE_SELECTION_OVERLAP"] == 0,
        gate["FINAL_VS_TCN_SELECTION_OVERLAP"] == 0,
        gate["FINAL_VS_GAMMA_CALIBRATION_OVERLAP"] == 0,
    )
    gate["FORMAL_E2_ROLLOUT_AUTHORIZED"] = "YES" if all(required) else "NO"
    write_json(OUT / "formal_provenance_gate.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    if gate["FORMAL_E2_ROLLOUT_AUTHORIZED"] != "YES":
        raise RuntimeError("formal provenance gate refused rollout")
    return gate


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Leakage-free E2 formal protocol v2; no model training")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check-only", action="store_true", help="verify frozen artifacts only; no case generation or rollout")
    parser.add_argument("--prepare-only", action="store_true", help="generate/freeze/audit case manifests; no model rollout")
    parser.add_argument("--max-new-rollouts", type=int, default=None, help="optional clean pause for resumability")
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.max_new_rollouts is not None and args.max_new_rollouts <= 0:
        raise ValueError("--max-new-rollouts must be positive")
    report = static_preflight()
    if args.check_only:
        print(json.dumps(report | {"STATIC_PREFLIGHT": "PASS", "ROLLOUT_RUN": "NO"}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    splits, split_audit = prepare_and_audit_splits()
    if args.prepare_only:
        print(json.dumps({"CASE_SPLITS_FROZEN": "YES", **split_audit}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_MODEL_ROLLOUT; CPU fallback forbidden")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    runner = ResumableRunner(OUT, args.max_new_rollouts)
    try:
        pilot = HybridCheckpointPredictor(PILOT)
        causal_tcn = CausalTCN(device)
        tcn_selection, tcn_checkpoint = select_tcn_checkpoint(
            runner, splits["tcn_selection"]["cases"], pilot, causal_tcn, device,
        )
        controllers = {
            "Global": AssistanceController("Global", GLOBAL_DIR / "checkpoints/step_110000.pt", device),
            "Oracle-V2": AssistanceController("Oracle-V2", ORACLE_DIR / "checkpoints/step_090000.pt", device),
            "TCN-V2": AssistanceController("TCN-V2", tcn_checkpoint, device),
        }
        gamma_selection = calibrate_gammas(
            runner, splits["gamma_calibration"]["cases"], pilot, causal_tcn, controllers,
        )
        gate = provenance_gate(split_audit, tcn_selection, gamma_selection, splits, tcn_checkpoint)
        final_rows = []
        for case in splits["final"]["cases"]:
            final_rows.append(runner.run(
                case=case, phase="final", method="NoAssist", gamma=0.0, pilot=pilot,
                controller=None, causal_tcn=causal_tcn, checkpoint_step=None,
                trace_path=OUT / "final" / "NoAssist" / f"{case['case_id']}.json",
            ))
            for method in METHODS[1:]:
                final_rows.append(runner.run(
                    case=case, phase="final", method=method,
                    gamma=float(gamma_selection["selections"][method]["selected_gamma"]),
                    pilot=pilot, controller=controllers[method], causal_tcn=causal_tcn,
                    checkpoint_step=controllers[method].step,
                    trace_path=OUT / "final" / method / f"{case['case_id']}.json",
                ))
        write_csv(OUT / "final_rollouts.csv", final_rows)
        statistics = paired_statistics(final_rows)
        write_json(OUT / "paired_statistics.json", statistics)
        completion = {
            "E2_FORMAL_V2_COMPLETE": "YES", "PROTOCOL_VALID": "YES",
            "provenance_gate_sha256": sha256(OUT / "formal_provenance_gate.json"),
            "final_summary": {method: summarize([row for row in final_rows if row["method"] == method]) for method in METHODS},
            "paired_statistics": statistics, "provenance": gate,
        }
        write_json(OUT / "completion.json", completion)
        print(json.dumps({"E2_FORMAL_V2_COMPLETE": "YES", "output": str(OUT)}, indent=2))
    except RolloutLimitReached:
        write_json(OUT / "resume_status.json", {
            "status": "PAUSED_CLEANLY", "new_rollouts_this_invocation": runner.new,
            "cached_completed_traces": len(runner.cached),
            "resume_command": "./.venv/bin/python scripts/evaluate_e2_formal_v2.py",
        })
        print(json.dumps({"E2_FORMAL_V2_COMPLETE": "NO", "status": "PAUSED_CLEANLY", "new_rollouts": runner.new}, indent=2))


if __name__ == "__main__":
    main()

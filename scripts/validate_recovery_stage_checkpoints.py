#!/usr/bin/env python3
"""Fixed-case 80k-120k validation for Recovery Stage V1/V2 checkpoints.

Cases are generated once from an independent seed bank and then replayed from
serialized post-failure simulator snapshots for every checkpoint.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import torch

import build_e2_valid_failure_snapshot_bank as bank
from run_e2_awac25k_global_formal import AWAC, DT, IKMAX, MAX
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.global_evaluation import GRIPPER_OPEN_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v1_120k_20260820"
V2_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820"
STAGE_NAMES = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")
KINDS = ("NORMAL", "GRASP_RECOVERY", "TRANSPORT_RECOVERY", "PLACE_RECOVERY")
FAILURE_TO_BANK = {"GRASP_RECOVERY": "GRASP_FAILURE", "TRANSPORT_RECOVERY": "TRANSPORT_EARLY", "PLACE_RECOVERY": "PLACE_FAILURE"}
STEPS = tuple(range(10_000, 120_001, 10_000))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state43(env, obs):
    return np.r_[env.get_policy_observation(obs), np.float32(bool(obs["object_grasped"]))].astype(np.float32)


def make_cases(root: Path, n: int, seed_base: int = 7_500_000) -> list[dict[str, Any]]:
    manifest_path = root / "validation_case_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        return payload["cases"]
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    # Separate, deterministic ranges from the E2 seed banks.
    for kind_index, kind in enumerate(KINDS):
        if kind == "NORMAL":
            for i in range(n):
                seed = seed_base + i
                env = PickPlaceEnv(render_mode=None, control_timestep=DT, max_episode_steps=MAX, enable_camera=False)
                try:
                    obs, _ = env.reset(seed=seed, options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
                    initial = state43(env, obs)
                    cases.append({"case_id": f"NORMAL_{i:03d}", "kind": kind, "environment_seed": seed, "sampling_seed": seed_base + 10_000_000 + i, "failure_rng_seed": None, "failure_step": None, "regression_step": None, "snapshot_hash": None, "simulator_state_hash": None, "initial_physical_state": initial.tolist(), "target_state": obs["goal_pose"][:3, 3].tolist()})
                finally:
                    env.close()
            continue
        accepted = 0
        candidate = 0
        while accepted < n:
            seed = seed_base + (kind_index * 100_000) + candidate
            candidate += 1
            spec = {"condition": FAILURE_TO_BANK[kind], "environment_seed": seed, "pilot_seed": seed + 17, "failure_rng_seed": seed + 101, "candidate_index": candidate, "transport_bucket": "EARLY" if kind == "TRANSPORT_RECOVERY" else None, "transport_progress_threshold": 0.25 if kind == "TRANSPORT_RECOVERY" else None}
            env = PickPlaceEnv(render_mode=None, control_timestep=DT, max_episode_steps=MAX, enable_camera=False)
            pilot = RuleBasedRecoveryPilot(); adapter = ExpertCommandAdapter(env.ik_controller, pilot.action_spec)
            try:
                result, reason = bank.make_snapshot(spec, (env, pilot, adapter))
            finally:
                env.close()
            if result is None:
                continue
            meta, snapshot = result
            snapshot_path = root / "snapshots" / f"{kind}_{accepted:03d}.pkl"
            snapshot_path.write_bytes(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
            meta = dict(meta); meta.update({"case_id": f"{kind}_{accepted:03d}", "kind": kind, "sampling_seed": seed_base + 10_000_000 + kind_index * 100_000 + accepted, "snapshot_path": str(snapshot_path.resolve()), "snapshot_sha256": sha(snapshot_path), "simulator_state_hash": meta["full_simulator_state_hash"], "initial_physical_state": meta["state43"], "target_state": meta["goal_pose"]})
            cases.append(meta); accepted += 1
            if accepted % 10 == 0:
                print({"case_generation": kind, "accepted": accepted, "attempts": candidate}, flush=True)
    payload = {"version": "recovery-stage-validation-v2", "independent_from_e2_formal": True, "N_per_kind": n, "seed_base": seed_base, "cases": cases}
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return cases


class Predictor:
    def __init__(self, kind: str, checkpoint: Path, normalization: Path, device: torch.device):
        self.kind = kind; self.checkpoint = checkpoint; self.device = device
        self.payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = self.payload["diffusion_config"]
        if kind == "V1":
            self.model = RSS2023Diffusion(DiffusionConfig(**config))
        else:
            self.model = StageEmbeddingDiffusion(StageEmbeddingDiffusionConfig(**{k: config[k] for k in StageEmbeddingDiffusionConfig.__dataclass_fields__ if k in config}))
        self.model.load_state_dict(self.payload["model"]); self.model.to(device).eval()
        with np.load(normalization, allow_pickle=False) as stats:
            self.pm = np.asarray(stats["physical_mean"], np.float32); self.ps = np.asarray(stats["physical_std"], np.float32); self.am = np.asarray(stats["action_mean"], np.float32); self.ass = np.asarray(stats["action_std"], np.float32)
        if self.pm.shape != (43,) or self.am.shape != (7,): raise ValueError("normalization shape mismatch")
        self.generator = None
        self.spec = ExpertActionSpec()

    def reset(self, seed: int):
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self, physical: np.ndarray, stage: int) -> np.ndarray:
        obs = np.r_[(physical - self.pm) / self.ps, np.eye(5, dtype=np.float32)[stage]].astype(np.float32)
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        human = torch.zeros((1, 7), device=self.device)
        if self.kind == "V1":
            out = self.model.assist(o, human, gamma=1.0, generator=self.generator)
        else:
            out = self.model.assist(o, human, gamma=1.0, generator=self.generator)
        return (out.squeeze(0).cpu().numpy() * self.ass + self.am).astype(np.float32)


def evaluate_case(case: dict[str, Any], predictor: Predictor, trace_dir: Path) -> dict[str, Any]:
    env = PickPlaceEnv(render_mode=None, control_timestep=DT, max_episode_steps=MAX, enable_camera=False)
    adapter = ExpertCommandAdapter(env.ik_controller, predictor.spec); tracker = RuleBasedRecoveryPilot()
    try:
        if case["kind"] == "NORMAL":
            obs, _ = env.reset(seed=case["environment_seed"], options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
            adapter.reset(obs["ee_pose"], obs["q_obs"]); tracker.reset(float(obs["object_pose"][2, 3]), case["environment_seed"] + 17); reward = AWACRewardV1Online(state43(env, obs)); consecutive = 0
        else:
            initial, _ = env.reset(seed=case["environment_seed"], options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
            adapter.reset(initial["ee_pose"], initial["q_obs"]); tracker.reset(float(initial["object_pose"][2, 3]), case["environment_seed"] + 17); reward = AWACRewardV1Online(state43(env, initial)); obs, consecutive = bank.restore(env, adapter, tracker, reward, pickle.loads(Path(case["snapshot_path"]).read_bytes()))
        predictor.reset(case["sampling_seed"]); reason = "timeout"; rows = []; seen = set(); pre_stage = None; regression = None; milestones = np.zeros(5, bool)
        for step in range(MAX):
            physical = state43(env, obs); command, stage = tracker.predict(_expert_observation(case["case_id"], 0, step, obs, physical[:42], None, None)); stage = int(stage); seen.add(stage)
            if pre_stage is None and case["kind"] != "NORMAL": pre_stage = stage
            raw = predictor.sample(physical, stage); bounded = np.clip(raw, -1.0, 1.0); bounded[6] = -1.0 if bounded[6] < GRIPPER_OPEN_THRESHOLD else 1.0
            adapted = adapter.adapt(predictor.spec.denormalize(bounded)); next_obs, *_ = env.step(adapted.joint_target); next_state = state43(env, next_obs); consecutive = 0 if adapted.accepted else consecutive + 1
            result = reward.step(physical, next_state, ik_failure=consecutive >= IKMAX, time_limit=step + 1 >= MAX); milestones = reward.tracker.current.copy();
            if case["kind"] != "NORMAL" and regression is None and pre_stage is not None and stage == 0: regression = {"from": pre_stage, "to": 0, "step": step}
            rows.append({"step": step, "stage": stage, "stage_name": STAGE_NAMES[stage], "object_grasped": bool(obs["object_grasped"]), "raw_action": raw.tolist(), "executed_action": bounded.tolist(), "termination": result.termination_reason})
            obs = next_obs
            if result.task_success: reason = "task_success"; break
            if result.terminated or result.truncated: reason = result.termination_reason; break
        trace_dir.mkdir(parents=True, exist_ok=True); (trace_dir / f"{case['case_id']}.json").write_text(json.dumps({"case": case["case_id"], "rows": rows}, indent=2) + "\n")
        return {"case_id": case["case_id"], "kind": case["kind"], "success": reason == "task_success", "grasp": bool(milestones[0]), "lift": bool(milestones[1]), "transport": bool(milestones[2]), "place_release": bool(milestones[3]), "retreat": bool(milestones[4]), "illegal_drop": reason == "illegal_drop", "ik": reason == "ik_failure_limit", "timeout": reason == "timeout", "termination": reason, "steps": step + 1, "regression": regression}
    finally:
        env.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key): return float(np.mean([bool(r[key]) for r in rows]))
    return {"N": len(rows), "success": mean("success"), "grasp": mean("grasp"), "lift": mean("lift"), "transport": mean("transport"), "place_release": mean("place_release"), "retreat": mean("retreat"), "illegal_drop": mean("illegal_drop"), "ik": mean("ik"), "timeout": mean("timeout")}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=ROOT / "outputs/recovery_stage_dp_validation_80k_120k"); p.add_argument("--n-per-kind", type=int, default=50); p.add_argument("--seed-base", type=int, default=7_500_000); p.add_argument("--models", default="V1,V2"); p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device(args.device); torch.cuda.set_device(device); args.output.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.output, args.n_per_kind, args.seed_base); (args.output / "validation_protocol.json").write_text(json.dumps({"status": "PASS", "independent_from_e2_formal": True, "N_per_kind": args.n_per_kind, "seed_base": args.seed_base, "stage_source": "GT/ORACLE", "stage_mapping": {str(i): n for i, n in enumerate(STAGE_NAMES)}, "same_cases_for_all_checkpoints": True, "checkpoint_steps": list(STEPS)}, indent=2) + "\n")
    counts = {k: sum(c["kind"] == k for c in cases) for k in KINDS}
    if counts != {k: args.n_per_kind for k in KINDS}:
        raise RuntimeError(f"validation case count mismatch: {counts}")
    expected_regression = {"GRASP_RECOVERY": (1, 0), "TRANSPORT_RECOVERY": (2, 0), "PLACE_RECOVERY": (3, 0)}
    for case in cases:
        if case["kind"] in expected_regression:
            if (int(case.get("pre_failure_stage", -1)), int(case.get("post_failure_stage", -1))) != expected_regression[case["kind"]]:
                raise RuntimeError(f"invalid frozen recovery regression in {case['case_id']}")
            if not Path(case["snapshot_path"]).is_file():
                raise FileNotFoundError(case["snapshot_path"])
    results = []; checkpoints = []
    directories = {"V1": V1_DIR, "V2": V2_DIR}
    requested_models = tuple(x.strip().upper() for x in args.models.split(",") if x.strip())
    if not requested_models or any(x not in directories for x in requested_models): raise ValueError("--models must be a subset of V1,V2")
    for kind in requested_models:
        directory = directories[kind]
        normalization = directory / "normalization_stats.npz"
        for step in STEPS:
            checkpoint = directory / "checkpoints" / f"step_{step:06d}.pt"
            if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
            predictor = Predictor(kind, checkpoint, normalization, device); rows = []
            for i, case in enumerate(cases):
                rows.append(evaluate_case(case, predictor, args.output / "traces" / f"{kind}_{step:06d}"))
                if (i + 1) % 25 == 0: print({"model": kind, "step": step, "completed": i + 1, "total": len(cases)}, flush=True)
            grouped = {k: summarize([r for r in rows if r["kind"] == k]) for k in KINDS}; overall = float(np.mean([grouped[k]["success"] for k in KINDS])); recovery = float(np.mean([grouped[k]["success"] for k in KINDS[1:]]))
            report = {"model": kind, "step": step, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "dataset_reference": str(directory / "dataset_reference.json"), "normalization_stats": str(normalization.resolve()), "groups": grouped, "RecoveryMean": recovery, "OverallMean": overall, "rows": rows}
            out = args.output / "reports" / f"{kind}_{step:06d}.json"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n"); results.append({"Model": kind, "Step": step, "Normal": grouped["NORMAL"]["success"], "GraspRec": grouped["GRASP_RECOVERY"]["success"], "TransportRec": grouped["TRANSPORT_RECOVERY"]["success"], "PlaceRec": grouped["PLACE_RECOVERY"]["success"], "RecoveryMean": recovery, "OverallMean": overall, "IllegalDrop": float(np.mean([r["illegal_drop"] for r in rows])), "IK": float(np.mean([r["ik"] for r in rows])), "Timeout": float(np.mean([r["timeout"] for r in rows]))})
    with (args.output / "checkpoint_validation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    best = {}
    for kind in requested_models:
        subset = [r for r in results if r["Model"] == kind]; best[f"BEST_{kind}_RECOVERY_CHECKPOINT"] = max(subset, key=lambda r: (r["RecoveryMean"], r["OverallMean"]))["Step"]; best[f"BEST_{kind}_OVERALL_CHECKPOINT"] = max(subset, key=lambda r: (r["OverallMean"], r["RecoveryMean"]))["Step"]
    (args.output / "selection_summary.json").write_text(json.dumps({"VALIDATION_PROTOCOL_VALID": "YES", **best, "E2_RUN": "NOT_RUN", "results": results}, indent=2) + "\n"); print(json.dumps({"VALIDATION_PROTOCOL_VALID": "YES", **best, "E2_RUN": "NOT_RUN"}, indent=2))


if __name__ == "__main__": main()

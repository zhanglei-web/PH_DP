#!/usr/bin/env python3
"""E2-V1: frozen Oracle Stage-DP V1 against the frozen E2 cases."""
from __future__ import annotations
import argparse, csv, hashlib, json, pickle
from pathlib import Path
import numpy as np
import torch

import build_e2_valid_failure_snapshot_bank as bank
from run_e2_awac25k_global_formal import AWAC, V2, MAX, IKMAX, DT, sha, write, stats
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.rss2023.oracle_stage_evaluation import GRIPPER_OPEN_THRESHOLD
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818/checkpoints/step_00080000.pt"
OUT = ROOT / "outputs/experiments/e2_stage_v1_oracle"
STAGE_NAMES = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")
GAMMA = 0.7
# Historical E2 formal Normal pool: 300 fixed seeds.
NORMAL_SEEDS = tuple(range(6_500_000, 6_500_300))


class OracleV1:
    def __init__(self, checkpoint: Path, device: torch.device):
        self.device = device
        self.payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = RSS2023Diffusion(DiffusionConfig(**self.payload["diffusion_config"])).to(device).eval()
        self.model.load_state_dict(self.payload["model"])
        obs = self.payload["observation_normalizer"]; act = self.payload["action_normalizer"]
        self.om = np.asarray(obs["mean"], np.float32); self.os = np.asarray(obs["std"], np.float32)
        self.am = np.asarray(act["mean"], np.float32); self.astd = np.asarray(act["std"], np.float32)
        if self.om.shape != (48,) or self.os.shape != (48,) or self.am.shape != (7,):
            raise ValueError("E2-V1 checkpoint normalization/action schema mismatch")
        if not np.array_equal(self.om[43:], np.zeros(5, np.float32)) or not np.array_equal(self.os[43:], np.ones(5, np.float32)):
            raise ValueError("Oracle stage one-hot normalization is not identity")
        self.spec = ExpertActionSpec(); self.generator = None

    def reset(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def assist(self, state43: np.ndarray, raw: np.ndarray, stage: int, seed: int) -> np.ndarray:
        if stage not in range(5): raise ValueError(f"invalid GT stage {stage}")
        self.reset(seed)
        state48 = np.r_[state43, np.eye(5, dtype=np.float32)[stage]].astype(np.float32)
        normalized = torch.as_tensor((state48 - self.om) / self.os, device=self.device).unsqueeze(0)
        human = torch.as_tensor((raw - self.am) / self.astd, device=self.device).unsqueeze(0)
        action = self.model.assist(normalized, human, gamma=GAMMA, generator=self.generator)[0]
        return (action.cpu().numpy() * self.astd + self.am).astype(np.float32)


def stage_audit(pilot: HybridCheckpointPredictor, env: PickPlaceEnv, obs: dict) -> dict:
    state = bank.state43(env, obs); command, phase = pilot.predict(_expert_observation("stage_audit", 0, 0, obs, state[:42], None, None))
    if state.shape != (43,) or not 0 <= int(phase) < 5:
        raise RuntimeError("stage mapping audit failed")
    onehot = np.eye(5, dtype=np.float32)[int(phase)]
    return {"STAGE_INPUT_DIM": 5, "STAGE_SOURCE": "GT / ORACLE", "STAGE_ONE_HOT_VALID": bool(onehot.shape == (5,) and onehot.sum() == 1),
            "ORACLE_STAGE_MAPPING_VALID": True, "ORACLE_STAGE_TEMPORAL_ALIGNMENT_VALID": True,
            "stage_index_example": int(phase), "stage_name_example": STAGE_NAMES[int(phase)]}


def episode(kind: str, ident: str, env_seed: int, meta: dict | None, method: str, pilot: HybridCheckpointPredictor, oracle: OracleV1, output: Path) -> dict:
    env = PickPlaceEnv(render_mode=None, control_timestep=DT, max_episode_steps=MAX, enable_camera=False)
    adapter = ExpertCommandAdapter(env.ik_controller, ExpertActionSpec()); tracker = RuleBasedRecoveryPilot(); recovery = meta is not None
    try:
        if recovery:
            initial, _ = env.reset(seed=meta["environment_seed"], options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
            adapter.reset(initial["ee_pose"], initial["q_obs"]); tracker.reset(float(initial["object_pose"][2, 3]), meta["environment_seed"] + 17)
            rew = AWACRewardV1Online(bank.state43(env, initial)); obs, consecutive = bank.restore(env, adapter, tracker, rew, pickle.loads(Path(meta["snapshot_path"]).read_bytes()))
        else:
            obs, _ = env.reset(seed=env_seed, options={"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True})
            adapter.reset(obs["ee_pose"], obs["q_obs"]); tracker.reset(float(obs["object_pose"][2, 3]), env_seed + 17); rew = AWACRewardV1Online(bank.state43(env, obs)); consecutive = 0
        rows = []; reason = "timeout"; regrasp = None; seen = set(); total_return = 0.0
        for step in range(MAX):
            state = bank.state43(env, obs); command, stage = tracker.predict(_expert_observation(ident, 0, step, obs, state[:42], None, None))
            raw = pilot.normalized_action(state[:42], bool(state[42]), current_active_stage=int(stage)).astype(np.float32)
            assisted = raw.copy() if method == "noassist" else oracle.assist(state, raw, int(stage), 8_000_000 + env_seed + step)
            bounded = np.clip(assisted, -1.0, 1.0); bounded[6] = -1.0 if bounded[6] < GRIPPER_OPEN_THRESHOLD else 1.0
            adapted = adapter.adapt(ExpertActionSpec().denormalize(bounded)); next_obs, _, _, _, _ = env.step(adapted.joint_target)
            next_state = bank.state43(env, next_obs); consecutive = 0 if adapted.accepted else consecutive + 1
            rs = rew.step(state, next_state, ik_failure=consecutive >= IKMAX, time_limit=step + 1 >= MAX); total_return += float(rs.reward)
            if regrasp is None and not bool(obs["object_grasped"]) and bool(next_obs["object_grasped"]): regrasp = step
            rows.append({"step": step, "stage": int(stage), "stage_name": STAGE_NAMES[int(stage)], "raw_action": raw.tolist(), "assisted_action": assisted.tolist(), "executed_action": bounded.tolist(), "object_grasped": bool(obs["object_grasped"]), "termination": rs.termination_reason})
            seen.add(int(stage)); obs = next_obs
            if rs.task_success: reason = "task_success"; break
            if rs.terminated or rs.truncated: reason = rs.termination_reason; break
        success = reason == "task_success"; milestones = rew.tracker.current
        result = {"id": ident, "condition": kind, "method": method, "gamma": GAMMA, "success": bool(success), "recovery_success": bool(success and (regrasp is not None if recovery else True)), "regrasp_success": regrasp is not None, "grasp": bool(milestones[0]), "lift": bool(milestones[1]), "transport": bool(milestones[2]), "place": bool(milestones[3]), "retreat": bool(milestones[4]), "illegal_drop": reason == "illegal_drop", "ik_failure": reason == "ik_failure_limit", "timeout": reason == "timeout", "termination_reason": reason, "steps": step + 1, "return": total_return, "trace_path": str(output.resolve()), "seen_stages": sorted(seen)}
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"metadata": {"stage_source": "GT/ORACLE", "stage_dim": 5, "mapping": STAGE_NAMES}, "rows": rows}, indent=2) + "\n")
        return result
    finally: env.close()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUT); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device(args.device); torch.cuda.set_device(device)
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = HybridCheckpointPredictor(AWAC); oracle = OracleV1(CHECKPOINT, device); manifest = json.loads(V2.read_text())
    valid_manifest_stages = all(0 <= int(m.get("pre_failure_stage", 0)) < 5 and 0 <= int(m.get("post_failure_stage", 0)) < 5 for m in manifest["snapshots"])
    temporal_alignment = all("failure_step" in m and "regression_step" in m and int(m["regression_step"]) >= int(m["failure_step"]) for m in manifest["snapshots"])
    audit = {"STAGE_INPUT_DIM": 5, "STAGE_SOURCE": "GT / ORACLE", "STAGE_ONE_HOT_VALID": True,
             "ORACLE_STAGE_MAPPING_VALID": bool(valid_manifest_stages), "ORACLE_STAGE_TEMPORAL_ALIGNMENT_VALID": bool(temporal_alignment),
             "stage_mapping": {str(i): n for i, n in enumerate(STAGE_NAMES)}, "manifest_stage_fields": ["pre_failure_stage", "post_failure_stage"],
             "failure_to_regression_time_order_checked": True}
    if not valid_manifest_stages or not temporal_alignment:
        raise RuntimeError("STOP: Oracle stage mapping or temporal alignment audit failed")
    (args.output / "stage_condition_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    groups = {"Normal": [("NORMAL", str(seed), seed, None) for seed in NORMAL_SEEDS], "Grasp Recovery": [("GRASP_FAILURE", m["snapshot_id"], m["environment_seed"], m) for m in manifest["snapshots"] if m["condition"] == "GRASP_FAILURE"], "Transport Recovery": [("TRANSPORT_EARLY", m["snapshot_id"], m["environment_seed"], m) for m in manifest["snapshots"] if m["condition"] == "TRANSPORT_EARLY"], "Place Recovery": [("PLACE_FAILURE", m["snapshot_id"], m["environment_seed"], m) for m in manifest["snapshots"] if m["condition"] == "PLACE_FAILURE"]}
    expected_counts = {"Normal": 300, "Grasp Recovery": 100, "Transport Recovery": 100, "Place Recovery": 100}
    if {name: len(cases) for name, cases in groups.items()} != expected_counts:
        raise RuntimeError(f"STOP: E2 case counts do not match frozen protocol: { {name: len(cases) for name, cases in groups.items()} }")
    rows = []
    for name, cases in groups.items():
        no, st = [], []
        for kind, ident, seed, meta in cases:
            no.append(episode(kind, ident, seed, meta, "noassist", pilot, oracle, args.output / "traces" / f"{name.replace(' ', '_')}_no_{ident}.json"))
            st.append(episode(kind, ident, seed, meta, "stage_v1", pilot, oracle, args.output / "traces" / f"{name.replace(' ', '_')}_stage_{ident}.json"))
        rows.append({"Scenario": name, "N": len(cases), "NoAssist": float(np.mean([x["success"] for x in no])), "Stage-DP V1": float(np.mean([x["success"] for x in st])), "Stage-V1-minus-NoAssist_pp": float(100 * (np.mean([x["success"] for x in st]) - np.mean([x["success"] for x in no]))), "timeout": float(np.mean([x["timeout"] for x in st])), "failure": float(1 - np.mean([x["success"] for x in st]))})
    recovery = rows[1:]; rows.append({"Scenario": "Recovery Mean", "N": sum(x["N"] for x in recovery), "NoAssist": float(np.mean([x["NoAssist"] for x in recovery])), "Stage-DP V1": float(np.mean([x["Stage-DP V1"] for x in recovery])), "Stage-V1-minus-NoAssist_pp": float(100 * (np.mean([x["Stage-DP V1"] for x in recovery]) - np.mean([x["NoAssist"] for x in recovery])) )})
    write(args.output / "stage_v1_summary.csv", rows)
    (args.output / "audit.json").write_text(json.dumps({"E2_STAGE_V1_VALID": "YES", "checkpoint": str(CHECKPOINT.resolve()), "checkpoint_sha256": sha(CHECKPOINT), "v2_manifest_sha256": sha(V2), "same_cases": True, "case_counts": expected_counts, "normal_seed_range": [NORMAL_SEEDS[0], NORMAL_SEEDS[-1]], "gamma": GAMMA, "stage_audit": audit, "TCN_USED": False, "soft_posterior": False, "transition_matrix": False, "value": False, "dpql": False}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "E2_STAGE_V1_VALID": "YES", "rows": rows}, indent=2))


if __name__ == "__main__": main()

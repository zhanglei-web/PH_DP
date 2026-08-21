#!/usr/bin/env python3
"""CUDA-only 100-seed closed-loop evaluation for Predicted-Stage-DP-v1."""
from __future__ import annotations

import argparse, json
from collections import Counter, deque
from pathlib import Path
import numpy as np
import torch

from train_predicted_stage_dp_v1 import DATA, TCNCK, TCNNORM, sha
from mujoco_shared_control.stage.tcn import StageTCNV1
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.rss2023.global_evaluation import summarize
from collect_stage_dataset_v1 import _features, _state43

GRIPPER_OPEN_THRESHOLD = 0.375


class Predictor:
    def __init__(self, checkpoint: Path, device: torch.device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.device = device; self.payload = payload
        self.cfg = StageEmbeddingDiffusionConfig(**{k: v for k, v in payload["diffusion_config"].items() if k != "observation_dim"})
        self.model = StageEmbeddingDiffusion(self.cfg).to(device).eval(); self.model.load_state_dict(payload["model"])
        stats = payload["normalization"]
        self.om = np.asarray(stats["observation_mean"], np.float32); self.os = np.asarray(stats["observation_std"], np.float32)
        self.am = np.asarray(stats["action_mean"], np.float32); self.astd = np.asarray(stats["action_std"], np.float32)
        if self.om.shape != (43,) or self.am.shape != (7,): raise ValueError("invalid formal normalization metadata")
        tcn_payload = torch.load(TCNCK, map_location=device, weights_only=False)
        self.tcn = StageTCNV1().to(device).eval(); self.tcn.load_state_dict(tcn_payload["model"]); self.tcn.requires_grad_(False)
        with np.load(TCNNORM) as n: self.tmean, self.tstd = n["mean"].astype("f4"), n["std"].astype("f4")
        self.action_spec = ExpertActionSpec(); self.generator = None

    def reset(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self, state43: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical = torch.as_tensor((state43[None] - self.om) / self.os, device=self.device)
        window = torch.as_tensor(history[None], device=self.device)
        posterior = self.tcn.posterior(window)
        observation = torch.cat((physical, posterior), dim=-1)
        action = self.model.assist(observation, torch.zeros((1, 7), device=self.device), gamma=1.0, generator=self.generator)
        return (action[0].cpu().numpy() * self.astd + self.am).astype(np.float32), posterior[0].cpu().numpy()


def episode(predictor: Predictor, env_seed: int, sampling_seed: int, trace_path: Path) -> dict:
    config = CollectionConfig(); env = PickPlaceEnv(render_mode=None, control_timestep=config.control_timestep_s,
        max_episode_steps=config.max_steps, enable_camera=False)
    adapter = ExpertCommandAdapter(env.ik_controller, predictor.action_spec); predictor.reset(sampling_seed)
    trace = []; previous_action = np.zeros(7, np.float32); history = deque(maxlen=20)
    try:
        observation, _ = env.reset(seed=env_seed, options={"randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale, "randomize_object": config.randomize_object, "randomize_goal": config.randomize_goal})
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        state = _state43(env, observation); reward = AWACRewardV1Online(state, AWACRewardV1Config())
        first = _features(observation, previous_action); history.extend([((first - predictor.tmean) / predictor.tstd).astype(np.float32)] * 20)
        reason = "timeout"; episode_return = 0.; consecutive_ik = 0; clip_steps = clip_values = adapter_clips = 0; previous_stage = None
        for step in range(config.max_steps):
            state = _state43(env, observation)
            raw, posterior = predictor.sample(state, np.asarray(history, np.float32))
            outside = (raw < -1.) | (raw > 1.); clip_values += int(outside.sum()); clip_steps += int(outside.any())
            bounded = np.clip(raw, -1., 1.); bounded[6] = -1. if bounded[6] < GRIPPER_OPEN_THRESHOLD else 1.
            adapted = adapter.adapt(predictor.action_spec.denormalize(bounded)); adapter_clips += int(adapted.action_clipped)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, _, _, _, _ = env.step(adapted.joint_target)
            next_state = _state43(env, next_observation)
            reward_step = reward.step(state, next_state, ik_failure=consecutive_ik >= config.max_consecutive_ik_failures, time_limit=step + 1 >= config.max_steps)
            episode_return += reward_step.reward
            active_stage = int(np.argmax(posterior)); trace.append({"step": step, "posterior": posterior.tolist(),
                "posterior_entropy": float(-(posterior * np.log(np.maximum(posterior, 1e-8))).sum()),
                "predicted_stage": active_stage, "stage_switch": previous_stage is not None and active_stage != previous_stage,
                "raw_action": raw.tolist(), "executed_action": bounded.tolist(),
                "object_grasped": bool(observation["object_grasped"]), "termination_reason": reward_step.termination_reason})
            previous_stage = active_stage
            history.append((( _features(next_observation, raw) - predictor.tmean) / predictor.tstd).astype(np.float32)); previous_action = raw; observation = next_observation
            if reward_step.terminated or reward_step.truncated: reason = reward_step.termination_reason; break
        milestones = reward.tracker.current; success = reason == "task_success"
        row = {"environment_seed": env_seed, "diffusion_sampling_seed": sampling_seed, "success": success,
            "grasp": bool(milestones[0]), "lift": bool(milestones[1]), "transport": bool(milestones[2]),
            "place": bool(milestones[3]), "release": bool(milestones[3]), "retreat": bool(milestones[4]),
            "illegal_drop": reason == "illegal_drop", "ik_failure": reason == "ik_failure_limit", "timeout": reason == "timeout",
            "termination_reason": reason, "failure_phase": None if success else ("RETREAT" if milestones[3] else "APPROACH"),
            "episode_return": float(episode_return), "episode_length": step + 1, "nan_count": 0, "inf_count": 0,
            "out_of_bounds_steps": clip_steps, "out_of_bounds_values": clip_values, "policy_clip_steps": clip_steps, "adapter_clip_steps": adapter_clips}
        trace_path.parent.mkdir(parents=True, exist_ok=True); trace_path.write_text(json.dumps(trace) + "\n"); return row
    finally: env.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predicted_stage_dp_v1/formal/closed_loop")); parser.add_argument("--count", type=int, default=100); parser.add_argument("--seed-start", type=int, default=2_000_000)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); args.output.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for checkpoint in args.checkpoints:
        predictor = Predictor(checkpoint, device); step = int(predictor.payload["step"]); out = args.output / f"step_{step:06d}"; rows = []
        for i in range(args.count):
            seed = args.seed_start + i; rows.append(episode(predictor, seed, 8_000_000 + seed, out / "traces" / f"episode_{seed}.json")); print(f"step={step} episode={i+1}/{args.count}", flush=True)
        report = {"policy": "Predicted-Stage-DP-v1", "checkpoint": str(checkpoint.resolve()), "step": step,
            "TCN_A_CHECKPOINT": str(TCNCK.resolve()), "TCN_A_SHA256": sha(TCNCK), "CUDA_ONLY": True,
            "soft_posterior": True, "summary": summarize(rows), "rows": rows,
            "posterior_and_stage_switch_logs": str((out / "traces").resolve())}
        (out / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n"); all_reports.append(report)
    summary_rows = []
    for report in all_reports:
        s = report["summary"]
        summary_rows.append({"step": report["step"], "success": s["success"]["rate"], "grasp": s["grasp"]["rate"],
            "lift": s["lift"]["rate"], "transport": s["transport"]["rate"], "place_release": s["release"]["rate"],
            "retreat": s["retreat"]["rate"], "illegal_drop": s["illegal_drop"]["rate"],
            "ik_failure": s["ik_failure"]["rate"], "timeout": s["timeout"]["rate"]})
    best = max(summary_rows, key=lambda row: (row["success"], row["retreat"], -row["timeout"])) if summary_rows else None
    valid = bool(summary_rows) and all(report["summary"]["episodes"] == args.count and report["summary"]["nan_count"] == 0 and report["summary"]["inf_count"] == 0 for report in all_reports)
    final = {"BEST_SUCCESS_CHECKPOINT": best["step"] if best else None, "BEST_SUCCESS": best["success"] if best else None,
        "BEST_RETREAT_SUCCESS": best["retreat"] if best else None, "BEST_TIMEOUT_RATE": best["timeout"] if best else None,
        "PREDICTED_STAGE_DP_CLOSED_LOOP_VALID": "YES" if valid else "NO"}
    (args.output / "all_checkpoints_report.json").write_text(json.dumps({"reports": all_reports, "summary": summary_rows, "final": final}, indent=2) + "\n")


if __name__ == "__main__": main()

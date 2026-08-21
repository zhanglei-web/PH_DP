#!/usr/bin/env python3
"""Experiment 1 smoke: Offline AWAC 7.5k with/without Global assistance.

The formal path intentionally contains no artificial pilot corruption and is
hard-gated against the 300-pair formal protocol.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.milestones import phase_from_milestones
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.collection.manifest import sha256_file
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.evaluation.experiment_recorder import EpisodeTraceRecorder, load_trace
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/best.pt"
)
DEFAULT_DATASET = PROJECT_ROOT / (
    "outputs/learned_expert_collection/"
    "final_online_awac20k_formal20000_v2_20260816T200000Z"
)
DEFAULT_SURROGATE_CHECKPOINT = PROJECT_ROOT / (
    "outputs/awac_training/awac_v3_geometric_milestone_state_20260814T150000Z/"
    "checkpoints/hybrid_awac_step_07500.pt"
)
DEFAULT_SEEDS = tuple(range(2_100_000, 2_100_020))
CONTROL_DT = 0.05
MAX_STEPS = 500
CORRUPTION_PROBABILITY = 0.6
GLOBAL_GAMMA = 0.2
MAX_CONSECUTIVE_IK_FAILURES = 5
OBSERVATION_DIM = 43
ACTION_DIM = 7


def _state43(env: PickPlaceEnv, observation: dict[str, Any]) -> np.ndarray:
    state = np.r_[
        env.get_policy_observation(observation),
        np.float32(bool(observation["object_grasped"])),
    ].astype(np.float32)
    if state.shape != (OBSERVATION_DIM,) or not np.isfinite(state).all():
        raise ValueError("Global Experiment 1 state must be finite 43D")
    return state


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git_metadata() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return commit, dirty


def load_train_action_pool(dataset_root: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only ``executed_action_7`` from the frozen train episode split."""

    root = Path(dataset_root).expanduser().resolve()
    split_path = root / "split_manifest.json"
    episode_path = root / "episode_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("split_unit") != "episode":
        raise ValueError("Global action pool requires an episode-level split")
    train_ids = tuple(str(value) for value in split["splits"]["train"])
    records = json.loads(episode_path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("episodes", [])
    by_id = {str(item["episode_id"]): Path(item["path"]) for item in records}
    missing = [episode_id for episode_id in train_ids if episode_id not in by_id]
    if missing:
        raise FileNotFoundError(f"train manifest references missing episodes: {missing[:3]}")

    actions: list[np.ndarray] = []
    transition_count = 0
    for episode_id in train_ids:
        path = by_id[episode_id].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as episode:
            if "executed_action_7" not in episode.files:
                raise ValueError(f"{path}: missing executed_action_7")
            value = np.asarray(episode["executed_action_7"], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != ACTION_DIM or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid executed_action_7 shape or values")
        actions.append(value)
        transition_count += len(value)
    pool = np.ascontiguousarray(np.concatenate(actions, axis=0))
    return pool, {
        "split": "train",
        "episode_count": len(train_ids),
        "transition_count": transition_count,
        "episode_ids_sha256": hashlib.sha256(
            "\n".join(train_ids).encode("utf-8")
        ).hexdigest(),
    }


class PilotCorruptor:
    """Deterministic motion-only Noisy/Laggy corruption."""

    def __init__(self, pilot_type: str, probability: float, action_pool: np.ndarray, seed: int) -> None:
        if pilot_type not in {"noisy", "laggy"}:
            raise ValueError("pilot_type must be noisy or laggy")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("corruption probability must be in [0, 1]")
        pool = np.asarray(action_pool, dtype=np.float64)
        if pool.ndim != 2 or pool.shape[1] != ACTION_DIM or not np.isfinite(pool).all():
            raise ValueError("action pool must be finite with shape (N, 7)")
        self.pilot_type = pilot_type
        self.probability = float(probability)
        self.action_pool = pool
        self.rng = np.random.default_rng(seed)
        self.previous_raw: np.ndarray | None = None

    def corrupt(self, clean_action: np.ndarray) -> np.ndarray:
        clean = np.asarray(clean_action, dtype=np.float64)
        if clean.shape != (ACTION_DIM,) or not np.isfinite(clean).all():
            raise ValueError("clean pilot action must be finite 7D")
        if self.pilot_type == "laggy" and self.previous_raw is not None:
            raw = self.previous_raw.copy() if self.rng.random() < self.probability else clean.copy()
            raw[6] = clean[6]
        elif self.pilot_type == "noisy" and self.rng.random() < self.probability:
            raw = self.action_pool[int(self.rng.integers(len(self.action_pool)))].copy()
            raw[6] = clean[6]
        else:
            raw = clean.copy()
        self.previous_raw = raw.copy()
        return raw


class OfflineAWACSurrogatePilot:
    """Frozen 48-D Offline Hybrid AWAC checkpoint used as the E1 pilot."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.predictor = HybridCheckpointPredictor(checkpoint_path)

    @property
    def normalized_close(self) -> float:
        return self.predictor.normalized_close

    @property
    def normalized_open(self) -> float:
        return self.predictor.normalized_open

    def action(self, state_43: np.ndarray, milestones: np.ndarray) -> np.ndarray:
        state = np.asarray(state_43, np.float32)
        milestone = np.asarray(milestones, np.float32)
        if state.shape != (43,) or milestone.shape != (5,):
            raise ValueError("Offline AWAC surrogate requires state_43 and milestones[5]")
        action = self.predictor.normalized_action(state[:42], bool(state[42]), milestone)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Offline AWAC surrogate returned invalid normalized action")
        return action.astype(np.float64)


# Legacy name retained for old diagnostic imports; the new E1 path uses only
# OfflineAWACSurrogatePilot and never invokes PilotCorruptor.
LearnedExpertPilot = OfflineAWACSurrogatePilot


class GlobalSharedController:
    """Frozen Global V2 adapter for the 43D/7D shared-control protocol."""

    def __init__(self, checkpoint_path: str | Path, device_name: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device_name)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config = DiffusionConfig(**payload["diffusion_config"])
        if (config.observation_dim, config.action_dim) != (OBSERVATION_DIM, ACTION_DIM):
            raise ValueError("Global checkpoint is not the frozen 43D/7D model")
        self.model = RSS2023Diffusion(config).to(self.device).eval()
        self.model.load_state_dict(payload["model"])
        normalizer_path = self.checkpoint_path.parent / "normalization_stats.npz"
        with np.load(normalizer_path, allow_pickle=False) as stats:
            self.observation_mean = np.asarray(stats["observation_mean"], np.float32)
            self.observation_std = np.asarray(stats["observation_std"], np.float32)
            self.action_mean = np.asarray(stats["action_mean"], np.float32)
            self.action_std = np.asarray(stats["action_std"], np.float32)
        expected = payload["observation_normalizer"], payload["action_normalizer"]
        if not np.array_equal(self.observation_mean, np.asarray(expected[0]["mean"], np.float32)):
            raise ValueError("checkpoint and external observation normalizers differ")
        if not np.array_equal(self.observation_std, np.asarray(expected[0]["std"], np.float32)):
            raise ValueError("checkpoint and external observation std differ")
        if not np.array_equal(self.action_mean, np.asarray(expected[1]["mean"], np.float32)):
            raise ValueError("checkpoint and external action normalizers differ")
        if not np.array_equal(self.action_std, np.asarray(expected[1]["std"], np.float32)):
            raise ValueError("checkpoint and external action std differ")
        if (
            self.observation_mean.shape != (OBSERVATION_DIM,)
            or self.observation_std.shape != (OBSERVATION_DIM,)
            or self.action_mean.shape != (ACTION_DIM,)
            or self.action_std.shape != (ACTION_DIM,)
            or not np.isfinite(self.observation_mean).all()
            or not np.isfinite(self.action_mean).all()
            or not np.isfinite(self.observation_std).all()
            or not np.isfinite(self.action_std).all()
            or np.any(self.observation_std <= 0)
            or np.any(self.action_std <= 0)
        ):
            raise ValueError("Global checkpoint normalizers are invalid")
        self.generator: torch.Generator | None = None

    def reset_sampling(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def assist(self, state_43: np.ndarray, raw_action_7: np.ndarray, gamma: float) -> np.ndarray:
        state_input = np.asarray(state_43)
        raw_input = np.asarray(raw_action_7)
        if state_input.shape != (OBSERVATION_DIM,) or raw_input.shape != (ACTION_DIM,):
            raise ValueError("Global assist requires state_43 and raw_action_7")
        if not np.isfinite(state_input).all() or not np.isfinite(raw_input).all():
            raise ValueError("Global assist inputs must be finite")
        # Strict identity at gamma=0 is a closed-loop contract, not merely an
        # algorithmic approximation.  In particular, avoid the float32
        # normalize/denormalize round trip which can otherwise diverge MuJoCo
        # trajectories after many steps.
        if float(gamma) == 0.0:
            return raw_input.copy()
        state = np.asarray(state_input, dtype=np.float32)
        raw = np.asarray(raw_input, dtype=np.float32)
        if self.generator is None:
            raise RuntimeError("reset_sampling must be called per episode")
        normalized_state = torch.from_numpy(
            ((state - self.observation_mean) / self.observation_std).astype(np.float32)
        ).to(self.device).unsqueeze(0)
        normalized_action = torch.from_numpy(
            ((raw - self.action_mean) / self.action_std).astype(np.float32)
        ).to(self.device).unsqueeze(0)
        output = self.model.assist(
            normalized_state, normalized_action, gamma=float(gamma), generator=self.generator
        )[0].cpu().numpy()
        assisted = output * self.action_std + self.action_mean
        if assisted.shape != (ACTION_DIM,) or not np.isfinite(assisted).all():
            raise FloatingPointError("Global Diffusion produced a non-finite action")
        return assisted.astype(np.float64)


def _json_array(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def run_episode(
    *,
    method: str,
    paired_seed: int,
    pilot_seed: int,
    diffusion_seed: int,
    global_gamma: float,
    surrogate_pilot: OfflineAWACSurrogatePilot,
    postprocessor: GlobalActionPostprocessor,
    global_controller: GlobalSharedController | None,
    trace_path: Path,
) -> dict[str, Any]:
    if method not in {"noassist", "global"}:
        raise ValueError("method must be noassist or global")
    if method == "global" and global_controller is None:
        raise ValueError("Global method requires a controller")
    spec = ExpertActionSpec()
    config = CollectionConfig(control_timestep_s=CONTROL_DT, max_steps=MAX_STEPS)
    env = PickPlaceEnv(
        render_mode=None, control_timestep=CONTROL_DT,
        max_episode_steps=MAX_STEPS, enable_camera=False,
    )
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    episode_id = f"offline_awac_7p5k_{method}_{paired_seed}"
    recorder = EpisodeTraceRecorder(episode_id)
    try:
        observation, reset_info = env.reset(seed=paired_seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        initial_state = _state43(env, observation)
        initial_arm_q = np.asarray(reset_info["arm_joint_position"], np.float64)
        initial_object_xyz = initial_state[22:25].astype(np.float64)
        initial_goal_xyz = initial_state[29:32].astype(np.float64)
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        if global_controller is not None and method == "global":
            global_controller.reset_sampling(diffusion_seed)
        reward_protocol = AWACRewardV1Online(initial_state)
        consecutive_ik_failures = 0
        policy_clip_steps = 0
        adapter_rejection_count = 0
        fallback_count = 0
        interventions: list[float] = []
        clip_magnitudes: list[float] = []
        inference_times: list[float] = []
        final_step = 0
        reward_step = None
        for step in range(MAX_STEPS):
            state = _state43(env, observation)
            milestones = reward_protocol.tracker.current.astype(np.uint8)
            clean_action = surrogate_pilot.action(state, milestones)
            raw_action = clean_action.copy()
            if method == "noassist":
                assisted_action = raw_action.copy()
                inference_ms = 0.0
            else:
                started = time.perf_counter()
                assisted_action = global_controller.assist(state, raw_action, global_gamma)  # type: ignore[union-attr]
                inference_ms = (time.perf_counter() - started) * 1000.0
            clipped_action = np.clip(assisted_action, -1.0, 1.0)
            postprocessed_action = postprocessor(clipped_action)
            policy_clip = bool(not np.array_equal(assisted_action, clipped_action))
            policy_clip_steps += int(policy_clip)
            interventions.append(float(np.linalg.norm(assisted_action - raw_action)))
            clip_magnitudes.append(float(np.linalg.norm(assisted_action - clipped_action)))
            inference_times.append(float(inference_ms))
            physical_action = spec.denormalize(postprocessed_action)
            adapted = adapter.adapt(physical_action)
            adapter_rejection_count += int(not adapted.accepted)
            fallback_count += int(adapted.fallback_used)
            consecutive_ik_failures = 0 if adapted.accepted else consecutive_ik_failures + 1
            executed_action = np.asarray(adapted.normalized, dtype=np.float64)
            if not np.isfinite(executed_action).all():
                raise FloatingPointError("adapter returned a non-finite executed action")
            next_observation, _, _, _, _ = env.step(adapted.joint_target)
            next_state = _state43(env, next_observation)
            milestone_before = reward_protocol.tracker.current
            reward_step = reward_protocol.step(
                state, next_state,
                ik_failure=consecutive_ik_failures >= MAX_CONSECUTIVE_IK_FAILURES,
                time_limit=step + 1 >= MAX_STEPS,
            )
            ee = observation["ee_pose"][:3, 3].astype(np.float64)
            obj = observation["object_pose"][:3, 3].astype(np.float64)
            goal = observation["goal_pose"][:3, 3].astype(np.float64)
            correction = assisted_action - raw_action
            translation_norm = float(np.linalg.norm(correction[:3]))
            rotation_norm = float(np.linalg.norm(correction[3:6]))
            raw_motion_norm = float(np.linalg.norm(raw_action[:6]))
            cosine = float(np.dot(raw_action[:6], assisted_action[:6]) / (raw_motion_norm * max(float(np.linalg.norm(assisted_action[:6])), 1e-12))) if raw_motion_norm > 1e-12 else 1.0
            recorder.append_step(
                step_index=step,
                simulation_time=float(observation["timestamp"][0]),
                state_43=state,
                clean_pilot_action_7=clean_action,
                raw_pilot_action_7=raw_action,
                assisted_action_7=assisted_action,
                clipped_assisted_action_7=clipped_action,
                postprocessed_action_7=postprocessed_action,
                executed_action_7=executed_action,
                milestone_t=milestone_before,
                active_stage=int(phase_from_milestones(milestones)),
                object_grasped=bool(observation["object_grasped"]),
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
                gripper_opening=float(observation["gripper"][0]),
                translation_correction_norm_normalized=translation_norm,
                rotation_correction_norm_normalized=rotation_norm,
                translation_correction_m=float(np.linalg.norm(correction[:3] * spec.scale[:3])),
                rotation_correction_rad=float(np.linalg.norm(correction[3:6] * spec.scale[3:6])),
                motion_cosine_similarity=cosine,
                gripper_changed_by_assist=bool(postprocessed_action[6] != raw_action[6]),
            )
            observation = next_observation
            final_step = step
            if reward_step.terminated or reward_step.truncated:
                break
        if reward_step is None:
            raise RuntimeError("episode produced no transition")
        arrays = recorder.arrays()
        recorder.save(trace_path)
        total_steps = len(arrays["step_index"])
        result = {
            "episode_id": episode_id,
            "paired_seed": int(paired_seed),
            "environment_seed": int(paired_seed),
            "pilot_seed": int(pilot_seed),
            "diffusion_seed": int(diffusion_seed),
            "surrogate_pilot": "offline_awac_7p5k",
            "base_pilot": "offline_hybrid_awac",
            "artificial_corruption": False,
            "method": method,
            "gamma": 0.0 if method == "noassist" else global_gamma,
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
            "timeout": reward_step.termination_reason == "timeout",
            "ik_failure": reward_step.termination_reason == "ik_failure_limit",
            "policy_clip_steps": policy_clip_steps,
            "policy_clip_fraction": policy_clip_steps / total_steps,
            "adapter_rejection_count": adapter_rejection_count,
            "fallback_count": fallback_count,
            "mean_diffusion_intervention": float(np.mean(interventions)),
            "mean_translation_correction_m": float(np.mean(arrays["translation_correction_m"])),
            "mean_rotation_correction_rad": float(np.mean(arrays["rotation_correction_rad"])),
            "mean_motion_cosine_similarity": float(np.mean(arrays["motion_cosine_similarity"])),
            "retreat_translation_correction_m": float(np.mean(arrays["translation_correction_m"][arrays["active_stage"] == 4])) if np.any(arrays["active_stage"] == 4) else 0.0,
            "mean_clip_magnitude": float(np.mean(clip_magnitudes)),
            "mean_inference_ms": float(np.mean(inference_times)),
            "initial_arm_q": _json_array(initial_arm_q),
            "initial_object_xyz": _json_array(initial_object_xyz),
            "initial_goal_xyz": _json_array(initial_goal_xyz),
            "trace_path": str(trace_path.resolve()),
            "trace_length": total_steps,
            "nan_count": int(sum(np.isnan(value).sum() for value in arrays.values() if value.dtype.kind in "fc")),
            "inf_count": int(sum(np.isinf(value).sum() for value in arrays.values() if value.dtype.kind in "fc")),
            "last_step": final_step,
        }
        return result
    finally:
        env.close()


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mcnemar_exact(noassist: np.ndarray, global_values: np.ndarray) -> dict[str, Any]:
    noassist = np.asarray(noassist, bool)
    global_values = np.asarray(global_values, bool)
    b = int(np.sum(~noassist & global_values))
    c = int(np.sum(noassist & ~global_values))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            float(math.comb(discordant, k))
            for k in range(min(b, c) + 1)
        ) / (2.0 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"global_only": b, "noassist_only": c, "discordant": discordant, "p_value": p_value}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.write_text("", encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in results}):
        selected = [row for row in results if row["method"] == method]
        steps = np.asarray([row["episode_steps"] for row in selected], np.float64)
        total_steps = max(1, int(steps.sum()))
        summary_rows.append({
            "surrogate_pilot": "offline_awac_7p5k", "method": method, "N": len(selected),
            "task_success_rate": float(np.mean([row["success"] for row in selected])),
            "timeout_rate": float(np.mean([row["timeout"] for row in selected])),
            "illegal_drop_rate": float(np.mean([row["illegal_drop"] for row in selected])),
            "ik_failure_rate": float(np.mean([row["ik_failure"] for row in selected])),
            "episode_steps_mean": float(steps.mean()), "episode_steps_median": float(np.median(steps)),
            "grasp_rate": float(np.mean([row["grasp"] for row in selected])),
            "lift_rate": float(np.mean([row["lift"] for row in selected])),
            "transport_rate": float(np.mean([row["transport"] for row in selected])),
            "place_rate": float(np.mean([row["place"] for row in selected])),
            "retreat_rate": float(np.mean([row["retreat"] for row in selected])),
            "policy_clip_fraction": float(sum(row["policy_clip_steps"] for row in selected) / total_steps),
            "adapter_rejection_rate": float(sum(row["adapter_rejection_count"] for row in selected) / total_steps),
            "fallback_rate": float(sum(row["fallback_count"] for row in selected) / total_steps),
            "mean_intervention": float(np.mean([row["mean_diffusion_intervention"] for row in selected])),
            "mean_inference_ms": float(np.mean([row["mean_inference_ms"] for row in selected])),
        })
    noassist = {row["paired_seed"]: row for row in results if row["method"] == "noassist"}
    global_rows = {row["paired_seed"]: row for row in results if row["method"] == "global"}
    seeds = sorted(set(noassist) & set(global_rows))
    if not seeds:
        return summary_rows, {}
    no_success = np.asarray([noassist[seed]["success"] for seed in seeds], bool)
    global_success = np.asarray([global_rows[seed]["success"] for seed in seeds], bool)
    success_difference = global_success.astype(np.float64) - no_success.astype(np.float64)
    step_difference = np.asarray([global_rows[s]["episode_steps"] - noassist[s]["episode_steps"] for s in seeds], np.float64)
    no_drop = np.asarray([noassist[s]["illegal_drop"] for s in seeds], bool)
    global_drop = np.asarray([global_rows[s]["illegal_drop"] for s in seeds], bool)
    paired = {
        "N": len(seeds), "paired_seeds": seeds,
        "success_difference_global_minus_noassist": float(success_difference.mean()),
        "success_difference_95_bootstrap_ci": _bootstrap_ci(success_difference, 20260817),
        "mcnemar_exact": _mcnemar_exact(no_success, global_success),
        "episode_steps_difference_mean_global_minus_noassist": float(step_difference.mean()),
        "episode_steps_difference_median_global_minus_noassist": float(np.median(step_difference)),
        "episode_steps_difference_95_bootstrap_ci": _bootstrap_ci(step_difference, 20260818),
        "illegal_drop_difference_global_minus_noassist": float(global_drop.astype(np.float64).mean() - no_drop.astype(np.float64).mean()),
        "illegal_drop_paired_counts": {
            "both_illegal_drop": int(np.sum(no_drop & global_drop)),
            "neither_illegal_drop": int(np.sum(~no_drop & ~global_drop)),
            "noassist_only_illegal_drop": int(np.sum(no_drop & ~global_drop)),
            "global_only_illegal_drop": int(np.sum(~no_drop & global_drop)),
        },
    }
    return summary_rows, paired


def smoke_audit(
    results: list[dict[str, Any]],
    controller: GlobalSharedController,
    *,
    surrogate_checkpoint: Path,
    expected_rollouts: int,
) -> dict[str, Any]:
    failures: list[str] = []
    if any(row["nan_count"] or row["inf_count"] for row in results):
        failures.append("NaN/Inf occurred in a rollout")
    groups: dict[int, dict[str, dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(row["paired_seed"], {})[row["method"]] = row
        trace = load_trace(row["trace_path"])
        expected_gamma = 0.0 if row["method"] == "noassist" else GLOBAL_GAMMA
        if row["gamma"] != expected_gamma:
            failures.append(f"gamma mismatch: {row['episode_id']}")
        if len({len(value) for value in trace.values()}) != 1:
            failures.append(f"misaligned trace: {row['episode_id']}")
        clean_raw_error = float(np.max(np.abs(trace["clean_pilot_action_7"] - trace["raw_pilot_action_7"])))
        if clean_raw_error > 1e-7:
            failures.append(f"raw/clean mismatch {row['episode_id']}: {clean_raw_error}")
        if row["method"] == "noassist":
            error = float(np.max(np.abs(trace["assisted_action_7"] - trace["raw_pilot_action_7"])))
            if error > 1e-7:
                failures.append(f"No Assist action mismatch {row['episode_id']}: {error}")
        if row["method"] == "global":
            if trace["state_43"].shape[1:] != (43,) or trace["raw_pilot_action_7"].shape[1:] != (7,):
                failures.append(f"Global contract mismatch: {row['episode_id']}")
            for name in ("assisted_action_7", "clipped_assisted_action_7", "executed_action_7"):
                if not np.isfinite(trace[name]).all():
                    failures.append(f"non-finite Global {name}: {row['episode_id']}")
            if np.any(np.abs(trace["clipped_assisted_action_7"]) > 1.0 + 1e-7):
                failures.append(f"Global policy clip outside [-1,1]: {row['episode_id']}")
    reset_mismatches = 0
    for paired_seed, methods in groups.items():
        if set(methods) != {"noassist", "global"}:
            failures.append(f"missing paired method: {paired_seed}")
            continue
        for field in ("initial_arm_q", "initial_object_xyz", "initial_goal_xyz"):
            if json.loads(methods["noassist"][field]) != json.loads(methods["global"][field]):
                reset_mismatches += 1
    if reset_mismatches:
        failures.append(f"paired reset mismatch count={reset_mismatches}")
    finite_normalizers = (
        controller.observation_mean, controller.observation_std,
        controller.action_mean, controller.action_std,
    )
    if any(not np.isfinite(value).all() for value in finite_normalizers):
        failures.append("Global normalizer contains NaN/Inf")
    if np.any(controller.observation_std <= 0) or np.any(controller.action_std <= 0):
        failures.append("Global normalizer contains non-positive std")
    if len(results) != expected_rollouts:
        failures.append(f"rollout count is {len(results)}, expected {expected_rollouts}")
    noassist_success = float(np.mean([row["success"] for row in results if row["method"] == "noassist"]))
    warnings: list[str] = []
    if noassist_success > 0.90: warnings.append(f"No Assist has ceiling-effect success={noassist_success:.3f}")
    elif noassist_success < 0.10: warnings.append(f"No Assist has floor-effect success={noassist_success:.3f}")
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "nan_count": int(sum(row["nan_count"] for row in results)),
        "inf_count": int(sum(row["inf_count"] for row in results)),
        "paired_reset_mismatch_count": reset_mismatches,
        "noassist_success_rate": noassist_success,
        "pilot_checkpoint_sha256": _sha256(surrogate_checkpoint),
        "raw_equals_clean": True,
        "artificial_corruption": False,
        "noassist_action_identity_max_error": max(
            float(np.max(np.abs(load_trace(row["trace_path"])["assisted_action_7"] - load_trace(row["trace_path"])["raw_pilot_action_7"])))
            for row in results if row["method"] == "noassist"
        ),
        "global_contract": {"observation_dim": 43, "action_dim": 7, "gamma": GLOBAL_GAMMA},
        "global_input_fields": ["state_43", "raw_pilot_action_7"],
        "stage_and_milestone_leak": False,
    }


def build_metadata(
    checkpoint: Path, surrogate_checkpoint: Path, output: Path,
    controller: GlobalSharedController, device: str, seeds: tuple[int, ...],
) -> dict[str, Any]:
    commit, dirty = _git_metadata()
    return {
        "experiment_name": "exp1_global_effectiveness",
        "date": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output.resolve()),
        "global_checkpoint": str(checkpoint.resolve()),
        "global_checkpoint_sha256": _sha256(checkpoint),
        "git_commit_sha": commit,
        "git_dirty": dirty,
        "gamma": {"noassist": 0.0, "global": GLOBAL_GAMMA},
        "surrogate_pilot": {
            "type": "Offline Hybrid AWAC",
            "checkpoint_step": 7500,
            "checkpoint_path": str(surrogate_checkpoint.resolve()),
            "sha256": _sha256(surrogate_checkpoint),
            "historical_success_reference": 0.56,
        },
        "surrogate_selection_reason": "natural intermediate checkpoint between 5k=22% and 10k=86%",
        "expert_source": "Final Online AWAC 20k",
        "global_model": "Global Diffusion V2",
        "artificial_corruption": False,
        "control_dt": CONTROL_DT,
        "max_steps": MAX_STEPS,
        "pilot_types": ["offline_awac_7p5k"],
        "methods": ["noassist", "global"],
        "paired_seed_count": len(seeds),
        "paired_seeds": list(seeds),
        "seed_mapping": {
            "environment_seed": "base_seed",
            "pilot_seed": "base_seed + 100003",
            "diffusion_seed": "base_seed + 300003",
        },
        "device": device,
        "diffusion_weights": "model",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "expert_action_spec": {
            "max_translation_step_m": 0.025,
            "max_rotation_step_rad": 0.10,
            "gripper_min_m": 0.0,
            "gripper_max_m": 0.08,
        },
        "canonical_gripper": {"close": -0.25, "open": 1.0, "threshold": 0.375},
        "normalization_shapes": {
            "observation_mean": list(controller.observation_mean.shape),
            "observation_std": list(controller.observation_std.shape),
            "action_mean": list(controller.action_mean.shape),
            "action_std": list(controller.action_std.shape),
        },
        "reward_version": "awac_reward_v1",
        "termination_rules": {
            "source": "AWACRewardV1Online",
            "max_consecutive_ik_failures": MAX_CONSECUTIVE_IK_FAILURES,
            "timeout": "AWACRewardV1Online time_limit at max_steps",
        },
        "global_input_contract": {
            "observation": "normalized state_43",
            "surrogate_action": "standardized ExpertActionSpec normalized raw_action_7",
            "stage_or_milestone_input": False,
        },
        "surrogate_input_contract": {
            "policy_state": 42, "object_grasped": 1, "geometric_milestones": 5,
            "assembled_observation_dim": 48,
        },
    }


def gripper_semantics(action_pool: np.ndarray, postprocessor: GlobalActionPostprocessor) -> dict[str, Any]:
    values = np.asarray(action_pool[:, 6], np.float64)
    close, open_ = postprocessor.normalized_close, postprocessor.normalized_open
    close_count = int(np.isclose(values, close, atol=1e-6).sum())
    open_count = int(np.isclose(values, open_, atol=1e-6).sum())
    modes, counts = np.unique(np.round(values, 6), return_counts=True)
    report = postprocessor.report({"close": close_count, "open": open_count})
    report.update({
        "min": float(values.min()), "max": float(values.max()),
        "quantiles": [float(x) for x in np.quantile(values, [0, .25, .5, .75, 1])],
        "unique": [{"value": float(v), "count": int(c)} for v, c in zip(modes, counts)],
        "matches_canonical_modes": bool(close_count + open_count == len(values)),
    })
    return report


def stage_correction_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = ["APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT", "COMPLETE"]
    for result in results:
        trace = load_trace(result["trace_path"])
        for stage, name in enumerate(names):
            mask = trace["active_stage"] == stage
            if not mask.any():
                continue
            row: dict[str, Any] = {"episode_id": result["episode_id"], "surrogate_pilot": "offline_awac_7p5k", "method": result["method"], "stage": name, "frames": int(mask.sum())}
            for field in ("translation_correction_norm_normalized", "rotation_correction_norm_normalized", "translation_correction_m", "rotation_correction_rad", "motion_cosine_similarity"):
                value = np.asarray(trace[field][mask], np.float64)
                row[f"{field}_mean"] = float(value.mean())
                row[f"{field}_median"] = float(np.median(value))
                row[f"{field}_p95"] = float(np.quantile(value, .95))
            for suffix in ("mean", "median", "p95"):
                row[f"translation_correction_mm_{suffix}"] = 1000.0 * row[f"translation_correction_m_{suffix}"]
            rows.append(row)
    return rows


def aggregate_stage_corrections(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = ["APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT"]
    traces = [load_trace(row["trace_path"]) for row in results if row["method"] == "global"]
    for stage, name in enumerate(names):
        selected = {field: [] for field in (
            "translation_correction_norm_normalized", "rotation_correction_norm_normalized",
            "translation_correction_m", "rotation_correction_rad", "motion_cosine_similarity",
        )}
        for trace in traces:
            mask = trace["active_stage"] == stage
            for field in selected:
                if mask.any(): selected[field].append(np.asarray(trace[field][mask], np.float64))
        row: dict[str, Any] = {"stage": name, "frames": int(sum(len(x) for x in selected["translation_correction_m"]))}
        for field, chunks in selected.items():
            value = np.concatenate(chunks) if chunks else np.empty(0)
            for statistic, function in (("mean", np.mean), ("median", np.median), ("p95", lambda x: np.quantile(x, .95))):
                row[f"{field}_{statistic}"] = float(function(value)) if len(value) else 0.0
        for statistic in ("mean", "median", "p95"):
            row[f"translation_correction_mm_{statistic}"] = 1000.0 * row[f"translation_correction_m_{statistic}"]
        rows.append(row)
    return rows


def run_batch(
    *, output: Path, seeds: tuple[int, ...], methods: tuple[str, ...],
    surrogate_pilot: OfflineAWACSurrogatePilot,
    postprocessor: GlobalActionPostprocessor, controller: GlobalSharedController,
    global_gamma: float = GLOBAL_GAMMA,
) -> list[dict[str, Any]]:
    traces = output / "traces"; traces.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            result = run_episode(
                method=method, paired_seed=seed, pilot_seed=seed + 100_003,
                diffusion_seed=seed + 300_003, global_gamma=global_gamma,
                surrogate_pilot=surrogate_pilot, postprocessor=postprocessor,
                global_controller=controller if method == "global" else None,
                trace_path=traces / f"offline_awac_7p5k_{method}_seed_{seed}.npz",
            )
            results.append(result)
            if len(results) % 10 == 0:
                print(f"{output.name}: {len(results)} rollouts complete", flush=True)
    _write_csv(output / "episode_results.csv", results)
    _write_csv(output / "stage_correction_summary.csv", stage_correction_summary(results))
    summary, paired = summarize(results)
    _write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps({"summary": summary, "paired_statistics": paired}, indent=2) + "\n")
    return results


def choose_calibration(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if .40 <= row["success_rate"] <= .75]
    return min(eligible, key=lambda row: (abs(row["success_rate"] - .60), row["probability"])) if eligible else None


def gamma_zero_equivalence(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    exact_pass_pairs = 0
    termination_mismatch = episode_length_mismatch = 0
    max_assisted_difference = max_raw_difference = max_postprocessed_difference = max_executed_difference = 0.0
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in results: by_seed.setdefault(row["paired_seed"], {})[row["method"]] = row
    for seed, pair in by_seed.items():
        left, right = pair["noassist"], pair["global"]
        pair_failed = False
        for field in ("success", "termination_reason", "episode_steps", "illegal_drop", "timeout", "ik_failure"):
            if left[field] != right[field]:
                if field == "termination_reason": termination_mismatch += 1
                if field == "episode_steps": episode_length_mismatch += 1
                failures.append({"seed": seed, "field": field, "noassist": left[field], "global": right[field]}); pair_failed = True; break
        a, b = load_trace(left["trace_path"]), load_trace(right["trace_path"])
        fields = ("raw_pilot_action_7", "assisted_action_7", "postprocessed_action_7", "executed_action_7")
        differences = {}
        for field in fields:
            error = float(np.max(np.abs(a[field] - b[field]))) if a[field].shape == b[field].shape else float("inf")
            differences[field] = error
        max_raw_difference = max(max_raw_difference, differences["raw_pilot_action_7"])
        max_assisted_difference = max(max_assisted_difference, differences["assisted_action_7"])
        max_postprocessed_difference = max(max_postprocessed_difference, differences["postprocessed_action_7"])
        max_executed_difference = max(max_executed_difference, differences["executed_action_7"])
        if not all(a[field].shape == b[field].shape and np.array_equal(a[field], b[field]) for field in fields):
            failures.append({"seed": seed, "field": "trace_actions", "max_abs_error": differences}); pair_failed = True
        if not pair_failed: exact_pass_pairs += 1
    return {
        "status": "PASS" if exact_pass_pairs == len(by_seed) else "FAIL", "pairs": len(by_seed),
        "exact_pass_pairs": exact_pass_pairs, "termination_mismatch": termination_mismatch,
        "episode_length_mismatch": episode_length_mismatch,
        "max_raw_difference": max_raw_difference, "max_assisted_difference": max_assisted_difference,
        "max_postprocessed_difference": max_postprocessed_difference, "max_executed_difference": max_executed_difference,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 Offline-AWAC-7.5k Global gated evaluation")
    parser.add_argument("--mode", choices=("all", "formal"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--surrogate-checkpoint", type=Path, default=DEFAULT_SURROGATE_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/experiments/exp1_offline_awac75_global")
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.mode == "formal":
        raise SystemExit("Formal E1 is launched only by the all-mode gated pipeline.")
    checkpoint = args.checkpoint.expanduser().resolve()
    surrogate_checkpoint = args.surrogate_checkpoint.expanduser().resolve()
    payload = torch.load(surrogate_checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 7500 or np.asarray(payload["observation_mean"]).shape != (48,):
        raise SystemExit("Surrogate checkpoint is not the frozen 48-D Offline Hybrid AWAC step 7500")
    historical = json.loads((surrogate_checkpoint.parents[1] / "closed_loop" / "hybrid_awac_step_07500.json").read_text())
    if int(historical["task_success"]) != 56 or int(historical["episodes"]) != 100:
        raise SystemExit("Surrogate checkpoint does not match the historical 56/100 run")
    root = args.output_root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gate_root = root / f"gates_{stamp}"; gate_root.mkdir()
    controller = GlobalSharedController(checkpoint, args.device)
    surrogate = OfflineAWACSurrogatePilot(surrogate_checkpoint)
    postprocessor = GlobalActionPostprocessor(surrogate.normalized_close, surrogate.normalized_open)
    if not np.isclose(postprocessor.normalized_close, -0.25) or not np.isclose(postprocessor.normalized_open, 1.0) or not np.isclose(postprocessor.threshold, .375):
        raise SystemExit("canonical gripper contract mismatch")
    gamma0_dir = gate_root / "gamma0_equivalence"; gamma0_dir.mkdir()
    gamma0 = run_batch(output=gamma0_dir, seeds=tuple(range(2_110_000, 2_110_020)), methods=("noassist", "global"), surrogate_pilot=surrogate, postprocessor=postprocessor, controller=controller, global_gamma=0.0)
    equivalence = gamma_zero_equivalence(gamma0)
    (gamma0_dir / "gamma0_equivalence_v2.json").write_text(json.dumps(equivalence, indent=2) + "\n")
    if equivalence["status"] != "PASS" or equivalence["exact_pass_pairs"] != 20:
        raise SystemExit("gamma=0 strict equivalence gate failed")

    sanity_dir = gate_root / "noassist_sanity"; sanity_dir.mkdir()
    sanity = run_batch(output=sanity_dir, seeds=tuple(range(2_120_000, 2_120_020)), methods=("noassist",), surrogate_pilot=surrogate, postprocessor=postprocessor, controller=controller)
    sanity_summary, _ = summarize(sanity); sanity_rate = float(sanity_summary[0]["task_success_rate"])
    sanity_nan = int(sum(row["nan_count"] for row in sanity)); sanity_inf = int(sum(row["inf_count"] for row in sanity))
    canonical_sanity = all(np.isin(load_trace(row["trace_path"])["executed_action_7"][:, 6], [postprocessor.normalized_close, postprocessor.normalized_open]).all() for row in sanity)
    sanity_gate = {"status": "FAIL" if sanity_rate <= .10 or sanity_rate >= .95 or sanity_nan or sanity_inf or not canonical_sanity else "PASS", "summary": sanity_summary[0], "nan_count": sanity_nan, "inf_count": sanity_inf, "canonical_gripper": canonical_sanity}
    (sanity_dir / "sanity_gate.json").write_text(json.dumps(sanity_gate, indent=2) + "\n")
    if sanity_gate["status"] == "FAIL": raise SystemExit("7.5k NoAssist sanity is inconsistent with a medium-capability checkpoint")

    smoke_dir = gate_root / "smoke_gamma02"; smoke_dir.mkdir()
    results = run_batch(output=smoke_dir, seeds=tuple(range(2_130_000, 2_130_010)), methods=("noassist", "global"), surrogate_pilot=surrogate, postprocessor=postprocessor, controller=controller)
    audit = smoke_audit(results, controller, surrogate_checkpoint=surrogate_checkpoint, expected_rollouts=20)
    for row in results:
        trace = load_trace(row["trace_path"])
        for field in ("raw_pilot_action_7", "postprocessed_action_7", "executed_action_7"):
            if not np.isin(trace[field][:, 6], [postprocessor.normalized_close, postprocessor.normalized_open]).all():
                audit["failures"].append(f"non-canonical {field} gripper: {row['episode_id']}")
    audit.update({
        "gamma_zero_equivalence": equivalence["status"], "noassist_sanity": sanity_gate,
        "pilot_input_dim": 48, "global_observation_dim": 43, "global_action_dim": 7,
        "formal_mode_blocked": False,
    })
    global_drop_rate = float(np.mean([row["illegal_drop"] for row in results if row["method"] == "global"]))
    audit["global_illegal_drop_rate"] = global_drop_rate
    if global_drop_rate >= .80: audit["failures"].append("Global illegal-drop rate indicates an execution bug")
    audit["status"] = "FAIL" if audit["failures"] else "PASS"
    (smoke_dir / "smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    corrections = aggregate_stage_corrections(results)
    (smoke_dir / "stage_correction_aggregate.json").write_text(json.dumps(corrections, indent=2) + "\n")
    if audit["status"] == "FAIL": raise SystemExit("Smoke structural audit failed")

    formal_dir = root / f"formal_{stamp}"; formal_dir.mkdir()
    formal_seeds = tuple(range(2_200_000, 2_200_300))
    metadata = build_metadata(checkpoint, surrogate_checkpoint, formal_dir, controller, args.device, formal_seeds)
    metadata["formal_seed_range"] = [formal_seeds[0], formal_seeds[-1]]
    metadata["gates"] = {"gamma0": equivalence["status"], "sanity": sanity_gate["status"], "smoke": audit["status"]}
    (formal_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    formal = run_batch(output=formal_dir, seeds=formal_seeds, methods=("noassist", "global"), surrogate_pilot=surrogate, postprocessor=postprocessor, controller=controller)
    _summary, paired = summarize(formal)
    (formal_dir / "paired_statistics.json").write_text(json.dumps(paired, indent=2) + "\n")
    formal_audit = smoke_audit(formal, controller, surrogate_checkpoint=surrogate_checkpoint, expected_rollouts=600)
    for row in formal:
        trace = load_trace(row["trace_path"])
        for field in ("raw_pilot_action_7", "postprocessed_action_7", "executed_action_7"):
            if not np.isin(trace[field][:, 6], [postprocessor.normalized_close, postprocessor.normalized_open]).all():
                formal_audit["failures"].append(f"non-canonical {field} gripper: {row['episode_id']}")
    formal_audit.update({"gamma": GLOBAL_GAMMA, "pilot_input_dim": 48, "global_observation_dim": 43, "global_action_dim": 7})
    formal_audit["status"] = "FAIL" if formal_audit["failures"] else "PASS"
    (formal_dir / "formal_audit.json").write_text(json.dumps(formal_audit, indent=2) + "\n")
    (formal_dir / "stage_correction_aggregate.json").write_text(json.dumps(aggregate_stage_corrections(formal), indent=2) + "\n")
    print(json.dumps({"gates": {"gamma0": equivalence["status"], "sanity": sanity_gate["status"], "smoke": audit["status"]}, "formal": str(formal_dir), "formal_audit": formal_audit["status"]}, indent=2))


if __name__ == "__main__":
    main()

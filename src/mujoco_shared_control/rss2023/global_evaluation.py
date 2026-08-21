"""Autonomous closed-loop evaluation for the frozen 43D->7D Global Diffusion."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.milestones import (
    GeometricTaskPhase,
    phase_from_milestones,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor


# Historical threshold used by the frozen Global evaluator.
GRIPPER_OPEN_THRESHOLD = 0.375


class GlobalDiffusionPredictor:
    def __init__(
        self, checkpoint_path: str | Path, normalization_path: str | Path,
        *, device_name: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available() else
            "cpu" if device_name == "auto" else device_name
        )
        payload = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        config = DiffusionConfig(**payload["diffusion_config"])
        if (config.observation_dim, config.action_dim, config.num_diffusion_steps) != (43, 7, 50):
            raise ValueError("checkpoint is not the frozen 43D/7D/50-step Global Diffusion")
        self.model = RSS2023Diffusion(config).to(self.device).eval()
        self.model.load_state_dict(payload["model"])
        with np.load(normalization_path, allow_pickle=False) as stats:
            self.observation_mean = np.asarray(stats["observation_mean"], np.float32)
            self.observation_std = np.asarray(stats["observation_std"], np.float32)
            self.action_mean = np.asarray(stats["action_mean"], np.float32)
            self.action_std = np.asarray(stats["action_std"], np.float32)
        expected = {
            "observation_mean": np.asarray(payload["observation_normalizer"]["mean"]),
            "observation_std": np.asarray(payload["observation_normalizer"]["std"]),
            "action_mean": np.asarray(payload["action_normalizer"]["mean"]),
            "action_std": np.asarray(payload["action_normalizer"]["std"]),
        }
        for name, value in expected.items():
            if not np.array_equal(getattr(self, name), value):
                raise ValueError(f"external normalization differs from checkpoint: {name}")
        if self.observation_mean.shape != (43,) or self.action_mean.shape != (7,):
            raise ValueError("normalization dimensions are invalid")
        self.action_spec = ExpertActionSpec()
        self.postprocessor = GlobalActionPostprocessor.from_expert_spec(self.action_spec)
        self.generator: torch.Generator | None = None

    def reset_sampling(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self, observation_43: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation_43, np.float32)
        if observation.shape != (43,) or not np.isfinite(observation).all():
            raise ValueError("Global Diffusion observation must be finite 43D")
        if self.generator is None:
            raise RuntimeError("sampling RNG must be reset for each episode")
        normalized = torch.from_numpy(
            (observation - self.observation_mean) / self.observation_std
        ).to(self.device).unsqueeze(0)
        # The project's released RSS2023 sampler at gamma=1 is the global p(a|s)
        # sampler; no phase or milestone is passed to the model.
        action_normalized = self.model.assist(
            normalized, torch.zeros((1, 7), device=self.device), gamma=1.0,
            generator=self.generator,
        ).squeeze(0).cpu().numpy()
        return np.asarray(action_normalized * self.action_std + self.action_mean, np.float64)


def _failure_phase(milestones: np.ndarray) -> str:
    phase = phase_from_milestones(milestones)
    if phase == GeometricTaskPhase.COMPLETE:
        phase = GeometricTaskPhase.RETREAT
    return phase.name


def evaluate_episode(
    predictor: GlobalDiffusionPredictor, environment_seed: int, sampling_seed: int,
    reward_config: AWACRewardV1Config = AWACRewardV1Config(),
) -> dict[str, Any]:
    config = CollectionConfig()
    env = PickPlaceEnv(
        render_mode=None, control_timestep=config.control_timestep_s,
        max_episode_steps=config.max_steps, enable_camera=False,
    )
    adapter = ExpertCommandAdapter(env.ik_controller, predictor.action_spec)
    predictor.reset_sampling(sampling_seed)
    try:
        observation, _ = env.reset(seed=environment_seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        initial_43 = np.r_[
            env.get_policy_observation(observation),
            np.float32(bool(observation["object_grasped"])),
        ].astype(np.float32)
        reward = AWACRewardV1Online(initial_43, reward_config)
        consecutive_ik = 0
        episode_return = 0.0
        reason = "timeout"
        out_of_bounds_steps = out_of_bounds_values = 0
        policy_clip_steps = adapter_clip_steps = 0
        nan_count = inf_count = 0
        for step in range(config.max_steps):
            state_43 = np.r_[
                env.get_policy_observation(observation),
                np.float32(bool(observation["object_grasped"])),
            ].astype(np.float32)
            raw_action = predictor.sample(state_43)
            nan_count += int(np.isnan(raw_action).sum())
            inf_count += int(np.isinf(raw_action).sum())
            if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
                reason = "non_finite_diffusion_action"
                episode_return += reward_config.failure_penalty
                break
            outside = (raw_action < -1.0) | (raw_action > 1.0)
            out_of_bounds_steps += int(outside.any())
            out_of_bounds_values += int(outside.sum())
            bounded = np.clip(raw_action, -1.0, 1.0)
            policy_clip_steps += int(not np.array_equal(raw_action, bounded))
            # Preserve the historical binary gripper execution semantics.
            bounded[6] = -1.0 if bounded[6] < GRIPPER_OPEN_THRESHOLD else 1.0
            adapted = adapter.adapt(predictor.action_spec.denormalize(bounded))
            adapter_clip_steps += int(adapted.action_clipped)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, _, _, _, _ = env.step(adapted.joint_target)
            next_state_43 = np.r_[
                env.get_policy_observation(next_observation),
                np.float32(bool(next_observation["object_grasped"])),
            ].astype(np.float32)
            reward_step = reward.step(
                state_43, next_state_43,
                ik_failure=consecutive_ik >= config.max_consecutive_ik_failures,
                time_limit=step + 1 >= config.max_steps,
            )
            episode_return += reward_step.reward
            observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                reason = reward_step.termination_reason
                break
        milestones = reward.tracker.current
        success = reason == "task_success"
        return {
            "environment_seed": environment_seed, "diffusion_sampling_seed": sampling_seed,
            "success": success, "grasp": bool(milestones[0]), "lift": bool(milestones[1]),
            "transport": bool(milestones[2]), "place": bool(milestones[3]),
            "release": bool(milestones[3]), "retreat": bool(milestones[4]),
            "illegal_drop": reason == "illegal_drop",
            "ik_failure": reason == "ik_failure_limit", "timeout": reason == "timeout",
            "termination_reason": reason, "failure_phase": None if success else _failure_phase(milestones),
            "episode_return": float(episode_return), "episode_length": step + 1,
            "nan_count": nan_count, "inf_count": inf_count,
            "out_of_bounds_steps": out_of_bounds_steps,
            "out_of_bounds_values": out_of_bounds_values,
            "policy_clip_steps": policy_clip_steps,
            "adapter_clip_steps": adapter_clip_steps,
        }
    finally:
        env.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    def metric(name: str) -> dict[str, Any]:
        value = int(sum(bool(row[name]) for row in rows))
        return {"count": value, "rate": value / count}
    returns = np.asarray([row["episode_return"] for row in rows])
    lengths = np.asarray([row["episode_length"] for row in rows])
    places = sum(row["place"] for row in rows)
    return {
        "episodes": count, "success": metric("success"), "grasp": metric("grasp"),
        "lift": metric("lift"), "transport": metric("transport"), "place": metric("place"),
        "release": metric("release"), "retreat": metric("retreat"),
        "illegal_drop": metric("illegal_drop"), "ik_failure": metric("ik_failure"),
        "timeout": metric("timeout"), "average_return": float(returns.mean()),
        "place_to_success": float(sum(row["success"] for row in rows) / places) if places else 0.0,
        "episode_length": {"mean": float(lengths.mean()), "std": float(lengths.std()),
                           "min": int(lengths.min()), "max": int(lengths.max())},
        "failure_by_phase": dict(Counter(row["failure_phase"] for row in rows if not row["success"])),
        "nan_count": int(sum(row["nan_count"] for row in rows)),
        "inf_count": int(sum(row["inf_count"] for row in rows)),
        "out_of_bounds_steps": int(sum(row["out_of_bounds_steps"] for row in rows)),
        "out_of_bounds_values": int(sum(row["out_of_bounds_values"] for row in rows)),
        "policy_clip_steps": int(sum(row["policy_clip_steps"] for row in rows)),
        "adapter_clip_steps": int(sum(row["adapter_clip_steps"] for row in rows)),
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
    }


def run_evaluation(
    checkpoint: Path, normalization: Path, output: Path,
    *, smoke_seeds: range = range(1_900_000, 1_900_010),
    formal_seeds: range = range(2_000_000, 2_000_300), device: str = "auto",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    predictor = GlobalDiffusionPredictor(checkpoint, normalization, device_name=device)
    smoke_rows = []
    for index, seed in enumerate(smoke_seeds):
        smoke_rows.append(evaluate_episode(predictor, seed, 7_000_000 + seed))
        print(f"smoke {index + 1}/{len(smoke_seeds)}", flush=True)
    smoke_summary = summarize(smoke_rows)
    structural = (
        smoke_summary["nan_count"] or smoke_summary["inf_count"]
        or any(row["termination_reason"] == "non_finite_diffusion_action" for row in smoke_rows)
    )
    smoke_report = {"status": "FAIL" if structural else "PASS", "summary": smoke_summary,
                    "rows": smoke_rows}
    (output / "smoke_report.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
    if structural:
        raise RuntimeError("closed-loop smoke test found a structural policy error")

    rows = []
    for index, seed in enumerate(formal_seeds):
        rows.append(evaluate_episode(predictor, seed, 8_000_000 + seed))
        if (index + 1) % 10 == 0:
            print(f"formal {index + 1}/{len(formal_seeds)}", flush=True)
    report = {
        "policy": "Global Diffusion", "checkpoint": str(checkpoint.resolve()),
        "normalization": str(normalization.resolve()), "diffusion_steps": 50,
        "policy_observation": "diffusion_observation_43", "policy_uses_phase": False,
        "policy_uses_milestones": False, "environment_seeds": [formal_seeds.start, formal_seeds.stop - 1],
        "summary": summarize(rows), "rows": rows,
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

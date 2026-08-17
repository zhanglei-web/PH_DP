"""Deterministic Dataset Aggregation for the frozen Rule Expert v1."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import nn

from mujoco_shared_control.collection.automatic import CollectionConfig, _expert_observation
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import EpisodeContext, ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RulePickPlaceExpert
from mujoco_shared_control.sac.constrained_actor import (
    SACConstrainedGaussianActor, configure_constrained_distillation,
)
from mujoco_shared_control.sac.evaluation import evaluate_sac


PHASES = ("P1", "P2", "P3", "P4")
PHASE_MAP = {
    "PRE_GRASP": "P1", "GRASP": "P2", "TRANSPORT": "P3",
    "PLACE_AND_RELEASE": "P4", "PLACE_AND_RETREAT": "P4",
}


@dataclass(frozen=True)
class DAggerConfig:
    reward_version: str = "sac_reward_v2_candidate"
    episodes_per_round: int = 1000
    temporal_stride: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer_steps: int = 10_000
    training_seed: int = 20260815
    max_episode_steps: int = 500
    max_consecutive_ik_failures: int = 5

    def __post_init__(self) -> None:
        if self.temporal_stride < 1 or self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("stride must be positive and batch size must be positive/even")
        if self.episodes_per_round < 1 or self.optimizer_steps < 1:
            raise ValueError("episode and optimizer-step counts must be positive")


def module_checksum(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def action_is_admissible(action: np.ndarray, tolerance: float = 1e-6) -> bool:
    action = np.asarray(action)
    return bool(
        action.shape == (7,) and np.isfinite(action).all()
        and np.linalg.norm(action[:3]) <= 1.0 + tolerance
        and np.linalg.norm(action[3:6]) <= 1.0 + tolerance
        and -1.0 - tolerance <= action[6] <= 1.0 + tolerance
    )


def round_mixture_counts(batch_size: int, round_count: int) -> list[int]:
    """Exact 50% D0; remaining rows distributed as evenly as integers allow."""
    if batch_size % 2 or round_count < 1:
        raise ValueError("batch_size must be even and round_count positive")
    d0 = batch_size // 2
    remainder = batch_size - d0
    base, extra = divmod(remainder, round_count)
    return [d0] + [base + int(index < extra) for index in range(round_count)]


def load_d0(manifest: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = ManifestActorDataset(manifest, "train")
    validation = ManifestActorDataset(manifest, "validation")
    if len(dataset.entries) != 900 or len(dataset) != 115_021:
        raise ValueError("D0 must be frozen 900-episode/115021-transition train split")
    if len(validation.entries) != 100 or len(validation) != 12_817:
        raise ValueError("D0 validation must be frozen 100-episode/12817-transition split")
    states, actions = [], []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as handle:
            states.append(np.asarray(handle["observations/policy_state_42"], np.float32))
            actions.append(np.asarray(handle["actions/normalized"], np.float32))
    return np.concatenate(states), np.concatenate(actions), {
        "manifest": str(manifest.resolve()), "manifest_content_sha": dataset.manifest["content_sha256"],
        "train_episodes": 900, "train_transitions": len(dataset),
        "validation_episodes": 100, "validation_transitions": len(validation),
    }


def normalized_actor_action(actor: SACConstrainedGaussianActor, state: np.ndarray,
                            mean: torch.Tensor, std: torch.Tensor,
                            device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
        action = actor.deterministic_action(((tensor - mean) / std).unsqueeze(0))[0]
    result = action.cpu().numpy()
    if not action_is_admissible(result):
        raise RuntimeError("native constrained Actor emitted inadmissible action")
    return result


def collect_round(actor: SACConstrainedGaussianActor, observation_mean: np.ndarray,
                  observation_std: np.ndarray, seeds: Sequence[int], output: Path,
                  config: DAggerConfig, round_index: int,
                  device: torch.device | str = "cpu") -> dict[str, Any]:
    """Pure learner rollout; Rule Expert is queried but never executed."""
    device = torch.device(device); actor = actor.to(device).eval().requires_grad_(False)
    mean = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    actor_before = module_checksum(actor)
    env = PickPlaceEnv(enable_camera=False, reward_version=config.reward_version,
                       control_timestep=CollectionConfig().control_timestep_s,
                       max_episode_steps=config.max_episode_steps)
    expert = RulePickPlaceExpert(); spec = ExpertActionSpec()
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    stored: dict[str, list[Any]] = defaultdict(list)
    episodes: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter(); selected_phase_counts: Counter[str] = Counter()
    correction: dict[str, list[float]] = defaultdict(list)
    oracle_failed = irrecoverable = queried = selected = 0
    try:
        for episode_number, seed in enumerate(seeds):
            options = {"randomize_arm": True, "arm_joint_noise_scale": 1.0,
                       "randomize_object": True, "randomize_goal": True}
            obs, info = env.reset(seed=int(seed), options=options)
            episode_id = f"dagger_r{round_index}_{seed}"
            context = EpisodeContext(episode_id, "pick_box", f"dagger_v1_r{round_index}", 0,
                                     episode_number, int(seed), int(seed), int(seed), options)
            expert.reset(context); adapter.reset(obs["ee_pose"], obs["q_obs"])
            previous_oracle_command = None; previous_executed_action = None
            consecutive_ik = 0; initial_z = float(obs["object_pose"][2, 3])
            milestones = np.zeros(4, dtype=bool); reason = "time_limit"
            episode_rows: list[int] = []
            for step in range(config.max_episode_steps):
                policy_state = np.asarray(info["policy_obs"], np.float32)
                learner_action = normalized_actor_action(actor, policy_state, mean, std, device)
                phase = PHASE_MAP.get(env.sac_task.phase.name, "P4")
                expert_obs = _expert_observation(
                    episode_id, 0, step, obs, policy_state,
                    previous_oracle_command, previous_executed_action,
                )
                command = expert.predict(expert_obs)
                valid = bool(command.valid and command.control_active)
                rule_action = np.full(7, np.nan, np.float32)
                if valid:
                    candidate = spec.normalize(command.delta_pose_gripper).astype(np.float32)
                    valid = action_is_admissible(candidate)
                    if valid: rule_action = candidate
                queried += 1
                if not valid:
                    oracle_failed += 1
                    irrecoverable += int(expert.stage.name == "FAILED")
                take = bool(valid and step % config.temporal_stride == 0)
                phase_counts[phase] += 1
                if valid:
                    correction[phase].append(float(np.linalg.norm(rule_action - learner_action)))
                if take:
                    selected += 1; selected_phase_counts[phase] += 1
                    episode_rows.append(len(stored["state"]))
                    stored["state"].append(policy_state.copy())
                    stored["rule_action"].append(rule_action.copy())
                    stored["learner_action"].append(learner_action.copy())
                    stored["phase"].append(PHASES.index(phase))
                    stored["episode_index"].append(episode_number)
                    stored["timestep"].append(step)
                    stored["seed"].append(int(seed))
                adapted = adapter.adapt(spec.denormalize(learner_action))
                consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
                safety = consecutive_ik >= config.max_consecutive_ik_failures
                next_obs, _reward, terminated, truncated, next_info = env.step(
                    adapted.joint_target, true_failure=safety,
                    failure_reason="ik_failure_limit",
                )
                grasped = bool(next_obs["object_grasped"])
                obj = next_obs["object_pose"][:3, 3]; goal = next_obs["goal_pose"][:3, 3]
                milestones[0] |= grasped
                milestones[1] |= bool(milestones[0] and grasped and obj[2] - initial_z >= .10)
                milestones[2] |= bool(milestones[1] and grasped and np.linalg.norm(obj[:2]-goal[:2]) < .055)
                milestones[3] |= bool(next_info.get("successful_release", False))
                previous_oracle_command = command.delta_pose_gripper.copy() if valid else previous_oracle_command
                previous_executed_action = adapted.joint_target.copy()
                obs, info = next_obs, next_info
                if terminated or truncated:
                    reason = str(next_info.get("termination_reason") or
                                 ("time_limit" if truncated else "other_failure"))
                    break
            outcome = "success" if reason == "task_success" else reason
            for row in episode_rows: stored["outcome"].append(outcome)
            episodes.append({"episode_id": episode_id, "seed": int(seed), "length": step + 1,
                             "outcome": outcome, "success": outcome == "success",
                             "milestones": milestones.astype(int).tolist()})
    finally:
        env.close(); expert.close()
    if module_checksum(actor) != actor_before:
        raise RuntimeError("Actor changed during deterministic collection")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "state": np.asarray(stored["state"], np.float32),
        "rule_action": np.asarray(stored["rule_action"], np.float32),
        "learner_action": np.asarray(stored["learner_action"], np.float32),
        "phase": np.asarray(stored["phase"], np.int8),
        "episode_index": np.asarray(stored["episode_index"], np.int32),
        "timestep": np.asarray(stored["timestep"], np.int16),
        "seed": np.asarray(stored["seed"], np.int32),
        "outcome": np.asarray(stored["outcome"], dtype="U32"),
    }
    np.savez_compressed(output / "oracle_labels.npz", **arrays)
    outcomes = Counter(item["outcome"] for item in episodes)
    correction_stats = {}
    for phase in PHASES:
        values = np.asarray(correction[phase], float)
        correction_stats[phase] = {"count": len(values), "mean": float(values.mean()) if len(values) else None,
                                   "median": float(np.median(values)) if len(values) else None,
                                   "p95": float(np.percentile(values, 95)) if len(values) else None}
    return {
        "actor_checksum": actor_before, "episodes": len(episodes), "seeds": [int(min(seeds)), int(max(seeds))],
        "success": int(outcomes["success"]), "outcome_counts": dict(outcomes),
        "milestones": {PHASES[i]: int(sum(e["milestones"][i] for e in episodes)) for i in range(4)},
        "queried_transitions": queried, "oracle_query_success": queried - oracle_failed,
        "oracle_query_failed": oracle_failed, "irrecoverable": irrecoverable,
        "subsampled_transitions": selected, "temporal_stride": config.temporal_stride,
        "phase_distribution_all": dict(phase_counts),
        "phase_distribution_subsampled": dict(selected_phase_counts),
        "action_correction_stats": correction_stats, "episodes_detail": episodes,
    }


def load_round(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["state"], np.float32)
        actions = np.asarray(data["rule_action"], np.float32)
    if states.ndim != 2 or states.shape[1] != 42 or actions.shape != (len(states), 7):
        raise ValueError("invalid DAgger dataset shape")
    if not np.isfinite(states).all() or not all(action_is_admissible(a) for a in actions):
        raise ValueError("invalid DAgger states/actions")
    return states, actions


def train_round(actor: SACConstrainedGaussianActor, observation_mean: np.ndarray,
                observation_std: np.ndarray, d0: tuple[np.ndarray, np.ndarray],
                rounds: Sequence[tuple[np.ndarray, np.ndarray]], output: Path,
                config: DAggerConfig, device: torch.device | str = "cpu") -> dict[str, Any]:
    device = torch.device(device); actor = actor.to(device)
    trainable = configure_constrained_distillation(actor)
    log_std_before = {k: v.clone() for k, v in actor.log_std_head.state_dict().items()}
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.training_seed + len(rounds))
    mean = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    counts = round_mixture_counts(config.batch_size, len(rounds))
    checkpoints = {0, 1000, 2000, 5000, 10000}; output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    sources = [d0, *rounds]
    for step in range(config.optimizer_steps + 1):
        if step in checkpoints:
            torch.save({"format_version": "dagger_actor_v1", "actor_state_dict": actor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "optimizer_step": step,
                        "observation_mean": np.asarray(observation_mean, np.float32),
                        "observation_std": np.asarray(observation_std, np.float32),
                        "mixture_counts": counts, "config": asdict(config)},
                       output / f"actor_step_{step:05d}.pt")
        if step == config.optimizer_steps: break
        state_parts, action_parts = [], []
        for count, (states, actions) in zip(counts, sources, strict=True):
            index = rng.integers(0, len(states), size=count)
            state_parts.append(states[index]); action_parts.append(actions[index])
        state = torch.as_tensor(np.concatenate(state_parts), dtype=torch.float32, device=device)
        target = torch.as_tensor(np.concatenate(action_parts), dtype=torch.float32, device=device)
        prediction = actor.deterministic_action((state - mean) / std)
        loss = nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = float(nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm))
        optimizer.step()
        if (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": gradient})
    for key, value in log_std_before.items():
        if not torch.equal(value, actor.log_std_head.state_dict()[key]):
            raise RuntimeError("DAgger modified frozen log_std head")
    return {"mixture_counts": counts, "optimizer_steps": config.optimizer_steps,
            "history": history, "log_std_unchanged": True}


class _EvaluationCore:
    """Small read-only adapter for the established isolated SAC evaluator."""
    def __init__(self, actor: SACConstrainedGaussianActor, mean: np.ndarray,
                 std: np.ndarray, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device); self.actor = actor.to(self.device)
        self.observation_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.observation_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        self.action_spec = asdict(ExpertActionSpec())
        self.config = type("EvalConfig", (), {"gamma": 0.995})()
        self.alpha = torch.zeros((), device=self.device)

    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation.to(self.device, dtype=torch.float32) - self.observation_mean) / self.observation_std

    @torch.no_grad()
    def select_action(self, policy_state: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            raise RuntimeError("DAgger evaluation is deterministic only")
        state = torch.as_tensor(policy_state, dtype=torch.float32, device=self.device)
        return self.actor.deterministic_action(self.normalize_observation(state).unsqueeze(0))[0].cpu().numpy()


def evaluate_actor(actor: SACConstrainedGaussianActor, mean: np.ndarray, std: np.ndarray,
                   seeds: Sequence[int], device: torch.device | str = "cpu") -> dict[str, Any]:
    checksum = module_checksum(actor)
    result = evaluate_sac(_EvaluationCore(actor, mean, std, device), list(map(int, seeds)),
                          reward_version="sac_reward_v2_candidate")
    if module_checksum(actor) != checksum:
        raise RuntimeError("Actor changed during evaluation")
    return result

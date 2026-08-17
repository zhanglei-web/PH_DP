"""Online SAC v1 rollout, logging, evaluation, checkpoint, and resume protocol."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig
from mujoco_shared_control.sac.evaluation import evaluate_sac
from mujoco_shared_control.sac.replay_buffer import SACReplayBuffer
from mujoco_shared_control.sac.constrained_actor import (
    SACConstrainedGaussianActor,
    project_to_admissible,
)
from mujoco_shared_control.sac.policy_anchor import (
    InitialPolicyAnchor,
    InitialPolicyKLTrustRegion,
    InitialPolicyTrustRegionConfig,
    PolicyAnchorConfig,
)
from mujoco_shared_control.actor_bc.evaluate import _validation_arrays
from mujoco_shared_control.collection.datasets import ManifestActorDataset


ACTOR_ARTIFACT = Path(
    "outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt"
)
FORMAL_MANIFEST = Path("manifests/rule_expert_v1_formal.json")
REWARD_COMPONENTS = (
    "p1_progress", "grasp_event", "p3_progress", "p4_place_progress",
    "place_event", "retreat_progress", "success_terminal",
    "failure_terminal", "illegal_drop",
)


@dataclass(frozen=True)
class TrainingProtocol:
    total_env_steps: int = 30_000
    training_seed_start: int = 700_000
    validation_seed_start: int = 410_000
    validation_episodes: int = 20
    evaluation_steps: tuple[int, ...] = (10_000, 15_000, 20_000, 25_000, 30_000)
    checkpoint_frequency: int = 50_000
    logging_frequency: int = 1_000
    replay_seed: int = 20260813
    torch_seed: int = 20260813
    checkpoint_replay: bool = True
    critic_learning_starts: int = 10_000
    actor_learning_starts: int = 10_000
    alpha_learning_starts: int = 10_000
    deterministic_collection_until: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.critic_learning_starts <= self.actor_learning_starts:
            raise ValueError("learning-start bounds must be positive and ordered")
        if self.alpha_learning_starts != self.actor_learning_starts:
            raise ValueError("Actor and alpha must start together")
        if not 0 <= self.deterministic_collection_until <= self.critic_learning_starts:
            raise ValueError(
                "deterministic collection must end no later than Critic updates"
            )
        if tuple(sorted(set(self.evaluation_steps))) != self.evaluation_steps:
            raise ValueError("evaluation_steps must be sorted and unique")


class TrainingStage(str, Enum):
    COLLECT = "COLLECT"
    CRITIC_WARMUP = "CRITIC_WARMUP"
    FULL_SAC = "FULL_SAC"


class InteractionMode(str, Enum):
    """Action source for environment interaction, independent of update stage."""

    DETERMINISTIC_BC = "DETERMINISTIC_BC"
    DETERMINISTIC_MEAN = "DETERMINISTIC_MEAN"
    STOCHASTIC_SAC = "STOCHASTIC_SAC"


@dataclass(frozen=True)
class EntropyWarmStartConfig:
    """Match the initialized BC policy first, then return to standard SAC."""

    enabled: bool = False
    initial_target_entropy: float = -16.7
    final_target_entropy: float = -7.0
    hold_actor_updates: int = 50_000
    transition_end_actor_updates: int = 200_000

    def __post_init__(self) -> None:
        if self.initial_target_entropy >= 0 or self.final_target_entropy >= 0:
            raise ValueError("entropy targets must be negative")
        if self.hold_actor_updates < 0:
            raise ValueError("entropy hold interval must be non-negative")
        if self.transition_end_actor_updates <= self.hold_actor_updates:
            raise ValueError("entropy transition must end after hold interval")

    def target_at(self, actor_updates: int) -> float:
        if actor_updates < 0:
            raise ValueError("actor_updates must be non-negative")
        if not self.enabled:
            return self.final_target_entropy
        if actor_updates <= self.hold_actor_updates:
            return self.initial_target_entropy
        if actor_updates >= self.transition_end_actor_updates:
            return self.final_target_entropy
        progress = (
            (actor_updates - self.hold_actor_updates)
            / (self.transition_end_actor_updates - self.hold_actor_updates)
        )
        return float(
            self.initial_target_entropy
            + progress * (self.final_target_entropy - self.initial_target_entropy)
        )


@dataclass(frozen=True)
class MediumHorizonReleaseConfig:
    """Explicit 30k-100k trust-region and entropy release schedule."""

    enabled: bool = False
    start_env_step: int = 30_000
    boundaries: tuple[int, ...] = (40_000, 60_000, 80_000, 100_001)
    kl_limits: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4)
    parameter_limits: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    target_entropies: tuple[float, ...] = (-16.7, -14.0, -10.0, -7.0)

    def __post_init__(self) -> None:
        size = len(self.boundaries)
        if size == 0 or not (
            len(self.kl_limits) == len(self.parameter_limits)
            == len(self.target_entropies) == size
        ):
            raise ValueError("medium-horizon schedule lengths must match")
        if tuple(sorted(self.boundaries)) != self.boundaries:
            raise ValueError("release boundaries must be sorted")
        if self.boundaries[0] <= self.start_env_step:
            raise ValueError("first release boundary must follow start_env_step")
        if min(self.kl_limits) <= 0 or min(self.parameter_limits) <= 0:
            raise ValueError("release trust-region limits must be positive")

    def values_at(self, env_step: int) -> dict[str, float | int | str]:
        if env_step < self.start_env_step:
            raise ValueError("medium-horizon release has not started")
        for index, boundary in enumerate(self.boundaries):
            if env_step < boundary:
                return {
                    "release_stage_index": index,
                    "release_stage": f"R{index}",
                    "kl_limit": self.kl_limits[index],
                    "parameter_limit": self.parameter_limits[index],
                    "target_entropy": self.target_entropies[index],
                }
        raise ValueError("env_step exceeds configured release horizon")


@dataclass(frozen=True)
class MeanPolicyImprovementConfig:
    """Conservative post-30k curriculum that decouples mean and entropy."""

    enabled: bool = False
    start_env_step: int = 30_000
    deterministic_episode_fraction: float = 0.5
    target_entropy: float = -16.7
    freeze_log_std: bool = True
    freeze_alpha: bool = True
    stable_kl_limit: float = 1e-4
    # An anomaly guard, deliberately much looser than the former 1e-4 primary
    # constraint. Behavioral KL, not parameterization, governs normal updates.
    stable_parameter_limit: float | None = 1e-2
    promote_only_on_100_episode_evaluation: bool = True
    promotion_margin: float = 0.02
    rollback_drop: float = 0.15
    rollback_kl_factor: float = 0.3
    minimum_kl_limit: float = 1e-5

    def __post_init__(self) -> None:
        if self.start_env_step < 0:
            raise ValueError("mean-policy curriculum start must be non-negative")
        if not 0.0 <= self.deterministic_episode_fraction <= 1.0:
            raise ValueError("deterministic episode fraction must be in [0,1]")
        if self.target_entropy >= 0 or self.stable_kl_limit <= 0:
            raise ValueError("invalid mean-policy entropy or KL limit")
        if self.stable_parameter_limit is not None and self.stable_parameter_limit <= 0:
            raise ValueError("stable parameter limit must be positive or None")
        if self.promotion_margin < 0 or self.rollback_drop <= 0:
            raise ValueError("invalid promotion or rollback threshold")
        if not 0 < self.rollback_kl_factor < 1 or not 0 < self.minimum_kl_limit <= self.stable_kl_limit:
            raise ValueError("invalid rollback KL schedule")

    def kl_limit_after_rollbacks(self, rollbacks: int) -> float:
        if rollbacks < 0:
            raise ValueError("rollbacks must be non-negative")
        return max(
            self.minimum_kl_limit,
            self.stable_kl_limit * self.rollback_kl_factor ** rollbacks,
        )


def _deployed_normalized_action(adapted: Any, spec: ExpertActionSpec) -> np.ndarray:
    if adapted.accepted:
        return np.asarray(adapted.normalized, dtype=np.float32)
    # IK fallback holds the existing Cartesian target; only its safe gripper command
    # is physically deployed. Replay must not claim the rejected Cartesian delta ran.
    physical = np.zeros(7, dtype=np.float64)
    physical[6] = float(adapted.joint_target[7])
    return spec.normalize(physical).astype(np.float32)


def _action_semantics_info(
    policy_action: np.ndarray, adapted: Any, spec: ExpertActionSpec
) -> dict[str, Any]:
    """Keep attempted RL action distinct from the safety-deployed command."""
    attempted = np.asarray(policy_action, dtype=np.float32)
    if attempted.shape != (7,) or not np.isfinite(attempted).all():
        raise ValueError("policy_action must be finite and 7-D")
    deployed = _deployed_normalized_action(adapted, spec)
    return {
        "policy_action": attempted.copy(),
        "deployed_action": deployed,
        "adapter_projected": bool(adapted.action_clipped),
        "fallback_used": bool(adapted.fallback_used),
    }


def is_better_evaluation(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    def key(report: dict[str, Any]) -> tuple[float, float, float]:
        failures = report["termination_reason_counts"]
        illegal_rate = failures.get("illegal_drop", 0) / report["episodes"]
        return report["success_rate"], -illegal_rate, report["episode_return"]["mean"]
    return key(candidate) > key(incumbent)


class SACTrainer:
    def __init__(
        self, run_directory: str | Path, protocol: TrainingProtocol = TrainingProtocol(),
        core_config: SACCoreConfig = SACCoreConfig(), device: str = "cpu",
        policy_anchor_config: PolicyAnchorConfig = PolicyAnchorConfig(enabled=False),
        policy_trust_region_config: InitialPolicyTrustRegionConfig = (
            InitialPolicyTrustRegionConfig()
        ),
        entropy_warm_start_config: EntropyWarmStartConfig = EntropyWarmStartConfig(),
        medium_horizon_release_config: MediumHorizonReleaseConfig = (
            MediumHorizonReleaseConfig()
        ),
        mean_policy_improvement_config: MeanPolicyImprovementConfig = (
            MeanPolicyImprovementConfig()
        ),
    ) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory = self.run_directory / "checkpoints"
        self.checkpoint_directory.mkdir(exist_ok=True)
        self.protocol, self.core_config = protocol, core_config
        self.policy_anchor_config = policy_anchor_config
        self.policy_trust_region_config = policy_trust_region_config
        self.entropy_warm_start_config = entropy_warm_start_config
        self.medium_horizon_release_config = medium_horizon_release_config
        self.mean_policy_improvement_config = mean_policy_improvement_config
        if medium_horizon_release_config.enabled and mean_policy_improvement_config.enabled:
            raise ValueError("release experiments are mutually exclusive")
        if policy_trust_region_config.enabled and not policy_anchor_config.enabled:
            raise ValueError("policy trust region requires the initial-policy anchor")
        random.seed(protocol.torch_seed)
        np.random.seed(protocol.torch_seed)
        torch.manual_seed(protocol.torch_seed)
        # Construct optimizers before native MuJoCo workers (documented runtime order).
        self.core = SACCore(ACTOR_ARTIFACT, core_config, device)
        if not isinstance(self.core.actor, SACConstrainedGaussianActor):
            raise TypeError("clean SAC v2 requires SACConstrainedGaussianActor")
        self.policy_anchor = (
            InitialPolicyAnchor(
                policy_anchor_config,
                self.core.actor_artifact,
                self.core.observation_mean,
                self.core.observation_std,
                self.core.device,
            )
            if policy_anchor_config.enabled else None
        )
        self.policy_trust_region = (
            InitialPolicyKLTrustRegion(policy_trust_region_config, self.policy_anchor)
            if policy_trust_region_config.enabled and self.policy_anchor is not None
            else None
        )
        self.replay = SACReplayBuffer(core_config.replay_capacity, protocol.replay_seed)
        self.global_env_steps = 0
        self.gradient_updates = 0
        self.actor_updates = 0
        self.alpha_updates = 0
        self.episode_count = 0
        self.training_reset_count = 0
        self.best_evaluation: dict[str, Any] | None = None
        self.stable_evaluation: dict[str, Any] | None = None
        self.stable_anchor_promotions = 0
        self.stable_checkpoint_path: str | None = None
        self.policy_rollbacks = 0
        self._recent_metrics: list[dict[str, float]] = []
        self._recent_rewards: list[float] = []
        self._component_window: defaultdict[str, float] = defaultdict(float)
        self._policy_translation_norms: list[float] = []
        self._policy_rotation_norms: list[float] = []
        self._adapter_difference_norms: list[float] = []
        self._adapter_translation_projections = 0
        self._adapter_rotation_projections = 0
        self._adapter_gripper_clips = 0
        self._fallback_count = 0
        self._replay_policy_mismatch_count = 0
        self._translation_numeric_boundary_count = 0
        self._rotation_numeric_boundary_count = 0
        self._interaction_mode_counts = {mode.value: 0 for mode in InteractionMode}
        self._interaction_event_counts = {
            mode.value: {
                name: 0 for name in (
                    "grasp_event", "place_event", "success_terminal",
                    "failure_terminal", "illegal_drop",
                )
            }
            for mode in InteractionMode
        }
        self._last_interaction_mode: InteractionMode | None = None
        self._last_log_report: dict[str, Any] | None = None
        self._metrics_path = self.run_directory / "training_metrics.jsonl"
        self._episode_path = self.run_directory / "episode_metrics.csv"
        self._evaluation_path = self.run_directory / "evaluation_metrics.jsonl"
        self._trust_region_path = self.run_directory / "trust_region_metrics.jsonl"
        self._entropy_path = self.run_directory / "entropy_metrics.jsonl"
        self._critic_health_path = self.run_directory / "critic_health.jsonl"
        self._actor_drift_path = self.run_directory / "actor_drift.jsonl"
        validation = ManifestActorDataset(FORMAL_MANIFEST, "validation")
        validation_states, _ = _validation_arrays(validation)
        self._reference_states = torch.as_tensor(
            validation_states, dtype=torch.float32, device=self.core.device
        )
        with torch.no_grad():
            normalized = self.core.normalize_observation(self._reference_states)
            self._initial_actions = self.core.actor.deterministic_action(normalized).cpu()
            initial_mean, initial_log_std, _ = self.core.actor.distribution_stats(normalized)
            self._initial_mean = initial_mean.cpu()
            self._initial_log_std = initial_log_std.cpu()
            sampled = self.core.actor.sample_action(normalized[:1024])[0]
            if torch.any(torch.linalg.vector_norm(sampled[:, :3], dim=-1) >= 1.0):
                raise RuntimeError("constrained Actor translation action left unit ball")
            if torch.any(torch.linalg.vector_norm(sampled[:, 3:6], dim=-1) >= 1.0):
                raise RuntimeError("constrained Actor rotation action left unit ball")
        self._initial_actor_checksum = self.actor_checksum()
        self._initial_log_alpha = self.core.log_alpha.detach().clone()
        self._warmup_start_actor_checksum: str | None = None
        self._warmup_end_actor_checksum: str | None = None
        self._write_config()

    def training_stage(self, env_steps: int | None = None) -> TrainingStage:
        step = self.global_env_steps if env_steps is None else env_steps
        if step <= self.protocol.critic_learning_starts:
            return TrainingStage.COLLECT
        if (
            self.protocol.critic_learning_starts < self.protocol.actor_learning_starts
            and step <= self.protocol.actor_learning_starts
        ):
            return TrainingStage.CRITIC_WARMUP
        return TrainingStage.FULL_SAC

    def interaction_mode(self, env_steps: int | None = None) -> InteractionMode:
        """Return the policy used for the *next* environment transition.

        ``env_steps`` is the number of already completed transitions. Thus a
        cutoff of 10,000 makes transitions 1..10,000 deterministic and starts
        stochastic interaction at transition 10,001.
        """
        step = self.global_env_steps if env_steps is None else env_steps
        if step < self.protocol.deterministic_collection_until:
            return InteractionMode.DETERMINISTIC_BC
        mean_config = getattr(
            self, "mean_policy_improvement_config", MeanPolicyImprovementConfig()
        )
        if mean_config.enabled and step >= mean_config.start_env_step:
            # Episode-level deterministic/stochastic mixture.  Alternation is
            # exact for the frozen 0.5 protocol and remains deterministic on resume.
            fraction = mean_config.deterministic_episode_fraction
            cycle = 1000
            deterministic_slots = round(fraction * cycle)
            # Multiplication distributes deterministic slots across the cycle;
            # for the frozen 0.5 protocol this alternates whole episodes.
            if (self.episode_count * deterministic_slots) % cycle < deterministic_slots:
                return InteractionMode.DETERMINISTIC_MEAN
        return InteractionMode.STOCHASTIC_SAC

    def actor_checksum(self) -> str:
        digest = hashlib.sha256()
        for name, value in self.core.actor.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def actor_drift(self) -> dict[str, Any]:
        normalized = self.core.normalize_observation(self._reference_states)
        current = self.core.actor.deterministic_action(normalized).cpu()
        current_mean, current_log_std, _ = self.core.actor.distribution_stats(normalized)
        current_mean, current_log_std = current_mean.cpu(), current_log_std.cpu()
        difference = (current - self._initial_actions).abs()
        initial_variance = torch.exp(2.0 * self._initial_log_std)
        current_variance = torch.exp(2.0 * current_log_std)
        kl_per_dimension = (
            self._initial_log_std - current_log_std
            + (current_variance + (current_mean - self._initial_mean).square())
            / (2.0 * initial_variance)
            - 0.5
        )
        return {
            "samples": len(current), "normalized_mae": float(difference.mean()),
            "per_dimension_mae": difference.mean(0).tolist(),
            "xyz_physical_vector_error_m": float(
                torch.linalg.vector_norm(difference[:, :3] * .025, dim=-1).mean()
            ),
            "rotation_physical_vector_error_rad": float(
                torch.linalg.vector_norm(difference[:, 3:6] * .10, dim=-1).mean()
            ),
            "gripper_normalized_mae": float(difference[:, 6].mean()),
            "gripper_physical_mae_m": float(difference[:, 6].mean() * 0.04),
            "max_abs_error": float(difference.max()),
            "initial_policy_kl_per_dimension": float(kl_per_dimension.mean()),
            "log_std_absolute_delta": float(
                (current_log_std - self._initial_log_std).abs().mean()
            ),
            "actor_checksum": self.actor_checksum(),
            "equals_initial_checksum": self.actor_checksum() == self._initial_actor_checksum,
        }

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | int]:
        data = np.asarray(values, dtype=np.float64)
        if not len(data):
            return {"count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "count": len(data), "mean": float(data.mean()),
            "p95": float(np.quantile(data, .95)), "max": float(data.max()),
        }

    @torch.no_grad()
    def action_support_report(self) -> dict[str, Any]:
        replay_actions = torch.as_tensor(
            self.replay.action[: len(self.replay)], dtype=torch.float32, device=self.core.device
        )
        if len(replay_actions):
            projected = project_to_admissible(replay_actions)
            replay_difference = torch.linalg.vector_norm(replay_actions - projected, dim=-1)
            sample_count = min(2048, len(replay_actions))
            indices = torch.linspace(0, len(replay_actions) - 1, sample_count).long()
            observations = self.core.normalize_observation(torch.as_tensor(
                self.replay.observation[: len(self.replay)], device=self.core.device
            )[indices])
            policy = replay_actions[indices]
            projected_policy = projected[indices]
            q1_policy, q2_policy = self.core.critics(observations, policy)
            q1_projected, q2_projected = self.core.critics(observations, projected_policy)
            q_difference = {
                "samples": sample_count,
                "q1_max_abs": float((q1_policy - q1_projected).abs().max()),
                "q2_max_abs": float((q2_policy - q2_projected).abs().max()),
            }
            replay_stats = {
                "translation_norm_max": float(torch.linalg.vector_norm(replay_actions[:, :3], dim=-1).max()),
                "rotation_norm_max": float(torch.linalg.vector_norm(replay_actions[:, 3:6], dim=-1).max()),
                "projection_difference_max": float(replay_difference.max()),
            }
        else:
            replay_stats = {"translation_norm_max": 0.0, "rotation_norm_max": 0.0,
                            "projection_difference_max": 0.0}
            q_difference = {"samples": 0, "q1_max_abs": 0.0, "q2_max_abs": 0.0}
        return {
            "global_env_steps": self.global_env_steps,
            "policy_translation_norm": self._distribution(self._policy_translation_norms),
            "policy_rotation_norm": self._distribution(self._policy_rotation_norms),
            "normal_adapter_difference_l2": self._distribution(self._adapter_difference_norms),
            "adapter_translation_projection_count": self._adapter_translation_projections,
            "adapter_rotation_projection_count": self._adapter_rotation_projections,
            "adapter_gripper_clip_count": self._adapter_gripper_clips,
            "fallback_count": self._fallback_count,
            "replay_policy_mismatch_count": self._replay_policy_mismatch_count,
            "translation_numeric_boundary_count": self._translation_numeric_boundary_count,
            "rotation_numeric_boundary_count": self._rotation_numeric_boundary_count,
            "interaction_mode_counts": dict(self._interaction_mode_counts),
            "interaction_event_counts": {
                mode: dict(counts)
                for mode, counts in self._interaction_event_counts.items()
            },
            "last_interaction_mode": (
                self._last_interaction_mode.value
                if self._last_interaction_mode is not None else None
            ),
            "replay": replay_stats, "q_policy_vs_adapter": q_difference,
        }

    def _write_support_report(self) -> None:
        (self.run_directory / "action_support_stats.json").write_text(
            json.dumps(self.action_support_report(), indent=2) + "\n"
        )

    def _write_config(self) -> None:
        path = self.run_directory / "config.json"
        if not path.exists():
            path.write_text(json.dumps({
                "algorithm": "BC-initialized online off-policy SAC",
                "reward_version": "sac_reward_v1", "core": asdict(self.core_config),
                "protocol": asdict(self.protocol), "actor": self.core.specification(),
                "policy_anchor": asdict(self.policy_anchor_config),
                "policy_trust_region": asdict(self.policy_trust_region_config),
                "entropy_warm_start": asdict(self.entropy_warm_start_config),
                "medium_horizon_release": asdict(self.medium_horizon_release_config),
                "mean_policy_improvement": asdict(self.mean_policy_improvement_config),
                "initialization_reference_seeds": [300000, 300099],
                "validation_seeds": [410000, 410099], "final_test_seeds": [500000, 500099],
            }, indent=2) + "\n")

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoint_directory / name
        payload = {
            "format_version": "sac_training_pipeline_v1",
            "core": self.core.training_state_dict(), "core_config": asdict(self.core_config),
            "protocol": asdict(self.protocol), "global_env_steps": self.global_env_steps,
            "gradient_updates": self.gradient_updates, "episode_count": self.episode_count,
            "actor_updates": self.actor_updates, "alpha_updates": self.alpha_updates,
            "training_reset_count": self.training_reset_count,
            "best_evaluation": self.best_evaluation,
            "policy_anchor_config": asdict(self.policy_anchor_config),
            "policy_trust_region_config": asdict(self.policy_trust_region_config),
            "entropy_warm_start_config": asdict(self.entropy_warm_start_config),
            "medium_horizon_release_config": asdict(self.medium_horizon_release_config),
            "mean_policy_improvement_config": asdict(self.mean_policy_improvement_config),
            "stable_evaluation": self.stable_evaluation,
            "stable_anchor_promotions": self.stable_anchor_promotions,
            "stable_checkpoint_path": self.stable_checkpoint_path,
            "policy_rollbacks": self.policy_rollbacks,
            "policy_anchor_state": (
                self.policy_anchor.state_dict() if self.policy_anchor is not None else None
            ),
            "initial_actor_checksum": self._initial_actor_checksum,
            "warmup_start_actor_checksum": self._warmup_start_actor_checksum,
            "warmup_end_actor_checksum": self._warmup_end_actor_checksum,
            "action_semantics_counters": {
                "policy_translation_norms": self._policy_translation_norms,
                "policy_rotation_norms": self._policy_rotation_norms,
                "adapter_difference_norms": self._adapter_difference_norms,
                "adapter_translation_projections": self._adapter_translation_projections,
                "adapter_rotation_projections": self._adapter_rotation_projections,
                "adapter_gripper_clips": self._adapter_gripper_clips,
                "fallback_count": self._fallback_count,
                "replay_policy_mismatch_count": self._replay_policy_mismatch_count,
                "translation_numeric_boundary_count": self._translation_numeric_boundary_count,
                "rotation_numeric_boundary_count": self._rotation_numeric_boundary_count,
                "interaction_mode_counts": dict(self._interaction_mode_counts),
                "interaction_event_counts": {
                    mode: dict(counts)
                    for mode, counts in self._interaction_event_counts.items()
                },
                "last_interaction_mode": (
                    self._last_interaction_mode.value
                    if self._last_interaction_mode is not None else None
                ),
            },
            "observation_mean": self.core.observation_mean.cpu(),
            "observation_std": self.core.observation_std.cpu(),
            "actor_artifact": str(self.core.actor_artifact),
            "actor_artifact_sha256": self.core.actor_artifact_sha256,
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "replay": self.replay.state_dict() if self.protocol.checkpoint_replay else None,
        }
        temporary = path.with_suffix(path.suffix + ".inprogress")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.core.device, weights_only=False)
        if payload["format_version"] != "sac_training_pipeline_v1":
            raise ValueError("unsupported training checkpoint")
        if payload["core_config"] != asdict(self.core_config):
            raise ValueError("frozen SAC Core config mismatch")
        if payload["actor_artifact_sha256"] != self.core.actor_artifact_sha256:
            raise ValueError("Actor initialization checksum mismatch")
        checkpoint_anchor_config = payload.get(
            "policy_anchor_config", asdict(PolicyAnchorConfig(enabled=False))
        )
        if checkpoint_anchor_config != asdict(self.policy_anchor_config):
            raise ValueError("policy-anchor config mismatch")
        checkpoint_trust_region_config = payload.get(
            "policy_trust_region_config", asdict(InitialPolicyTrustRegionConfig())
        )
        if checkpoint_trust_region_config != asdict(self.policy_trust_region_config):
            raise ValueError("policy trust-region config mismatch")
        checkpoint_entropy_config = payload.get(
            "entropy_warm_start_config", asdict(EntropyWarmStartConfig())
        )
        if checkpoint_entropy_config != asdict(self.entropy_warm_start_config):
            raise ValueError("entropy warm-start config mismatch")
        checkpoint_release_config = payload.get("medium_horizon_release_config")
        if checkpoint_release_config is not None:
            if checkpoint_release_config != asdict(self.medium_horizon_release_config):
                raise ValueError("medium-horizon release config mismatch")
        elif self.medium_horizon_release_config.enabled and int(payload["global_env_steps"]) != 30_000:
            raise ValueError("release schedule may only fork the validated 30k checkpoint")
        checkpoint_mean_config = payload.get("mean_policy_improvement_config")
        if checkpoint_mean_config is not None:
            current_mean_config = asdict(self.mean_policy_improvement_config)
            # Backward-compatible extension of a local, not-yet-frozen
            # experiment checkpoint with rollback KL controls.
            checkpoint_mean_config = {
                **current_mean_config, **checkpoint_mean_config,
            }
            if checkpoint_mean_config != current_mean_config:
                raise ValueError("mean-policy curriculum config mismatch")
        elif self.mean_policy_improvement_config.enabled and int(payload["global_env_steps"]) != 30_000:
            raise ValueError("mean-policy curriculum may only fork the validated 30k checkpoint")
        self.core.load_training_state_dict(payload["core"])
        if payload["replay"] is None:
            raise ValueError("SAC v1 true resume requires checkpointed replay")
        self.replay.load_state_dict(payload["replay"])
        self.global_env_steps = int(payload["global_env_steps"])
        self.gradient_updates = int(payload["gradient_updates"])
        self.actor_updates = int(payload.get("actor_updates", self.gradient_updates))
        self.alpha_updates = int(payload.get("alpha_updates", self.gradient_updates))
        self.episode_count = int(payload["episode_count"])
        self.training_reset_count = int(payload["training_reset_count"])
        self.best_evaluation = payload["best_evaluation"]
        self.stable_evaluation = payload.get("stable_evaluation")
        self.stable_anchor_promotions = int(payload.get("stable_anchor_promotions", 0))
        self.stable_checkpoint_path = payload.get("stable_checkpoint_path")
        self.policy_rollbacks = int(payload.get("policy_rollbacks", 0))
        checkpoint_anchor_state = payload.get("policy_anchor_state")
        if self.policy_anchor is None:
            if checkpoint_anchor_state is not None:
                raise ValueError("checkpoint uses a policy anchor but trainer does not")
        else:
            if checkpoint_anchor_state is None:
                raise ValueError("policy-anchor checkpoint state is missing")
            self.policy_anchor.load_state_dict(checkpoint_anchor_state)
        if payload.get("initial_actor_checksum", self._initial_actor_checksum) != self._initial_actor_checksum:
            raise ValueError("initial Actor reference checksum mismatch")
        self._warmup_start_actor_checksum = payload.get("warmup_start_actor_checksum")
        self._warmup_end_actor_checksum = payload.get("warmup_end_actor_checksum")
        counters = payload.get("action_semantics_counters", {})
        self._policy_translation_norms = list(counters.get("policy_translation_norms", []))
        self._policy_rotation_norms = list(counters.get("policy_rotation_norms", []))
        self._adapter_difference_norms = list(counters.get("adapter_difference_norms", []))
        self._adapter_translation_projections = int(counters.get("adapter_translation_projections", 0))
        self._adapter_rotation_projections = int(counters.get("adapter_rotation_projections", 0))
        self._adapter_gripper_clips = int(counters.get("adapter_gripper_clips", 0))
        self._fallback_count = int(counters.get("fallback_count", 0))
        self._replay_policy_mismatch_count = int(counters.get("replay_policy_mismatch_count", 0))
        self._translation_numeric_boundary_count = int(counters.get("translation_numeric_boundary_count", 0))
        self._rotation_numeric_boundary_count = int(counters.get("rotation_numeric_boundary_count", 0))
        self._interaction_mode_counts = {
            mode.value: int(counters.get("interaction_mode_counts", {}).get(mode.value, 0))
            for mode in InteractionMode
        }
        saved_event_counts = counters.get("interaction_event_counts", {})
        self._interaction_event_counts = {
            mode.value: {
                name: int(saved_event_counts.get(mode.value, {}).get(name, 0))
                for name in (
                    "grasp_event", "place_event", "success_terminal",
                    "failure_terminal", "illegal_drop",
                )
            }
            for mode in InteractionMode
        }
        last_interaction_mode = counters.get("last_interaction_mode")
        self._last_interaction_mode = (
            InteractionMode(last_interaction_mode)
            if last_interaction_mode is not None else None
        )
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_random_state"].cpu())
        if self.mean_policy_improvement_config.enabled and self.global_env_steps == 30_000:
            # The validated 30k policy, not the original BC artifact, is the
            # first movable stable anchor for this fork.
            self.policy_trust_region.set_reference(self.core.actor)
        elif self.mean_policy_improvement_config.enabled:
            # Restore the last promoted reference carried by the anchor state.
            self.policy_trust_region.set_reference(self.policy_anchor.teacher)

    def _append_episode(self, row: dict[str, Any]) -> None:
        exists = self._episode_path.exists()
        with self._episode_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)

    def _restore_stable_actor(self) -> None:
        """Rollback policy only; retain newer Critic, Replay and alpha data.

        The promoted teacher is checkpointed with the policy anchor, so this
        remains valid even when a resume writes into a different run directory.
        Adam moments from the rejected proposal are discarded deliberately.
        """
        if self.policy_anchor is None:
            raise RuntimeError("stable policy rollback requires policy anchor")
        self.core.actor.load_state_dict(self.policy_anchor.teacher.state_dict())
        self.core.actor_optimizer.state.clear()
        self.policy_trust_region.set_reference(self.core.actor)
        self.policy_rollbacks += 1

    def _log_window(self) -> dict[str, Any]:
        metrics = self._recent_metrics
        report: dict[str, Any] = {
            "global_env_steps": self.global_env_steps, "episodes": self.episode_count,
            "gradient_updates": self.gradient_updates, "replay_size": len(self.replay),
            "actor_updates": self.actor_updates, "alpha_updates": self.alpha_updates,
            "training_stage": self.training_stage().value,
            "interaction_mode": (
                self._last_interaction_mode.value
                if self._last_interaction_mode is not None
                else self.interaction_mode().value
            ),
            "next_interaction_mode": self.interaction_mode().value,
            "interaction_mode_counts": dict(self._interaction_mode_counts),
            "interaction_event_counts": {
                mode: dict(counts)
                for mode, counts in self._interaction_event_counts.items()
            },
            "reward_mean": float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0,
            "reward_components": dict(self._component_window),
        }
        for key in ("critic_loss", "actor_loss", "actor_sac_loss", "anchor_loss",
                    "policy_q_mean", "entropy_term_mean",
                    "anchor_weight", "anchor_kl", "anchor_formal_kl", "anchor_replay_kl",
                    "anchor_action_mae",
                    "anchor_log_std_abs_delta", "target_entropy", "alpha_loss", "alpha",
                    "alpha_updated", "log_std_frozen",
                    "trust_region_enabled", "trust_region_triggered",
                    "trust_region_proposal_kl", "trust_region_final_kl",
                    "trust_region_final_formal_kl", "trust_region_final_replay_kl",
                    "trust_region_scale", "trust_region_backtracks",
                    "trust_region_parameter_global_relative_radius",
                    "trust_region_parameter_max_group_relative_radius",
                    "trust_region_configured_kl_limit",
                    "trust_region_configured_parameter_limit",
                    "trust_region_final_formal_action_mae",
                    "trust_region_final_replay_action_mae",
                    "release_stage_index",
                    "mean_log_prob", "entropy_gap",
                    "q1_mean", "q2_mean", "q1_std", "q2_std", "q1_min", "q1_max",
                    "q2_min", "q2_max", "target_mean", "target_std",
                    "td_error_mean", "td_error_std"):
            values = [m[key] for m in metrics if key in m]
            report[key] = float(np.mean(values)) if values else None
        trust_attempts = sum("trust_region_enabled" in metric for metric in metrics)
        projected = sum(metric.get("trust_region_triggered", 0.0) > 0.5 for metric in metrics)
        rollback = sum(metric.get("trust_region_scale", 1.0) == 0.0 for metric in metrics)
        report.update({
            "actor_update_attempts": trust_attempts,
            "accepted_updates": trust_attempts - projected,
            "projected_updates": projected,
            "rollback_updates": rollback,
            "acceptance_rate": (
                (trust_attempts - projected) / trust_attempts if trust_attempts else None
            ),
        })
        with torch.no_grad():
            if len(self.replay) >= 256:
                # Read-only logging must not advance the replay sampling RNG.
                states = self.core.normalize_observation(torch.as_tensor(
                    self.replay.observation[:256], device=self.core.device
                ))
            else:
                states = torch.zeros(256, 42, device=self.core.device)
            _, log_std, _ = self.core.actor.distribution_stats(states)
        report.update({"mean_log_std": float(log_std.mean()), "min_log_std": float(log_std.min()), "max_log_std": float(log_std.max())})
        with self._metrics_path.open("a") as stream: stream.write(json.dumps(report) + "\n")
        drift = self.actor_drift()
        for path, keys in (
            (self._trust_region_path, tuple(key for key in report if key.startswith("trust_") or key in ("global_env_steps", "actor_update_attempts", "accepted_updates", "projected_updates", "rollback_updates", "acceptance_rate"))),
            (self._entropy_path, ("global_env_steps", "target_entropy", "alpha", "mean_log_prob", "entropy_gap", "mean_log_std", "min_log_std", "max_log_std")),
            (self._critic_health_path, ("global_env_steps", "critic_loss", "q1_mean", "q2_mean", "q1_std", "q2_std", "q1_min", "q1_max", "q2_min", "q2_max", "target_mean", "target_std", "td_error_mean", "td_error_std")),
        ):
            with path.open("a") as stream:
                stream.write(json.dumps({key: report.get(key) for key in keys}) + "\n")
        with self._actor_drift_path.open("a") as stream:
            stream.write(json.dumps({"global_env_steps": self.global_env_steps, **drift}) + "\n")
        (self.run_directory / "replay_outcomes.json").write_text(json.dumps({
            "global_env_steps": self.global_env_steps,
            "interaction_mode_counts": self._interaction_mode_counts,
            "interaction_event_counts": self._interaction_event_counts,
            "deterministic_bc_success_episodes": self._interaction_event_counts[
                InteractionMode.DETERMINISTIC_BC.value
            ]["success_terminal"],
            "online_sac_success_episodes": self._interaction_event_counts[
                InteractionMode.STOCHASTIC_SAC.value
            ]["success_terminal"],
            "online_deterministic_success_episodes": self._interaction_event_counts[
                InteractionMode.DETERMINISTIC_MEAN.value
            ]["success_terminal"],
        }, indent=2) + "\n")
        self._last_log_report = report
        self._recent_metrics.clear(); self._recent_rewards.clear(); self._component_window.clear()
        return report

    def evaluate(self, episodes: int | None = None, *, consider_best: bool = True) -> dict[str, Any]:
        report = evaluate_sac(self.core, list(range(
            self.protocol.validation_seed_start,
            self.protocol.validation_seed_start + (episodes or self.protocol.validation_episodes),
        )))
        report["global_env_steps"] = self.global_env_steps
        report["gradient_updates"] = self.gradient_updates
        report["actor_updates"] = self.actor_updates
        report["alpha_updates"] = self.alpha_updates
        report["training_stage"] = self.training_stage().value
        report["next_interaction_mode"] = self.interaction_mode().value
        report["interaction_mode_counts"] = dict(self._interaction_mode_counts)
        report["interaction_event_counts"] = {
            mode: dict(counts)
            for mode, counts in self._interaction_event_counts.items()
        }
        report["replay_size"] = len(self.replay)
        report["actor_drift"] = self.actor_drift()
        report["warmup_actor_checksums"] = {
            "initial": self._initial_actor_checksum,
            "start": self._warmup_start_actor_checksum,
            "end": self._warmup_end_actor_checksum,
        }
        report["training_window"] = self._last_log_report
        report["action_support"] = self.action_support_report()
        report["evaluation_scope"] = f"validation_{episodes or self.protocol.validation_episodes}"
        with self._evaluation_path.open("a") as stream: stream.write(json.dumps(report) + "\n")
        if consider_best and is_better_evaluation(report, self.best_evaluation):
            self.best_evaluation = report
            self.save_checkpoint("best.pt")
        if (
            self.mean_policy_improvement_config.enabled
            and (episodes or self.protocol.validation_episodes) == 100
        ):
            stable_rate = (
                self.stable_evaluation["success_rate"]
                if self.stable_evaluation is not None else float("-inf")
            )
            if report["success_rate"] >= stable_rate + self.mean_policy_improvement_config.promotion_margin:
                self.stable_evaluation = report
                self.stable_anchor_promotions += 1
                self.policy_trust_region.set_reference(self.core.actor)
                self.stable_checkpoint_path = str(self.checkpoint_directory / "stable.pt")
                self.save_checkpoint("stable.pt")
                report["stable_anchor_decision"] = "PROMOTE"
            elif report["success_rate"] <= stable_rate - self.mean_policy_improvement_config.rollback_drop:
                report["stable_anchor_decision"] = "ROLLBACK_REQUIRED"
                self._restore_stable_actor()
                report["stable_anchor_decision"] = "ROLLED_BACK"
            else:
                report["stable_anchor_decision"] = "HOLD"
        return report

    def train(self, target_env_steps: int | None = None) -> dict[str, Any]:
        target = target_env_steps or self.protocol.total_env_steps
        if target < self.global_env_steps:
            raise ValueError("target_env_steps precedes checkpoint state")
        env = PickPlaceEnv(
            enable_camera=False, reward_version="sac_reward_v1", control_timestep=.05,
            max_episode_steps=CollectionConfig().max_steps,
        )
        assert env.reward_version == "sac_reward_v1"
        spec = ExpertActionSpec(**self.core.action_spec)
        adapter = ExpertCommandAdapter(env.ik_controller, spec)
        episode_return = episode_discounted = 0.0
        episode_length = 0
        try:
            seed = self.protocol.training_seed_start + self.training_reset_count
            self.training_reset_count += 1
            obs, info = env.reset(seed=seed)
            assert info["policy_obs"].shape == (42,)
            adapter.reset(obs["ee_pose"], obs["q_obs"])
            consecutive_ik = 0
            while self.global_env_steps < target:
                policy_state = info["policy_obs"]
                interaction_mode = self.interaction_mode()
                deterministic_interaction = (
                    interaction_mode != InteractionMode.STOCHASTIC_SAC
                )
                actor_action = self.core.select_action(
                    policy_state, deterministic=deterministic_interaction
                )
                self._last_interaction_mode = interaction_mode
                self._interaction_mode_counts[interaction_mode.value] += 1
                translation_norm = float(np.linalg.norm(actor_action[:3]))
                rotation_norm = float(np.linalg.norm(actor_action[3:6]))
                if translation_norm > 1.0 + 1e-6 or rotation_norm > 1.0 + 1e-6:
                    raise RuntimeError("native constrained policy produced inadmissible action")
                self._translation_numeric_boundary_count += int(translation_norm >= 1.0)
                self._rotation_numeric_boundary_count += int(rotation_norm >= 1.0)
                self._policy_translation_norms.append(translation_norm)
                self._policy_rotation_norms.append(rotation_norm)
                adapted = adapter.adapt(spec.denormalize(actor_action))
                self._fallback_count += int(adapted.fallback_used)
                self._adapter_translation_projections += int(
                    not np.allclose(adapted.requested[:3], adapted.clipped[:3])
                )
                self._adapter_rotation_projections += int(
                    not np.allclose(adapted.requested[3:6], adapted.clipped[3:6])
                )
                self._adapter_gripper_clips += int(
                    not np.isclose(adapted.requested[6], adapted.clipped[6])
                )
                if adapted.accepted:
                    self._adapter_difference_norms.append(float(np.linalg.norm(
                        np.asarray(adapted.normalized) - np.asarray(actor_action)
                    )))
                consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
                safety = consecutive_ik >= CollectionConfig().max_consecutive_ik_failures
                next_obs, reward, terminated, truncated, next_info = env.step(
                    adapted.joint_target, true_failure=safety,
                    failure_reason="ik_failure_limit",
                )
                if next_info.get("reward_version") != "sac_reward_v1":
                    raise RuntimeError("training environment is not using sac_reward_v1")
                next_info.update(_action_semantics_info(actor_action, adapted, spec))
                # RL attribution is to the agent-selected action. A safety fallback
                # may execute another command, but must never overwrite Replay.
                replay_index = self.replay.position
                self.replay.add(
                    policy_state, np.asarray(actor_action, dtype=np.float32), reward,
                    next_info["policy_obs"], terminated, truncated,
                )
                if not np.array_equal(self.replay.action[replay_index], np.asarray(actor_action, np.float32)):
                    self._replay_policy_mismatch_count += 1
                    raise RuntimeError("Replay action differs from agent-selected policy_action")
                self.global_env_steps += 1
                episode_return += reward
                episode_discounted += self.core_config.gamma ** episode_length * reward
                episode_length += 1
                self._recent_rewards.append(reward)
                for key, value in next_info["reward_components"].items():
                    if key in REWARD_COMPONENTS:
                        self._component_window[key] += float(value)
                    if (
                        key in self._interaction_event_counts[interaction_mode.value]
                        and float(value) != 0.0
                    ):
                        self._interaction_event_counts[interaction_mode.value][key] += 1
                stage = self.training_stage()
                if self.global_env_steps == self.protocol.critic_learning_starts:
                    self._warmup_start_actor_checksum = self.actor_checksum()
                if stage != TrainingStage.COLLECT:
                    batch = self.replay.sample(self.core_config.batch_size, self.core.device)
                    # There is exactly one SAC temperature.  Entropy warm-start
                    # changes the target used by the alpha loss; it must never
                    # create a second, fixed coefficient in the Bellman target.
                    metrics = self.core.update_critics(batch)
                    if stage == TrainingStage.FULL_SAC:
                        anchor_weight = self.policy_anchor_config.weight_at(self.actor_updates)
                        release_values = None
                        if self.medium_horizon_release_config.enabled:
                            release_values = self.medium_horizon_release_config.values_at(
                                max(self.medium_horizon_release_config.start_env_step,
                                    self.global_env_steps - 1)
                            )
                        target_entropy = (
                            float(release_values["target_entropy"])
                            if release_values is not None
                            else self.entropy_warm_start_config.target_at(self.actor_updates)
                        )
                        mean_curriculum = self.mean_policy_improvement_config.enabled
                        if mean_curriculum:
                            target_entropy = self.mean_policy_improvement_config.target_entropy
                        metrics.update(self.core.update_actor_and_alpha(
                            batch,
                            policy_anchor=self.policy_anchor,
                            anchor_weight=anchor_weight,
                            target_entropy=target_entropy,
                            update_alpha=not (
                                mean_curriculum
                                and self.mean_policy_improvement_config.freeze_alpha
                            ),
                            freeze_log_std=(
                                mean_curriculum
                                and self.mean_policy_improvement_config.freeze_log_std
                            ),
                        ))
                        if self.policy_trust_region is not None:
                            metrics.update(self.policy_trust_region.project(
                                self.core.actor,
                                replay_states=self.core.normalize_observation(
                                    batch.observation
                                ),
                                max_kl=(
                                    self.mean_policy_improvement_config.kl_limit_after_rollbacks(
                                        self.policy_rollbacks
                                    )
                                    if mean_curriculum else (
                                        float(release_values["kl_limit"])
                                        if release_values is not None else None
                                    )
                                ),
                                max_parameter_relative_radius=(
                                    self.mean_policy_improvement_config.stable_parameter_limit
                                    if mean_curriculum else (
                                        float(release_values["parameter_limit"])
                                        if release_values is not None else None
                                    )
                                ),
                            ))
                        if release_values is not None:
                            metrics["release_stage_index"] = float(
                                release_values["release_stage_index"]
                            )
                        self.actor_updates += 1
                        if not (
                            mean_curriculum
                            and self.mean_policy_improvement_config.freeze_alpha
                        ):
                            self.alpha_updates += 1
                    else:
                        metrics["alpha"] = float(self.core.alpha.detach())
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError(f"non-finite SAC metrics: {metrics}")
                    if max(abs(metrics["q1_mean"]), abs(metrics["q2_mean"]), abs(metrics["target_mean"])) > 1e6:
                        raise FloatingPointError(f"SAC Q magnitude exceeded safety limit: {metrics}")
                    if not 1e-8 < metrics["alpha"] < 1e3:
                        raise FloatingPointError(f"SAC alpha outside safety interval: {metrics}")
                    self.gradient_updates += 1
                    self._recent_metrics.append(metrics)
                if self.global_env_steps == self.protocol.actor_learning_starts:
                    self._warmup_end_actor_checksum = self.actor_checksum()
                    if self._warmup_end_actor_checksum != self._warmup_start_actor_checksum:
                        raise RuntimeError("Actor changed during critic-only warmup")
                    if not torch.equal(self.core.log_alpha.detach(), self._initial_log_alpha):
                        raise RuntimeError("alpha changed during critic-only warmup")
                if terminated or truncated:
                    self.episode_count += 1
                    self._append_episode({
                        "episode": self.episode_count, "seed": seed,
                        "end_env_step": self.global_env_steps, "length": episode_length,
                        "return": episode_return, "discounted_return": episode_discounted,
                        "terminated": terminated, "truncated": truncated,
                        "termination_reason": next_info["termination_reason"],
                    })
                    seed = self.protocol.training_seed_start + self.training_reset_count
                    self.training_reset_count += 1
                    obs, info = env.reset(seed=seed)
                    adapter.reset(obs["ee_pose"], obs["q_obs"])
                    consecutive_ik = 0; episode_return = episode_discounted = 0.0; episode_length = 0
                else:
                    obs, info = next_obs, next_info
                if self.global_env_steps % self.protocol.logging_frequency == 0:
                    window = self._log_window()
                    print(json.dumps(window), flush=True)
                if self.global_env_steps in self.protocol.evaluation_steps:
                    self.evaluate()
                    if (
                        self.mean_policy_improvement_config.enabled
                        or self.global_env_steps in (50_000, 60_000, 80_000, 100_000)
                    ):
                        self.evaluate(100, consider_best=False)
                    self._write_support_report()
                    self.save_checkpoint("latest.pt")
                if self.global_env_steps % self.protocol.checkpoint_frequency == 0:
                    self.save_checkpoint(f"step_{self.global_env_steps:09d}.pt")
            self.save_checkpoint("latest.pt")
            self._write_support_report()
            return {"global_env_steps": self.global_env_steps, "gradient_updates": self.gradient_updates,
                    "actor_updates": self.actor_updates, "alpha_updates": self.alpha_updates,
                    "episode_count": self.episode_count, "replay_size": len(self.replay),
                    "best_evaluation": self.best_evaluation}
        finally:
            env.close()

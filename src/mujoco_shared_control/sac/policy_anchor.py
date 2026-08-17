"""Behavior-preserving policy anchor for BC-initialized online SAC.

The anchor is deliberately independent of Replay.  It samples only the frozen
formal Actor-BC training split and penalizes drift from the initialized
constrained Gaussian policy.  Integration with the SAC update loop is kept
outside this module so that the existing reward, Critic target, entropy loss,
and Replay action semantics remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from mujoco_shared_control.actor_bc.model import ACTOR_ACTION_DIM, POLICY_STATE_DIM
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor


FORMAL_MANIFEST = Path(__file__).resolve().parents[3] / "manifests/rule_expert_v1_formal.json"
EXPECTED_TRAIN_EPISODES = 900
EXPECTED_TRAIN_TRANSITIONS = 115_021


@dataclass(frozen=True)
class PolicyAnchorConfig:
    """Frozen first policy-anchor schedule; ``actor_updates`` is zero based."""

    enabled: bool = True
    weight: float = 0.1
    hold_actor_updates: int = 50_000
    decay_end_actor_updates: int = 200_000
    batch_size: int = 256
    seed: int = 20260813
    manifest_path: str = str(FORMAL_MANIFEST)

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("policy-anchor weight must be non-negative")
        if self.hold_actor_updates < 0:
            raise ValueError("hold_actor_updates must be non-negative")
        if self.decay_end_actor_updates <= self.hold_actor_updates:
            raise ValueError("decay must end after the hold interval")
        if self.batch_size <= 0:
            raise ValueError("policy-anchor batch_size must be positive")

    def weight_at(self, actor_updates: int) -> float:
        """Return 0.1 through 50k, then linearly decay to zero at 200k."""
        if actor_updates < 0:
            raise ValueError("actor_updates must be non-negative")
        if not self.enabled or self.weight == 0.0:
            return 0.0
        if actor_updates <= self.hold_actor_updates:
            return float(self.weight)
        if actor_updates >= self.decay_end_actor_updates:
            return 0.0
        remaining = self.decay_end_actor_updates - actor_updates
        duration = self.decay_end_actor_updates - self.hold_actor_updates
        return float(self.weight * remaining / duration)


@dataclass(frozen=True)
class InitialPolicyTrustRegionConfig:
    """Hard empirical KL constraint around the initialized policy.

    The first revision is disabled by default.  A sanity experiment must opt in
    explicitly with ``max_kl=0.01``; this prevents an unreviewed trust region
    from silently changing the existing SAC baseline.
    """

    enabled: bool = False
    max_kl: float = 0.01
    backtrack_ratio: float = 0.5
    max_backtracks: int = 24
    numerical_tolerance: float = 1e-6
    # Maximum relative L2 displacement from the initialized Actor, applied to
    # each architectural group (trunk, mean head, and log-std head).  This
    # catches large internal parameter compensation that an empirical KL batch
    # can miss.  ``None`` explicitly disables the parameter-space guard while
    # retaining the KL constraint.
    max_parameter_relative_radius: float | None = 0.01
    parameter_norm_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_kl <= 0.0:
            raise ValueError("trust-region max_kl must be positive")
        if not 0.0 < self.backtrack_ratio < 1.0:
            raise ValueError("backtrack_ratio must be in (0,1)")
        if self.max_backtracks <= 0:
            raise ValueError("max_backtracks must be positive")
        if self.numerical_tolerance < 0.0:
            raise ValueError("numerical_tolerance must be non-negative")
        if (
            self.max_parameter_relative_radius is not None
            and self.max_parameter_relative_radius <= 0.0
        ):
            raise ValueError("max_parameter_relative_radius must be positive or None")
        if self.parameter_norm_epsilon <= 0.0:
            raise ValueError("parameter_norm_epsilon must be positive")


def _numpy_normalization(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (POLICY_STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape (42,)")
    return result


def _load_policy_states(dataset: ManifestActorDataset) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            chunks.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
    if not chunks:
        raise ValueError("formal Actor training split is empty")
    return np.concatenate(chunks, axis=0)


def _module_checksum(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class InitialPolicyAnchor:
    """KL anchor from a current Actor to its frozen constrained-v2 initializer.

    Both policies use the same invertible constrained transform.  Consequently,
    KL in their pre-transform Gaussian space equals KL between their transformed
    action distributions, while remaining cheap and numerically stable.
    """

    def __init__(
        self,
        config: PolicyAnchorConfig,
        actor_artifact: str | Path,
        observation_mean: np.ndarray | torch.Tensor,
        observation_std: np.ndarray | torch.Tensor,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor_artifact = Path(actor_artifact).expanduser().resolve()
        payload: dict[str, Any] = torch.load(
            self.actor_artifact, map_location="cpu", weights_only=False
        )
        if payload.get("format_version") != "sac_constrained_actor_v2_full_mean_path_distilled":
            raise ValueError("policy anchor requires constrained SAC Actor v2 artifact")

        mean = _numpy_normalization(observation_mean, "observation_mean")
        std = _numpy_normalization(observation_std, "observation_std")
        if np.any(std <= 0.0):
            raise ValueError("observation_std must be positive")
        artifact_mean = _numpy_normalization(payload["observation_mean"], "artifact mean")
        artifact_std = _numpy_normalization(payload["observation_std"], "artifact std")
        if not np.array_equal(mean, artifact_mean) or not np.array_equal(std, artifact_std):
            raise ValueError("policy anchor normalization differs from Actor artifact")

        # Constructing an nn.Module consumes the global Torch RNG even though
        # its random initialization is immediately overwritten.  Preserve that
        # RNG so merely enabling the anchor cannot alter the online exploration
        # stream before the first Actor update (required for a clean A/B).
        with torch.random.fork_rng(devices=[]):
            self.teacher = SACConstrainedGaussianActor().to(self.device)
        self.teacher.load_state_dict(payload["actor_state_dict"])
        self.teacher.eval().requires_grad_(False)
        self.teacher_checksum = _module_checksum(self.teacher)

        manifest_path = Path(config.manifest_path).expanduser().resolve()
        dataset = ManifestActorDataset(manifest_path, "train", verify_checksums=False)
        if len(dataset.entries) != EXPECTED_TRAIN_EPISODES or len(dataset) != EXPECTED_TRAIN_TRANSITIONS:
            raise ValueError(
                "policy anchor requires the frozen 900-episode/115021-transition train split"
            )
        if any(entry.outcome != "success" or entry.variant != "nominal" for entry in dataset.entries):
            raise ValueError("policy anchor train split contains non-nominal-success data")
        raw_states = _load_policy_states(dataset)
        if raw_states.shape != (EXPECTED_TRAIN_TRANSITIONS, POLICY_STATE_DIM):
            raise ValueError("policy anchor policy_state array has unexpected shape")
        normalized = (raw_states - mean) / std
        if not np.isfinite(normalized).all():
            raise ValueError("policy anchor normalized states contain NaN/Inf")
        # Keep this CPU resident (~18.4 MiB); only sampled batches move to the
        # training device, avoiding an unnecessary persistent GPU allocation.
        self.states = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
        self.episode_count = len(dataset.entries)
        self.transition_count = len(dataset)
        self.manifest_path = manifest_path
        self.rng = np.random.default_rng(config.seed)
        self.batches_sampled = 0

    def _sample_states(self) -> torch.Tensor:
        replace = self.config.batch_size > self.transition_count
        indices = self.rng.choice(
            self.transition_count, size=self.config.batch_size, replace=replace
        )
        self.batches_sampled += 1
        return self.states[torch.from_numpy(indices.astype(np.int64))].to(self.device)

    def sample_state_mixture(
        self, replay_states: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """Return one fixed formal/Replay mixture and its formal row count."""
        formal_states = self._sample_states()
        formal_count = len(formal_states)
        if replay_states is not None:
            replay_states = replay_states.detach().to(self.device, dtype=torch.float32)
            if replay_states.ndim != 2 or replay_states.shape[1] != POLICY_STATE_DIM:
                raise ValueError("normalized Replay states must have shape [B,42]")
            if not torch.isfinite(replay_states).all():
                raise ValueError("normalized Replay states contain NaN/Inf")
            states = torch.cat((formal_states, replay_states), dim=0)
        else:
            states = formal_states
        return states, formal_count

    def loss_and_metrics_on_states(
        self,
        current_actor: SACConstrainedGaussianActor,
        states: torch.Tensor,
        formal_count: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate ``KL(current||initial)/7`` on an already fixed mixture."""
        if not isinstance(current_actor, SACConstrainedGaussianActor):
            raise TypeError("policy anchor requires SACConstrainedGaussianActor")
        if states.ndim != 2 or states.shape[1] != POLICY_STATE_DIM:
            raise ValueError("policy-anchor states must have shape [B,42]")
        if not 0 < formal_count <= len(states):
            raise ValueError("formal_count must identify a non-empty prefix")
        if not torch.isfinite(states).all():
            raise ValueError("policy-anchor states contain NaN/Inf")
        with torch.no_grad():
            teacher_mean, teacher_log_std, _ = self.teacher.distribution_stats(states)
            teacher_action = self.teacher.deterministic_action(states)
        current_mean, current_log_std, _ = current_actor.distribution_stats(states)
        current_variance = torch.exp(2.0 * current_log_std)
        teacher_variance = torch.exp(2.0 * teacher_log_std)
        kl_per_dimension = (
            teacher_log_std - current_log_std
            + (current_variance + (current_mean - teacher_mean).square())
            / (2.0 * teacher_variance)
            - 0.5
        )
        loss = kl_per_dimension.sum(dim=-1).mean() / float(ACTOR_ACTION_DIM)
        formal_kl = kl_per_dimension[:formal_count].sum(dim=-1).mean() / float(ACTOR_ACTION_DIM)
        replay_kl = (
            kl_per_dimension[formal_count:].sum(dim=-1).mean() / float(ACTOR_ACTION_DIM)
            if len(states) > formal_count else torch.zeros((), device=self.device)
        )

        with torch.no_grad():
            current_action = current_actor.deterministic_action(states)
            action_error = (current_action - teacher_action).abs()
            log_std_error = (current_log_std - teacher_log_std).abs()
            formal_action_error = action_error[:formal_count]
            replay_action_error = action_error[formal_count:]
            metrics = {
                "anchor_kl": float(loss.detach()),
                "anchor_formal_kl": float(formal_kl.detach()),
                "anchor_replay_kl": float(replay_kl.detach()),
                "anchor_formal_samples": float(formal_count),
                "anchor_replay_samples": float(len(states) - formal_count),
                "anchor_action_mae": float(action_error.mean()),
                "anchor_formal_action_mae": float(formal_action_error.mean()),
                "anchor_replay_action_mae": (
                    float(replay_action_error.mean()) if len(replay_action_error) else 0.0
                ),
                "anchor_log_std_abs_delta": float(log_std_error.mean()),
                "anchor_teacher_log_std_mean": float(teacher_log_std.mean()),
                "anchor_current_log_std_mean": float(current_log_std.mean()),
            }
            metrics.update({
                f"anchor_action_mae_dim_{dimension}": float(action_error[:, dimension].mean())
                for dimension in range(ACTOR_ACTION_DIM)
            })
        return loss, metrics

    def loss_and_metrics(
        self,
        current_actor: SACConstrainedGaussianActor,
        replay_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return unweighted ``KL(current||initial)/7`` on a state mixture.

        Every call contains a formal BC-train batch.  During online SAC the
        caller also supplies the *normalized* Replay observation batch.  This
        closes the gap where the policy stayed close on demonstrations but
        changed drastically immediately after a small closed-loop deviation.
        No expert action or Critic target is consumed here.
        """
        states, formal_count = self.sample_state_mixture(replay_states)
        return self.loss_and_metrics_on_states(current_actor, states, formal_count)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "initial_policy_anchor_v1",
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "batches_sampled": self.batches_sampled,
            "teacher_checksum": self.teacher_checksum,
            "manifest_path": str(self.manifest_path),
            "teacher_state_dict": self.teacher.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != "initial_policy_anchor_v1":
            raise ValueError("unsupported policy-anchor state format")
        if "teacher_state_dict" in state:
            self.teacher.load_state_dict(state["teacher_state_dict"])
            self.teacher.eval().requires_grad_(False)
            self.teacher_checksum = _module_checksum(self.teacher)
        if state.get("teacher_checksum") != self.teacher_checksum:
            raise ValueError("policy-anchor teacher checksum mismatch")
        if Path(state.get("manifest_path", "")).resolve() != self.manifest_path:
            raise ValueError("policy-anchor manifest mismatch")
        batches_sampled = int(state["batches_sampled"])
        if batches_sampled < 0:
            raise ValueError("invalid policy-anchor batch count")
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self.batches_sampled = batches_sampled

    @torch.no_grad()
    def set_reference(self, actor: SACConstrainedGaussianActor) -> None:
        """Promote an evaluated Actor to the behavioral anchor."""
        self.teacher.load_state_dict(actor.state_dict())
        self.teacher.eval().requires_grad_(False)
        self.teacher_checksum = _module_checksum(self.teacher)


class InitialPolicyKLTrustRegion:
    """Backtrack an Actor proposal onto an empirical initial-policy KL ball.

    Projection follows the line from the immutable initialized Actor to the
    proposal.  All candidates are evaluated on exactly the same sampled formal
    and current Replay states, so a projection decision cannot be an artifact
    of changing batches.  Replay contents, the teacher, and action semantics are
    never mutated.
    """

    def __init__(
        self,
        config: InitialPolicyTrustRegionConfig,
        policy_anchor: InitialPolicyAnchor,
    ) -> None:
        self.config = config
        self.policy_anchor = policy_anchor
        self.initial_parameters = {
            name: value.detach().clone()
            for name, value in policy_anchor.teacher.state_dict().items()
        }
        self.initial_checksum = policy_anchor.teacher_checksum

    @torch.no_grad()
    def set_reference(self, actor: SACConstrainedGaussianActor) -> None:
        """Move the trust-region centre only after external evaluation approval."""
        self.policy_anchor.set_reference(actor)
        self.initial_parameters = {
            name: value.detach().clone() for name, value in actor.state_dict().items()
        }
        self.initial_checksum = self.policy_anchor.teacher_checksum

    @staticmethod
    def _parameter_group(name: str) -> str:
        """Map the frozen Actor parameter names to stable trust-region groups."""
        if name.startswith("trunk."):
            return "trunk"
        if name.startswith("mean_head."):
            return "mean_head"
        if name.startswith("log_std_head."):
            return "log_std_head"
        # Fail closed if the Actor later gains a parameter outside the frozen
        # v2 architecture.  Silently pooling a new module could weaken the
        # intended per-group radius constraint.
        raise ValueError(f"unrecognized constrained-Actor parameter group: {name}")

    @torch.no_grad()
    def _parameter_radius_metrics(
        self,
        current_actor: SACConstrainedGaussianActor,
    ) -> dict[str, float]:
        """Return initial-centered global and per-group relative L2 radii."""
        current = current_actor.state_dict()
        if current.keys() != self.initial_parameters.keys():
            raise ValueError("trust-region Actor state does not match initialized policy")

        group_delta_squared: dict[str, float] = {}
        group_initial_squared: dict[str, float] = {}
        for name, value in current.items():
            if not value.is_floating_point():
                continue
            group = self._parameter_group(name)
            initial = self.initial_parameters[name].to(value.device)
            delta = (value - initial).to(dtype=torch.float64)
            initial64 = initial.to(dtype=torch.float64)
            group_delta_squared[group] = group_delta_squared.get(group, 0.0) + float(
                delta.square().sum()
            )
            group_initial_squared[group] = group_initial_squared.get(group, 0.0) + float(
                initial64.square().sum()
            )

        epsilon = self.config.parameter_norm_epsilon
        group_radii = {
            group: np.sqrt(group_delta_squared[group])
            / max(np.sqrt(group_initial_squared[group]), epsilon)
            for group in sorted(group_delta_squared)
        }
        global_delta = np.sqrt(sum(group_delta_squared.values()))
        global_initial = np.sqrt(sum(group_initial_squared.values()))
        metrics = {
            "trust_region_parameter_global_relative_radius": float(
                global_delta / max(global_initial, epsilon)
            ),
            "trust_region_parameter_max_group_relative_radius": float(
                max(group_radii.values(), default=0.0)
            ),
        }
        metrics.update({
            f"trust_region_parameter_{group}_relative_radius": float(radius)
            for group, radius in group_radii.items()
        })
        return metrics

    def _parameter_radius_is_admissible(self, metrics: dict[str, float]) -> bool:
        limit = self.config.max_parameter_relative_radius
        if limit is None:
            return True
        radius = metrics["trust_region_parameter_max_group_relative_radius"]
        return bool(
            np.isfinite(radius)
            and radius <= limit + self.config.numerical_tolerance
        )

    @torch.no_grad()
    def _set_interpolation(
        self,
        current_actor: SACConstrainedGaussianActor,
        proposal: dict[str, torch.Tensor],
        scale: float,
    ) -> None:
        current = current_actor.state_dict()
        if current.keys() != self.initial_parameters.keys() or current.keys() != proposal.keys():
            raise ValueError("trust-region Actor state does not match initialized policy")
        for name, destination in current.items():
            initial = self.initial_parameters[name].to(destination.device)
            proposed = proposal[name].to(destination.device)
            destination.copy_(initial + scale * (proposed - initial))

    @torch.no_grad()
    def project(
        self,
        current_actor: SACConstrainedGaussianActor,
        replay_states: torch.Tensor | None,
        *,
        max_kl: float | None = None,
        max_parameter_relative_radius: float | None = None,
    ) -> dict[str, float]:
        """Project a completed Actor optimizer proposal, if explicitly enabled."""
        if not isinstance(current_actor, SACConstrainedGaussianActor):
            raise TypeError("trust region requires SACConstrainedGaussianActor")
        if _module_checksum(self.policy_anchor.teacher) != self.initial_checksum:
            raise RuntimeError("initial-policy teacher changed before projection")
        if not self.config.enabled:
            return {
                "trust_region_enabled": 0.0,
                "trust_region_triggered": 0.0,
                "trust_region_proposal_kl": 0.0,
                "trust_region_final_kl": 0.0,
                "trust_region_scale": 1.0,
                "trust_region_backtracks": 0.0,
                "trust_region_parameter_guard_enabled": 0.0,
            }

        states, formal_count = self.policy_anchor.sample_state_mixture(replay_states)
        proposal = {
            name: value.detach().clone()
            for name, value in current_actor.state_dict().items()
        }

        def evaluate() -> tuple[float, dict[str, float]]:
            loss, metrics = self.policy_anchor.loss_and_metrics_on_states(
                current_actor, states, formal_count
            )
            return float(loss), metrics

        proposal_kl, final_metrics = evaluate()
        proposal_radius_metrics = self._parameter_radius_metrics(current_actor)
        effective_max_kl = self.config.max_kl if max_kl is None else float(max_kl)
        effective_parameter_radius = (
            self.config.max_parameter_relative_radius
            if max_parameter_relative_radius is None
            else float(max_parameter_relative_radius)
        )
        if effective_max_kl <= 0.0 or (
            effective_parameter_radius is not None
            and effective_parameter_radius <= 0.0
        ):
            raise ValueError("effective trust-region limits must be positive")
        threshold = effective_max_kl + self.config.numerical_tolerance
        def radius_admissible(metrics: dict[str, float]) -> bool:
            if effective_parameter_radius is None:
                return True
            radius = metrics["trust_region_parameter_max_group_relative_radius"]
            return bool(
                np.isfinite(radius)
                and radius <= effective_parameter_radius + self.config.numerical_tolerance
            )
        proposal_kl_admissible = np.isfinite(proposal_kl) and proposal_kl <= threshold
        proposal_radius_admissible = radius_admissible(proposal_radius_metrics)
        if proposal_kl_admissible and proposal_radius_admissible:
            return {
                "trust_region_enabled": 1.0,
                "trust_region_triggered": 0.0,
                "trust_region_proposal_kl": proposal_kl,
                "trust_region_final_kl": proposal_kl,
                "trust_region_final_formal_kl": final_metrics["anchor_formal_kl"],
                "trust_region_final_replay_kl": final_metrics["anchor_replay_kl"],
                "trust_region_final_formal_action_mae": final_metrics["anchor_formal_action_mae"],
                "trust_region_final_replay_action_mae": final_metrics["anchor_replay_action_mae"],
                "trust_region_configured_kl_limit": effective_max_kl,
                "trust_region_configured_parameter_limit": (
                    effective_parameter_radius if effective_parameter_radius is not None
                    else float("nan")
                ),
                "trust_region_scale": 1.0,
                "trust_region_backtracks": 0.0,
                "trust_region_parameter_guard_enabled": float(
                    self.config.max_parameter_relative_radius is not None
                ),
                "trust_region_proposal_parameter_max_group_relative_radius": (
                    proposal_radius_metrics[
                        "trust_region_parameter_max_group_relative_radius"
                    ]
                ),
                **proposal_radius_metrics,
            }

        scale = 1.0
        final_kl = proposal_kl
        backtracks = 0
        for backtracks in range(1, self.config.max_backtracks + 1):
            scale *= self.config.backtrack_ratio
            self._set_interpolation(current_actor, proposal, scale)
            final_kl, final_metrics = evaluate()
            final_radius_metrics = self._parameter_radius_metrics(current_actor)
            if (
                np.isfinite(final_kl)
                and final_kl <= threshold
                and radius_admissible(final_radius_metrics)
            ):
                break
        else:
            # The exact initialized policy has zero KL and is the safe fallback
            # if finite precision or an unusually curved proposal defeats the
            # configured number of geometric backtracks.
            scale = 0.0
            self._set_interpolation(current_actor, proposal, scale)
            final_kl, final_metrics = evaluate()
            final_radius_metrics = self._parameter_radius_metrics(current_actor)
            backtracks = self.config.max_backtracks + 1

        if (
            not np.isfinite(final_kl)
            or final_kl > threshold
            or not radius_admissible(final_radius_metrics)
        ):
            raise RuntimeError("initial-policy trust-region projection failed")
        if _module_checksum(self.policy_anchor.teacher) != self.initial_checksum:
            raise RuntimeError("initial-policy teacher changed during projection")
        return {
            "trust_region_enabled": 1.0,
            "trust_region_triggered": 1.0,
            "trust_region_proposal_kl": proposal_kl,
            "trust_region_final_kl": final_kl,
            "trust_region_final_formal_kl": final_metrics["anchor_formal_kl"],
            "trust_region_final_replay_kl": final_metrics["anchor_replay_kl"],
            "trust_region_final_formal_action_mae": final_metrics["anchor_formal_action_mae"],
            "trust_region_final_replay_action_mae": final_metrics["anchor_replay_action_mae"],
            "trust_region_configured_kl_limit": effective_max_kl,
            "trust_region_configured_parameter_limit": (
                effective_parameter_radius if effective_parameter_radius is not None
                else float("nan")
            ),
            "trust_region_scale": scale,
            "trust_region_backtracks": float(backtracks),
            "trust_region_parameter_guard_enabled": float(
                self.config.max_parameter_relative_radius is not None
            ),
            "trust_region_proposal_parameter_max_group_relative_radius": (
                proposal_radius_metrics[
                    "trust_region_parameter_max_group_relative_radius"
                ]
            ),
            **final_radius_metrics,
        }

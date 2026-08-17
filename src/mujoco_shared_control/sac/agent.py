"""Frozen SAC Core v1 losses, target updates, and initialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mujoco_shared_control.actor_bc.model import ACTOR_ACTION_DIM, POLICY_STATE_DIM
from mujoco_shared_control.sac.actor import SACGaussianActor
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic
from mujoco_shared_control.sac.replay_buffer import ReplayBatch

if TYPE_CHECKING:
    from mujoco_shared_control.sac.policy_anchor import InitialPolicyAnchor


@dataclass(frozen=True)
class SACCoreConfig:
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    learning_starts: int = 10_000
    target_entropy: float = -7.0
    alpha_init: float = 0.1
    update_to_data_ratio: int = 1
    actor_update_frequency: int = 1
    target_update_frequency: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma/tau are outside their valid ranges")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.alpha_init) <= 0:
            raise ValueError("learning rates and alpha_init must be positive")
        if min(self.batch_size, self.replay_capacity, self.learning_starts) <= 0:
            raise ValueError("batch/capacity/learning_starts must be positive")
        if self.target_entropy != -float(ACTOR_ACTION_DIM):
            raise ValueError("SAC Core v1 target entropy is frozen to -action_dim")
        if (
            self.update_to_data_ratio != 1
            or self.actor_update_frequency != 1
            or self.target_update_frequency != 1
        ):
            raise ValueError("SAC Core v1 freezes UTD and update frequencies to 1")


def bootstrap_mask(terminated: torch.Tensor) -> torch.Tensor:
    """Time-limit truncations bootstrap; only true terminals mask the target."""
    return 1.0 - terminated.to(dtype=torch.float32)


@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    for online_parameter, target_parameter in zip(
        online.parameters(), target.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


class SACCore:
    """One-step SAC optimizer core; rollout scheduling intentionally lives elsewhere."""

    def __init__(
        self,
        actor_artifact: str | Path,
        config: SACCoreConfig = SACCoreConfig(),
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor_artifact = Path(actor_artifact).resolve()
        self.actor_artifact_sha256 = hashlib.sha256(self.actor_artifact.read_bytes()).hexdigest()
        payload: dict[str, Any] = torch.load(
            self.actor_artifact, map_location="cpu", weights_only=False
        )
        actor_format = payload.get("format_version")
        if actor_format == "sac_actor_v1_full_mean_path_distilled":
            self.actor = SACGaussianActor().to(self.device)
        elif actor_format == "sac_constrained_actor_v2_full_mean_path_distilled":
            self.actor = SACConstrainedGaussianActor().to(self.device)
        else:
            raise ValueError("unsupported SAC Actor artifact format")
        self.actor_format = str(actor_format)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.observation_mean = torch.as_tensor(
            np.asarray(payload["observation_mean"], dtype=np.float32), device=self.device
        )
        self.observation_std = torch.as_tensor(
            np.asarray(payload["observation_std"], dtype=np.float32), device=self.device
        )
        if self.observation_mean.shape != (POLICY_STATE_DIM,) or self.observation_std.shape != (POLICY_STATE_DIM,):
            raise ValueError("Actor normalization must be 42-D")
        if torch.any(self.observation_std <= 0) or not torch.isfinite(self.observation_std).all():
            raise ValueError("Actor observation std must be finite and positive")
        self.action_spec = dict(payload["action_spec"])
        expected_action_spec = {
            "max_translation_step_m": 0.025,
            "max_rotation_step_rad": 0.1,
            "gripper_min_m": 0.0,
            "gripper_max_m": 0.08,
        }
        if self.action_spec != expected_action_spec:
            raise ValueError("Actor action normalization is not the frozen v1 definition")
        initial_log_std = self.actor.state_dict()
        if torch.count_nonzero(initial_log_std["log_std_head.weight"]) != 0 or not torch.equal(
            initial_log_std["log_std_head.bias"], torch.full((ACTOR_ACTION_DIM,), -3.0)
        ):
            raise ValueError("Actor artifact does not have the frozen initial log_std=-3")

        self.critics = TwinSACCritic().to(self.device)
        self.target_critics = TwinSACCritic().to(self.device)
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.target_critics.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.log_alpha = nn.Parameter(
            torch.tensor(log(config.alpha_init), dtype=torch.float32, device=self.device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.update_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation.to(self.device, dtype=torch.float32) - self.observation_mean) / self.observation_std

    @torch.no_grad()
    def select_action(
        self, policy_state: np.ndarray, *, deterministic: bool = False
    ) -> np.ndarray:
        state = torch.as_tensor(policy_state, dtype=torch.float32, device=self.device)
        if state.shape != (POLICY_STATE_DIM,) or not torch.isfinite(state).all():
            raise ValueError("policy_state must be finite with shape (42,)")
        normalized = self.normalize_observation(state).unsqueeze(0)
        action = (
            self.actor.deterministic_action(normalized)
            if deterministic else self.actor.sample_action(normalized)[0]
        )
        return action.squeeze(0).cpu().numpy()

    def critic_target(self, batch: ReplayBatch) -> torch.Tensor:
        """Build the soft Bellman target with the current learned temperature.

        There is deliberately no coefficient override: the Critic target and
        Actor objective must describe the same maximum-entropy policy.  A dynamic
        target-entropy schedule controls the ``log_alpha`` loss; it is not a
        second temperature for the Bellman backup.
        """
        with torch.no_grad():
            next_obs = self.normalize_observation(batch.next_observation)
            next_action, next_log_prob, _ = self.actor.sample_action(next_obs)
            target_q1, target_q2 = self.target_critics(next_obs, next_action)
            soft_value = (
                torch.minimum(target_q1, target_q2)
                - self.alpha.detach() * next_log_prob
            )
            return batch.reward.to(self.device) + self.config.gamma * bootstrap_mask(
                batch.terminated.to(self.device)
            ) * soft_value

    def alpha_loss(
        self, log_prob: torch.Tensor, target_entropy: float | None = None
    ) -> torch.Tensor:
        entropy_target = (
            self.config.target_entropy if target_entropy is None else float(target_entropy)
        )
        return -(
            self.log_alpha * (log_prob + entropy_target).detach()
        ).mean()

    def update_critics(self, batch: ReplayBatch) -> dict[str, float]:
        """Update Q1/Q2 and their targets without touching Actor or alpha."""
        obs = self.normalize_observation(batch.observation)
        action = batch.action.to(self.device, dtype=torch.float32)
        target = self.critic_target(batch)

        q1, q2 = self.critics(obs, action)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_step += 1
        if self.update_step % self.config.target_update_frequency == 0:
            polyak_update(self.critics, self.target_critics, self.config.tau)
        td1, td2 = target - q1.detach(), target - q2.detach()
        values = {
            "critic_loss": critic_loss, "q1_loss": q1_loss, "q2_loss": q2_loss,
            "target_mean": target.mean(), "target_std": target.std(unbiased=False),
            "q1_mean": q1.mean(), "q2_mean": q2.mean(),
            "q1_std": q1.std(unbiased=False), "q2_std": q2.std(unbiased=False),
            "q1_min": q1.min(), "q1_max": q1.max(),
            "q2_min": q2.min(), "q2_max": q2.max(),
            "td_error_mean": torch.cat((td1, td2)).mean(),
            "td_error_std": torch.cat((td1, td2)).std(unbiased=False),
        }
        return {name: float(value.detach()) for name, value in values.items()}

    def update_actor_and_alpha(
        self,
        batch: ReplayBatch,
        *,
        policy_anchor: InitialPolicyAnchor | None = None,
        anchor_weight: float = 0.0,
        target_entropy: float | None = None,
        update_alpha: bool = True,
        freeze_log_std: bool = False,
    ) -> dict[str, float]:
        """Update Actor/alpha, optionally anchoring the warm-start policy.

        The anchor is deliberately Actor-only.  It neither changes the SAC
        transition batch nor contributes gradients to Critic or alpha.
        """
        if anchor_weight < 0.0:
            raise ValueError("anchor_weight must be non-negative")
        if anchor_weight and policy_anchor is None:
            raise ValueError("positive anchor_weight requires a policy anchor")
        obs = self.normalize_observation(batch.observation)

        self.critics.requires_grad_(False)
        log_std_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.actor.log_std_head.named_parameters()
        }
        if freeze_log_std:
            self.actor.log_std_head.requires_grad_(False)
        policy_action, log_prob, _ = self.actor.sample_action(obs)
        policy_q1, policy_q2 = self.critics(obs, policy_action)
        actor_sac_loss = (
            self.alpha.detach() * log_prob - torch.minimum(policy_q1, policy_q2)
        ).mean()
        policy_q_mean = torch.minimum(policy_q1, policy_q2).mean()
        entropy_term_mean = (self.alpha.detach() * log_prob).mean()
        anchor_metrics: dict[str, float] = {}
        if policy_anchor is not None and anchor_weight > 0.0:
            anchor_loss, anchor_metrics = policy_anchor.loss_and_metrics(
                self.actor, replay_states=obs
            )
        else:
            anchor_loss = torch.zeros((), device=self.device)
        actor_loss = actor_sac_loss + anchor_weight * anchor_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        if freeze_log_std:
            for name, parameter in self.actor.log_std_head.named_parameters():
                parameter.requires_grad_(log_std_requires_grad[name])
        self.critics.requires_grad_(True)

        entropy_target = (
            self.config.target_entropy if target_entropy is None else float(target_entropy)
        )
        alpha_loss = self.alpha_loss(log_prob, entropy_target)
        if update_alpha:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        values = {
            "actor_loss": actor_loss, "actor_sac_loss": actor_sac_loss,
            "policy_q_mean": policy_q_mean,
            "entropy_term_mean": entropy_term_mean,
            "anchor_loss": anchor_loss,
            "anchor_weight": torch.as_tensor(anchor_weight, device=self.device),
            "target_entropy": torch.as_tensor(entropy_target, device=self.device),
            "alpha_loss": alpha_loss, "alpha": self.alpha,
            "alpha_updated": torch.as_tensor(float(update_alpha), device=self.device),
            "log_std_frozen": torch.as_tensor(float(freeze_log_std), device=self.device),
            "mean_log_prob": log_prob.mean(),
            "entropy_gap": log_prob.mean() + entropy_target,
        }
        return {
            **{name: float(value.detach()) for name, value in values.items()},
            **anchor_metrics,
        }

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """Standard full SAC update, preserved as the Stage-C operation."""
        return {**self.update_critics(batch), **self.update_actor_and_alpha(batch)}

    def specification(self) -> dict[str, Any]:
        return {
            **asdict(self.config), "observation_dim": POLICY_STATE_DIM,
            "action_dim": ACTOR_ACTION_DIM, "bootstrap_mask": "1 - terminated",
            "actor_artifact": str(self.actor_artifact),
            "actor_artifact_sha256": self.actor_artifact_sha256,
            "actor_format": self.actor_format,
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        self.target_critics.requires_grad_(False)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_step = int(state["update_step"])

"""Fixed positive-advantage filtering for conservative offline Actor updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic
from mujoco_shared_control.sac.critic_pretraining import CriticArrays


@dataclass(frozen=True)
class OfflineAdvBCConfig:
    learning_rate: float = 3e-4
    batch_size: int = 256
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    max_optimizer_steps: int = 10_000
    seed: int = 20260814
    positive_advantage_threshold: float = 0.0


@dataclass
class AdvantageArrays:
    observation: np.ndarray
    data_action: np.ndarray
    initial_actor_action: np.ndarray
    q_data: np.ndarray
    q_initial_actor: np.ndarray
    advantage: np.ndarray
    positive: np.ndarray
    category: np.ndarray
    phase: np.ndarray
    episode_id: np.ndarray

    def __len__(self) -> int: return len(self.advantage)


@torch.no_grad()
def build_advantage_arrays(
    data: CriticArrays,
    reference_actor: SACConstrainedGaussianActor,
    frozen_critic: TwinSACCritic,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    *,
    batch_size: int = 4096,
    threshold: float = 0.0,
) -> AdvantageArrays:
    if threshold != 0.0: raise ValueError("v1 positive-advantage threshold is exactly zero")
    actor_actions, q_data_values, q_actor_values = [], [], []
    reference_actor.eval(); frozen_critic.eval()
    for start in range(0, len(data), batch_size):
        stop = start + batch_size
        observation = torch.from_numpy(data.observation[start:stop])
        normalized = (observation - observation_mean) / observation_std
        data_action = torch.from_numpy(data.action[start:stop])
        actor_action = reference_actor.deterministic_action(normalized)
        q1d, q2d = frozen_critic(normalized, data_action)
        q1a, q2a = frozen_critic(normalized, actor_action)
        actor_actions.append(actor_action.cpu().numpy())
        q_data_values.append(torch.minimum(q1d, q2d).cpu().numpy())
        q_actor_values.append(torch.minimum(q1a, q2a).cpu().numpy())
    q_data = np.concatenate(q_data_values).reshape(-1)
    q_actor = np.concatenate(q_actor_values).reshape(-1)
    advantage = q_data - q_actor
    return AdvantageArrays(
        data.observation.copy(), data.action.copy(), np.concatenate(actor_actions),
        q_data, q_actor, advantage, advantage > threshold, data.category.copy(),
        data.phase.copy(), data.episode_id.copy(),
    )


def actor_action_mse(
    actor: SACConstrainedGaussianActor, observation: torch.Tensor,
    target_action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(actor.deterministic_action((observation - mean) / std), target_action)


def train_step(
    actor: SACConstrainedGaussianActor,
    optimizer: torch.optim.Optimizer,
    anchor_observation: torch.Tensor,
    anchor_action: torch.Tensor,
    improve_observation: torch.Tensor,
    improve_action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    gradient_clip_norm: float,
) -> dict[str, float]:
    anchor_loss = actor_action_mse(actor, anchor_observation, anchor_action, mean, std)
    improve_loss = actor_action_mse(actor, improve_observation, improve_action, mean, std)
    total = anchor_loss + improve_loss
    optimizer.zero_grad(set_to_none=True); total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        gradient_clip_norm,
    )
    optimizer.step()
    return {"anchor_loss": float(anchor_loss.detach()),
            "improve_loss": float(improve_loss.detach()),
            "total_loss": float(total.detach()), "gradient_norm": float(gradient_norm)}


def state_checksum(module: torch.nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_trainable_actor(actor: SACConstrainedGaussianActor) -> list[str]:
    trainable = []
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(not name.startswith("log_std_head."))
        if parameter.requires_grad: trainable.append(name)
    return trainable

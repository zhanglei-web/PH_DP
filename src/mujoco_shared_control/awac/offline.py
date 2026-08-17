"""Standard offline Advantage-Weighted Actor-Critic (AWAC)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


OBSERVATION_DIM = 42
ACTION_DIM = 7


@dataclass(frozen=True)
class OfflineAWACConfig:
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 0.3
    max_advantage_weight: float = 20.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    offline_updates: int = 25_000
    seed: int = 20260814
    gradient_clip_norm: float = 10.0


def _mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, hidden), nn.ReLU()))
        current = hidden
    return nn.Sequential(*layers)


class AWACGaussianActor(nn.Module):
    """42 -> 256 ReLU x4 -> state-dependent squashed 7-D Gaussian."""

    def __init__(self, config: OfflineAWACConfig = OfflineAWACConfig()) -> None:
        super().__init__()
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.trunk = _mlp(OBSERVATION_DIM, config.actor_hidden_dims)
        final_dim = config.actor_hidden_dims[-1]
        self.mean_head = nn.Linear(final_dim, ACTION_DIM)
        self.log_std_head = nn.Linear(final_dim, ACTION_DIM)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def distribution_stats(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observation)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    @staticmethod
    def _log_prob_from_pre_squash(normal: Normal, pre_squash: torch.Tensor) -> torch.Tensor:
        correction = 2.0 * (np.log(2.0) - pre_squash - F.softplus(-2.0 * pre_squash))
        return (normal.log_prob(pre_squash) - correction).sum(dim=-1, keepdim=True)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_stats(observation)
        normal = Normal(mean, log_std.exp())
        pre_squash = normal.rsample()
        return torch.tanh(pre_squash), self._log_prob_from_pre_squash(normal, pre_squash)

    def log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        bounded = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_squash = torch.atanh(bounded)
        mean, log_std = self.distribution_stats(observation)
        return self._log_prob_from_pre_squash(Normal(mean, log_std.exp()), pre_squash)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_stats(observation)
        return torch.tanh(mean)


class AWACCritic(nn.Module):
    """49 -> 256 ReLU x4 -> scalar Q."""

    def __init__(self, config: OfflineAWACConfig = OfflineAWACConfig()) -> None:
        super().__init__()
        self.trunk = _mlp(OBSERVATION_DIM + ACTION_DIM, config.critic_hidden_dims)
        self.output = nn.Linear(config.critic_hidden_dims[-1], 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != OBSERVATION_DIM or action.shape[-1] != ACTION_DIM:
            raise ValueError("AWAC critic expects (...,42) state and (...,7) action")
        return self.output(self.trunk(torch.cat((observation, action), dim=-1)))


@dataclass
class TransitionBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


class OfflineReplayBuffer:
    def __init__(self, npz_path: str | Path, *, device: torch.device) -> None:
        with np.load(Path(npz_path), allow_pickle=False) as data:
            self.observation = torch.from_numpy(np.asarray(data["obs"], np.float32)).to(device)
            self.action = torch.from_numpy(np.asarray(data["action"], np.float32)).to(device)
            self.reward = torch.from_numpy(np.asarray(data["reward"], np.float32)).to(device).unsqueeze(1)
            self.next_observation = torch.from_numpy(np.asarray(data["next_obs"], np.float32)).to(device)
            done = np.asarray(data["terminated"], bool) | np.asarray(data["truncated"], bool)
            self.done = torch.from_numpy(done.astype(np.float32)).to(device).unsqueeze(1)
        if self.observation.shape[1:] != (42,) or self.action.shape[1:] != (7,):
            raise ValueError("offline replay dimensions do not match AWAC-v1")

    def __len__(self) -> int:
        return len(self.observation)

    def sample(self, batch_size: int, generator: torch.Generator) -> TransitionBatch:
        indices = torch.randint(len(self), (batch_size,), generator=generator, device=self.observation.device)
        return TransitionBatch(
            self.observation[indices], self.action[indices], self.reward[indices],
            self.next_observation[indices], self.done[indices],
        )


def observation_statistics(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(npz_path), allow_pickle=False) as data:
        observation = np.asarray(data["obs"], np.float64)
    mean = observation.mean(axis=0).astype(np.float32)
    std = observation.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


class OfflineAWACTrainer:
    def __init__(
        self,
        config: OfflineAWACConfig,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        *,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.actor = AWACGaussianActor(config).to(device)
        self.q1 = AWACCritic(config).to(device)
        self.q2 = AWACCritic(config).to(device)
        self.target_q1 = AWACCritic(config).to(device)
        self.target_q2 = AWACCritic(config).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.mean = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)
        self.step = 0

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.mean) / self.std

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        obs = self.normalize(batch.observation)
        next_obs = self.normalize(batch.next_observation)
        with torch.no_grad():
            next_action, _ = self.actor.sample(next_obs)
            target_value = torch.minimum(
                self.target_q1(next_obs, next_action), self.target_q2(next_obs, next_action)
            )
            target = batch.reward + self.config.gamma * (1.0 - batch.done) * target_value

        q1 = self.q1(obs, batch.action)
        q2 = self.q2(obs, batch.action)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.gradient_clip_norm)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.gradient_clip_norm)
        self.q2_optimizer.step()

        with torch.no_grad():
            policy_action, _ = self.actor.sample(obs)
            dataset_q = torch.minimum(self.q1(obs, batch.action), self.q2(obs, batch.action))
            value = torch.minimum(self.q1(obs, policy_action), self.q2(obs, policy_action))
            advantage = dataset_q - value
            log_weight = torch.clamp(
                advantage / self.config.awac_lambda,
                max=float(np.log(self.config.max_advantage_weight)),
            )
            weight = torch.exp(log_weight)
        log_prob = self.actor.log_prob(obs, batch.action)
        actor_loss = -(weight * log_prob).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            for source, target_network in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(source.parameters(), target_network.parameters(), strict=True):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.step += 1
        weight_percentiles = torch.quantile(weight.flatten(), torch.tensor([.5, .9, .99], device=self.device))
        metrics = {
            "step": self.step,
            "critic_loss_q1": float(q1_loss.detach()),
            "critic_loss_q2": float(q2_loss.detach()),
            "q1_mean": float(q1.detach().mean()), "q1_std": float(q1.detach().std(unbiased=False)),
            "q2_mean": float(q2.detach().mean()), "q2_std": float(q2.detach().std(unbiased=False)),
            "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std(unbiased=False)),
            "advantage_min": float(advantage.min()), "advantage_max": float(advantage.max()),
            "awac_weight_mean": float(weight.mean()),
            "awac_weight_p50": float(weight_percentiles[0]),
            "awac_weight_p90": float(weight_percentiles[1]),
            "awac_weight_p99": float(weight_percentiles[2]),
            "awac_weight_max": float(weight.max()),
            "actor_loss": float(actor_loss.detach()),
            "actor_log_prob": float(log_prob.detach().mean()),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite AWAC metric at step {self.step}")
        return metrics

    def checkpoint_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "format_version": "offline_awac_v1",
            "step": self.step,
            "actor": self.actor.state_dict(),
            "critic_q1": self.q1.state_dict(), "critic_q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_q1_optimizer": self.q1_optimizer.state_dict(),
            "critic_q2_optimizer": self.q2_optimizer.state_dict(),
            "observation_mean": self.mean.detach().cpu(),
            "observation_std": self.std.detach().cpu(),
            "training_config": asdict(self.config),
            **metadata,
        }


"""AWAC-v2 Hybrid Actor, supervised warm start, and offline updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F


OBSERVATION_DIM = 43
CONTINUOUS_ACTION_DIM = 6


@dataclass(frozen=True)
class HybridAWACConfig:
    observation_dim: int = OBSERVATION_DIM
    hidden_dims: tuple[int, ...] = (256, 256, 256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 0.3
    max_advantage_weight: float = 20.0
    beta_gripper: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    gradient_clip_norm: float = 10.0
    awac_updates: int = 5_000
    seed: int = 20260814
    bc_max_epochs: int = 50
    bc_early_stopping_patience: int = 8


def _backbone(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for output_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        input_dim = output_dim
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    def __init__(self, config: HybridAWACConfig = HybridAWACConfig()) -> None:
        super().__init__()
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.observation_dim = config.observation_dim
        self.backbone = _backbone(config.observation_dim, config.hidden_dims)
        width = config.hidden_dims[-1]
        self.continuous_mean = nn.Linear(width, CONTINUOUS_ACTION_DIM)
        self.continuous_log_std = nn.Linear(width, CONTINUOUS_ACTION_DIM)
        self.gripper_logit = nn.Linear(width, 1)
        for head in (self.continuous_mean, self.gripper_logit):
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.continuous_log_std.weight)
        nn.init.constant_(self.continuous_log_std.bias, -0.5)

    def distribution_stats(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = self.continuous_mean(features)
        log_std = torch.clamp(self.continuous_log_std(features), self.log_std_min, self.log_std_max)
        return mean, log_std, self.gripper_logit(features)

    @staticmethod
    def _continuous_log_prob(normal: Normal, pre_squash: torch.Tensor) -> torch.Tensor:
        correction = 2.0 * (np.log(2.0) - pre_squash - F.softplus(-2.0 * pre_squash))
        return (normal.log_prob(pre_squash) - correction).sum(dim=-1, keepdim=True)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, logit = self.distribution_stats(observation)
        normal = Normal(mean, log_std.exp())
        pre_squash = normal.rsample()
        continuous = torch.tanh(pre_squash)
        gripper = Bernoulli(logits=logit).sample()
        joint_log_prob = self._continuous_log_prob(normal, pre_squash) + Bernoulli(logits=logit).log_prob(gripper)
        return continuous, gripper, joint_log_prob

    def dataset_log_prob(
        self, observation: torch.Tensor, continuous: torch.Tensor,
        gripper: torch.Tensor, beta_gripper: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bounded = torch.clamp(continuous, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_squash = torch.atanh(bounded)
        mean, log_std, logit = self.distribution_stats(observation)
        continuous_log_prob = self._continuous_log_prob(Normal(mean, log_std.exp()), pre_squash)
        gripper_log_prob = -F.binary_cross_entropy_with_logits(logit, gripper, reduction="none")
        return continuous_log_prob + beta_gripper * gripper_log_prob, continuous_log_prob, gripper_log_prob

    def deterministic_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _log_std, logit = self.distribution_stats(observation)
        probability_close = torch.sigmoid(logit)
        return torch.tanh(mean), (probability_close >= 0.5).float(), probability_close


class HybridCritic(nn.Module):
    def __init__(self, config: HybridAWACConfig = HybridAWACConfig()) -> None:
        super().__init__()
        self.observation_dim = config.observation_dim
        self.backbone = _backbone(config.observation_dim + CONTINUOUS_ACTION_DIM + 1, config.hidden_dims)
        self.output = nn.Linear(config.hidden_dims[-1], 1)

    def forward(self, observation: torch.Tensor, continuous: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        if (
            observation.shape[-1] != self.observation_dim
            or continuous.shape[-1] != CONTINUOUS_ACTION_DIM
            or gripper.shape[-1] != 1
        ):
            raise ValueError(
                f"Hybrid critic expects state{self.observation_dim} + continuous6 + gripper1"
            )
        return self.output(self.backbone(torch.cat((observation, continuous, gripper), dim=-1)))


@dataclass
class HybridBatch:
    observation: torch.Tensor
    continuous_action: torch.Tensor
    gripper_action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


class HybridReplay:
    def __init__(self, path: str | Path, device: torch.device) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            self.observation = torch.from_numpy(np.asarray(data["obs"], np.float32)).to(device)
            self.continuous_action = torch.from_numpy(np.asarray(data["continuous_action"], np.float32)).to(device)
            self.gripper_action = torch.from_numpy(np.asarray(data["gripper_action"], np.float32)).to(device).unsqueeze(1)
            self.reward = torch.from_numpy(np.asarray(data["reward"], np.float32)).to(device).unsqueeze(1)
            self.next_observation = torch.from_numpy(np.asarray(data["next_obs"], np.float32)).to(device)
            done = np.asarray(data["terminated"], bool) | np.asarray(data["truncated"], bool)
            self.done = torch.from_numpy(done.astype(np.float32)).to(device).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.observation)

    def sample(self, size: int, generator: torch.Generator) -> HybridBatch:
        indices = torch.randint(len(self), (size,), generator=generator, device=self.observation.device)
        return HybridBatch(
            self.observation[indices], self.continuous_action[indices], self.gripper_action[indices],
            self.reward[indices], self.next_observation[indices], self.done[indices],
        )


def observation_statistics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        observation = np.asarray(data["obs"], np.float64)
    mean = observation.mean(0).astype(np.float32)
    std = np.maximum(observation.std(0).astype(np.float32), np.float32(1e-6))
    return mean, std


@torch.no_grad()
def actor_metrics(
    actor: HybridActor, observation: torch.Tensor, continuous: torch.Tensor,
    gripper: torch.Tensor, stage: torch.Tensor,
) -> dict[str, Any]:
    predicted_continuous, predicted_gripper, close_probability = actor.deterministic_action(observation)
    result: dict[str, Any] = {
        "continuous_action_mse": float(F.mse_loss(predicted_continuous, continuous)),
        "overall_gripper_accuracy": float((predicted_gripper == gripper).float().mean()),
        "overall_predicted_close_ratio": float(predicted_gripper.mean()),
        "overall_target_close_ratio": float(gripper.mean()),
        "overall_mean_close_probability": float(close_probability.mean()),
    }
    for name, value, target_is_close in (("CLOSE_GRIPPER", 2, True), ("OPEN_GRIPPER", 6, False)):
        mask = stage == value
        target_close = gripper[mask]
        predicted_close = predicted_gripper[mask]
        probability = close_probability[mask]
        result[name] = {
            "transitions": int(mask.sum()),
            "accuracy": float((predicted_close == target_close).float().mean()),
            "target_close_ratio": float(target_close.mean()),
            "predicted_close_ratio": float(predicted_close.mean()),
            "mean_close_probability": float(probability.mean()),
            "target_open_ratio": float(1.0 - target_close.mean()),
            "predicted_open_ratio": float(1.0 - predicted_close.mean()),
            "mean_open_probability": float(1.0 - probability.mean()),
            "target_semantic": "CLOSE" if target_is_close else "OPEN",
        }
    return result


class HybridAWACTrainer:
    def __init__(
        self, config: HybridAWACConfig, mean: np.ndarray, std: np.ndarray,
        warm_actor_state: dict[str, torch.Tensor], device: torch.device,
    ) -> None:
        if mean.shape != (config.observation_dim,) or std.shape != (config.observation_dim,):
            raise ValueError("Hybrid normalization shape does not match observation_dim")
        self.config = config; self.device = device; self.step = 0
        self.actor = HybridActor(config).to(device); self.actor.load_state_dict(warm_actor_state)
        self.q1 = HybridCritic(config).to(device); self.q2 = HybridCritic(config).to(device)
        self.target_q1 = HybridCritic(config).to(device); self.target_q2 = HybridCritic(config).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict()); self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False); self.target_q2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed + 1)

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.mean) / self.std

    def update(self, batch: HybridBatch) -> dict[str, float]:
        observation = self.normalize(batch.observation); next_observation = self.normalize(batch.next_observation)
        with torch.no_grad():
            next_continuous, next_gripper, _ = self.actor.sample(next_observation)
            target_value = torch.minimum(
                self.target_q1(next_observation, next_continuous, next_gripper),
                self.target_q2(next_observation, next_continuous, next_gripper),
            )
            target = batch.reward + self.config.gamma * (1.0 - batch.done) * target_value
        q1 = self.q1(observation, batch.continuous_action, batch.gripper_action)
        q2 = self.q2(observation, batch.continuous_action, batch.gripper_action)
        q1_loss = F.mse_loss(q1, target); q2_loss = F.mse_loss(q2, target)
        self.q1_optimizer.zero_grad(set_to_none=True); q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.gradient_clip_norm); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(set_to_none=True); q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.gradient_clip_norm); self.q2_optimizer.step()
        with torch.no_grad():
            policy_continuous, policy_gripper, _ = self.actor.sample(observation)
            dataset_q = torch.minimum(
                self.q1(observation, batch.continuous_action, batch.gripper_action),
                self.q2(observation, batch.continuous_action, batch.gripper_action),
            )
            value = torch.minimum(
                self.q1(observation, policy_continuous, policy_gripper),
                self.q2(observation, policy_continuous, policy_gripper),
            )
            advantage = dataset_q - value
            weight = torch.exp(torch.clamp(
                advantage / self.config.awac_lambda,
                max=float(np.log(self.config.max_advantage_weight)),
            ))
        joint_log_prob, continuous_log_prob, gripper_log_prob = self.actor.dataset_log_prob(
            observation, batch.continuous_action, batch.gripper_action, self.config.beta_gripper,
        )
        actor_loss = -(weight * joint_log_prob).mean()
        self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm); self.actor_optimizer.step()
        with torch.no_grad():
            for source, target_network in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(source.parameters(), target_network.parameters(), strict=True):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.step += 1
        percentiles = torch.quantile(weight.flatten(), torch.tensor([.5, .9, .99], device=self.device))
        metrics = {
            "step": self.step, "critic_loss_q1": float(q1_loss.detach()), "critic_loss_q2": float(q2_loss.detach()),
            "q1_mean": float(q1.detach().mean()), "q1_std": float(q1.detach().std(unbiased=False)),
            "q1_min": float(q1.detach().min()), "q1_max": float(q1.detach().max()),
            "q2_mean": float(q2.detach().mean()), "q2_std": float(q2.detach().std(unbiased=False)),
            "q2_min": float(q2.detach().min()), "q2_max": float(q2.detach().max()),
            "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std(unbiased=False)),
            "advantage_min": float(advantage.min()), "advantage_max": float(advantage.max()),
            "awac_weight_mean": float(weight.mean()), "awac_weight_p50": float(percentiles[0]),
            "awac_weight_p90": float(percentiles[1]), "awac_weight_p99": float(percentiles[2]),
            "awac_weight_max": float(weight.max()), "actor_loss": float(actor_loss.detach()),
            "actor_log_prob": float(joint_log_prob.detach().mean()),
            "continuous_log_prob": float(continuous_log_prob.detach().mean()),
            "gripper_log_prob": float(gripper_log_prob.detach().mean()),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite Hybrid AWAC metric at {self.step}")
        return metrics

    def checkpoint(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "format_version": "offline_awac_v2_hybrid", "step": self.step,
            "actor": self.actor.state_dict(), "critic_q1": self.q1.state_dict(), "critic_q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_q1_optimizer": self.q1_optimizer.state_dict(), "critic_q2_optimizer": self.q2_optimizer.state_dict(),
            "observation_mean": self.mean.detach().cpu(), "observation_std": self.std.detach().cpu(),
            "training_config": asdict(self.config), **metadata,
        }

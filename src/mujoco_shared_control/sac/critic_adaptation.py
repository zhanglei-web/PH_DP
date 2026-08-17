"""Fixed-policy, Critic-only adaptation on Expert and deterministic Actor data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic


GAMMA = 0.995
TAU = 0.005


@dataclass(frozen=True)
class CriticAdaptationConfig:
    gamma: float = GAMMA
    tau: float = TAU
    learning_rate: float = 3e-4
    batch_size: int = 256
    offline_fraction: float = 0.5
    max_updates: int = 20_000
    gradient_clip: float = 1.0
    seed: int = 20260815

    def __post_init__(self) -> None:
        if self.gamma != GAMMA or self.tau != TAU:
            raise ValueError("v1 gamma/tau are frozen")
        if self.batch_size % 2 or self.offline_fraction != 0.5:
            raise ValueError("v1 requires an exact 50/50 batch")

    @property
    def half_batch(self) -> int:
        return self.batch_size // 2


@dataclass
class ActorTransitionArrays:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    mc_return: np.ndarray
    phase: np.ndarray
    episode_id: np.ndarray
    seed: np.ndarray
    step: np.ndarray
    outcome: np.ndarray

    def __len__(self) -> int:
        return len(self.reward)

    def subset(self, mask: np.ndarray) -> "ActorTransitionArrays":
        return ActorTransitionArrays(**{
            name: getattr(self, name)[mask] for name in self.__dataclass_fields__
        })

    def save(self, path: Any) -> None:
        np.savez_compressed(path, **{
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def load(cls, path: Any) -> "ActorTransitionArrays":
        with np.load(path) as payload:
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


def monte_carlo_returns(
    reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    reward = np.asarray(reward, np.float64).reshape(-1)
    terminated = np.asarray(terminated, bool).reshape(-1)
    truncated = np.asarray(truncated, bool).reshape(-1)
    result = np.empty_like(reward)
    running = 0.0
    for index in range(len(reward) - 1, -1, -1):
        if terminated[index] or truncated[index]:
            running = 0.0
        running = reward[index] + gamma * running
        result[index] = running
    return result.astype(np.float32)[:, None]


@torch.no_grad()
def fixed_policy_td_target(
    actor: SACConstrainedGaussianActor,
    target_critic: TwinSACCritic,
    reward: torch.Tensor,
    next_observation: torch.Tensor,
    terminated: torch.Tensor,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    gamma: float = GAMMA,
) -> torch.Tensor:
    """Deterministic Q^pi target; truncation deliberately keeps bootstrapping."""
    normalized_next = (next_observation - observation_mean) / observation_std
    next_action = actor.deterministic_action(normalized_next)
    q1, q2 = target_critic(normalized_next, next_action)
    return reward + gamma * (1.0 - terminated.float()) * torch.minimum(q1, q2)


def critic_update(
    critic: TwinSACCritic,
    target_critic: TwinSACCritic,
    actor: SACConstrainedGaussianActor,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    config: CriticAdaptationConfig,
) -> dict[str, float]:
    target = fixed_policy_td_target(
        actor, target_critic, batch["reward"], batch["next_observation"],
        batch["terminated"], observation_mean, observation_std, config.gamma,
    )
    normalized = (batch["observation"] - observation_mean) / observation_std
    q1, q2 = critic(normalized, batch["action"])
    loss1 = F.mse_loss(q1, target)
    loss2 = F.mse_loss(q2, target)
    loss = loss1 + loss2
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
    optimizer.step()
    polyak_update(critic, target_critic, config.tau)
    return {
        "critic_loss": float(loss.detach()), "q1_loss": float(loss1.detach()),
        "q2_loss": float(loss2.detach()), "target_mean": float(target.mean().detach()),
        "target_std": float(target.std().detach()), "q1_mean": float(q1.mean().detach()),
        "q2_mean": float(q2.mean().detach()),
        "q_disagreement": float((q1-q2).abs().mean().detach()),
        "gradient_norm": float(gradient_norm),
    }


@torch.no_grad()
def polyak_update(online: TwinSACCritic, target: TwinSACCritic, tau: float = TAU) -> None:
    for source, destination in zip(online.parameters(), target.parameters(), strict=True):
        destination.mul_(1.0 - tau).add_(source, alpha=tau)


def exact_mixed_indices(
    offline_size: int, online_size: int, config: CriticAdaptationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if offline_size <= 0 or online_size <= 0:
        raise ValueError("both buffers must be non-empty")
    return (
        rng.integers(offline_size, size=config.half_batch),
        rng.integers(online_size, size=config.half_batch),
    )


def concatenate_batch(offline: Any, online: ActorTransitionArrays,
                      offline_indices: np.ndarray, online_indices: np.ndarray) -> dict[str, torch.Tensor]:
    fields = ("observation", "action", "reward", "next_observation", "terminated", "truncated")
    return {name: torch.from_numpy(np.concatenate((
        getattr(offline, name)[offline_indices], getattr(online, name)[online_indices]
    ))) for name in fields}


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    x = np.asarray(x).reshape(-1); y = np.asarray(y).reshape(-1)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction).reshape(-1); target = np.asarray(target).reshape(-1)
    error = prediction - target
    return {
        "count": len(target), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": _correlation(prediction, target, "pearson"),
        "spearman": _correlation(prediction, target, "spearman"),
        "prediction_mean": float(prediction.mean()), "prediction_std": float(prediction.std()),
        "target_mean": float(target.mean()), "target_std": float(target.std()),
    }


@torch.no_grad()
def evaluate_actor_data(
    critic: TwinSACCritic, arrays: ActorTransitionArrays,
    observation_mean: torch.Tensor, observation_std: torch.Tensor,
) -> dict[str, Any]:
    predictions1, predictions2 = [], []
    for start in range(0, len(arrays), 4096):
        observation = torch.from_numpy(arrays.observation[start:start+4096])
        normalized = (observation - observation_mean) / observation_std
        action = torch.from_numpy(arrays.action[start:start+4096])
        q1, q2 = critic(normalized, action)
        predictions1.append(q1.numpy()); predictions2.append(q2.numpy())
    q1 = np.concatenate(predictions1); q2 = np.concatenate(predictions2)
    prediction = (q1 + q2) / 2
    result: dict[str, Any] = {
        "overall": regression_metrics(prediction, arrays.mc_return),
        "q1_q2_disagreement_mae": float(np.mean(np.abs(q1-q2))),
        "q1": {"mean": float(q1.mean()), "std": float(q1.std()),
               "min": float(q1.min()), "max": float(q1.max())},
        "q2": {"mean": float(q2.mean()), "std": float(q2.std()),
               "min": float(q2.min()), "max": float(q2.max())},
        "outcome": {}, "phase": {},
    }
    for name, labels in (("outcome", np.unique(arrays.outcome)),
                         ("phase", np.unique(arrays.phase))):
        source = getattr(arrays, name)
        for label in labels:
            mask = source == label
            result[name][str(label)] = regression_metrics(prediction[mask], arrays.mc_return[mask])
    return result


def module_checksum(module: torch.nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

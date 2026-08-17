"""Unified offline+online replay and exact Hybrid AWAC checkpoint continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import (
    HybridAWACConfig, HybridAWACTrainer, HybridActor, HybridBatch,
)


@torch.no_grad()
def low_noise_behavior_action(
    actor: HybridActor,
    observation: torch.Tensor,
    *,
    exploration_std_scale: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample only the collection policy with reduced continuous variance.

    This intentionally does not call or alter the actor's training distribution:
    HybridAWACTrainer.update still uses HybridActor.sample and dataset_log_prob.
    The gripper is deterministic because its validated Bernoulli classifier is not
    being used as an exploration source during environment interaction.
    """
    if not 0.0 <= exploration_std_scale <= 1.0:
        raise ValueError("exploration_std_scale must be within [0, 1]")
    mean, log_std, logit = actor.distribution_stats(observation)
    policy_std = log_std.exp()
    effective_std = policy_std * exploration_std_scale
    # At scale zero this is the exact deterministic actor action and deliberately
    # consumes no Gaussian RNG state during the precision retreat phase.
    pre_squash = (
        mean if exploration_std_scale == 0.0
        else mean + effective_std * torch.randn_like(mean)
    )
    close_probability = torch.sigmoid(logit)
    return (
        torch.tanh(pre_squash),
        (close_probability >= 0.5).float(),
        policy_std,
        effective_std,
        close_probability,
    )


class UnifiedHybridReplay:
    def __init__(
        self, offline_path: str | Path, *, capacity: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity); self.device = device
        with np.load(Path(offline_path), allow_pickle=False) as data:
            offline = len(data["obs"])
            if offline > capacity:
                raise ValueError("replay capacity is smaller than offline dataset")
            observation_dim = int(data["obs"].shape[1])
            if data["obs"].ndim != 2 or data["next_obs"].shape != (offline, observation_dim):
                raise ValueError("offline replay observation arrays are inconsistent")
            self.observation_dim = observation_dim
            self.observation = torch.empty((capacity, observation_dim), dtype=torch.float32, device=device)
            self.continuous = torch.empty((capacity, 6), dtype=torch.float32, device=device)
            self.gripper = torch.empty((capacity, 1), dtype=torch.float32, device=device)
            self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=device)
            self.next_observation = torch.empty((capacity, observation_dim), dtype=torch.float32, device=device)
            self.done = torch.empty((capacity, 1), dtype=torch.float32, device=device)
            self.observation[:offline] = torch.from_numpy(np.asarray(data["obs"], np.float32)).to(device)
            self.continuous[:offline] = torch.from_numpy(np.asarray(data["continuous_action"], np.float32)).to(device)
            self.gripper[:offline, 0] = torch.from_numpy(np.asarray(data["gripper_action"], np.float32)).to(device)
            self.reward[:offline, 0] = torch.from_numpy(np.asarray(data["reward"], np.float32)).to(device)
            self.next_observation[:offline] = torch.from_numpy(np.asarray(data["next_obs"], np.float32)).to(device)
            terminal = np.asarray(data["terminated"], bool) | np.asarray(data["truncated"], bool)
            self.done[:offline, 0] = torch.from_numpy(terminal.astype(np.float32)).to(device)
        self.offline_count = offline
        self.online_count = 0
        self.size = offline
        self.last_sample_online_count = 0

    def __len__(self) -> int:
        return self.size

    def append(
        self, observation: np.ndarray, continuous: np.ndarray, gripper: float,
        reward: float, next_observation: np.ndarray, terminated: bool,
        truncated: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("unified replay capacity exhausted")
        observation = np.asarray(observation, np.float32)
        next_observation = np.asarray(next_observation, np.float32)
        continuous = np.asarray(continuous, np.float32)
        if (
            observation.shape != (self.observation_dim,)
            or next_observation.shape != (self.observation_dim,)
            or continuous.shape != (6,)
            or not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
            or not np.isfinite(continuous).all()
            or not np.isfinite(reward)
            or gripper not in (0.0, 1.0)
        ):
            raise ValueError("invalid online Hybrid replay transition")
        index = self.size
        self.observation[index] = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        self.continuous[index] = torch.as_tensor(continuous, dtype=torch.float32, device=self.device)
        self.gripper[index, 0] = float(gripper)
        self.reward[index, 0] = float(reward)
        self.next_observation[index] = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
        self.done[index, 0] = float(terminated or truncated)
        self.size += 1; self.online_count += 1

    def sample(self, batch_size: int, generator: torch.Generator) -> HybridBatch:
        indices = torch.randint(
            self.size, (batch_size,), generator=generator, device=self.device
        )
        self.last_sample_online_count = int((indices >= self.offline_count).sum())
        return HybridBatch(
            self.observation[indices], self.continuous[indices], self.gripper[indices],
            self.reward[indices], self.next_observation[indices], self.done[indices],
        )

    def metadata(self) -> dict[str, int]:
        return {
            "capacity": self.capacity, "size": self.size,
            "offline_transition_count": self.offline_count,
            "online_transition_count": self.online_count,
        }


def restore_hybrid_awac_trainer(
    checkpoint_path: str | Path, *, device: torch.device,
) -> tuple[HybridAWACTrainer, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if payload.get("format_version") not in {
        "offline_awac_v2_hybrid", "online_awac_v2_hybrid",
        "offline_awac_v3_geometric_milestone_state",
        "online_awac_v3_geometric_milestone_state",
    }:
        raise ValueError("Online AWAC requires an Offline/Online Hybrid AWAC checkpoint")
    config = HybridAWACConfig(**payload["training_config"])
    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    trainer = HybridAWACTrainer(config, mean, std, payload["actor"], device)
    trainer.q1.load_state_dict(payload["critic_q1"])
    trainer.q2.load_state_dict(payload["critic_q2"])
    trainer.target_q1.load_state_dict(payload["target_q1"])
    trainer.target_q2.load_state_dict(payload["target_q2"])
    trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    trainer.q1_optimizer.load_state_dict(payload["critic_q1_optimizer"])
    trainer.q2_optimizer.load_state_dict(payload["critic_q2_optimizer"])
    trainer.step = int(payload["step"])
    return trainer, payload

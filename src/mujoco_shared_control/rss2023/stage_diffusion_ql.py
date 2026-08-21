"""Stage-conditioned Diffusion-QL V1 training on the frozen 4k replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion


@dataclass(frozen=True)
class DiffusionQLConfig:
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    batch_size: int = 512
    eta_q: float = 0.0
    steps: int = 80_000
    seed: int = 20260819
    grad_clip: float = 1.0


class TwinQ(nn.Module):
    def __init__(self, state_dim: int = 48, action_dim: int = 7) -> None:
        super().__init__()
        def net() -> nn.Sequential:
            return nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q1, self.q2 = net(), net()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((state, action), dim=-1)
        return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, path: str | Path, device: torch.device, action_mean: np.ndarray | None = None, action_std: np.ndarray | None = None) -> None:
        with np.load(path, allow_pickle=False) as d:
            self.obs = torch.as_tensor(d['obs'], dtype=torch.float32, device=device)
            self.next_obs = torch.as_tensor(d['next_obs'], dtype=torch.float32, device=device)
            self.action = torch.as_tensor(d['action'], dtype=torch.float32, device=device)
            if action_mean is not None and action_std is not None:
                self.action = (self.action - torch.as_tensor(action_mean, dtype=torch.float32, device=device)) / torch.as_tensor(action_std, dtype=torch.float32, device=device)
            self.reward = torch.as_tensor(d['reward'], dtype=torch.float32, device=device).reshape(-1, 1)
            self.done = torch.as_tensor(np.asarray(d['done']), dtype=torch.float32, device=device).reshape(-1, 1)

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        i = torch.randint(len(self.obs), (n,), generator=generator, device=self.obs.device)
        return self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.done[i]


def _normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _policy_action(model: RSS2023Diffusion, state: torch.Tensor, action_seed: torch.Tensor) -> torch.Tensor:
    """Differentiable t=0 denoising proxy used only for the Q actor term."""
    t = torch.zeros((state.shape[0],), dtype=torch.long, device=state.device)
    pred = model.denoiser(torch.cat((state, action_seed), dim=-1), t)[..., model.config.observation_dim:]
    return torch.clamp(action_seed - pred, -1.0, 1.0)


class DiffusionQLTrainer:
    def __init__(self, cfg: DiffusionQLConfig, diffusion_cfg: DiffusionConfig,
                 obs_mean: np.ndarray, obs_std: np.ndarray, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.actor = RSS2023Diffusion(diffusion_cfg).to(device)
        self.q = TwinQ().to(device)
        self.target_q = TwinQ().to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.mean = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(obs_std, dtype=torch.float32, device=device)
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    def update(self, replay: Replay) -> dict[str, float]:
        obs, action, reward, next_obs, done = replay.sample(self.cfg.batch_size, self.rng)
        obs_n, next_n = _normalise(obs, self.mean, self.std), _normalise(next_obs, self.mean, self.std)
        with torch.no_grad():
            next_seed = torch.randn(action.shape, generator=self.rng, device=action.device)
            next_action = _policy_action(self.actor, next_n, next_seed).clamp(-1, 1)
            tq1, tq2 = self.target_q(next_n, next_action)
            target = reward + self.cfg.gamma * (1.0 - done) * torch.minimum(tq1, tq2)
        q1, q2 = self.q(obs_n, action)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True); q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip); self.q_opt.step()
        bc_loss = self.actor.loss(obs_n, action)
        policy_seed = torch.randn(action.shape, generator=self.rng, device=action.device)
        policy_action = _policy_action(self.actor, obs_n, policy_seed)
        pq1, pq2 = self.q(obs_n, policy_action)
        q_loss_raw = -torch.minimum(pq1, pq2).mean()
        q_scale = torch.minimum(pq1.detach(), pq2.detach()).std().clamp_min(1e-3)
        actor_loss = bc_loss + self.cfg.eta_q * q_loss_raw / q_scale
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip); self.actor_opt.step()
        with torch.no_grad():
            for src, dst in zip(self.q.parameters(), self.target_q.parameters(), strict=True): dst.lerp_(src, self.cfg.tau)
        return {'actor_loss': float(actor_loss.item()), 'bc_loss': float(bc_loss.item()), 'q_loss': float(q_loss.item()),
                'q_loss_raw': float(q_loss_raw.item()), 'q_scale': float(q_scale.item()),
                'q_mean': float(torch.minimum(pq1, pq2).mean().item())}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_replay_stats(replay_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(replay_path, allow_pickle=False) as d:
        obs = np.asarray(d['obs'], np.float32); action = np.asarray(d['action'], np.float32)
    om=obs.mean(0).astype(np.float32); os=np.maximum(obs.std(0), 1e-6).astype(np.float32); om[43:]=0.; os[43:]=1.
    return om, os, action.mean(0).astype(np.float32), np.maximum(action.std(0),1e-6).astype(np.float32)


def checkpoint_payload(trainer: DiffusionQLTrainer, step: int, cfg: DiffusionQLConfig, diffusion_cfg: DiffusionConfig, action_mean: np.ndarray | None = None, action_std: np.ndarray | None = None) -> dict[str, Any]:
    if action_mean is None: action_mean, action_std = np.zeros(7, np.float32), np.ones(7, np.float32)
    return {'format_version': 'stage-diffusion-ql-v1', 'step': step, 'model': trainer.actor.state_dict(),
            'diffusion_config': diffusion_cfg.state_dict(), 'training_config': asdict(cfg),
            'observation_normalizer': {'mean': trainer.mean.detach().cpu().numpy(), 'std': trainer.std.detach().cpu().numpy()},
            'action_normalizer': {'mean': np.asarray(action_mean, np.float32), 'std': np.asarray(action_std, np.float32)},
            'q1': trainer.q.q1.state_dict(), 'q2': trainer.q.q2.state_dict(), 'target_q1': trainer.target_q.q1.state_dict(), 'target_q2': trainer.target_q.q2.state_dict()}

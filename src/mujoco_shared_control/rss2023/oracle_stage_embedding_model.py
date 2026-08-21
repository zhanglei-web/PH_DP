"""Oracle Stage V2: independent physical/stage encoders with latent concatenation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mujoco_shared_control.rss2023.model import _extract, make_beta_schedule


@dataclass(frozen=True)
class StageEmbeddingDiffusionConfig:
    physical_dim: int = 43
    stage_dim: int = 5
    stage_embedding_dim: int = 32
    condition_hidden_dim: int = 128
    action_dim: int = 7
    num_diffusion_steps: int = 50
    beta_schedule: str = "sigmoid"
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 128

    @property
    def observation_dim(self) -> int:
        return self.physical_dim + self.stage_dim

    def state_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation_dim"] = self.observation_dim
        return value


class StageEmbeddingCondition(nn.Module):
    def __init__(self, config: StageEmbeddingDiffusionConfig) -> None:
        super().__init__()
        self.physical_encoder = nn.Sequential(nn.Linear(config.physical_dim, 128), nn.SiLU())
        self.stage_encoder = nn.Sequential(nn.Linear(config.stage_dim, config.stage_embedding_dim), nn.SiLU())
        self.condition_projection = nn.Sequential(nn.Linear(128 + config.stage_embedding_dim, config.condition_hidden_dim), nn.SiLU())

    def forward(self, physical: Tensor, stage: Tensor) -> Tensor:
        physical_feature = self.physical_encoder(physical)
        stage_feature = self.stage_encoder(stage)
        return self.condition_projection(torch.cat((physical_feature, stage_feature), dim=-1))


class _ConditionDenoiser(nn.Module):
    def __init__(self, config: StageEmbeddingDiffusionConfig) -> None:
        super().__init__()
        input_dim = config.condition_hidden_dim + config.action_dim
        self.layer1 = nn.Linear(input_dim, config.hidden_dim)
        self.layer1_embedding = nn.Embedding(config.num_diffusion_steps, config.hidden_dim)
        self.layer2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layer2_embedding = nn.Embedding(config.num_diffusion_steps, config.hidden_dim)
        self.layer3 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layer3_embedding = nn.Embedding(config.num_diffusion_steps, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, input_dim)

    def _conditional(self, values: Tensor, timestep: Tensor, layer: nn.Linear, embedding: nn.Embedding) -> Tensor:
        return layer(values) * embedding(timestep)

    def forward(self, values: Tensor, timesteps: Tensor) -> Tensor:
        values = F.silu(self._conditional(values, timesteps, self.layer1, self.layer1_embedding))
        values = F.silu(self._conditional(values, timesteps, self.layer2, self.layer2_embedding))
        values = F.silu(self._conditional(values, timesteps, self.layer3, self.layer3_embedding))
        return self.output(values)


class StageEmbeddingDiffusion(nn.Module):
    def __init__(self, config: StageEmbeddingDiffusionConfig = StageEmbeddingDiffusionConfig()) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = StageEmbeddingCondition(config)
        self.denoiser = _ConditionDenoiser(config)
        betas = make_beta_schedule(type("Schedule", (), {"num_diffusion_steps": config.num_diffusion_steps, "beta_min": config.beta_min, "beta_max": config.beta_max, "beta_schedule": config.beta_schedule})())
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _condition(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != self.config.observation_dim:
            raise ValueError("unexpected V2 observation dimension")
        return self.condition_encoder(observation[..., :self.config.physical_dim], observation[..., self.config.physical_dim:])

    def q_sample(self, clean_action: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(clean_action) if noise is None else noise
        noisy = _extract(self.sqrt_alphas_cumprod, timesteps, clean_action) * clean_action + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_action) * noise
        return noisy, noise

    def loss(self, observation: Tensor, clean_action: Tensor, timesteps: Tensor | None = None) -> Tensor:
        batch = observation.shape[0]
        timesteps = torch.randint(self.config.num_diffusion_steps, (batch,), device=observation.device) if timesteps is None else timesteps
        noisy, target = self.q_sample(clean_action, timesteps)
        predicted = self.denoiser(torch.cat((self._condition(observation), noisy), dim=-1), timesteps)
        return F.mse_loss(predicted[..., self.config.condition_hidden_dim:], target)

    @torch.no_grad()
    def assist(self, observation: Tensor, human_action: Tensor, gamma: float, *, generator: torch.Generator | None = None) -> Tensor:
        if not 0.0 <= gamma <= 1.0: raise ValueError("gamma must be between zero and one")
        squeeze = observation.ndim == 1
        if squeeze: observation, human_action = observation.unsqueeze(0), human_action.unsqueeze(0)
        if observation.shape[-1] != self.config.observation_dim or human_action.shape[-1] != self.config.action_dim: raise ValueError("unexpected V2 input dimension")
        step = int((self.config.num_diffusion_steps - 1) * gamma)
        if step == 0: return human_action.squeeze(0) if squeeze else human_action.clone()
        timesteps = torch.full((human_action.shape[0],), step, dtype=torch.long, device=human_action.device)
        action, _ = self.q_sample(human_action, timesteps, torch.randn(human_action.shape, dtype=human_action.dtype, device=human_action.device, generator=generator))
        condition = self._condition(observation)
        for timestep in reversed(range(step)):
            ts = torch.full((action.shape[0],), timestep, dtype=torch.long, device=action.device)
            epsilon = self.denoiser(torch.cat((condition, action), dim=-1), ts)[..., self.config.condition_hidden_dim:]
            alpha = _extract(self.alphas, ts, action); noise_scale = _extract(self.sqrt_one_minus_alphas_cumprod, ts, action)
            mean = (action - ((1.0 - alpha) / noise_scale) * epsilon) / torch.sqrt(alpha)
            action = mean + torch.sqrt(_extract(self.betas, ts, action)) * torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator) if timestep > 0 else mean
        return action.squeeze(0) if squeeze else action

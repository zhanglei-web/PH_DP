"""Oracle Stage FiLM conditioning for the existing vector diffusion denoiser."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mujoco_shared_control.rss2023.model import _extract, make_beta_schedule


@dataclass(frozen=True)
class OracleFiLMStageConfig:
    physical_dim: int = 43
    stage_dim: int = 5
    stage_embedding_dim: int = 64
    action_dim: int = 7
    num_diffusion_steps: int = 50
    beta_schedule: str = "sigmoid"
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 128

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


class OracleFiLMStageDenoiser(nn.Module):
    def __init__(self, config: OracleFiLMStageConfig):
        super().__init__(); self.config = config
        self.stage_encoder = nn.Sequential(nn.Linear(config.stage_dim, 32), nn.SiLU(), nn.Linear(32, config.stage_embedding_dim), nn.SiLU())
        input_dim = config.physical_dim + config.action_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim)])
        self.time_embeddings = nn.ModuleList([nn.Embedding(config.num_diffusion_steps, config.hidden_dim) for _ in range(3)])
        self.film_heads = nn.ModuleList([nn.Linear(config.stage_embedding_dim, 2 * config.hidden_dim) for _ in range(3)])
        for head in self.film_heads:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.output = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, physical: Tensor, noisy_action: Tensor, timesteps: Tensor, stage: Tensor) -> Tensor:
        if physical.shape[-1] != self.config.physical_dim or stage.shape[-1] != self.config.stage_dim: raise ValueError("invalid FiLM condition shape")
        stage_embedding = self.stage_encoder(stage); values = torch.cat((physical, noisy_action), dim=-1)
        for layer, time_embedding, film_head in zip(self.layers, self.time_embeddings, self.film_heads, strict=True):
            values = layer(values) * time_embedding(timesteps)
            gamma, beta = film_head(stage_embedding).chunk(2, dim=-1)
            values = (1.0 + gamma) * values + beta
            values = F.softplus(values)
        return self.output(values)

    def film_parameters(self):
        return self.film_heads.parameters()


class OracleFiLMStageDiffusion(nn.Module):
    def __init__(self, config: OracleFiLMStageConfig = OracleFiLMStageConfig()):
        super().__init__(); self.config = config; self.denoiser = OracleFiLMStageDenoiser(config)
        schedule = type("Schedule", (), {"num_diffusion_steps": config.num_diffusion_steps, "beta_min": config.beta_min, "beta_max": config.beta_max, "beta_schedule": config.beta_schedule})()
        betas = make_beta_schedule(schedule); self.register_buffer("betas", betas); self.register_buffer("alphas", 1.0 - betas)
        ac = torch.cumprod(1.0 - betas, dim=0); self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(ac)); self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - ac))

    def q_sample(self, clean_action: Tensor, timesteps: Tensor, noise: Tensor | None = None):
        noise = torch.randn_like(clean_action) if noise is None else noise
        return _extract(self.sqrt_alphas_cumprod, timesteps, clean_action) * clean_action + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_action) * noise, noise

    def loss(self, physical: Tensor, stage: Tensor, clean_action: Tensor, timesteps: Tensor | None = None):
        timesteps = torch.randint(self.config.num_diffusion_steps, (physical.shape[0],), device=physical.device) if timesteps is None else timesteps
        noisy, target = self.q_sample(clean_action, timesteps)
        return F.mse_loss(self.denoiser(physical, noisy, timesteps, stage), target)

    @torch.no_grad()
    def assist(self, physical: Tensor, stage: Tensor, human_action: Tensor, gamma: float, *, generator: torch.Generator | None = None) -> Tensor:
        if not 0.0 <= gamma <= 1.0: raise ValueError("gamma must be between zero and one")
        squeeze = physical.ndim == 1
        if squeeze: physical, stage, human_action = physical.unsqueeze(0), stage.unsqueeze(0), human_action.unsqueeze(0)
        step = int((self.config.num_diffusion_steps - 1) * gamma)
        if step == 0: return human_action.squeeze(0) if squeeze else human_action.clone()
        ts = torch.full((human_action.shape[0],), step, dtype=torch.long, device=human_action.device)
        noise = torch.randn(human_action.shape, dtype=human_action.dtype, device=human_action.device, generator=generator)
        action, _ = self.q_sample(human_action, ts, noise)
        for timestep in reversed(range(step)):
            current = torch.full((action.shape[0],), timestep, dtype=torch.long, device=action.device)
            epsilon = self.denoiser(physical, action, current, stage)
            alpha = _extract(self.alphas, current, action); noise_scale = _extract(self.sqrt_one_minus_alphas_cumprod, current, action)
            mean = (action - ((1.0 - alpha) / noise_scale) * epsilon) / torch.sqrt(alpha)
            action = mean if timestep == 0 else mean + torch.sqrt(_extract(self.betas, current, action)) * torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator)
        return action.squeeze(0) if squeeze else action

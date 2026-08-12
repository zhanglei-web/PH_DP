"""Conditional vector DDPM following the RSS 2023 Diffusha formulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    observation_dim: int = 29
    action_dim: int = 8
    num_diffusion_steps: int = 50
    beta_schedule: str = "sigmoid"
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 128

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_beta_schedule(config: DiffusionConfig) -> Tensor:
    if config.num_diffusion_steps < 2:
        raise ValueError("num_diffusion_steps must be at least 2")
    if not 0.0 < config.beta_min < config.beta_max < 1.0:
        raise ValueError("betas must satisfy 0 < beta_min < beta_max < 1")
    if config.beta_schedule == "linear":
        return torch.linspace(
            config.beta_min, config.beta_max, config.num_diffusion_steps
        )
    if config.beta_schedule == "sigmoid":
        positions = torch.linspace(-6.0, 6.0, config.num_diffusion_steps)
        return torch.sigmoid(positions) * (
            config.beta_max - config.beta_min
        ) + config.beta_min
    raise ValueError(f"unsupported beta schedule: {config.beta_schedule}")


def _extract(values: Tensor, timesteps: Tensor, sample: Tensor) -> Tensor:
    result = values.gather(0, timesteps.to(values.device))
    return result.reshape(timesteps.shape[0], *([1] * (sample.ndim - 1)))


class ConditionalLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, diffusion_steps: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.embedding = nn.Embedding(diffusion_steps, output_dim)
        nn.init.uniform_(self.embedding.weight)

    def forward(self, values: Tensor, timesteps: Tensor) -> Tensor:
        return self.linear(values) * self.embedding(timesteps)


class ConditionalDenoiser(nn.Module):
    """The three-layer timestep-conditioned MLP used by the original code."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        input_dim = config.observation_dim + config.action_dim
        self.layer1 = ConditionalLinear(
            input_dim, config.hidden_dim, config.num_diffusion_steps
        )
        self.layer2 = ConditionalLinear(
            config.hidden_dim, config.hidden_dim, config.num_diffusion_steps
        )
        self.layer3 = ConditionalLinear(
            config.hidden_dim, config.hidden_dim, config.num_diffusion_steps
        )
        self.output = nn.Linear(config.hidden_dim, input_dim)

    def forward(self, values: Tensor, timesteps: Tensor) -> Tensor:
        values = F.softplus(self.layer1(values, timesteps))
        values = F.softplus(self.layer2(values, timesteps))
        values = F.softplus(self.layer3(values, timesteps))
        return self.output(values)


class RSS2023Diffusion(nn.Module):
    """Diffuse only Cartesian actions while hard-conditioning on observations."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = ConditionalDenoiser(config)
        betas = make_beta_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self,
        clean_action: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(clean_action)
        noisy_action = (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean_action) * clean_action
            + _extract(
                self.sqrt_one_minus_alphas_cumprod, timesteps, clean_action
            )
            * noise
        )
        return noisy_action, noise

    def loss(
        self,
        observation: Tensor,
        clean_action: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        batch_size = observation.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                self.config.num_diffusion_steps,
                (batch_size,),
                device=observation.device,
            )
        noisy_action, target_noise = self.q_sample(clean_action, timesteps)
        model_input = torch.cat((observation, noisy_action), dim=-1)
        predicted = self.denoiser(model_input, timesteps)
        predicted_action_noise = predicted[..., self.config.observation_dim :]
        return F.mse_loss(predicted_action_noise, target_noise)

    @torch.no_grad()
    def assist(
        self,
        observation: Tensor,
        human_action: Tensor,
        gamma: float,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Forward diffuse a human command and reverse it under a fixed state."""
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        squeeze = observation.ndim == 1
        if squeeze:
            observation = observation.unsqueeze(0)
            human_action = human_action.unsqueeze(0)
        if observation.shape[-1] != self.config.observation_dim:
            raise ValueError("unexpected observation dimension")
        if human_action.shape[-1] != self.config.action_dim:
            raise ValueError("unexpected action dimension")
        step = int((self.config.num_diffusion_steps - 1) * gamma)
        if step == 0:
            return human_action.squeeze(0) if squeeze else human_action.clone()

        timesteps = torch.full(
            (human_action.shape[0],), step, dtype=torch.long, device=human_action.device
        )
        noise = torch.randn(
            human_action.shape,
            dtype=human_action.dtype,
            device=human_action.device,
            generator=generator,
        )
        action, _ = self.q_sample(human_action, timesteps, noise=noise)
        for timestep in reversed(range(step)):
            batch_timesteps = torch.full(
                (action.shape[0],),
                timestep,
                dtype=torch.long,
                device=action.device,
            )
            model_input = torch.cat((observation, action), dim=-1)
            predicted = self.denoiser(model_input, batch_timesteps)
            epsilon = predicted[..., self.config.observation_dim :]
            alpha = _extract(self.alphas, batch_timesteps, action)
            noise_scale = _extract(
                self.sqrt_one_minus_alphas_cumprod, batch_timesteps, action
            )
            mean = (action - ((1.0 - alpha) / noise_scale) * epsilon) / torch.sqrt(
                alpha
            )
            if timestep > 0:
                random_noise = torch.randn(
                    action.shape,
                    dtype=action.dtype,
                    device=action.device,
                    generator=generator,
                )
                action = mean + torch.sqrt(
                    _extract(self.betas, batch_timesteps, action)
                ) * random_noise
            else:
                action = mean
        return action.squeeze(0) if squeeze else action

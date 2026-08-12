"""The training and sampling behavior of the released RSS 2023 Diffusha code."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mujoco_shared_control.rss2023.model import (
    ConditionalDenoiser,
    DiffusionConfig,
    _extract,
    make_beta_schedule,
)


class PaperRSS2023Diffusion(nn.Module):
    """Diffuse state+action and hard-condition state exactly as released."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = ConditionalDenoiser(config)
        betas = make_beta_schedule(config)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_cumulative", torch.sqrt(cumulative))
        self.register_buffer("sqrt_one_minus_cumulative", torch.sqrt(1.0 - cumulative))

    def _diffuse(self, clean: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(clean)
        noisy = _extract(self.sqrt_cumulative, timesteps, clean) * clean
        noisy += _extract(self.sqrt_one_minus_cumulative, timesteps, clean) * noise
        noisy[..., : self.config.observation_dim] = clean[..., : self.config.observation_dim]
        return noisy, noise

    def loss(self, observation: Tensor, action: Tensor) -> Tensor:
        clean = torch.cat((observation, action), dim=-1)
        half = clean.shape[0] // 2 + 1
        timesteps = torch.randint(
            0, self.config.num_diffusion_steps, (half,), device=clean.device
        )
        timesteps = torch.cat(
            (timesteps, self.config.num_diffusion_steps - timesteps - 1)
        )[: clean.shape[0]]
        noisy, target_noise = self._diffuse(clean, timesteps)
        predicted_noise = self.denoiser(noisy, timesteps)
        # The released implementation includes conditional dimensions in this
        # loss; preserve that behavior for a genuine protocol reproduction.
        return (target_noise - predicted_noise).square().mean()

    @torch.no_grad()
    def assist(
        self,
        observation: Tensor,
        human_action: Tensor,
        gamma: float,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        squeeze = observation.ndim == 1
        if squeeze:
            observation = observation.unsqueeze(0)
            human_action = human_action.unsqueeze(0)
        step = int((self.config.num_diffusion_steps - 1) * gamma)
        if step == 0:
            return human_action.squeeze(0) if squeeze else human_action.clone()
        state = torch.cat((observation, human_action), dim=-1)
        timestep = torch.full((len(state),), step, device=state.device, dtype=torch.long)
        noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
        sample = _extract(self.sqrt_cumulative, timestep, state) * state
        sample += _extract(self.sqrt_one_minus_cumulative, timestep, state) * noise
        sample[..., : self.config.observation_dim] = observation
        for index in reversed(range(step)):
            times = torch.full((len(state),), index, device=state.device, dtype=torch.long)
            epsilon = self.denoiser(sample, times)
            alpha = _extract(self.alphas, times, sample)
            factor = (1.0 - alpha) / _extract(
                self.sqrt_one_minus_cumulative, times, sample
            )
            mean = (sample - factor * epsilon) / torch.sqrt(alpha)
            random_noise = torch.randn(
                sample.shape, device=sample.device, dtype=sample.dtype, generator=generator
            )
            sample = mean + torch.sqrt(_extract(self.betas, times, sample)) * random_noise
            sample[..., : self.config.observation_dim] = observation
        result = sample[..., self.config.observation_dim :]
        result = torch.clamp(result, -1.0, 1.0)
        return result.squeeze(0) if squeeze else result

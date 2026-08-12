"""Checkpoint loading and NumPy inference for a trained RSS 2023 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from mujoco_shared_control.rss2023.dataset import ACTION_DIM, OBSERVATION_DIM
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion


def _normalizer_arrays(
    state: dict[str, Any], expected_dim: int, name: str
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
        raise ValueError(f"checkpoint has invalid {name} normalizer shape")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"checkpoint has invalid {name} normalizer values")
    return mean, std


class RSS2023Predictor:
    """Apply forward/reverse diffusion to one normalized Cartesian user command."""

    def __init__(
        self,
        model: RSS2023Diffusion,
        observation_mean: NDArray[np.float32],
        observation_std: NDArray[np.float32],
        action_mean: NDArray[np.float32],
        action_std: NDArray[np.float32],
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.observation_mean = observation_mean
        self.observation_std = observation_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device_name: str = "auto",
        use_ema: bool = False,
    ) -> "RSS2023Predictor":
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        config = DiffusionConfig(**checkpoint["diffusion_config"])
        if config.observation_dim != OBSERVATION_DIM or config.action_dim != ACTION_DIM:
            raise ValueError("checkpoint dimensions do not match the 29+8 interface")
        model = RSS2023Diffusion(config)
        if use_ema:
            model.load_state_dict(checkpoint["ema"]["shadow"])
        else:
            model.load_state_dict(checkpoint["model"])
        observation_mean, observation_std = _normalizer_arrays(
            checkpoint["observation_normalizer"], OBSERVATION_DIM, "observation"
        )
        action_mean, action_std = _normalizer_arrays(
            checkpoint["action_normalizer"], ACTION_DIM, "action"
        )
        return cls(
            model,
            observation_mean,
            observation_std,
            action_mean,
            action_std,
            device=device,
        )

    @torch.no_grad()
    def predict(
        self,
        observation: NDArray[np.floating[Any]],
        human_action: NDArray[np.floating[Any]],
        *,
        gamma: float = 0.4,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        observation_array = np.asarray(observation, dtype=np.float32)
        action_array = np.asarray(human_action, dtype=np.float32)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(f"observation must have shape ({OBSERVATION_DIM},)")
        if action_array.shape != (ACTION_DIM,):
            raise ValueError(f"human_action must have shape ({ACTION_DIM},)")
        if not np.isfinite(observation_array).all() or not np.isfinite(action_array).all():
            raise ValueError("observation and human_action must be finite")

        return self.predict_batch(
            observation_array[None, :],
            action_array[None, :],
            gamma=gamma,
            seed=seed,
        )[0]

    @torch.no_grad()
    def predict_batch(
        self,
        observations: NDArray[np.floating[Any]],
        human_actions: NDArray[np.floating[Any]],
        *,
        gamma: float = 0.4,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        observation_array = np.asarray(observations, dtype=np.float32)
        action_array = np.asarray(human_actions, dtype=np.float32)
        if observation_array.ndim != 2 or observation_array.shape[1] != OBSERVATION_DIM:
            raise ValueError(f"observations must have shape (N, {OBSERVATION_DIM})")
        if action_array.shape != (observation_array.shape[0], ACTION_DIM):
            raise ValueError(f"human_actions must have shape (N, {ACTION_DIM})")
        if not np.isfinite(observation_array).all() or not np.isfinite(action_array).all():
            raise ValueError("observations and human_actions must be finite")

        observation_normalized = torch.as_tensor(
            (observation_array - self.observation_mean) / self.observation_std,
            device=self.device,
        )
        action_normalized = torch.as_tensor(
            (action_array - self.action_mean) / self.action_std,
            device=self.device,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        assisted_normalized = self.model.assist(
            observation_normalized,
            action_normalized,
            gamma,
            generator=generator,
        )
        assisted = (
            assisted_normalized.cpu().numpy() * self.action_std + self.action_mean
        ).astype(np.float64)
        quaternion_norm = np.linalg.norm(assisted[:, 3:7], axis=1)
        if not np.isfinite(assisted).all() or np.any(quaternion_norm <= 1e-8):
            raise RuntimeError("diffusion model produced an invalid Cartesian command")
        assisted[:, 3:7] /= quaternion_norm[:, None]
        negative = assisted[:, 3] < 0.0
        assisted[negative, 3:7] *= -1.0
        assisted[:, 7] = np.clip(assisted[:, 7], 0.0, 0.08)
        return assisted

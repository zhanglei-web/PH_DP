"""Native unit-ball Gaussian Actor for constrained SAC action semantics v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from mujoco_shared_control.actor_bc.model import ACTOR_ACTION_DIM, HIDDEN_DIM, POLICY_STATE_DIM
from mujoco_shared_control.sac.actor import LOG_STD_INIT, LOG_STD_MAX, LOG_STD_MIN


_SMALL_RADIUS = 1e-4


def _log_sech_squared(value: torch.Tensor) -> torch.Tensor:
    return 2.0 * (np.log(2.0) - value - F.softplus(-2.0 * value))


def _radial_ratio(radius: torch.Tensor) -> torch.Tensor:
    """Stable tanh(r)/r with its analytic limit at zero."""
    radius2 = radius.square()
    series = 1.0 - radius2 / 3.0 + 2.0 * radius2.square() / 15.0
    regular = torch.tanh(radius) / radius.clamp_min(torch.finfo(radius.dtype).tiny)
    return torch.where(radius < _SMALL_RADIUS, series, regular)


def _radial_log_ratio(radius: torch.Tensor) -> torch.Tensor:
    """Stable log(tanh(r)/r), including r=0."""
    radius2 = radius.square()
    series = -radius2 / 3.0 + 7.0 * radius2.square() / 90.0
    regular = torch.log(torch.tanh(radius).clamp_min(torch.finfo(radius.dtype).tiny)) - torch.log(
        radius.clamp_min(torch.finfo(radius.dtype).tiny)
    )
    return torch.where(radius < _SMALL_RADIUS, series, regular)


def radial_squash(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map R^3 to the open unit L2 ball and return log|det J|."""
    if vector.shape[-1] != 3:
        raise ValueError("radial_squash expects a final dimension of 3")
    radius_squared = vector.square().sum(dim=-1, keepdim=True)
    small = radius_squared < _SMALL_RADIUS**2
    # Never differentiate sqrt at exactly zero: the selected small-radius branch
    # is an analytic series in r^2, while the regular branch gets a safe radius.
    regular_radius = torch.sqrt(
        torch.where(small, torch.full_like(radius_squared, _SMALL_RADIUS**2), radius_squared)
    )
    ratio_series = 1.0 - radius_squared / 3.0 + 2.0 * radius_squared.square() / 15.0
    ratio_regular = torch.tanh(regular_radius) / regular_radius
    action = vector * torch.where(small, ratio_series, ratio_regular)
    # In finite precision tanh(large r) may round to one. Pull only that numerical
    # boundary inward; the mathematical transform remains tanh(r) u/r.
    action_norm_squared = action.square().sum(dim=-1, keepdim=True)
    upper = torch.nextafter(
        torch.ones_like(action_norm_squared), torch.zeros_like(action_norm_squared)
    )
    saturated_scale = upper / torch.sqrt(action_norm_squared.clamp_min(1.0))
    action = action * torch.where(
        action_norm_squared >= 1.0, saturated_scale, torch.ones_like(action_norm_squared)
    )
    log_det_series = -5.0 * radius_squared / 3.0 + 29.0 * radius_squared.square() / 90.0
    log_det_regular = _log_sech_squared(regular_radius) + 2.0 * _radial_log_ratio(regular_radius)
    log_det = torch.where(small, log_det_series, log_det_regular)
    return action, log_det


def constrained_transform(pre_transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply two 3-D radial squashes plus scalar tanh; return summed logdet."""
    if pre_transform.shape[-1] != ACTOR_ACTION_DIM:
        raise ValueError("constrained transform expects 7-D input")
    xyz, xyz_log_det = radial_squash(pre_transform[..., :3])
    rotation, rotation_log_det = radial_squash(pre_transform[..., 3:6])
    gripper = torch.tanh(pre_transform[..., 6:7])
    gripper_log_det = _log_sech_squared(pre_transform[..., 6:7])
    action = torch.cat((xyz, rotation, gripper), dim=-1)
    return action, xyz_log_det + rotation_log_det + gripper_log_det


def project_to_admissible(action: torch.Tensor) -> torch.Tensor:
    """Exact normalized-space counterpart of the frozen environment adapter."""
    if action.shape[-1] != ACTOR_ACTION_DIM:
        raise ValueError("projection expects 7-D action")
    result = action.clone()
    for start in (0, 3):
        vector = result[..., start : start + 3]
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scale = torch.where(norm > 1.0, 1.0 / norm.clamp_min(1e-12), torch.ones_like(norm))
        result[..., start : start + 3] = vector * scale
    result[..., 6] = result[..., 6].clamp(-1.0, 1.0)
    return result


class SACConstrainedGaussianActor(nn.Module):
    """Gaussian Actor whose RL action is native to B3 x B3 x (-1,1)."""

    def __init__(self, log_std_min: float = LOG_STD_MIN, log_std_max: float = LOG_STD_MAX) -> None:
        super().__init__()
        if not log_std_min < log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.trunk = nn.Sequential(
            nn.Linear(POLICY_STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        )
        self.mean_head = nn.Linear(HIDDEN_DIM, ACTOR_ACTION_DIM)
        self.log_std_head = nn.Linear(HIDDEN_DIM, ACTOR_ACTION_DIM)

    def distribution_stats(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std, log_std.exp()

    def sample_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _log_std, std = self.distribution_stats(state)
        normal = Normal(mean, std)
        pre_transform = normal.rsample()
        action, log_det = constrained_transform(pre_transform)
        log_prob = normal.log_prob(pre_transform).sum(dim=-1, keepdim=True) - log_det
        mean_action, _ = constrained_transform(mean)
        return action, log_prob, mean_action

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        mean, _log_std, _std = self.distribution_stats(state)
        return constrained_transform(mean)[0]


@dataclass(frozen=True)
class ConstrainedBCInitialization:
    checkpoint: str
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_spec: dict[str, float]
    manifest_content_sha: str


def initialize_constrained_from_bc(
    actor: SACConstrainedGaussianActor,
    checkpoint_path: str | Path,
    log_std_init: float = LOG_STD_INIT,
) -> ConstrainedBCInitialization:
    """Exact-copy BC mean path before output-space redistillation."""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    mapping = {
        "network.0.weight": "trunk.0.weight", "network.0.bias": "trunk.0.bias",
        "network.2.weight": "trunk.2.weight", "network.2.bias": "trunk.2.bias",
        "network.4.weight": "trunk.4.weight", "network.4.bias": "trunk.4.bias",
        "network.6.weight": "mean_head.weight", "network.6.bias": "mean_head.bias",
    }
    own = actor.state_dict()
    with torch.no_grad():
        for source_key, target_key in mapping.items():
            own[target_key].copy_(source[source_key])
        own["log_std_head.weight"].zero_()
        own["log_std_head.bias"].fill_(log_std_init)
    actor.load_state_dict(own)
    return ConstrainedBCInitialization(
        str(checkpoint_path), np.asarray(checkpoint["observation_mean"], np.float32),
        np.asarray(checkpoint["observation_std"], np.float32), dict(checkpoint["action_spec"]),
        checkpoint["manifest_content_sha"],
    )


def configure_constrained_distillation(actor: SACConstrainedGaussianActor) -> tuple[nn.Parameter, ...]:
    for parameter in actor.trunk.parameters():
        parameter.requires_grad_(True)
    for parameter in actor.mean_head.parameters():
        parameter.requires_grad_(True)
    for parameter in actor.log_std_head.parameters():
        parameter.requires_grad_(False)
    return tuple(actor.trunk.parameters()) + tuple(actor.mean_head.parameters())

"""Standard state-dependent squashed Gaussian actor for SAC v1."""

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


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
LOG_STD_INIT = -3.0
ATANH_EPSILON = 1e-6


@dataclass(frozen=True)
class BCInitializationReport:
    checkpoint: str
    option: str
    copied_keys: tuple[str, ...]
    converted_keys: tuple[str, ...]
    new_keys: tuple[str, ...]
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_spec: dict[str, float]
    manifest_content_sha: str


class SACGaussianActor(nn.Module):
    """42 -> 256 x 3 trunk with state-dependent Gaussian heads and tanh squash."""

    def __init__(
        self,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        if not log_std_min < log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = nn.Sequential(
            nn.Linear(POLICY_STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        )
        self.mean_head = nn.Linear(HIDDEN_DIM, ACTOR_ACTION_DIM)
        self.log_std_head = nn.Linear(HIDDEN_DIM, ACTOR_ACTION_DIM)

    def distribution_stats(
        self, normalized_policy_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(normalized_policy_state)
        mean = self.mean_head(features)
        # Hard bounds make the chosen units explicit and avoid the asymmetric
        # initial value produced by a full-range tanh remapping.
        log_std = torch.clamp(
            self.log_std_head(features), self.log_std_min, self.log_std_max
        )
        return mean, log_std, log_std.exp()

    @staticmethod
    def _squashed_log_prob(
        normal: Normal, pre_squash: torch.Tensor
    ) -> torch.Tensor:
        # Equivalent to log(1-tanh(u)^2), but stable for large |u|.
        correction = 2.0 * (
            np.log(2.0) - pre_squash - F.softplus(-2.0 * pre_squash)
        )
        return (normal.log_prob(pre_squash) - correction).sum(dim=-1, keepdim=True)

    def sample_action(
        self, normalized_policy_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _log_std, std = self.distribution_stats(normalized_policy_state)
        normal = Normal(mean, std)
        pre_squash = normal.rsample()
        action = torch.tanh(pre_squash)
        log_prob = self._squashed_log_prob(normal, pre_squash)
        return action, log_prob, torch.tanh(mean)

    def deterministic_action(self, normalized_policy_state: torch.Tensor) -> torch.Tensor:
        mean, _log_std, _std = self.distribution_stats(normalized_policy_state)
        return torch.tanh(mean)


def initialize_from_bc(
    actor: SACGaussianActor,
    checkpoint_path: str | Path,
    *,
    option: str = "direct_head_copy",
    log_std_init: float = LOG_STD_INIT,
) -> BCInitializationReport:
    """Initialize without any optimizer or policy/reward gradient update.

    `direct_head_copy` is executable without calibration. `trunk_only` leaves the
    mean head at its PyTorch initialization. Behavior-preserving atanh calibration
    is deliberately not performed here because it is a supervised fitting step.
    """
    if option not in {"direct_head_copy", "trunk_only"}:
        raise ValueError("option must be 'direct_head_copy' or 'trunk_only'")
    if not actor.log_std_min <= log_std_init <= actor.log_std_max:
        raise ValueError("log_std_init must be inside actor bounds")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint["model_state_dict"]
    mapping = {
        "network.0.weight": "trunk.0.weight", "network.0.bias": "trunk.0.bias",
        "network.2.weight": "trunk.2.weight", "network.2.bias": "trunk.2.bias",
        "network.4.weight": "trunk.4.weight", "network.4.bias": "trunk.4.bias",
    }
    own = actor.state_dict()
    for source, target in mapping.items():
        if state[source].shape != own[target].shape:
            raise ValueError(f"BC/SAC shape mismatch: {source} -> {target}")
        own[target].copy_(state[source])
    copied = list(mapping)
    if option == "direct_head_copy":
        own["mean_head.weight"].copy_(state["network.6.weight"])
        own["mean_head.bias"].copy_(state["network.6.bias"])
        copied.extend(("network.6.weight", "network.6.bias"))
    own["log_std_head.weight"].zero_()
    own["log_std_head.bias"].fill_(float(log_std_init))
    actor.load_state_dict(own)
    return BCInitializationReport(
        str(checkpoint_path), option, tuple(copied), (),
        ("log_std_head.weight", "log_std_head.bias")
        + (() if option == "direct_head_copy" else ("mean_head.weight", "mean_head.bias")),
        np.asarray(checkpoint["observation_mean"], dtype=np.float32),
        np.asarray(checkpoint["observation_std"], dtype=np.float32),
        dict(checkpoint["action_spec"]), checkpoint["manifest_content_sha"],
    )


def freeze_for_mean_calibration(actor: SACGaussianActor) -> tuple[nn.Parameter, nn.Parameter]:
    """Freeze every parameter except the pre-tanh mean head."""
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.mean_head.weight.requires_grad_(True)
    actor.mean_head.bias.requires_grad_(True)
    return actor.mean_head.weight, actor.mean_head.bias


def configure_full_mean_path_distillation(actor: SACGaussianActor) -> tuple[nn.Parameter, ...]:
    """Train the student trunk and mean head while freezing exploration state."""
    for parameter in actor.trunk.parameters():
        parameter.requires_grad_(True)
    for parameter in actor.mean_head.parameters():
        parameter.requires_grad_(True)
    for parameter in actor.log_std_head.parameters():
        parameter.requires_grad_(False)
    return tuple(actor.trunk.parameters()) + tuple(actor.mean_head.parameters())

"""Twin state-action critics for SAC Core v1."""

from __future__ import annotations

import torch
from torch import nn

from mujoco_shared_control.actor_bc.model import ACTOR_ACTION_DIM, HIDDEN_DIM, POLICY_STATE_DIM


CRITIC_INPUT_DIM = POLICY_STATE_DIM + ACTOR_ACTION_DIM


class SACCritic(nn.Module):
    """49 -> 256 x 3 -> 1 scalar Q function."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CRITIC_INPUT_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != POLICY_STATE_DIM:
            raise ValueError(f"observation last dimension must be {POLICY_STATE_DIM}")
        if action.shape[-1] != ACTOR_ACTION_DIM:
            raise ValueError(f"action last dimension must be {ACTOR_ACTION_DIM}")
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("observation and action batch shapes must match")
        return self.network(torch.cat((observation, action), dim=-1))


class TwinSACCritic(nn.Module):
    """Two independent Q networks."""

    def __init__(self) -> None:
        super().__init__()
        self.q1 = SACCritic()
        self.q2 = SACCritic()

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)

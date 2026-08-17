"""Fixed Actor BC v1 network and inference boundary."""

from __future__ import annotations

import torch
from torch import nn


POLICY_STATE_DIM = 42
ACTOR_ACTION_DIM = 7
HIDDEN_DIM = 256


class ActorBC(nn.Module):
    """42 -> 256 -> 256 -> 256 -> 7 SiLU MLP without output activation."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(POLICY_STATE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, ACTOR_ACTION_DIM),
        )

    def forward(self, policy_state: torch.Tensor) -> torch.Tensor:
        return self.network(policy_state)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

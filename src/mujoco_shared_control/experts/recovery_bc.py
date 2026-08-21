"""Small feed-forward behavior-cloning policy for the E2 learned user."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RecoveryBCPolicy(nn.Module):
    """43D physical state to normalized 6D motion plus an OPEN/CLOSE logit."""

    input_dim = 43

    def __init__(self, input_dim: int = 43) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.motion_head = nn.Linear(256, 6)
        self.gripper_head = nn.Linear(256, 1)

    def forward(self, state_43: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state_43)
        return torch.tanh(self.motion_head(features)), self.gripper_head(features).squeeze(-1)

    @torch.no_grad()
    def action(self, state_43: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Return canonical ExpertActionSpec normalized action values."""
        value = (np.asarray(state_43, np.float32) - mean) / std
        motion, logit = self(torch.from_numpy(value[None]))
        action = np.empty(7, np.float32)
        action[:6] = motion[0].cpu().numpy()
        action[6] = 1.0 if float(logit[0]) >= 0.0 else -0.25
        return action

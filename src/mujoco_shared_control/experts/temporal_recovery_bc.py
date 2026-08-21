"""Causal temporal behavior-cloning policy for the E2 learned user."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CausalConvBlock(nn.Module):
    """A Conv1d block whose output at index t sees inputs at most through t."""

    def __init__(self, input_dim: int, output_dim: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=3, dilation=dilation, padding=self.padding)
        self.activation = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.conv(value)
        if self.padding:
            result = result[..., :-self.padding]
        return self.activation(result)


class UnifiedStageAwareTemporalBC(nn.Module):
    """48D current-stage features over a causal history to one 7D action."""

    input_dim = 48

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            CausalConvBlock(48, 64, 1),
            CausalConvBlock(64, 128, 2),
            CausalConvBlock(128, 128, 4),
        )
        self.representation = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.motion_head = nn.Linear(128, 6)
        self.gripper_head = nn.Linear(128, 1)

    def temporal_features(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError("history must have shape [B,L,48]")
        return self.temporal(history.transpose(1, 2)).transpose(1, 2)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.representation(self.temporal_features(history)[:, -1])
        return torch.tanh(self.motion_head(features)), self.gripper_head(features).squeeze(-1)

    @torch.no_grad()
    def action(self, history: np.ndarray) -> np.ndarray:
        """Return canonical ExpertActionSpec normalized action values."""
        motion, logit = self(torch.from_numpy(np.asarray(history, np.float32)[None]))
        action = np.empty(7, np.float32)
        action[:6] = motion[0].cpu().numpy()
        action[6] = 1.0 if float(logit[0]) >= 0.0 else -0.25
        return action

"""Small causal TCN for five-class active-stage prediction."""

from __future__ import annotations

import torch
from torch import nn


class CausalResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=3, dilation=dilation, padding=self.padding)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if input_dim == output_dim else nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.conv(value)
        if self.padding:
            result = result[..., :-self.padding]
        return self.activation(self.dropout(result) + self.residual(value))


class StageTCNV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            CausalResidualBlock(19, 64, 1, 0.1),
            CausalResidualBlock(64, 128, 2, 0.1),
            CausalResidualBlock(128, 128, 4, 0.1),
        )
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (20, 19):
            raise ValueError("StageTCNV1 input must have shape [B,20,19]")
        temporal = self.temporal(value.transpose(1, 2))
        return self.classifier(temporal[..., -1])

    def posterior(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(value), dim=-1)

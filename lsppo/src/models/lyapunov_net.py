"""Lyapunov certificate network and losses."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def build_lyapunov_mlp(input_dim: int, hidden_sizes: Iterable[int]) -> nn.Sequential:
    """Build a ReLU MLP that outputs a scalar pre-activation."""
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class LyapunovNet(nn.Module):
    """Non-negative Lyapunov certificate L_phi(s)."""

    def __init__(self, obs_dim: int, hidden_sizes: Iterable[int]) -> None:
        """Initialize Lyapunov network."""
        super().__init__()
        self.net = build_lyapunov_mlp(obs_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return non-negative Lyapunov values."""
        return F.softplus(self.net(obs).squeeze(-1))


def lyapunov_td_loss(
    current_l: torch.Tensor,
    step_cost: torch.Tensor,
    next_l: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute TD-style regression to cost-to-go."""
    target = step_cost + gamma * next_l
    return F.mse_loss(current_l, target.detach())


def lyapunov_decrease_loss(
    current_l: torch.Tensor,
    next_l: torch.Tensor,
    gamma: float,
    margin: float,
) -> torch.Tensor:
    """Enforce Lyapunov decrease condition."""
    violation = torch.relu(next_l - gamma * current_l + margin)
    return (violation ** 2).mean()

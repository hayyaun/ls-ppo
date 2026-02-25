"""Actor-critic model used by PPO variants."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
    activation: nn.Module = nn.Tanh,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a feed-forward MLP."""
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        layers.append(activation())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian actor and scalar critic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_sizes: Iterable[int],
        critic_hidden_sizes: Iterable[int],
        log_std_init: float = -0.5,
    ) -> None:
        """Create actor and critic networks."""
        super().__init__()
        self.actor = build_mlp(obs_dim, actor_hidden_sizes, act_dim, activation=nn.Tanh)
        self.critic = build_mlp(obs_dim, critic_hidden_sizes, 1, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Return action distribution for observation batch."""
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or return mean action and compute log-prob/value."""
        dist = self._distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value, dist.mean

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probability, entropy, and value for action batch."""
        dist = self._distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return critic values."""
        return self.critic(obs).squeeze(-1)

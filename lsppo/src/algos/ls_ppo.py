"""LS-PPO glue logic: Lyapunov updates and Lagrangian multiplier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from src.models.lyapunov_net import lyapunov_decrease_loss, lyapunov_td_loss


@dataclass
class LSPPOConfig:
    """Configuration for LS-PPO-specific components."""

    gamma: float = 0.99
    lam_lr: float = 1e-3
    lambda_update_every: int = 1
    cost_limit: float = 5.0
    lambda_max: float = 100.0
    lambda_init: float = 0.0
    lyap_loss_weight: float = 1.0
    lyap_weight_schedule: str = "constant"
    lyap_warmup_updates: int = 50
    lyap_td_weight: float = 1.0
    lyap_decrease_weight: float = 1.0
    lyap_margin: float = 0.01
    lyap_objective: str = "both"  # td, decrease, both


def estimate_empirical_discounted_cost(costs: np.ndarray, done: np.ndarray, gamma: float) -> float:
    """Estimate C_empirical as mean episodic discounted cost over completed episodes."""
    t_horizon, n_envs = costs.shape
    episodes: list[float] = []
    running_cost = np.zeros(n_envs, dtype=np.float32)
    running_discount = np.ones(n_envs, dtype=np.float32)

    for t in range(t_horizon):
        running_cost += running_discount * costs[t]
        running_discount *= gamma
        done_mask = done[t] > 0.5
        if np.any(done_mask):
            episodes.extend(running_cost[done_mask].tolist())
            running_cost[done_mask] = 0.0
            running_discount[done_mask] = 1.0

    # Fallback to partial episode estimate when no episode terminated in this batch.
    if episodes:
        return float(np.mean(episodes))
    return float(np.mean(running_cost))


class LSPPOCoordinator:
    """Coordinate Lyapunov optimization and lambda updates."""

    def __init__(
        self,
        lyap_net: torch.nn.Module,
        lyap_optimizer: torch.optim.Optimizer,
        cfg: LSPPOConfig,
        device: torch.device,
    ) -> None:
        """Initialize LS-PPO coordinator."""
        self.lyap_net = lyap_net
        self.lyap_optimizer = lyap_optimizer
        self.cfg = cfg
        self.device = device
        self.lambda_value = float(cfg.lambda_init)

    def current_lyap_weight(self, update_idx: int) -> float:
        """Return scheduled Lyapunov loss weight."""
        base = float(self.cfg.lyap_loss_weight)
        schedule = str(self.cfg.lyap_weight_schedule).lower()
        if schedule == "linear_warmup":
            warmup = max(1, int(self.cfg.lyap_warmup_updates))
            scale = min(1.0, float(update_idx + 1) / float(warmup))
            return base * scale
        return base

    def update_lyapunov(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        costs: np.ndarray,
        update_idx: int = 0,
    ) -> Dict[str, float]:
        """Run one Lyapunov-network update step."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        costs_t = torch.as_tensor(costs, dtype=torch.float32, device=self.device)

        current_l = self.lyap_net(obs_t)
        next_l = self.lyap_net(next_obs_t)

        td = lyapunov_td_loss(current_l, costs_t, next_l.detach(), self.cfg.gamma)
        dec = lyapunov_decrease_loss(current_l, next_l, self.cfg.gamma, self.cfg.lyap_margin)

        objective = self.cfg.lyap_objective.lower()
        if objective == "td":
            raw_loss = td
        elif objective == "decrease":
            raw_loss = dec
        else:
            raw_loss = self.cfg.lyap_td_weight * td + self.cfg.lyap_decrease_weight * dec
        effective_weight = self.current_lyap_weight(update_idx)
        loss = effective_weight * raw_loss

        self.lyap_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.lyap_optimizer.step()

        return {
            "lyap_loss": float(loss.detach().cpu().item()),
            "lyap_td_loss": float(td.detach().cpu().item()),
            "lyap_decrease_loss": float(dec.detach().cpu().item()),
            "lyap_effective_weight": float(effective_weight),
        }

    def update_lambda(self, empirical_cost: float) -> float:
        """Perform clipped dual update for Lagrangian multiplier."""
        updated = self.lambda_value + self.cfg.lam_lr * (empirical_cost - self.cfg.cost_limit)
        self.lambda_value = float(np.clip(updated, 0.0, self.cfg.lambda_max))
        return self.lambda_value

"""PPO optimization primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PPOConfig:
    """Hyperparameters used by PPO update."""

    clip_eps: float = 0.2
    epochs: int = 10
    mini_batch_size: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


class PPOTrainer:
    """PPO trainer with clipped surrogate objective."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: PPOConfig,
        device: torch.device,
    ) -> None:
        """Construct PPO trainer."""
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device

    def update(self, batch: Dict[str, np.ndarray], lambda_value: float = 0.0) -> Dict[str, float]:
        """Run PPO optimization and return scalar metrics."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        cost_adv = torch.as_tensor(batch["cost_advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std(unbiased=False) + 1e-8)
        combined_adv = advantages - lambda_value * cost_adv

        n_samples = obs.shape[0]
        indices = np.arange(n_samples)

        actor_losses = []
        value_losses = []
        entropy_terms = []
        approx_kls = []

        for _ in range(self.cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.cfg.mini_batch_size):
                mb = indices[start : start + self.cfg.mini_batch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_log_prob = old_log_prob[mb]
                mb_returns = returns[mb]
                mb_adv = combined_adv[mb]

                new_log_prob, entropy, value = self.model.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_prob - mb_old_log_prob)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_returns)
                entropy_term = entropy.mean()
                loss = actor_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_term

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = (mb_old_log_prob - new_log_prob).mean().detach()
                actor_losses.append(float(actor_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_terms.append(float(entropy_term.detach().cpu().item()))
                approx_kls.append(float(approx_kl.cpu().item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_terms)) if entropy_terms else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "lambda": float(lambda_value),
        }

from __future__ import annotations

import numpy as np
import torch

from src.algos.ls_ppo import LSPPOConfig, LSPPOCoordinator, estimate_empirical_discounted_cost
from src.algos.ppo import PPOConfig, PPOTrainer
from src.models.actor_critic import ActorCritic
from src.models.lyapunov_net import LyapunovNet


def test_estimate_empirical_discounted_cost_episodic_mean() -> None:
    costs = np.array(
        [
            [1.0, 2.0],
            [1.0, 0.0],
            [3.0, 1.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    done = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = estimate_empirical_discounted_cost(costs, done, gamma=0.9)
    expected = np.mean([1.9, 2.81, 6.6, 5.0])
    assert abs(out - expected) < 1e-5


def test_lsppo_coordinator_updates_lyapunov_and_lambda() -> None:
    torch.manual_seed(7)
    net = LyapunovNet(obs_dim=4, hidden_sizes=[8])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    coord = LSPPOCoordinator(
        lyap_net=net,
        lyap_optimizer=optimizer,
        cfg=LSPPOConfig(gamma=0.99, lam_lr=0.1, cost_limit=0.5, lambda_max=10.0, lambda_init=0.0),
        device=torch.device("cpu"),
    )
    obs = np.random.randn(16, 4).astype(np.float32)
    next_obs = np.random.randn(16, 4).astype(np.float32)
    costs = np.abs(np.random.randn(16).astype(np.float32))
    stats = coord.update_lyapunov(obs=obs, next_obs=next_obs, costs=costs, update_idx=3)
    assert set(stats.keys()) >= {"lyap_loss", "lyap_td_loss", "lyap_decrease_loss", "lyap_effective_weight"}
    assert np.isfinite(stats["lyap_loss"])

    old_lambda = coord.lambda_value
    new_lambda = coord.update_lambda(empirical_cost=1.5)
    assert new_lambda >= old_lambda
    assert 0.0 <= new_lambda <= coord.cfg.lambda_max


def test_ppo_trainer_update_returns_finite_metrics() -> None:
    torch.manual_seed(11)
    np.random.seed(11)
    model = ActorCritic(obs_dim=6, act_dim=2, actor_hidden_sizes=[16], critic_hidden_sizes=[16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        cfg=PPOConfig(epochs=2, mini_batch_size=4),
        device=torch.device("cpu"),
    )

    n = 8
    obs = torch.randn(n, 6)
    with torch.no_grad():
        actions, old_log_prob, values, _ = model.act(obs, deterministic=False)
    batch = {
        "obs": obs.numpy().astype(np.float32),
        "actions": actions.numpy().astype(np.float32),
        "log_prob": old_log_prob.numpy().astype(np.float32),
        "returns": (values + 0.1 * torch.randn_like(values)).numpy().astype(np.float32),
        "advantages": np.random.randn(n).astype(np.float32),
        "cost_advantages": np.random.randn(n).astype(np.float32),
    }
    metrics = trainer.update(batch, lambda_value=0.3)
    for key in ("actor_loss", "value_loss", "entropy", "approx_kl", "lambda"):
        assert key in metrics
        assert np.isfinite(metrics[key])

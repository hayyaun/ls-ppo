from __future__ import annotations

import numpy as np
import torch

from src.models.actor_critic import ActorCritic
from src.models.lyapunov_net import LyapunovNet, lyapunov_decrease_loss, lyapunov_td_loss
from src.models import safety_projection as safety_projection_module
from src.models.safety_projection import SafetyProjector


class DummyLyapunov(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sum(obs * obs, dim=-1) + 1.0


def test_actor_critic_output_shapes() -> None:
    model = ActorCritic(obs_dim=6, act_dim=2, actor_hidden_sizes=[16], critic_hidden_sizes=[16])
    obs = torch.randn(4, 6)
    action, log_prob, value, mean = model.act(obs, deterministic=False)
    assert action.shape == (4, 2)
    assert mean.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)

    log_prob2, entropy, value2 = model.evaluate_actions(obs, action)
    assert log_prob2.shape == (4,)
    assert entropy.shape == (4,)
    assert value2.shape == (4,)


def test_lyapunov_outputs_and_losses() -> None:
    net = LyapunovNet(obs_dim=6, hidden_sizes=[8, 8])
    obs = torch.randn(5, 6)
    next_obs = torch.randn(5, 6)
    costs = torch.rand(5)

    current_l = net(obs)
    next_l = net(next_obs)
    assert torch.all(current_l >= 0.0)
    assert torch.all(next_l >= 0.0)

    td = lyapunov_td_loss(current_l, costs, next_l, gamma=0.99)
    dec = lyapunov_decrease_loss(current_l, next_l, gamma=0.99, margin=0.01)
    assert float(td.item()) >= 0.0
    assert float(dec.item()) >= 0.0


def test_safety_projector_finite_diff_respects_action_bounds() -> None:
    projector = SafetyProjector(
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        method="finite_diff",
    )
    lyap = DummyLyapunov()
    state = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    action = np.array([3.0, -2.0], dtype=np.float32)

    result = projector.project(
        action_theta=action,
        state=state,
        lyap_net=lyap,
        gamma=0.99,
        epsilon=0.01,
        env_name="PointGoalSafe-v0",
        env_params={},
        device=torch.device("cpu"),
    )
    assert result.action.shape == (2,)
    assert np.all(result.action <= 1.0 + 1e-6)
    assert np.all(result.action >= -1.0 - 1e-6)


def test_safety_projector_qp_fallback_when_cvxpy_missing(monkeypatch) -> None:
    monkeypatch.setattr(safety_projection_module, "HAS_CVXPY", False)
    projector = SafetyProjector(
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        method="qp",
    )
    lyap = DummyLyapunov()
    state = np.array([0.2, -0.1, 1.0, 1.0, 0.8, 1.1], dtype=np.float32)
    action = np.array([0.5, 0.25], dtype=np.float32)
    result = projector.project(
        action_theta=action,
        state=state,
        lyap_net=lyap,
        gamma=0.99,
        epsilon=0.01,
        env_name="PointGoalSafe-v0",
        env_params={},
        device=torch.device("cpu"),
    )
    assert result.method == "qp"
    assert "cvxpy_not_installed" in result.message

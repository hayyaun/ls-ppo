"""Safety projection for LS-PPO actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False
    cp = None


@dataclass
class ProjectionResult:
    """Result of projecting a sampled policy action."""

    action: np.ndarray
    projected: bool
    feasible: bool
    method: str
    message: str


class SafetyProjector:
    """Project actions into a linearized safe set."""

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        method: str = "finite_diff",
        finite_diff_eps: float = 1e-3,
        qp_solver: str = "OSQP",
        qp_max_iter: int = 5000,
        qp_eps_abs: float = 1e-5,
        qp_eps_rel: float = 1e-5,
    ) -> None:
        """Initialize projection solver options."""
        self.action_low = action_low.astype(np.float64)
        self.action_high = action_high.astype(np.float64)
        self.method = method
        self.finite_diff_eps = float(finite_diff_eps)
        self.qp_solver = qp_solver
        self.qp_max_iter = int(qp_max_iter)
        self.qp_eps_abs = float(qp_eps_abs)
        self.qp_eps_rel = float(qp_eps_rel)

    def _predict_next_state_np(
        self,
        env_name: str,
        state: np.ndarray,
        action: np.ndarray,
        env_params: Dict[str, float],
    ) -> np.ndarray:
        """Predict next state using the known environment model."""
        if env_name == "PointGoalSafe-v0":
            action_scale = float(env_params.get("point_action_scale", 0.2))
            world_low = float(env_params.get("point_world_low", -3.0))
            world_high = float(env_params.get("point_world_high", 3.0))
            pos = state[:2].astype(np.float64)
            goal = state[2:4].astype(np.float64)
            next_pos = np.clip(pos + action_scale * np.clip(action, -1.0, 1.0), world_low, world_high)
            delta = goal - next_pos
            return np.array([next_pos[0], next_pos[1], goal[0], goal[1], delta[0], delta[1]], dtype=np.float32)

        if env_name == "CarGoalSafe-v0":
            dt = float(env_params.get("car_dt", 0.1))
            accel_scale = float(env_params.get("car_accel_scale", 1.0))
            steer_scale = float(env_params.get("car_steer_scale", 1.5))
            max_speed = float(env_params.get("car_max_speed", 2.5))
            world_low = float(env_params.get("car_world_low", -4.0))
            world_high = float(env_params.get("car_world_high", 4.0))
            x = float(state[0])
            y = float(state[1])
            heading = float(np.arctan2(state[3], state[2]))
            speed = float(state[4])
            goal_x = float(state[5])
            goal_y = float(state[6])

            accel = float(np.clip(action[0], -1.0, 1.0)) * accel_scale
            steer = float(np.clip(action[1], -1.0, 1.0)) * steer_scale
            speed = np.clip(speed + accel * dt, 0.0, max_speed)
            heading = heading + steer * dt
            x = np.clip(x + speed * np.cos(heading) * dt, world_low, world_high)
            y = np.clip(y + speed * np.sin(heading) * dt, world_low, world_high)
            distance = np.hypot(x - goal_x, y - goal_y)
            return np.array([x, y, np.cos(heading), np.sin(heading), speed, goal_x, goal_y, distance], dtype=np.float32)

        return state.astype(np.float32)

    def _predict_next_state_torch(
        self,
        env_name: str,
        state: torch.Tensor,
        action: torch.Tensor,
        env_params: Dict[str, float],
    ) -> torch.Tensor:
        """Differentiable next-state model for Jacobian-based projection."""
        if env_name == "PointGoalSafe-v0":
            action_scale = float(env_params.get("point_action_scale", 0.2))
            world_low = float(env_params.get("point_world_low", -3.0))
            world_high = float(env_params.get("point_world_high", 3.0))
            pos = state[:2]
            goal = state[2:4]
            next_pos = torch.clamp(pos + action_scale * torch.clamp(action, -1.0, 1.0), world_low, world_high)
            delta = goal - next_pos
            return torch.stack([next_pos[0], next_pos[1], goal[0], goal[1], delta[0], delta[1]])

        if env_name == "CarGoalSafe-v0":
            dt = float(env_params.get("car_dt", 0.1))
            accel_scale = float(env_params.get("car_accel_scale", 1.0))
            steer_scale = float(env_params.get("car_steer_scale", 1.5))
            max_speed = float(env_params.get("car_max_speed", 2.5))
            world_low = float(env_params.get("car_world_low", -4.0))
            world_high = float(env_params.get("car_world_high", 4.0))
            x = state[0]
            y = state[1]
            heading = torch.atan2(state[3], state[2])
            speed = state[4]
            goal_x = state[5]
            goal_y = state[6]

            accel = torch.clamp(action[0], -1.0, 1.0) * accel_scale
            steer = torch.clamp(action[1], -1.0, 1.0) * steer_scale
            speed = torch.clamp(speed + accel * dt, 0.0, max_speed)
            heading = heading + steer * dt
            x = torch.clamp(x + speed * torch.cos(heading) * dt, world_low, world_high)
            y = torch.clamp(y + speed * torch.sin(heading) * dt, world_low, world_high)
            distance = torch.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
            return torch.stack([x, y, torch.cos(heading), torch.sin(heading), speed, goal_x, goal_y, distance])

        return state

    def _lyapunov_value(self, lyap_net: torch.nn.Module, state: np.ndarray, device: torch.device) -> float:
        """Evaluate Lyapunov network on a numpy state."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            value = lyap_net(state_t).squeeze().item()
        return float(value)

    def _gradient_finite_diff(
        self,
        lyap_net: torch.nn.Module,
        state: np.ndarray,
        action: np.ndarray,
        env_name: str,
        env_params: Dict[str, float],
        device: torch.device,
    ) -> np.ndarray:
        """Estimate dL/d a by finite differences."""
        dim = action.shape[0]
        gradient = np.zeros(dim, dtype=np.float64)
        base_next = self._predict_next_state_np(env_name, state, action, env_params)
        base_val = self._lyapunov_value(lyap_net, base_next, device)
        for i in range(dim):
            perturbed = action.copy()
            perturbed[i] += self.finite_diff_eps
            perturbed = np.clip(perturbed, self.action_low, self.action_high)
            next_state = self._predict_next_state_np(env_name, state, perturbed, env_params)
            value = self._lyapunov_value(lyap_net, next_state, device)
            gradient[i] = (value - base_val) / self.finite_diff_eps
        return gradient

    def _gradient_jacobian(
        self,
        lyap_net: torch.nn.Module,
        state: np.ndarray,
        action: np.ndarray,
        env_name: str,
        env_params: Dict[str, float],
        device: torch.device,
    ) -> np.ndarray:
        """Estimate dL/d a by autograd through model dynamics."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device, requires_grad=True)
        next_state = self._predict_next_state_torch(env_name, state_t, action_t, env_params)
        l_value = lyap_net(next_state.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(l_value, action_t, retain_graph=False, create_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    def _closed_form_projection(self, action: np.ndarray, grad: np.ndarray, bound: float) -> Tuple[np.ndarray, bool]:
        """Project onto half-space and box constraints in closed form where possible."""
        candidate = np.clip(action, self.action_low, self.action_high)
        if float(np.dot(grad, candidate)) <= bound + 1e-9:
            return candidate, True
        norm_sq = float(np.dot(grad, grad))
        if norm_sq < 1e-12:
            return candidate, False

        candidate = candidate - ((np.dot(grad, candidate) - bound) / norm_sq) * grad
        candidate = np.clip(candidate, self.action_low, self.action_high)
        if float(np.dot(grad, candidate)) <= bound + 1e-7:
            return candidate, True

        idx = int(np.argmax(np.abs(grad)))
        if abs(grad[idx]) > 1e-12:
            rest = float(np.dot(grad, candidate) - grad[idx] * candidate[idx])
            candidate[idx] = np.clip((bound - rest) / grad[idx], self.action_low[idx], self.action_high[idx])
        feasible = float(np.dot(grad, candidate)) <= bound + 1e-7
        return candidate, feasible

    def _qp_projection(self, action: np.ndarray, grad: np.ndarray, bound: float) -> Tuple[np.ndarray, bool, str]:
        """Solve small QP projection when closed-form is insufficient."""
        if not HAS_CVXPY:
            return np.clip(action, self.action_low, self.action_high), False, "cvxpy_not_installed"

        var = cp.Variable(action.shape[0])
        objective = cp.Minimize(cp.sum_squares(var - action))
        constraints = [
            grad @ var <= bound,
            var >= self.action_low,
            var <= self.action_high,
        ]
        problem = cp.Problem(objective, constraints)
        try:
            solver = self.qp_solver.upper()
            kwargs = {
                "solver": solver,
                "verbose": False,
            }
            if solver == "OSQP":
                kwargs.update(
                    {
                        "max_iter": self.qp_max_iter,
                        "eps_abs": self.qp_eps_abs,
                        "eps_rel": self.qp_eps_rel,
                        "warm_start": False,
                        "adaptive_rho": False,
                        "adaptive_rho_interval": 25,
                        "polish": False,
                    }
                )
            problem.solve(**kwargs)
        except Exception as exc:
            return np.clip(action, self.action_low, self.action_high), False, f"qp_exception:{exc}"

        if var.value is None:
            return np.clip(action, self.action_low, self.action_high), False, f"qp_status:{problem.status}"
        projected = np.clip(np.asarray(var.value, dtype=np.float64), self.action_low, self.action_high)
        feasible = float(np.dot(grad, projected)) <= bound + 1e-6
        return projected, feasible, f"qp_status:{problem.status}"

    def project(
        self,
        action_theta: np.ndarray,
        state: np.ndarray,
        lyap_net: torch.nn.Module,
        gamma: float,
        epsilon: float,
        env_name: str,
        env_params: Dict[str, float],
        device: torch.device,
    ) -> ProjectionResult:
        """Project action into linearized safe set around action_theta."""
        if lyap_net is None:
            clipped = np.clip(action_theta, self.action_low, self.action_high)
            return ProjectionResult(clipped, False, True, "none", "lyapunov_disabled")

        action_theta = np.clip(action_theta.astype(np.float64), self.action_low, self.action_high)
        state = state.astype(np.float64)
        l_s = self._lyapunov_value(lyap_net, state, device)

        if self.method == "jacobian":
            grad = self._gradient_jacobian(lyap_net, state, action_theta, env_name, env_params, device)
            grad_method = "jacobian"
        else:
            grad = self._gradient_finite_diff(lyap_net, state, action_theta, env_name, env_params, device)
            grad_method = "finite_diff"

        # L(s') ~ L(s'_theta) + grad^T (a - a_theta) <= gamma * L(s) + epsilon
        next_state_theta = self._predict_next_state_np(env_name, state, action_theta, env_params)
        l_next_theta = self._lyapunov_value(lyap_net, next_state_theta, device)
        bound = gamma * l_s + epsilon - l_next_theta + float(np.dot(grad, action_theta))

        if self.method == "qp":
            qp_action, feasible, msg = self._qp_projection(action_theta, grad, bound)
            return ProjectionResult(qp_action.astype(np.float32), True, feasible, "qp", msg)

        projected, feasible = self._closed_form_projection(action_theta, grad, bound)
        if feasible:
            changed = not np.allclose(projected, action_theta)
            return ProjectionResult(projected.astype(np.float32), changed, True, grad_method, "closed_form")

        qp_action, qp_feasible, msg = self._qp_projection(projected, grad, bound)
        if qp_feasible:
            return ProjectionResult(qp_action.astype(np.float32), True, True, "qp_fallback", msg)

        return ProjectionResult(
            np.clip(projected, self.action_low, self.action_high).astype(np.float32),
            True,
            False,
            "fallback_clip",
            msg,
        )

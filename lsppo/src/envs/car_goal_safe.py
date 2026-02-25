"""2D kinematic car goal environment with safety costs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CarGoalSafeEnv(gym.Env):
    """Simple car-like integrator with continuous acceleration and steering rate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 200,
        dt: float = 0.1,
        accel_scale: float = 1.0,
        steer_scale: float = 1.5,
        max_speed: float = 2.5,
        world_low: float = -4.0,
        world_high: float = 4.0,
        goal_radius: float = 0.35,
        forbidden_circles: Optional[Tuple[Tuple[float, float, float], ...]] = None,
    ) -> None:
        """Initialize dynamics and spaces."""
        super().__init__()
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.accel_scale = float(accel_scale)
        self.steer_scale = float(steer_scale)
        self.max_speed = float(max_speed)
        self.world_low = float(world_low)
        self.world_high = float(world_high)
        self.goal_radius = float(goal_radius)
        self.forbidden_circles = forbidden_circles or (
            (0.0, 0.0, 1.0),
            (-1.5, 1.0, 0.7),
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        self.state = np.zeros(4, dtype=np.float64)  # x, y, heading, speed
        self.goal = np.array([2.8, 2.8], dtype=np.float64)
        self.step_count = 0

    def _is_forbidden(self, x: float, y: float) -> bool:
        """Return True when point lies in forbidden region."""
        for cx, cy, radius in self.forbidden_circles:
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                return True
        return False

    def _distance_to_goal(self, x: float, y: float) -> float:
        """Compute distance from current position to goal."""
        return float(np.hypot(x - self.goal[0], y - self.goal[1]))

    def _build_obs(self, state: np.ndarray) -> np.ndarray:
        """Build observation vector from latent state."""
        x, y, heading, speed = state
        distance = self._distance_to_goal(x, y)
        return np.array(
            [
                x,
                y,
                np.cos(heading),
                np.sin(heading),
                speed,
                self.goal[0],
                self.goal[1],
                distance,
            ],
            dtype=np.float32,
        )

    def predict_next_state(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Deterministic one-step model for projection and Jacobian estimation."""
        x = float(obs[0])
        y = float(obs[1])
        heading = float(np.arctan2(obs[3], obs[2]))
        speed = float(obs[4])
        goal_x = float(obs[5])
        goal_y = float(obs[6])

        accel = float(np.clip(action[0], -1.0, 1.0)) * self.accel_scale
        steer_rate = float(np.clip(action[1], -1.0, 1.0)) * self.steer_scale

        next_speed = np.clip(speed + accel * self.dt, 0.0, self.max_speed)
        next_heading = heading + steer_rate * self.dt
        next_x = np.clip(x + next_speed * np.cos(next_heading) * self.dt, self.world_low, self.world_high)
        next_y = np.clip(y + next_speed * np.sin(next_heading) * self.dt, self.world_low, self.world_high)
        distance = float(np.hypot(next_x - goal_x, next_y - goal_y))
        return np.array(
            [
                next_x,
                next_y,
                np.cos(next_heading),
                np.sin(next_heading),
                next_speed,
                goal_x,
                goal_y,
                distance,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Reset episode with deterministic RNG support."""
        super().reset(seed=seed)
        self.step_count = 0
        start_x, start_y = -3.0, -3.0
        heading = self.np_random.uniform(-np.pi, np.pi)
        speed = self.np_random.uniform(0.0, 0.3)
        self.state = np.array([start_x, start_y, heading, speed], dtype=np.float64)
        obs = self._build_obs(self.state)
        info = {"cost": 0.0, "distance": self._distance_to_goal(start_x, start_y)}
        return obs, info

    def step(self, action: np.ndarray):
        """Advance the car dynamics one step."""
        self.step_count += 1
        accel = float(np.clip(action[0], -1.0, 1.0)) * self.accel_scale
        steer_rate = float(np.clip(action[1], -1.0, 1.0)) * self.steer_scale

        x, y, heading, speed = self.state
        speed = np.clip(speed + accel * self.dt, 0.0, self.max_speed)
        heading = heading + steer_rate * self.dt
        x = np.clip(x + speed * np.cos(heading) * self.dt, self.world_low, self.world_high)
        y = np.clip(y + speed * np.sin(heading) * self.dt, self.world_low, self.world_high)
        self.state = np.array([x, y, heading, speed], dtype=np.float64)

        distance = self._distance_to_goal(x, y)
        reward = -distance - 0.01 * (accel ** 2 + steer_rate ** 2)
        cost = 1.0 if self._is_forbidden(x, y) else 0.0
        terminated = distance <= self.goal_radius
        truncated = self.step_count >= self.max_steps
        obs = self._build_obs(self.state)

        info = {
            "cost": float(cost),
            "distance": distance,
            "unsafe_step": int(cost > 0.0),
            "goal_reached": bool(terminated),
            "step": int(self.step_count),
        }

        assert np.isfinite(reward), "Reward became non-finite."
        assert np.isfinite(cost), "Cost became non-finite."
        assert np.all(np.isfinite(obs)), "Observation became non-finite."
        return obs, float(reward), bool(terminated), bool(truncated), info

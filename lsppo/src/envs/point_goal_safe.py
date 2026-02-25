"""2D point-mass goal environment with safety costs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointGoalSafeEnv(gym.Env):
    """Point robot reaches a goal while avoiding forbidden circles."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 200,
        action_scale: float = 0.2,
        world_low: float = -3.0,
        world_high: float = 3.0,
        goal_radius: float = 0.25,
        forbidden_circles: Optional[Tuple[Tuple[float, float, float], ...]] = None,
    ) -> None:
        """Initialize environment parameters and spaces."""
        super().__init__()
        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.world_low = float(world_low)
        self.world_high = float(world_high)
        self.goal_radius = float(goal_radius)
        self.forbidden_circles = forbidden_circles or (
            (0.0, 0.0, 0.75),
            (1.25, -1.0, 0.55),
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        self.position = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.step_count = 0

    def _is_forbidden(self, position: np.ndarray) -> bool:
        """Return True when position is inside any forbidden circle."""
        for cx, cy, radius in self.forbidden_circles:
            if (position[0] - cx) ** 2 + (position[1] - cy) ** 2 <= radius ** 2:
                return True
        return False

    def _distance_to_goal(self, position: np.ndarray) -> float:
        """Compute Euclidean distance to goal."""
        return float(np.linalg.norm(position - self.goal))

    def _build_obs(self, position: np.ndarray) -> np.ndarray:
        """Build observation vector."""
        delta = self.goal - position
        obs = np.array(
            [position[0], position[1], self.goal[0], self.goal[1], delta[0], delta[1]],
            dtype=np.float32,
        )
        return obs

    def predict_next_state(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Deterministic one-step model for projection and Jacobian estimation."""
        position = obs[:2].astype(np.float64)
        goal = obs[2:4].astype(np.float64)
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        next_position = np.clip(
            position + self.action_scale * clipped_action,
            self.world_low,
            self.world_high,
        )
        delta = goal - next_position
        return np.array(
            [next_position[0], next_position[1], goal[0], goal[1], delta[0], delta[1]],
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

        for _ in range(512):
            start = self.np_random.uniform(self.world_low, self.world_high, size=2)
            if not self._is_forbidden(start):
                self.position = start.astype(np.float64)
                break
        else:
            self.position = np.array([self.world_low, self.world_low], dtype=np.float64)

        for _ in range(512):
            goal = self.np_random.uniform(self.world_low, self.world_high, size=2)
            if not self._is_forbidden(goal) and np.linalg.norm(goal - self.position) > 1.0:
                self.goal = goal.astype(np.float64)
                break
        else:
            self.goal = np.array([self.world_high, self.world_high], dtype=np.float64)

        obs = self._build_obs(self.position)
        info = {"cost": 0.0, "distance": self._distance_to_goal(self.position)}
        return obs, info

    def step(self, action: np.ndarray):
        """Advance the environment by one step."""
        self.step_count += 1
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        next_position = np.clip(
            self.position + self.action_scale * clipped_action,
            self.world_low,
            self.world_high,
        )

        distance = self._distance_to_goal(next_position)
        reward = -distance
        cost = 1.0 if self._is_forbidden(next_position) else 0.0
        terminated = distance <= self.goal_radius
        truncated = self.step_count >= self.max_steps

        self.position = next_position
        obs = self._build_obs(self.position)
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

from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.envs import register_safe_envs


def test_point_goal_env_reset_step_and_cost_signal() -> None:
    register_safe_envs()
    env = gym.make("PointGoalSafe-v0")
    obs, info = env.reset(seed=123)
    assert obs.shape == (6,)
    assert "cost" in info

    action = np.array([0.25, -0.1], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cost" in step_info
    assert step_info["cost"] in (0.0, 1.0)
    env.close()


def test_car_goal_env_reset_step_and_cost_signal() -> None:
    register_safe_envs()
    env = gym.make("CarGoalSafe-v0")
    obs, info = env.reset(seed=456)
    assert obs.shape == (8,)
    assert "cost" in info

    action = np.array([0.1, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cost" in step_info
    assert step_info["cost"] in (0.0, 1.0)
    env.close()


def test_env_reset_is_deterministic_for_same_seed() -> None:
    register_safe_envs()
    env = gym.make("PointGoalSafe-v0")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2, atol=1e-7)
    env.close()

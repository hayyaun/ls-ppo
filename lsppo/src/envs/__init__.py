"""Environment registration for custom safety tasks."""

from __future__ import annotations

from gymnasium.envs.registration import register

_REGISTERED = False


def register_safe_envs() -> None:
    """Register custom safety environments once."""
    global _REGISTERED
    if _REGISTERED:
        return
    register(
        id="PointGoalSafe-v0",
        entry_point="src.envs.point_goal_safe:PointGoalSafeEnv",
    )
    register(
        id="CarGoalSafe-v0",
        entry_point="src.envs.car_goal_safe:CarGoalSafeEnv",
    )
    _REGISTERED = True

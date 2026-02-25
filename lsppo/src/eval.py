"""Evaluation script for deterministic policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from src.envs import register_safe_envs
from src.models.actor_critic import ActorCritic
from src.models.lyapunov_net import LyapunovNet
from src.models.safety_projection import SafetyProjector
from src.utils import load_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained policy.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--algo", type=str, default="ls_ppo", choices=["ppo", "ppo_lagrangian", "ls_ppo"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--projection", type=str, default="true")
    parser.add_argument("--out", type=str, default="experiments/results/eval_summary.json")
    parser.add_argument("--save_trajectory_csv", type=str, default="")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run deterministic evaluation episodes and save summary."""
    args = parse_args()
    register_safe_envs()
    cfg = load_config(args.config)
    if args.env is not None:
        cfg["env"] = args.env
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(args.seed, deterministic_torch=bool(cfg.get("deterministic_torch", True)))
    device = select_device(str(cfg.get("device", "auto")))

    env = gym.make(cfg["env"], **cfg.get("env_kwargs", {}))
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor_hidden_sizes=cfg["actor_hidden_sizes"],
        critic_hidden_sizes=cfg["critic_hidden_sizes"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_projection = args.projection.lower() in {"1", "true", "yes", "y"}
    lyap_net = None
    projector = None
    if args.algo == "ls_ppo" and use_projection:
        lyap_net = LyapunovNet(obs_dim=obs_dim, hidden_sizes=cfg["lyap_hidden_sizes"]).to(device)
        if checkpoint.get("lyap_state_dict") is not None:
            lyap_net.load_state_dict(checkpoint["lyap_state_dict"])
        lyap_net.eval()
        projector = SafetyProjector(
            action_low=action_low,
            action_high=action_high,
            method=str(cfg.get("projection_method", "finite_diff")),
            finite_diff_eps=float(cfg.get("finite_diff_eps", 1e-3)),
            qp_solver=str(cfg.get("qp_solver", "OSQP")),
        )

    episodes = int(args.episodes or cfg.get("episodes_per_eval", 20))
    gamma = float(cfg["gamma"])

    returns: List[float] = []
    discounted_costs: List[float] = []
    violation_rates: List[float] = []
    lengths: List[int] = []
    trajectory_rows = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=args.seed + ep_idx)
        done = False
        ep_return = 0.0
        ep_disc_cost = 0.0
        ep_violations = 0
        ep_len = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, _, mean_t = model.act(obs_t, deterministic=True)
            action = mean_t.squeeze(0).cpu().numpy()
            action = np.clip(action, action_low, action_high)

            if projector is not None and lyap_net is not None:
                result = projector.project(
                    action_theta=action,
                    state=obs,
                    lyap_net=lyap_net,
                    gamma=gamma,
                    epsilon=float(cfg.get("projection_epsilon", 0.01)),
                    env_name=str(cfg["env"]),
                    env_params=dict(cfg.get("projection_env_params", {})),
                    device=device,
                )
                action = result.action

            next_obs, reward, terminated, truncated, info = env.step(action)
            cost = float(info.get("cost", 0.0))
            ep_return += float(reward)
            ep_disc_cost += (gamma ** ep_len) * cost
            ep_violations += int(cost > 0.0)
            ep_len += 1

            if str(cfg["env"]) == "PointGoalSafe-v0":
                trajectory_rows.append(
                    {
                        "episode": ep_idx,
                        "x": float(next_obs[0]),
                        "y": float(next_obs[1]),
                        "unsafe": int(cost > 0.0),
                    }
                )

            obs = next_obs
            done = bool(terminated or truncated)

        returns.append(ep_return)
        discounted_costs.append(ep_disc_cost)
        violation_rates.append(ep_violations / max(1, ep_len))
        lengths.append(ep_len)

    summary = {
        "algo": args.algo,
        "env": str(cfg["env"]),
        "episodes": episodes,
        "seed": args.seed,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "mean_discounted_cost": float(np.mean(discounted_costs)),
        "std_discounted_cost": float(np.std(discounted_costs, ddof=1)) if len(discounted_costs) > 1 else 0.0,
        "mean_violation_rate": float(np.mean(violation_rates)),
        "std_violation_rate": float(np.std(violation_rates, ddof=1)) if len(violation_rates) > 1 else 0.0,
        "mean_length": float(np.mean(lengths)),
        "checkpoint": args.checkpoint,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    if args.save_trajectory_csv:
        traj_path = Path(args.save_trajectory_csv)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trajectory_rows).to_csv(traj_path, index=False)

    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()

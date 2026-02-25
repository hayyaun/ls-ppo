"""Main training entrypoint for PPO, PPO-Lagrangian, and LS-PPO."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

from src.algos.ls_ppo import LSPPOConfig, LSPPOCoordinator, estimate_empirical_discounted_cost
from src.algos.ppo import PPOConfig, PPOTrainer
from src.envs import register_safe_envs
from src.models.actor_critic import ActorCritic
from src.models.lyapunov_net import LyapunovNet
from src.models.safety_projection import SafetyProjector
from src.utils import (
    compute_gae,
    create_writers,
    ensure_dir,
    flatten_time_env,
    load_config,
    safe_mean,
    select_device,
    set_seed,
    str2bool,
)

try:
    import wandb
except Exception:
    wandb = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO-family algorithms.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--algo", type=str, default="ls_ppo", choices=["ppo", "ppo_lagrangian", "ls_ppo"])
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--quick_test", type=str, default="false")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def make_envs(env_id: str, n_envs: int, base_seed: int, env_kwargs: Dict[str, Any]) -> List[gym.Env]:
    """Create and seed independent environment instances."""
    envs: List[gym.Env] = []
    for idx in range(n_envs):
        env = gym.make(env_id, **env_kwargs)
        env.reset(seed=base_seed + idx)
        env.action_space.seed(base_seed + idx)
        envs.append(env)
    return envs


def save_checkpoint(
    checkpoint_path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    lyap_net: LyapunovNet | None,
    lyap_optimizer: torch.optim.Optimizer | None,
    lambda_value: float,
    update_idx: int,
    env_steps: int,
    config: Dict[str, Any],
    algo: str,
) -> None:
    """Serialize full training state for resume."""
    payload = {
        "algo": algo,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lyap_state_dict": lyap_net.state_dict() if lyap_net is not None else None,
        "lyap_optimizer_state_dict": lyap_optimizer.state_dict() if lyap_optimizer is not None else None,
        "lambda_value": float(lambda_value),
        "update_idx": int(update_idx),
        "env_steps": int(env_steps),
        "config": config,
        "random_state_py": random.getstate(),
        "random_state_np": np.random.get_state(),
        "random_state_torch": torch.random.get_rng_state(),
    }
    ensure_dir(checkpoint_path.parent)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    lyap_net: LyapunovNet | None,
    lyap_optimizer: torch.optim.Optimizer | None,
) -> Tuple[int, int, float]:
    """Load checkpoint and restore trainer state."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if lyap_net is not None and payload.get("lyap_state_dict") is not None:
        lyap_net.load_state_dict(payload["lyap_state_dict"])
    if lyap_optimizer is not None and payload.get("lyap_optimizer_state_dict") is not None:
        lyap_optimizer.load_state_dict(payload["lyap_optimizer_state_dict"])
    random.setstate(payload["random_state_py"])
    np.random.set_state(payload["random_state_np"])
    torch.random.set_rng_state(payload["random_state_torch"])
    return int(payload["update_idx"]), int(payload["env_steps"]), float(payload.get("lambda_value", 0.0))


def main() -> None:
    """Train selected algorithm and write logs/checkpoints."""
    args = parse_args()
    register_safe_envs()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.env is not None:
        cfg["env"] = args.env
    if args.device is not None:
        cfg["device"] = args.device
    quick_test = str2bool(args.quick_test)

    seed = int(cfg["seed"])
    set_seed(seed, deterministic_torch=bool(cfg.get("deterministic_torch", True)))
    device = select_device(str(cfg.get("device", "auto")))

    n_envs = int(cfg["n_envs"])
    steps_per_env = int(cfg["steps_per_env"])
    env_id = str(cfg["env"])
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    envs = make_envs(env_id, n_envs, seed, env_kwargs)

    obs, _ = zip(*[env.reset(seed=seed + idx) for idx, env in enumerate(envs)])
    obs_arr = np.stack(obs).astype(np.float32)
    obs_dim = obs_arr.shape[1]
    act_dim = int(envs[0].action_space.shape[0])
    action_low = envs[0].action_space.low.astype(np.float32)
    action_high = envs[0].action_space.high.astype(np.float32)

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor_hidden_sizes=cfg["actor_hidden_sizes"],
        critic_hidden_sizes=cfg["critic_hidden_sizes"],
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.actor.parameters(), "lr": float(cfg["lr_actor"])},
            {"params": [model.log_std], "lr": float(cfg["lr_actor"])},
            {"params": model.critic.parameters(), "lr": float(cfg["lr_critic"])},
        ]
    )

    ppo_trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        cfg=PPOConfig(
            clip_eps=float(cfg["clip_eps"]),
            epochs=int(cfg["epochs"]),
            mini_batch_size=int(cfg["mini_batch_size"]),
            vf_coef=float(cfg["vf_coef"]),
            ent_coef=float(cfg["ent_coef"]),
            max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        ),
        device=device,
    )

    use_lagrangian = args.algo in {"ppo_lagrangian", "ls_ppo"}
    use_projection = args.algo == "ls_ppo" and bool(cfg.get("projection", True))

    lyap_net = None
    lyap_optimizer = None
    ls_helper = None
    projector = None
    lambda_value = float(cfg.get("lambda_init", 0.0))

    if args.algo == "ls_ppo":
        lyap_net = LyapunovNet(obs_dim=obs_dim, hidden_sizes=cfg["lyap_hidden_sizes"]).to(device)
        lyap_optimizer = torch.optim.Adam(lyap_net.parameters(), lr=float(cfg["lr_lyap"]))
        ls_helper = LSPPOCoordinator(
            lyap_net=lyap_net,
            lyap_optimizer=lyap_optimizer,
            cfg=LSPPOConfig(
                gamma=float(cfg["gamma"]),
                lam_lr=float(cfg["lam_lr"]),
                lambda_update_every=int(cfg.get("lambda_update_every", 1)),
                cost_limit=float(cfg["cost_limit"]),
                lambda_max=float(cfg["lambda_max"]),
                lambda_init=float(cfg.get("lambda_init", 0.0)),
                lyap_loss_weight=float(cfg["lyap_loss_weight"]),
                lyap_weight_schedule=str(cfg.get("lyap_weight_schedule", "constant")),
                lyap_warmup_updates=int(cfg.get("lyap_warmup_updates", 50)),
                lyap_td_weight=float(cfg.get("lyap_td_weight", 1.0)),
                lyap_decrease_weight=float(cfg.get("lyap_decrease_weight", 1.0)),
                lyap_margin=float(cfg.get("lyap_margin", 0.01)),
                lyap_objective=str(cfg.get("lyap_objective", "both")),
            ),
            device=device,
        )
        lambda_value = ls_helper.lambda_value
        projector = SafetyProjector(
            action_low=action_low,
            action_high=action_high,
            method=str(cfg.get("projection_method", "finite_diff")),
            finite_diff_eps=float(cfg.get("finite_diff_eps", 1e-3)),
            qp_solver=str(cfg.get("qp_solver", "OSQP")),
        )

    logdir = ensure_dir(args.logdir)
    checkpoints_dir = ensure_dir(logdir / "checkpoints")
    tb_writer, json_logger = create_writers(logdir)
    projection_warned = False

    wandb_run = None
    if bool(cfg.get("log_wandb", False)):
        if wandb is None:
            print("[WARN] log_wandb=true but wandb is not installed; continuing without wandb logging.")
        else:
            wandb_run = wandb.init(
                project=str(cfg.get("wandb_project", "lsppo")),
                entity=cfg.get("wandb_entity", None),
                config=cfg,
                name=f"{args.algo}_{env_id}_seed{seed}",
                dir=str(logdir),
                reinit=True,
            )

    with (logdir / "resolved_config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    update_start = 0
    env_steps = 0
    if args.resume:
        update_start, env_steps, lambda_value = load_checkpoint(
            checkpoint_path=Path(args.resume),
            model=model,
            optimizer=optimizer,
            lyap_net=lyap_net,
            lyap_optimizer=lyap_optimizer,
        )
        if ls_helper is not None:
            ls_helper.lambda_value = lambda_value

    total_steps = int(cfg["quick_test_steps"] if quick_test else cfg["total_env_steps"])
    steps_per_update = n_envs * steps_per_env
    n_updates = int(math.ceil(total_steps / steps_per_update))

    gamma = float(cfg["gamma"])
    gae_lambda = float(cfg["gae_lambda"])

    ep_returns_window = deque(maxlen=100)
    ep_costs_window = deque(maxlen=100)
    ep_violation_window = deque(maxlen=100)
    ep_lengths_window = deque(maxlen=100)

    running_returns = np.zeros(n_envs, dtype=np.float32)
    running_discounted_cost = np.zeros(n_envs, dtype=np.float32)
    running_violations = np.zeros(n_envs, dtype=np.float32)
    running_lengths = np.zeros(n_envs, dtype=np.int32)

    checkpoint_interval = int(cfg["checkpoint_interval"])
    next_checkpoint_step = ((env_steps // checkpoint_interval) + 1) * checkpoint_interval

    for update_idx in range(update_start, n_updates):
        obs_buf = np.zeros((steps_per_env, n_envs, obs_dim), dtype=np.float32)
        next_obs_buf = np.zeros((steps_per_env, n_envs, obs_dim), dtype=np.float32)
        action_buf = np.zeros((steps_per_env, n_envs, act_dim), dtype=np.float32)
        logp_buf = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        reward_buf = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        cost_buf = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        value_buf = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        done_buf = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        projection_applied = 0
        projection_infeasible = 0

        for t in range(steps_per_env):
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_theta_t, _, value_t, _ = model.act(obs_t, deterministic=False)
            action_exec = action_theta_t.detach().cpu().numpy().astype(np.float32)

            if use_projection and projector is not None and lyap_net is not None:
                env_params = dict(cfg.get("projection_env_params", {}))
                for env_idx in range(n_envs):
                    result = projector.project(
                        action_theta=action_exec[env_idx],
                        state=obs_arr[env_idx],
                        lyap_net=lyap_net,
                        gamma=float(cfg["gamma"]),
                        epsilon=float(cfg.get("projection_epsilon", 0.01)),
                        env_name=env_id,
                        env_params=env_params,
                        device=device,
                    )
                    action_exec[env_idx] = result.action
                    projection_applied += int(result.projected)
                    projection_infeasible += int(not result.feasible)
                    if (not result.feasible or "cvxpy_not_installed" in result.message) and not projection_warned:
                        print(f"[WARN] Projection fallback active: method={result.method}, message={result.message}")
                        projection_warned = True

            action_exec = np.clip(action_exec, action_low, action_high)
            with torch.no_grad():
                logp_exec, _, _ = model.evaluate_actions(
                    obs_t, torch.as_tensor(action_exec, dtype=torch.float32, device=device)
                )

            next_obs_arr = np.zeros_like(obs_arr)
            for env_idx, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(action_exec[env_idx])
                cost = float(info.get("cost", 0.0))
                done_flag = float(terminated or truncated)

                obs_buf[t, env_idx] = obs_arr[env_idx]
                next_obs_buf[t, env_idx] = next_obs
                action_buf[t, env_idx] = action_exec[env_idx]
                logp_buf[t, env_idx] = float(logp_exec[env_idx].detach().cpu().item())
                reward_buf[t, env_idx] = float(reward)
                cost_buf[t, env_idx] = cost
                value_buf[t, env_idx] = float(value_t[env_idx].detach().cpu().item())
                done_buf[t, env_idx] = done_flag

                running_returns[env_idx] += reward
                running_discounted_cost[env_idx] += (gamma ** running_lengths[env_idx]) * cost
                running_violations[env_idx] += float(cost > 0.0)
                running_lengths[env_idx] += 1

                done = bool(done_flag)
                if done:
                    ep_returns_window.append(float(running_returns[env_idx]))
                    ep_costs_window.append(float(running_discounted_cost[env_idx]))
                    ep_violation_window.append(float(running_violations[env_idx] / max(1, running_lengths[env_idx])))
                    ep_lengths_window.append(int(running_lengths[env_idx]))
                    next_obs, _ = env.reset()
                    running_returns[env_idx] = 0.0
                    running_discounted_cost[env_idx] = 0.0
                    running_violations[env_idx] = 0.0
                    running_lengths[env_idx] = 0

                next_obs_arr[env_idx] = next_obs

            obs_arr = next_obs_arr
            env_steps += n_envs

        with torch.no_grad():
            last_values = model.get_value(torch.as_tensor(obs_arr, dtype=torch.float32, device=device)).cpu().numpy()

        adv_buf, returns_buf = compute_gae(
            rewards=reward_buf,
            values=value_buf,
            done=done_buf,
            last_values=last_values.astype(np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        zero_values = np.zeros_like(value_buf, dtype=np.float32)
        cost_adv_buf, _ = compute_gae(
            rewards=cost_buf,
            values=zero_values,
            done=done_buf,
            last_values=np.zeros(n_envs, dtype=np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batch = {
            "obs": flatten_time_env(obs_buf),
            "actions": flatten_time_env(action_buf),
            "log_prob": flatten_time_env(logp_buf),
            "returns": flatten_time_env(returns_buf),
            "advantages": flatten_time_env(adv_buf),
            "cost_advantages": flatten_time_env(cost_adv_buf),
        }

        lagrangian_value = lambda_value if use_lagrangian else 0.0
        ppo_stats = ppo_trainer.update(batch, lambda_value=lagrangian_value)

        c_empirical = estimate_empirical_discounted_cost(cost_buf, done_buf, gamma=gamma)
        lyap_stats: Dict[str, float] = {"lyap_loss": 0.0, "lyap_td_loss": 0.0, "lyap_decrease_loss": 0.0, "lyap_effective_weight": 0.0}
        if args.algo == "ls_ppo" and ls_helper is not None:
            lyap_stats = ls_helper.update_lyapunov(
                obs=flatten_time_env(obs_buf),
                next_obs=flatten_time_env(next_obs_buf),
                costs=flatten_time_env(cost_buf),
                update_idx=update_idx,
            )
            if ((update_idx + 1) % int(max(1, ls_helper.cfg.lambda_update_every))) == 0:
                lambda_value = ls_helper.update_lambda(c_empirical)
        elif args.algo == "ppo_lagrangian":
            lam_lr = float(cfg["lam_lr"])
            lam_max = float(cfg["lambda_max"])
            cost_limit = float(cfg["cost_limit"])
            lambda_update_every = int(max(1, cfg.get("lambda_update_every", 1)))
            if ((update_idx + 1) % lambda_update_every) == 0:
                lambda_value = float(np.clip(lambda_value + lam_lr * (c_empirical - cost_limit), 0.0, lam_max))

        metrics = {
            "algo": args.algo,
            "env": env_id,
            "seed": seed,
            "update": update_idx,
            "env_steps": env_steps,
            "episode_return_mean": safe_mean(ep_returns_window),
            "episode_discounted_cost_mean": safe_mean(ep_costs_window),
            "episode_violation_rate_mean": safe_mean(ep_violation_window),
            "episode_length_mean": safe_mean(ep_lengths_window),
            "projection_applied_rate": projection_applied / max(1, n_envs * steps_per_env),
            "projection_infeasible_rate": projection_infeasible / max(1, n_envs * steps_per_env),
            "c_empirical": c_empirical,
            "lambda_value": lambda_value,
            **ppo_stats,
            **lyap_stats,
        }

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                tb_writer.add_scalar(key, float(value), env_steps)
        json_logger.log(metrics)
        if wandb_run is not None:
            wandb_run.log({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}, step=env_steps)

        if env_steps >= next_checkpoint_step or update_idx == n_updates - 1:
            checkpoint_path = checkpoints_dir / f"step_{env_steps}.pt"
            latest_path = checkpoints_dir / "latest.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                lyap_net=lyap_net,
                lyap_optimizer=lyap_optimizer,
                lambda_value=lambda_value,
                update_idx=update_idx + 1,
                env_steps=env_steps,
                config=cfg,
                algo=args.algo,
            )
            save_checkpoint(
                checkpoint_path=latest_path,
                model=model,
                optimizer=optimizer,
                lyap_net=lyap_net,
                lyap_optimizer=lyap_optimizer,
                lambda_value=lambda_value,
                update_idx=update_idx + 1,
                env_steps=env_steps,
                config=cfg,
                algo=args.algo,
            )
            next_checkpoint_step += checkpoint_interval

        for key in ("actor_loss", "value_loss", "entropy", "approx_kl", "lambda_value", "c_empirical"):
            assert np.isfinite(metrics[key]), f"Non-finite metric detected: {key}={metrics[key]}"

    json_logger.close()
    tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()

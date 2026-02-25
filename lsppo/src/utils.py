"""Utility functions for configuration, reproducibility, logging, and returns."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def str2bool(value: Any) -> bool:
    """Parse booleans from CLI-style values."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config, optionally inheriting from base_config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    base_name = raw.pop("base_config", None)
    if base_name is None:
        return raw

    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    base = load_config(base_path)
    return deep_update(base, raw)


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(device_name: str = "auto") -> torch.device:
    """Choose train/eval device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def flatten_time_env(array: np.ndarray) -> np.ndarray:
    """Flatten [T, N, ...] arrays into [T*N, ...]."""
    return array.reshape((-1,) + array.shape[2:]) if array.ndim > 2 else array.reshape(-1)


def safe_mean(values: Iterable[float]) -> float:
    """Compute mean robustly for empty iterables."""
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    done: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and returns."""
    t_horizon, n_envs = rewards.shape
    advantages = np.zeros((t_horizon, n_envs), dtype=np.float32)
    gae = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(t_horizon)):
        next_values = last_values if t == t_horizon - 1 else values[t + 1]
        nonterminal = 1.0 - done[t]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class JsonlLogger:
    """Append-only JSONL logger for training metrics."""

    def __init__(self, file_path: str | Path) -> None:
        """Open JSONL file handle."""
        self.file_path = Path(file_path)
        ensure_dir(self.file_path.parent)
        self._fh = self.file_path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        """Write one JSON line."""
        item = dict(payload)
        item.setdefault("wall_time", time.time())
        self._fh.write(json.dumps(item, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close file handle."""
        self._fh.close()


def create_writers(logdir: str | Path) -> Tuple[SummaryWriter, JsonlLogger]:
    """Create TensorBoard and JSONL writers."""
    logdir_path = ensure_dir(logdir)
    tb_writer = SummaryWriter(log_dir=str(logdir_path))
    json_logger = JsonlLogger(logdir_path / "metrics.jsonl")
    return tb_writer, json_logger

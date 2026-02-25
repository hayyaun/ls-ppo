from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.utils import (
    JsonlLogger,
    compute_gae,
    create_writers,
    deep_update,
    load_config,
    safe_mean,
    set_seed,
    str2bool,
)


def test_str2bool_and_safe_mean() -> None:
    assert str2bool("true") is True
    assert str2bool("No") is False
    assert safe_mean([1.0, 2.0, 3.0]) == 2.0
    assert safe_mean([]) == 0.0


def test_deep_update_and_load_config(tmp_path: Path) -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    upd = {"a": {"y": 10}, "c": 4}
    merged = deep_update(base, upd)
    assert merged["a"]["x"] == 1
    assert merged["a"]["y"] == 10
    assert merged["c"] == 4

    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(yaml.safe_dump({"foo": 1, "nested": {"v": 2}}), encoding="utf-8")
    child_path.write_text(
        yaml.safe_dump({"base_config": "base.yaml", "nested": {"v": 9}, "bar": 5}),
        encoding="utf-8",
    )
    cfg = load_config(child_path)
    assert cfg["foo"] == 1
    assert cfg["bar"] == 5
    assert cfg["nested"]["v"] == 9


def test_set_seed_reproducibility() -> None:
    set_seed(123, deterministic_torch=False)
    a1 = np.random.rand(4)
    t1 = torch.rand(4)
    set_seed(123, deterministic_torch=False)
    a2 = np.random.rand(4)
    t2 = torch.rand(4)
    np.testing.assert_allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_compute_gae_shapes_and_values() -> None:
    rewards = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
    values = np.array([[0.2, 0.3], [0.1, 0.2]], dtype=np.float32)
    done = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    last_values = np.array([0.0, 0.0], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, done, last_values, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    assert np.all(np.isfinite(adv))
    assert np.all(np.isfinite(ret))


def test_jsonl_logger_and_tensorboard_writer(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path)
    logger.log({"a": 1, "b": 2.0})
    logger.close()
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["a"] == 1
    assert "wall_time" in rows[0]

    writer, json_logger = create_writers(tmp_path / "tb")
    writer.add_scalar("test/value", 1.0, 0)
    json_logger.log({"x": 1})
    writer.close()
    json_logger.close()

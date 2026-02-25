from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


def _make_tiny_config(tmp_path: Path) -> Path:
    with (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(
        {
            "n_envs": 1,
            "steps_per_env": 4,
            "mini_batch_size": 4,
            "epochs": 1,
            "actor_hidden_sizes": [16],
            "critic_hidden_sizes": [16],
            "lyap_hidden_sizes": [8],
            "quick_test_steps": 8,
            "total_env_steps": 8,
            "checkpoint_interval": 4,
            "eval_interval": 4,
            "episodes_per_eval": 2,
            "deterministic_torch": False,
            "log_wandb": False,
        }
    )
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _run_main(monkeypatch, module_main, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    module_main()


def test_end_to_end_lsppo_train_eval_collect_analyze(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _make_tiny_config(tmp_path)
    log_dir = tmp_path / "experiments" / "logs" / "ls_ppo" / "PointGoalSafe-v0" / "seed_0"
    result_dir = tmp_path / "experiments" / "results" / "ls_ppo" / "PointGoalSafe-v0" / "seed_0"
    result_dir.mkdir(parents=True, exist_ok=True)

    from src import analyze as analyze_module
    from src import collect_results as collect_results_module
    from src import eval as eval_module
    from src import train as train_module

    _run_main(
        monkeypatch,
        train_module.main,
        [
            "prog",
            "--config",
            str(cfg_path),
            "--algo",
            "ls_ppo",
            "--quick_test",
            "true",
            "--seed",
            "0",
            "--variant",
            "e2e_lsppo",
            "--logdir",
            str(log_dir),
        ],
    )
    metrics_path = log_dir / "metrics.jsonl"
    ckpt_path = log_dir / "checkpoints" / "latest.pt"
    assert metrics_path.exists()
    assert ckpt_path.exists()

    _run_main(
        monkeypatch,
        eval_module.main,
        [
            "prog",
            "--config",
            str(cfg_path),
            "--algo",
            "ls_ppo",
            "--checkpoint",
            str(ckpt_path),
            "--seed",
            "0",
            "--variant",
            "e2e_lsppo",
            "--out",
            str(result_dir / "eval_summary.json"),
        ],
    )
    assert (result_dir / "eval_summary.json").exists()

    out_csv = tmp_path / "experiments" / "results" / "main_results.csv"
    _run_main(
        monkeypatch,
        collect_results_module.main,
        [
            "prog",
            "--log_root",
            str(tmp_path / "experiments" / "logs"),
            "--eval_root",
            str(tmp_path / "experiments" / "results"),
            "--out_csv",
            str(out_csv),
        ],
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert not df.empty
    assert "eval_mean_return" in df.columns

    fig_dir = tmp_path / "paper"
    _run_main(
        monkeypatch,
        analyze_module.main,
        [
            "prog",
            "--logdir",
            str(tmp_path / "experiments" / "logs"),
            "--figdir",
            str(fig_dir),
            "--results_csv",
            str(out_csv),
        ],
    )
    assert (fig_dir / "LS-PPO_fig1_learning_curve.png").exists()
    assert (fig_dir / "pairwise_stats.csv").exists()


def test_train_quick_other_algorithms(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _make_tiny_config(tmp_path)
    from src import train as train_module

    for algo in ("ppo", "ppo_lagrangian"):
        log_dir = tmp_path / "experiments" / "logs" / algo / "PointGoalSafe-v0" / "seed_0"
        _run_main(
            monkeypatch,
            train_module.main,
            [
                "prog",
                "--config",
                str(cfg_path),
                "--algo",
                algo,
                "--quick_test",
                "true",
                "--seed",
                "0",
                "--variant",
                f"e2e_{algo}",
                "--logdir",
                str(log_dir),
            ],
        )
        assert (log_dir / "metrics.jsonl").exists()
        assert (log_dir / "checkpoints" / "latest.pt").exists()

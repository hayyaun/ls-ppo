"""Analysis, plotting, and statistics for LS-PPO experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def parse_args() -> argparse.Namespace:
    """Parse CLI args for analysis."""
    parser = argparse.ArgumentParser(description="Analyze LS-PPO results.")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--figdir", type=str, default="paper")
    parser.add_argument("--results_csv", type=str, default="experiments/results/main_results.csv")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--trajectory_csv", type=str, default="")
    return parser.parse_args()


def load_metrics(logdir: Path) -> pd.DataFrame:
    """Load metrics from one run dir or nested run dirs."""
    files = []
    direct = logdir / "metrics.jsonl"
    if direct.exists():
        files.append(direct)
    files.extend([p for p in logdir.rglob("metrics.jsonl") if p not in files])

    rows: List[Dict[str, float]] = []
    for path in files:
        rel = path.relative_to(logdir) if path != direct else Path(".")
        algo = "unknown"
        env_name = "unknown"
        seed = 0
        if rel != Path(".") and len(rel.parts) >= 4:
            algo, env_name, seed_dir = rel.parts[0], rel.parts[1], rel.parts[2]
            if seed_dir.startswith("seed_"):
                seed = int(seed_dir.replace("seed_", ""))
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                item.setdefault("algo", algo)
                item.setdefault("env", env_name)
                item.setdefault("seed", seed)
                rows.append(item)
    return pd.DataFrame(rows)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def bootstrap_pvalue(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0) -> float:
    """Compute permutation-style bootstrap p-value for mean difference."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    count = 0
    for _ in range(n_boot):
        rng.shuffle(pooled)
        x_s = pooled[: len(x)]
        y_s = pooled[len(x) :]
        if abs(np.mean(x_s) - np.mean(y_s)) >= observed:
            count += 1
    return float((count + 1) / (n_boot + 1))


def compare_groups(x: np.ndarray, y: np.ndarray) -> Tuple[str, float]:
    """Choose Welch t-test or bootstrap based on normality."""
    if len(x) >= 3 and len(y) >= 3:
        p_x = stats.shapiro(x).pvalue
        p_y = stats.shapiro(y).pvalue
        if p_x > 0.05 and p_y > 0.05:
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            return "welch_t", float(p)
    return "bootstrap", bootstrap_pvalue(x, y)


def plot_curve(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    """Plot mean +/-95% CI over env steps."""
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="env_steps",
        y=metric,
        hue="algo",
        style="env",
        errorbar=("ci", 95),
    )
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_violation_heatmap(trajectory_csv: Path, out_path: Path, bins: int = 40) -> None:
    """Plot unsafe-step fraction over state grid for PointGoal."""
    df = pd.read_csv(trajectory_csv)
    if df.empty or not {"x", "y", "unsafe"}.issubset(df.columns):
        return
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    unsafe = df["unsafe"].to_numpy()
    count, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    unsafe_count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=unsafe)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(unsafe_count, count, out=np.zeros_like(unsafe_count), where=count > 0)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        ratio.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="Unsafe-step fraction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """Generate plots, aggregate results, and run pairwise stats."""
    args = parse_args()
    sns.set_theme(style="whitegrid")
    logdir = Path(args.logdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(logdir)
    if df.empty:
        raise RuntimeError(f"No metrics found under {logdir}")

    if args.out:
        plot_curve(df, metric="episode_return_mean", out_path=Path(args.out), ylabel="Episode return")
    else:
        plot_curve(df, metric="episode_return_mean", out_path=figdir / "LS-PPO_fig1_learning_curve.png", ylabel="Episode return")
    plot_curve(df, metric="episode_discounted_cost_mean", out_path=figdir / "LS-PPO_fig2_cost_curve.png", ylabel="Discounted cost")

    final = (
        df.sort_values(["algo", "env", "seed", "env_steps"])
        .groupby(["algo", "env", "seed"], as_index=False)
        .tail(1)
        .copy()
    )
    final.rename(
        columns={
            "episode_return_mean": "return",
            "episode_discounted_cost_mean": "discounted_cost",
            "episode_violation_rate_mean": "violation_rate",
        },
        inplace=True,
    )
    final_cols = ["algo", "env", "seed", "env_steps", "return", "discounted_cost", "violation_rate", "lambda_value"]
    final = final[final_cols]
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(results_csv, index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=final, x="algo", y="return", hue="env", errorbar=("ci", 95))
    plt.ylabel("Return (final)")
    plt.xlabel("Variant")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "LS-PPO_fig3_ablation_return_bar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=final, x="algo", y="discounted_cost", hue="env", errorbar=("ci", 95))
    plt.ylabel("Discounted cost (final)")
    plt.xlabel("Variant")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "LS-PPO_fig4_ablation_cost_bar.png", dpi=300)
    plt.close()

    trajectory_csv = Path(args.trajectory_csv) if args.trajectory_csv else None
    if trajectory_csv and trajectory_csv.exists():
        plot_violation_heatmap(trajectory_csv, figdir / "LS-PPO_fig5_violation_heatmap.png")

    stats_rows: List[Dict[str, float]] = []
    for env_name, env_df in final.groupby("env"):
        if "ls_ppo" not in set(env_df["algo"]):
            continue
        ref = env_df[env_df["algo"] == "ls_ppo"]
        for algo_name, algo_df in env_df.groupby("algo"):
            if algo_name == "ls_ppo":
                continue
            for metric in ("return", "discounted_cost", "violation_rate"):
                x = ref[metric].to_numpy(dtype=float)
                y = algo_df[metric].to_numpy(dtype=float)
                test_name, p_value = compare_groups(x, y)
                stats_rows.append(
                    {
                        "env": env_name,
                        "baseline": algo_name,
                        "metric": metric,
                        "test": test_name,
                        "p_value": p_value,
                        "cohens_d_lsppo_minus_baseline": cohen_d(x, y),
                        "mean_lsppo": float(np.mean(x)),
                        "mean_baseline": float(np.mean(y)),
                    }
                )
    pd.DataFrame(stats_rows).to_csv(figdir / "pairwise_stats.csv", index=False)


if __name__ == "__main__":
    main()

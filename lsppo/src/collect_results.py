"""Aggregate run directories into a results CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Collect results from logs.")
    parser.add_argument("--log_root", type=str, required=True)
    parser.add_argument("--eval_root", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="experiments/results/main_results.csv")
    return parser.parse_args()


def load_last_jsonl(path: Path) -> Dict[str, float]:
    """Return last JSON object from JSONL file."""
    last = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                last = json.loads(text)
    return last


def main() -> None:
    """Collect final metrics for each run."""
    args = parse_args()
    root = Path(args.log_root)
    eval_root = Path(args.eval_root) if args.eval_root else None
    rows: List[Dict[str, float]] = []

    for metrics_file in root.rglob("metrics.jsonl"):
        rel = metrics_file.relative_to(root)
        if len(rel.parts) >= 4:
            algo, env_name, seed_dir = rel.parts[0], rel.parts[1], rel.parts[2]
        else:
            algo, env_name, seed_dir = "unknown", "unknown", "seed_0"
        seed = int(seed_dir.replace("seed_", "")) if seed_dir.startswith("seed_") else 0

        last = load_last_jsonl(metrics_file)
        if not last:
            continue
        rows.append(
            {
                "algo": last.get("algo", algo),
                "env": last.get("env", env_name),
                "seed": last.get("seed", seed),
                "variant": last.get("variant", "default"),
                "env_steps": last.get("env_steps", 0),
                "return": last.get("episode_return_mean", 0.0),
                "discounted_cost": last.get("episode_discounted_cost_mean", 0.0),
                "violation_rate": last.get("episode_violation_rate_mean", 0.0),
                "lambda_value": last.get("lambda_value", 0.0),
                "run_dir": str(metrics_file.parent),
            }
        )

    # Attach evaluation metrics when available.
    if eval_root is not None and eval_root.exists():
        eval_map: Dict[tuple, Dict[str, float]] = {}
        for eval_file in eval_root.rglob("eval_summary.json"):
            try:
                with eval_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            key = (
                payload.get("algo"),
                payload.get("env"),
                int(payload.get("seed", 0)),
                payload.get("variant", "default"),
            )
            eval_map[key] = payload

        for row in rows:
            key = (row["algo"], row["env"], int(row["seed"]), row.get("variant", "default"))
            eval_payload = eval_map.get(key)
            if not eval_payload:
                continue
            row["eval_mean_return"] = float(eval_payload.get("mean_return", 0.0))
            row["eval_std_return"] = float(eval_payload.get("std_return", 0.0))
            row["eval_mean_discounted_cost"] = float(eval_payload.get("mean_discounted_cost", 0.0))
            row["eval_std_discounted_cost"] = float(eval_payload.get("std_discounted_cost", 0.0))
            row["eval_mean_violation_rate"] = float(eval_payload.get("mean_violation_rate", 0.0))
            row["eval_std_violation_rate"] = float(eval_payload.get("std_violation_rate", 0.0)
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

# LS-PPO (Lyapunov-Shielded PPO)

Production-oriented reference implementation for constrained RL with:
- PPO actor-critic,
- Lyapunov certificate network,
- Action safety projection (closed-form + optional QP fallback),
- Lagrangian cost control.

## 1) Setup

### Pip (default)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### CPU-only PyTorch (optional)
```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### Conda (optional)
```bash
conda create -n lsppo python=3.10 -y
conda activate lsppo
pip install -r requirements.txt
```

### Docker
```bash
docker build -t lsppo:latest .
docker run --rm -it -v "$PWD:/workspace/lsppo" lsppo:latest
```

## 2) Quick smoke test

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml --quick_test true --seed 0 --logdir experiments/logs/smoke
python -m src.analyze --logdir experiments/logs/smoke --out paper/LS-PPO_fig1_learning_curve.png
```

## 3) Baselines and LS-PPO

```bash
# PPO baseline
python -m src.train --config configs/default.yaml --algo ppo --logdir experiments/logs/ppo/PointGoalSafe-v0/seed_0

# PPO-Lagrangian
python -m src.train --config configs/default.yaml --algo ppo_lagrangian --logdir experiments/logs/ppo_lagrangian/PointGoalSafe-v0/seed_0

# LS-PPO (ours)
python -m src.train --config configs/default.yaml --algo ls_ppo --logdir experiments/logs/ls_ppo/PointGoalSafe-v0/seed_0
```

## 4) Full experiment launcher

```bash
bash run_experiments.sh configs/default.yaml experiments/logs experiments/results
```

## 5) Evaluate and aggregate

```bash
python -m src.eval --config configs/default.yaml --algo ls_ppo --checkpoint experiments/logs/ls_ppo/PointGoalSafe-v0/seed_0/checkpoints/latest.pt --out experiments/results/ls_ppo_point_seed0_eval.json
python -m src.collect_results --log_root experiments/logs --out_csv experiments/results/main_results.csv
python -m src.analyze --logdir experiments/logs --figdir paper --results_csv experiments/results/main_results.csv
```

## 6) Logging layout

Each run directory stores:
- `metrics.jsonl`: update-level training logs (step, wall-clock, return/cost/violations, lambda, losses)
- `events.out.tfevents.*`: TensorBoard
- `checkpoints/*.pt`: periodic + latest checkpoints
- `eval_summary.json`: evaluation summary (optional)

Recommended run path:
`experiments/logs/{algo}/{env}/seed_{seed}/`

## 7) Determinism and integrity

- Seed all RNGs (`random`, `numpy`, `torch`, `torch.cuda`).
- Set `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Set `torch.backends.cudnn.benchmark=False` and `torch.backends.cudnn.deterministic=True`.
- For QP projection with OSQP: disable adaptive rho and use fixed tolerances.
- Save config snapshots and checkpoint metadata for exact replay.

## 8) Known limitations

- Projection relies on local model linearization; mismatch between true and approximate dynamics can reduce projection accuracy.
- CPO is not included in this codebase to keep dependencies minimal and deterministic. Comparison is made to PPO and PPO-Lagrangian.

## 9) Type-check gate (project policy compliance)

```bash
# Python sanity
python -m py_compile src/*.py src/envs/*.py src/models/*.py src/algos/*.py

# TypeScript gate (only if TS exists)
if ls tsconfig*.json >/dev/null 2>&1 || ls src/**/*.ts >/dev/null 2>&1 || ls src/**/*.tsx >/dev/null 2>&1; then
  npx tsc --noEmit
else
  echo "tsc gate: N/A (no TypeScript files detected)"
fi
```

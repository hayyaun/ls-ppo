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

## 3.1) Ablation commands (single-run examples)

```bash
# 1) Projection ON/OFF
python -m src.train --config configs/default.yaml --algo ls_ppo --variant proj_on --logdir experiments/logs/ablations/proj_on/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_proj_off.yaml --algo ls_ppo --variant proj_off --logdir experiments/logs/ablations/proj_off/PointGoalSafe-v0/seed_0

# 2) Lyapunov loss weight
python -m src.train --config configs/ablation_lyap_weight_0.yaml --algo ls_ppo --variant lyap_w0 --logdir experiments/logs/ablations/lyap_w0/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_lyap_weight_01.yaml --algo ls_ppo --variant lyap_w01 --logdir experiments/logs/ablations/lyap_w01/PointGoalSafe-v0/seed_0
python -m src.train --config configs/default.yaml --algo ls_ppo --variant lyap_w1 --logdir experiments/logs/ablations/lyap_w1/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_lyap_weight_10.yaml --algo ls_ppo --variant lyap_w10 --logdir experiments/logs/ablations/lyap_w10/PointGoalSafe-v0/seed_0

# 3) Projection method
python -m src.train --config configs/default.yaml --algo ls_ppo --variant proj_finite_diff --logdir experiments/logs/ablations/proj_finite_diff/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_projection_jacobian.yaml --algo ls_ppo --variant proj_jacobian --logdir experiments/logs/ablations/proj_jacobian/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_projection_qp.yaml --algo ls_ppo --variant proj_qp --logdir experiments/logs/ablations/proj_qp/PointGoalSafe-v0/seed_0

# 4) Lyapunov architecture
python -m src.train --config configs/default.yaml --algo ls_ppo --variant lyap_arch_small --logdir experiments/logs/ablations/lyap_arch_small/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_lyap_arch_deep.yaml --algo ls_ppo --variant lyap_arch_deep --logdir experiments/logs/ablations/lyap_arch_deep/PointGoalSafe-v0/seed_0

# 5) Lagrangian update aggressiveness
python -m src.train --config configs/default.yaml --algo ls_ppo --variant lambda_default --logdir experiments/logs/ablations/lambda_default/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_lambda_aggressive.yaml --algo ls_ppo --variant lambda_aggressive --logdir experiments/logs/ablations/lambda_aggressive/PointGoalSafe-v0/seed_0

# 6) Cost limit d
python -m src.train --config configs/ablation_cost_limit_1.yaml --algo ls_ppo --variant cost_d1 --logdir experiments/logs/ablations/cost_d1/PointGoalSafe-v0/seed_0
python -m src.train --config configs/ablation_cost_limit_3.yaml --algo ls_ppo --variant cost_d3 --logdir experiments/logs/ablations/cost_d3/PointGoalSafe-v0/seed_0
python -m src.train --config configs/default.yaml --algo ls_ppo --variant cost_d5 --logdir experiments/logs/ablations/cost_d5/PointGoalSafe-v0/seed_0
```

## 4) Full experiment launcher

```bash
bash run_experiments.sh configs/default.yaml experiments/logs experiments/results
```

## 4.1) Full ablation launcher

```bash
bash run_ablations.sh experiments/logs/ablations experiments/results/ablations
```

## 5) Evaluate and aggregate

```bash
python -m src.eval --config configs/default.yaml --algo ls_ppo --checkpoint experiments/logs/ls_ppo/PointGoalSafe-v0/seed_0/checkpoints/latest.pt --out experiments/results/ls_ppo_point_seed0_eval.json
python -m src.collect_results --log_root experiments/logs --eval_root experiments/results --out_csv experiments/results/main_results.csv
python -m src.collect_results --log_root experiments/logs/ablations --eval_root experiments/results/ablations --out_csv experiments/results/ablations/ablation_results.csv
python -m src.analyze --logdir experiments/logs --figdir paper --results_csv experiments/results/main_results.csv
python -m src.analyze --logdir experiments/logs/ablations --figdir paper --results_csv experiments/results/ablations/ablation_results.csv
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

## 8.1) Compute budget guidance

For the full matrix (`2 envs * 3 algos * 5 seeds = 30 runs`) at 1M steps/run:
- CPU-only (single worker): roughly 45-120 hours total.
- CPU-only (4-way parallel): roughly 12-30 hours total.
- GPU-assisted policy updates: roughly 8-20 hours total.

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

## 10) Test suite and coverage (run only in approved environment)

This repository includes `pytest` tests under `tests/` and a coverage gate in `pytest.ini`:
- minimum coverage: `50%` (`--cov-fail-under=50`)
- measured package: `src/`

Run only when you are on an approved machine/environment:

```bash
pytest
```

If you prefer explicit coverage output:

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=50
```

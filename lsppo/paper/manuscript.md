# Lyapunov-Shielded Proximal Policy Optimization for Constrained Continuous Control

**Authors:** Placeholder Author 1, Placeholder Author 2  
**Affiliations:** Placeholder Institution  

## Abstract

We present Lyapunov-Shielded Proximal Policy Optimization (LS-PPO), a constrained reinforcement learning method that augments PPO with a learned Lyapunov certificate and online action projection into a linearized safe set. The method combines reward optimization with dual cost control through a Lagrangian multiplier, while safety projection reduces instantaneous constraint violations during data collection. We implement LS-PPO on two lightweight Gymnasium-compatible safety tasks (PointGoalSafe and CarGoalSafe) without Safety-Gymnasium dependencies. Our experimental protocol uses fixed seeds, deterministic settings, and standardized logging to evaluate return, discounted cost, and violation rate against PPO and PPO-Lagrangian baselines. We provide ablations over projection usage, Lyapunov loss weighting, projection method, architecture size, dual update aggressiveness, and cost limits. Results are reported with confidence intervals, significance tests, and effect sizes. The full code path, experiment scripts, and analysis utilities are designed for direct reproducibility and manuscript integration.

## 1. Introduction

Constrained RL must jointly optimize task return and safety constraints in deployment-critical settings. Standard unconstrained policy optimization can produce strong reward but often violates safety limits during training and evaluation. We focus on continuous control where a practical method should be sample-efficient, stable, and straightforward to reproduce.

**Contributions**
- We introduce LS-PPO, combining PPO, a non-negative Lyapunov certificate, and projection-based shielding.
- We provide a deterministic and production-oriented implementation with explicit integrity checks and fallback behavior.
- We deliver a complete end-to-end reproducibility package: configs, scripts, figures, statistical tests, and manuscript assets.

## 2. Related Work

Constrained policy optimization methods include CPO and trust-region variants that enforce constraints through constrained updates. Lagrangian PPO variants optimize a reward-cost trade-off but may still violate constraints during rollout. Lyapunov-based constrained RL introduces certificate functions to enforce safe evolution. LS-PPO is aligned with this direction and operationalizes projection during action execution, following the broader Lyapunov-safe RL theme and recent developments (including work by Milosevic, ICML 2025).

## 3. Method

### 3.1 CMDP formulation

We define a CMDP with reward \(r(s,a)\), cost \(c(s,a)\), and policy \(\pi_\theta(a|s)\). The objective is:
\[
\max_\theta \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
\quad \text{s.t.} \quad
\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t c_t\right] \le d.
\]

### 3.2 Lyapunov certificate

We learn \(L_\phi(s) \ge 0\) with Softplus output. Two losses are supported:
\[
\mathcal{L}_{TD} = \left\|L_\phi(s_t) - \left(c_t + \gamma L_\phi(s_{t+1})\right)\right\|_2^2
\]
\[
\mathcal{L}_{dec} = \left[\max\left(0, L_\phi(s_{t+1}) - \gamma L_\phi(s_t) + m\right)\right]^2
\]
where \(m\) is a margin.

### 3.3 Action projection

Given sampled action \(a_\theta\), we linearize around \(a_\theta\):
\[
L(s'|a) \approx L(s'|a_\theta) + \nabla_a L(s'|a_\theta)^\top (a-a_\theta).
\]
Impose:
\[
L(s'|a) \le \gamma L(s) + \epsilon.
\]
This yields:
\[
\nabla_a L^\top a \le \gamma L(s)+\epsilon-L(s'|a_\theta)+\nabla_a L^\top a_\theta.
\]
We solve:
\[
\min_a \|a-a_\theta\|_2^2 \quad
\text{s.t.}\quad \nabla_a L^\top a \le b,\; a \in [a_{\min},a_{\max}].
\]
Closed-form projection is used when feasible; otherwise a deterministic QP (OSQP via CVXPY) is used if installed.

### 3.4 LS-PPO objective and dual update

PPO uses clipped surrogate objective with GAE. Lagrangian dual variable is updated by:
\[
\lambda \leftarrow \mathrm{clip}\left(\lambda + \alpha_\lambda \left(C_{empirical} - d\right), 0, \lambda_{max}\right).
\]
Rollouts always use projected actions in LS-PPO.

### 3.5 Algorithm (pseudocode)

```text
Initialize theta (actor-critic), phi (Lyapunov), lambda >= 0
for each update:
    collect rollout with action a = proj(a_theta, A_s)
    compute reward advantages (GAE) and cost advantages
    update theta with PPO clipped objective using (A_r - lambda A_c)
    update phi with selected Lyapunov losses
    estimate C_empirical from rollout costs
    lambda <- clip(lambda + alpha_lambda * (C_empirical - d), 0, lambda_max)
```

## 4. Experimental Setup

- **Environments:** PointGoalSafe-v0, CarGoalSafe-v0.
- **Algorithms:** PPO, PPO-Lagrangian, LS-PPO.
- **Seeds:** 5 per algorithm per environment (`0..4`).
- **Training horizon:** 1M environment steps per run.
- **Evaluation:** deterministic policy, projection enabled for LS-PPO, 20 episodes.
- **Metrics:** return, discounted cost, violation rate, lambda, Lyapunov losses.

## 5. Results

Figure placeholders:
- `paper/LS-PPO_fig1_learning_curve.png` (reward vs steps)
- `paper/LS-PPO_fig2_cost_curve.png` (cost vs steps)
- `paper/LS-PPO_fig5_violation_heatmap.png` (PointGoal safety map)

Main results table: `paper/LS-PPO_table1_main_results.md`.

Example interpretation template (replace with measured values):
- LS-PPO improves constraint satisfaction with modest reward trade-off relative to PPO.
- LS-PPO reduces violation rate relative to PPO-Lagrangian under tighter cost limits.

## 6. Ablation Study

Use the ablation matrix in `LS-PPO_Implementation.md` and report:
- return (mean +/- std),
- discounted cost (mean +/- std),
- violation rate,
- p-value vs LS-PPO default,
- effect size (Cohen's d).

## 7. Limitations and Broader Impact

**Limitations**
- Projection quality depends on local model linearization and Lyapunov approximation quality.
- Tight constraints can induce conservative behavior and reduced reward.
- Results may vary across hardware and dependency versions despite deterministic controls.

**Broader impact**
- Improved safety compliance in RL training can reduce unsafe exploration in practical systems.
- Overconfidence in learned certificates may still be risky; deployment requires domain-specific validation.

## 8. Reproducibility Appendix

- Full commands:
  - `python -m src.train --config configs/default.yaml --algo ls_ppo --logdir experiments/logs/ls_ppo/PointGoalSafe-v0/seed_0`
  - `python -m src.collect_results --log_root experiments/logs --out_csv experiments/results/main_results.csv`
  - `python -m src.analyze --logdir experiments/logs --figdir paper --results_csv experiments/results/main_results.csv`
- Hyperparameters: see `configs/default.yaml`.
- Seeds: `0,1,2,3,4`.
- Commit hash placeholder: `REPLACE_WITH_GIT_COMMIT_HASH`.
- Determinism: see README section `Determinism and integrity`.

## Suggested captions

- **Figure 1:** Learning curves (mean +/-95% CI across 5 seeds) on PointGoalSafe-v0 and CarGoalSafe-v0.
- **Figure 2:** Discounted-cost trajectories showing constraint satisfaction behavior over training.
- **Table 1:** Main comparison of reward-cost trade-offs and violation rates across baselines and LS-PPO.
- **Table 2:** Ablation summary with significance tests and effect sizes vs LS-PPO default.

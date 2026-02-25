# Cursor prompt — produce a single, complete `LS-PPO_Implementation.md` (ready-to-run, repo skeleton + experiments + paper material)

Use this prompt with Cursor (or paste into any code-generation agent). The agent should output **one** Markdown file named `LS-PPO_Implementation.md` that fully implements the Lyapunov-Shielded PPO project from code skeleton to experiment plan to manuscript-ready figures/tables and ablation studies. The Markdown must be actionable (contain exact file names, command lines, hyperparameters, experiment grids, plotting instructions, and LaTeX/markdown-ready figure/table stubs). Do **not** use Safety-Gymnasium. Use only Gymnasium built-ins and simple custom numpy environments (no MuJoCo, no pygame). Base your method on the referenced prior work: entity["people","Milosevic, N.","ICML 2025 paper author"].

---

## High-level instructions for the agent (what you must produce)
1. Create a repo skeleton (folders and placeholder files) and show exact file contents to implement (Python code) for each file listed below.  
2. Implement working code (PyTorch) for:
   - PPO actor-critic,
   - Lyapunov network \(L_\phi(s)\),
   - Safety projection layer that projects sampled actions \(a_\theta\) into a linearized safe set \(\mathcal{A}_s\) (closed-form when possible, fallback to small QP using `cvxpy` if available),
   - Lagrangian multiplier update loop,
   - Optional Lyapunov loss and its weighting schedule.
3. Provide complete `train.py`, `eval.py`, and `run_experiments.sh` scripts that can reproduce all main results and ablation studies with single command examples.
4. Provide a deterministic experiment plan: seeds, number of runs per seed, hardware assumptions, compute budget estimate, and logging (TensorBoard + optional Weights & Biases).
5. Provide exact hyperparameter grids for baselines and LS-PPO, with recommended defaults for a strong Q1-level submission.
6. Provide data analysis / plotting code that produces the exact figures and tables to include in a Q1 paper (learning curves, constraint-violation curves, box/tables with mean±std, ablation bar charts, p-values).
7. Provide a complete manuscript-ready skeleton (Markdown + LaTeX snippets) with Results, Methods, Ablation, Related Work, Broader Impact, Reproducibility Appendix, and figure/table captions — ready to paste into a Q1 submission system.
8. Provide an **Ablation Study matrix** listing every ablation, expected hypothesis, experiment command, and how to present results.
9. Provide a README with installation and run instructions, including a Dockerfile or `requirements.txt` and `conda` env example.
10. Keep everything reproducible (seed, random state, exact versions in `requirements.txt`).

---

# Required repository layout (agent must create these files and show full contents)

``` id="qlxlxl"
lsppo/
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ run_experiments.sh
├─ configs/
│  ├─ default.yaml
│  └─ ablation_*.yaml
├─ src/
│  ├─ __init__.py
│  ├─ envs/
│  │  ├─ __init__.py
│  │  ├─ point_goal_safe.py        # simple numpy Gymnasium env with cost signal
│  │  └─ car_goal_safe.py          # alternate simple env
│  ├─ models/
│  │  ├─ actor_critic.py
│  │  ├─ lyapunov_net.py
│  │  └─ safety_projection.py
│  ├─ algos/
│  │  ├─ ppo.py
│  │  └─ ls_ppo.py                 # glue: PPO + Lyapunov proj + Lagrangian + losses
│  ├─ train.py
│  ├─ eval.py
│  ├─ analyze.py                   # plotting and stats
│  └─ utils.py
├─ experiments/
│  ├─ logs/                        # logging structure explained in README
│  └─ results/
└─ paper/
   ├─ LS-PPO_fig1_learning_curve.png (generated)
   ├─ LS-PPO_table1_main_results.md
   └─ manuscript.md
```

---

# `requirements.txt` (exact)
Agent must provide this exact file:

``` id="0are2a"
torch>=1.12
numpy>=1.23
gymnasium>=0.28
matplotlib>=3.5
tensorboard>=2.9
pyyaml
scipy
pandas
scikit-learn
cvxpy      # optional; projection fallback
tqdm
seaborn
```

(also include instructions for CPU-only runs and minimal CUDA hints in README)

---

# Key design details the agent must implement

### Environments (no Safety-Gymnasium)
- Implement at least two simple Gymnasium-style safety environments (pure numpy):
  1. `PointGoalSafe-v0` — continuous 2D point reaches goal; reward = negative distance; cost = indicator of entering forbidden circle(s); episode length 200.
  2. `CarGoalSafe-v0` — discrete-time 2D car-like integrator; continuous actions; similar cost structure; more challenging dynamics.

Envs should expose `reward` and `cost` (float cost per step). Provide seeding and deterministic resets.

### Lyapunov certificate \(L_\phi(s)\)
- A small MLP with ReLU (architecture configurable via YAML). Output must be constrained to non-negative values (e.g., `softplus` final activation).
- Provide two learning objectives:
  - TD-style learning to approximate cost-to-go: target = `cost + gamma * L(next_state)`.
  - Lyapunov decrease loss: `max(0, L(s') - gamma * L(s) + margin)` squared (agent must implement margin parameter).
- The agent must be able to switch between those losses via config.

### Safety projection layer
- Linearize constraint: approximate `E[L(s') | s, a]` by finite difference or by Jacobian `\nabla_a L(f(s,a))`; derive a linear inequality in `a`.
- If inequality is linear and box constraints apply, provide closed-form projection (clamp + single-dim solve). If not solvable analytically, formulate a small QP:

``` id="ytpvr6"
min_a  ||a - a_theta||^2
s.t.  grad_L_a · a <= gamma * L(s) - L(s) + epsilon
      a ∈ [a_min, a_max]
```

- Implement the QP using `cvxpy` *only if installed*; otherwise, fall back to simple clipping and log a warning. The QP must be deterministic.

### Training algorithm (LS-PPO)
- PPO clipped surrogate objective for rewards (actor-critic).
- Lagrangian multiplier λ updated per batch: `λ ← clip(λ + α_λ * (C_empirical - d), 0, λ_max)` where `C_empirical` is mean episodic discounted cost over training buffer.
- Each training step:
  1. Collect rollouts using executed action `a = proj(a_theta, A_s)`.
  2. Compute PPO loss and update actor-critic.
  3. Update Lyapunov net using its losses.
  4. Update λ.
- Provide options to:
  - Enable/disable projection (for ablation).
  - Enable/disable Lyapunov loss (for ablation).
  - Use finite-diff vs analytic Jacobian (for ablation).
- Use GAE for advantage estimation; standard normalization.

---

# Hyperparameters — defaults (agent must include these in `configs/default.yaml`)

- env: `PointGoalSafe-v0`
- seed: `0`
- n_envs: `8`
- steps_per_env: `256`
- mini_batch_size: `64`
- epochs: `10`
- lr_actor: `3e-4`
- lr_critic: `1e-3`
- lr_lyap: `1e-3`
- gamma: `0.99`
- gae_lambda: `0.95`
- clip_eps: `0.2`
- vf_coef: `0.5`
- ent_coef: `0.01`
- lam_lr (Lagrangian step): `1e-3`
- cost_limit d: `5.0` (configurable)
- lyap_loss_weight: `1.0`
- projection: `true`
- projection_method: `finite_diff`  # or `jacobian`
- episodes_per_eval: `20`
- seeds_for_report: `[0,1,2,3,4]`  # at least 5 seeds

---

# Baselines (agent must implement and run)
- PPO (no constraint handling).
- PPO-Lagrangian (no projection, uses λ on cost penalty).
- CPO (if a simple reference implementation is available — otherwise note that we compare to PPO-Lagrangian only and cite the related work).
- LS-PPO (ours — projection + L_phi + λ).

Provide exact `train` commands for each baseline and for LS-PPO.

Examples:

```bash id="epo7mf"
# PPO baseline
python -m src.train --config configs/default.yaml --algo ppo --logdir experiments/logs/ppo

# PPO-Lagrangian
python -m src.train --config configs/default.yaml --algo ppo_lagrangian --logdir experiments/logs/ppo_lagr

# LS-PPO (ours)
python -m src.train --config configs/default.yaml --algo ls_ppo --logdir experiments/logs/ls_ppo
```

---

# Experiment plan & schedule (agent must output a reproducible plan)
- For each env (PointGoalSafe, CarGoalSafe), run 5 seeds for each algorithm (PPO, PPO-Lagrangian, LS-PPO).
- Each run: `n_updates` sufficient to reach convergence (e.g., 1M environment steps or configurable).
- Save checkpoints every `50k` steps.
- Logging: episode return, discounted cost, per-step safety violation count, λ, Lyapunov loss.
- Final evaluation: run `episodes_per_eval` episodes per seed with deterministic policy (mean action, projection enabled) and collect metrics.

Provide `run_experiments.sh` that wraps these loops in for-loops, creates `experiments/results/{algo}/{env}/{seed}`, launches training with `nohup` and records PID, and a `collect_results.py` to aggregate metrics into CSV.

---

# Ablation studies (must be a clear matrix — agent must produce commands & expected plotting)
Agent must implement these ablations as separate configs and include plotting code to generate the associated figure/table.

1. **Projection ON vs OFF**  
   - Hypothesis: Projection reduces per-step violations and improves constraint satisfaction with minimal reward loss.  
   - Commands: two trainings with `projection=true` and `projection=false`.

2. **Lyapunov loss weight** (`0.0`, `0.1`, `1.0`, `10.0`)  
   - Hypothesis: Moderate weight improves certificate accuracy; too high harms reward signal.

3. **Projection method** (`finite_diff`, `jacobian`, `qp`)  
   - Hypothesis: Jacobian-based projection is more accurate; QP is more conservative.

4. **Lyapunov architecture** (small MLP vs deep MLP)  
   - Hypothesis: Larger nets capture complex cost-to-go but overfit with low samples.

5. **Lagrangian update freq and step size**  
   - Hypothesis: Aggressive λ updates can destabilize reward learning.

6. **Cost limit (d)** (`1.0`, `3.0`, `5.0`)  
   - Hypothesis: Tight constraints reveal the advantage of projection.

For each ablation: include exact config file name (e.g., `configs/ablation_proj_off.yaml`) and a command to run it.

Agent must produce an **Ablation summary table template** (Markdown table) with columns:
`ablation`, `metric` (return ±std), `cost ±std`, `violation_rate`, `p-value vs LS-PPO`, `notes`.

---

# Analysis & plotting (agent must implement `src/analyze.py`)
Plots to produce:
1. Learning curve: reward vs env steps (mean ± 95% CI across seeds).
2. Cost curve: discounted cost vs env steps.
3. Violation heatmap: fraction of unsafe steps vs state grid (for PointGoal).
4. Ablation bar charts: mean reward and mean cost per ablation variant.
5. Statistical test: pairwise t-test (or bootstrap if non-normal) between LS-PPO and baselines; report p-values and effect sizes (Cohen's d).

Agent must include code to save figures as high-res PNGs suitable for paper (300dpi).

---

# Manuscript skeleton (file: `paper/manuscript.md`) — agent must fill with content
Agent must produce a full manuscript draft with these sections, each with content and placeholders for figures/tables produced by experiments. Include LaTeX-ready table code blocks and figure references.

Sections to include:
- Title, Authors (placeholder), Abstract (≤ 200 words).
- Introduction: motivation, problem statement, contributions (3 bullets).
- Related Work: short paragraph on constrained RL, CPO, C-TRPO, Lyapunov RL methods (cite entity["people","Milosevic, N.","ICML 2025 paper author"] and standard refs).
- Method: formal CMDP, Lyapunov certificate definition, projection derivation (show math; include linearized inequality, QP formulation).
- Algorithm: pseudocode block for LS-PPO (detailed steps).
- Experimental Setup: envs, baselines, metrics, seeds, compute.
- Results: include figure placeholders and markdown tables (agent must fill with example numbers if code isn't run; if code run, fill actual numbers).
- Ablation Study: table + short analysis for each ablation.
- Limitations and Broader Impact.
- Reproducibility Appendix: exact commands, full hyperparameter lists, random seeds, exact git commit hash (agent can put placeholder string to be replaced).

Also include **high-quality figure captions** and suggested table captions.

---

# Reproducibility checklist (agent must output a checklist to include in paper)
- All code, configs, and seeds provided.
- `requirements.txt` and Dockerfile included.
- Random seeds & deterministic settings documented.
- Checkpoints & logs saved to `experiments/logs/*`.
- Scripts for reproducing key figures included and executable.

---

# Deliverables (exact files & outputs the Cursor agent must create inside the Markdown)

1. Full content of each source file in `src/` (display code blocks for each file). The code must be runnable and syntactically correct.
2. `configs/default.yaml` and example `configs/ablation_*.yaml` files.
3. `run_experiments.sh` with loops to launch all experiments.
4. `paper/manuscript.md` pre-filled (as above).
5. `paper/LS-PPO_table1_main_results.md` — a Markdown table template.
6. `README.md` with installation, quick start, and recommended compute.
7. Example commands to produce each figure and table.
8. `analyze.py` that aggregates logs and produces figures and CSVs for tables.
9. `requirements.txt` and `Dockerfile`.
10. Ablation matrix (as a clear Markdown table with commands).

---

# Extra quality requirements (for Q1-level submission)
- Code must be modular and well-documented (docstrings for every function).
- Use good coding practices: type hints where useful, clear naming.
- Training loop must checkpoint and support resuming.
- Logging must include wall-clock timestamps, step counts, and per-episode metrics.
- Plots must include shaded confidence intervals and proper axis labels, legends, and captions suitable for a publication figure.
- Provide at least one short paragraph that critiques limitations and future work (honest, not overstated).

---

# Output format required from Cursor (what the agent should print)
- A single Markdown document `LS-PPO_Implementation.md` that contains:
  - The repo layout (as above).
  - Full source code snippets for each file (wrapped in triple-backtick code blocks with filenames as comments).
  - Config files content.
  - run scripts and exact commands.
  - Experiment plan, ablation matrix, and expected plots/tables.
  - The `paper/manuscript.md` contents embedded (or as a link placeholder to file path).
  - A short “How I tested this locally” section describing minimal smoke tests (e.g., run for 1000 steps on `PointGoalSafe-v0` with a tiny network).
- **Important**: The agent must **not** attempt to run experiments itself during generation — only produce code and instructions. (The user will run.)

---

# Minimal smoke-test commands to include in the Markdown
Include these exact commands near the top of the generated Markdown as "Quick smoke test":

```bash id="bt04yw"
# install
pip install -r requirements.txt

# quick smoke test (tiny run; 1k steps)
python -m src.train --config configs/default.yaml --quick_test true --seed 0 --logdir experiments/logs/smoke

# produce a single learning-curve figure from the smoke run
python -m src.analyze --logdir experiments/logs/smoke --out paper/LS-PPO_fig1_learning_curve.png
```

---

# Tone & style
Be succinct, precise, and practical. Use direct, coach-like language when giving instructions. Do not add speculative claims about experimental outcomes—state hypotheses for ablations and how to test them.

---

# Final note to Cursor agent
Produce exactly one Markdown file following the specs above. The markdown should be comprehensive enough that a competent researcher can clone the repository structure from the file, run the smoke test, then scale experiments to reproduce figures/tables for a Q1-level paper.

---

If you want, I can now convert that Markdown into a ready-to-open canvas or create the initial `src/` file contents here. Which would you like next?


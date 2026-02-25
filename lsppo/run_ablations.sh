#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${1:-experiments/logs/ablations}"
RESULT_ROOT="${2:-experiments/results/ablations}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: Neither python3 nor python is available in PATH."
    exit 1
  fi
fi

SEEDS=(0 1 2 3 4)
ENVS=("PointGoalSafe-v0" "CarGoalSafe-v0")
ABLATIONS=(
  "proj_off:configs/ablation_proj_off.yaml"
  "lyap_w0:configs/ablation_lyap_weight_0.yaml"
  "lyap_w01:configs/ablation_lyap_weight_01.yaml"
  "lyap_w1:configs/default.yaml"
  "lyap_w10:configs/ablation_lyap_weight_10.yaml"
  "proj_finite_diff:configs/default.yaml"
  "proj_jacobian:configs/ablation_projection_jacobian.yaml"
  "proj_qp:configs/ablation_projection_qp.yaml"
  "lyap_arch_small:configs/default.yaml"
  "lyap_arch_deep:configs/ablation_lyap_arch_deep.yaml"
  "lambda_default:configs/default.yaml"
  "lambda_aggressive:configs/ablation_lambda_aggressive.yaml"
  "cost_d1:configs/ablation_cost_limit_1.yaml"
  "cost_d3:configs/ablation_cost_limit_3.yaml"
  "cost_d5:configs/default.yaml"
)

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"
PID_FILE="${RESULT_ROOT}/pids.tsv"
echo -e "pid\tablation\tenv\tseed\tlogdir" > "${PID_FILE}"
MISSING_CKPT_FILE="${RESULT_ROOT}/missing_checkpoints.tsv"
echo -e "ablation\tenv\tseed\trun_dir" > "${MISSING_CKPT_FILE}"

running_jobs=0

launch_train() {
  local ablation="$1"
  local config="$2"
  local env_name="$3"
  local seed="$4"
  local run_dir="${LOG_ROOT}/${ablation}/${env_name}/seed_${seed}"
  mkdir -p "${run_dir}"

  nohup "${PYTHON_BIN}" -m src.train \
    --config "${config}" \
    --algo ls_ppo \
    --env "${env_name}" \
    --seed "${seed}" \
    --variant "${ablation}" \
    --logdir "${run_dir}" \
    > "${run_dir}/train.out" 2>&1 &

  local pid="$!"
  echo -e "${pid}\t${ablation}\t${env_name}\t${seed}\t${run_dir}" >> "${PID_FILE}"
  running_jobs=$((running_jobs + 1))

  if [[ "${running_jobs}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
}

run_eval() {
  local ablation="$1"
  local env_name="$2"
  local seed="$3"
  local run_dir="${LOG_ROOT}/${ablation}/${env_name}/seed_${seed}"
  local result_dir="${RESULT_ROOT}/${ablation}/${env_name}/seed_${seed}"
  local checkpoint="${run_dir}/checkpoints/latest.pt"
  mkdir -p "${result_dir}"

  if [[ -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" -m src.eval \
      --config "configs/default.yaml" \
      --algo ls_ppo \
      --env "${env_name}" \
      --seed "${seed}" \
      --variant "${ablation}" \
      --checkpoint "${checkpoint}" \
      --out "${result_dir}/eval_summary.json"
  else
    echo -e "${ablation}\t${env_name}\t${seed}\t${run_dir}" >> "${MISSING_CKPT_FILE}"
  fi
}

for entry in "${ABLATIONS[@]}"; do
  ablation="${entry%%:*}"
  config="${entry#*:}"
  for env_name in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_train "${ablation}" "${config}" "${env_name}" "${seed}"
    done
  done
done

wait

for entry in "${ABLATIONS[@]}"; do
  ablation="${entry%%:*}"
  for env_name in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_eval "${ablation}" "${env_name}" "${seed}"
    done
  done
done

"${PYTHON_BIN}" -m src.collect_results \
  --log_root "${LOG_ROOT}" \
  --eval_root "${RESULT_ROOT}" \
  --out_csv "${RESULT_ROOT}/ablation_results.csv"

echo "Finished ablations. PIDs recorded in ${PID_FILE}"

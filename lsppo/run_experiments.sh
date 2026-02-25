#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"
LOG_ROOT="${2:-experiments/logs}"
RESULT_ROOT="${3:-experiments/results}"
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

ALGORITHMS=("ppo" "ppo_lagrangian" "ls_ppo")
ENVS=("PointGoalSafe-v0" "CarGoalSafe-v0")
SEEDS=(0 1 2 3 4)

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"
PID_FILE="${RESULT_ROOT}/pids.tsv"
echo -e "pid\talgo\tenv\tseed\tlogdir" > "${PID_FILE}"
MISSING_CKPT_FILE="${RESULT_ROOT}/missing_checkpoints.tsv"
echo -e "algo\tenv\tseed\trun_dir" > "${MISSING_CKPT_FILE}"

running_jobs=0

launch_train() {
  local algo="$1"
  local env_name="$2"
  local seed="$3"
  local variant="$4"
  local run_dir="${LOG_ROOT}/${algo}/${env_name}/seed_${seed}"
  local result_dir="${RESULT_ROOT}/${algo}/${env_name}/seed_${seed}"
  mkdir -p "${run_dir}"
  mkdir -p "${result_dir}"

  nohup "${PYTHON_BIN}" -m src.train \
    --config "${CONFIG_PATH}" \
    --algo "${algo}" \
    --env "${env_name}" \
    --seed "${seed}" \
    --variant "${variant}" \
    --logdir "${run_dir}" \
    > "${run_dir}/train.out" 2>&1 &

  local pid="$!"
  echo -e "${pid}\t${algo}\t${env_name}\t${seed}\t${run_dir}" >> "${PID_FILE}"
  running_jobs=$((running_jobs + 1))

  if [[ "${running_jobs}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
}

run_eval() {
  local algo="$1"
  local env_name="$2"
  local seed="$3"
  local variant="$4"
  local run_dir="${LOG_ROOT}/${algo}/${env_name}/seed_${seed}"
  local result_dir="${RESULT_ROOT}/${algo}/${env_name}/seed_${seed}"
  local checkpoint="${run_dir}/checkpoints/latest.pt"
  mkdir -p "${result_dir}"

  if [[ -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" -m src.eval \
      --config "${CONFIG_PATH}" \
      --algo "${algo}" \
      --env "${env_name}" \
      --seed "${seed}" \
      --variant "${variant}" \
      --checkpoint "${checkpoint}" \
      --out "${result_dir}/eval_summary.json"
  else
    echo -e "${algo}\t${env_name}\t${seed}\t${run_dir}" >> "${MISSING_CKPT_FILE}"
  fi
}

for env_name in "${ENVS[@]}"; do
  for algo in "${ALGORITHMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_train "${algo}" "${env_name}" "${seed}" "main"
    done
  done
done

wait

for env_name in "${ENVS[@]}"; do
  for algo in "${ALGORITHMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_eval "${algo}" "${env_name}" "${seed}" "main"
    done
  done
done

"${PYTHON_BIN}" -m src.collect_results \
  --log_root "${LOG_ROOT}" \
  --eval_root "${RESULT_ROOT}" \
  --out_csv "${RESULT_ROOT}/main_results.csv"

echo "Finished. PIDs recorded in ${PID_FILE}"

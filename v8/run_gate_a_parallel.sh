#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="${GATEA_REPO:-${project_dir}/Mobiπ_code}"
experiment_root="${GATEA_ROOT:-${project_dir}}"
experiment_id="${GATEA_EXPERIMENT_ID:-gate_a_v8}"
output_root="${GATEA_OUTPUT_ROOT:-${experiment_root}/results/${experiment_id}}"
log_root="${GATEA_LOG_ROOT:-${experiment_root}/logs/${experiment_id}}"
python_bin="${GATEA_PYTHON:-python3}"

if [[ -f "${CONDA_ROOT:-${HOME}/miniforge3}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_ROOT:-${HOME}/miniforge3}/etc/profile.d/conda.sh"
  conda activate "${GATEA_CONDA_ENV:-mobipi_gatea}"
fi

mkdir -p "${output_root}" "${log_root}"
export MUJOCO_GL=egl
export PYTHONPATH="${script_dir}:${repo_root}:${repo_root}/external/robocasa:${repo_root}/external/robomimic:${repo_root}/external/mimicgen${PYTHONPATH:+:${PYTHONPATH}}"

if (( $# > 0 )); then
  exec "${python_bin}" "${script_dir}/eval_gate_a.py" "$@"
fi

IFS=',' read -r -a gpu_ids <<< "${GATEA_GPUS:-0,1,2}"
IFS=',' read -r -a tasks <<< "${GATEA_TASKS:-TurnOnStove,TurnOnSinkFaucet,TurnOnMicrowave}"
IFS=',' read -r -a scenes <<< "${GATEA_SCENES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a policy_seeds <<< "${GATEA_POLICY_SEEDS:-1,2,3}"
reference_args=()
if [[ -n "${GATEA_REFERENCE_REGISTRY:-}" ]]; then
  reference_args=(--reference-registry "${GATEA_REFERENCE_REGISTRY}")
fi

run_shard() {
  local task="$1"
  local scene="$2"
  local policy_seed="$3"
  local gpu="$4"
  local shard="${output_root}/shards/${task}_scene${scene}_seed${policy_seed}"
  local log="${log_root}/${task}_scene${scene}_seed${policy_seed}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID=0 \
    "${python_bin}" "${script_dir}/eval_gate_a.py" \
      --asset-root "${repo_root}/external/robocasa/robocasa/models/assets" \
      --ckpt-root "${repo_root}/ckpts" \
      --data-root "${repo_root}/data" \
      --clip-cache-root "${experiment_root}/cache/clip" \
      --output-root "${shard}" \
      --tasks "${task}" \
      --scenes "${scene}" \
      --policy-seeds "${policy_seed}" \
      --num-sampled-poses "${GATEA_NUM_POSES:-48}" \
      --rollouts-per-pose "${GATEA_ROLLOUTS_PER_POSE:-5}" \
      --square-side-m 1.0 \
      --target-radius-m 1.0 \
      --yaw-jitter-deg 30.0 \
      --max-sampling-attempts 5000 \
      --policy-version "${GATEA_POLICY_VERSION:-checkpoint-metadata-unavailable}" \
      --policy-git-commit "${GATEA_POLICY_GIT_COMMIT:-unknown}" \
      "${reference_args[@]}" \
      > "${log}" 2>&1
}

status=0
job_index=0
pids=()
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
}

for task in "${tasks[@]}"; do
  for scene in "${scenes[@]}"; do
    for policy_seed in "${policy_seeds[@]}"; do
      gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
      run_shard "${task}" "${scene}" "${policy_seed}" "${gpu}" &
      pids+=("$!")
      job_index=$((job_index + 1))
      if (( ${#pids[@]} == ${#gpu_ids[@]} )); then
        wait_batch
      fi
    done
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_batch
fi
if (( status != 0 )); then
  echo "At least one Gate A v8 shard failed; inspect ${log_root}." >&2
  exit "${status}"
fi

"${python_bin}" "${script_dir}/analyze_gate_a.py" "${output_root}" \
  --output-dir "${output_root}/analysis"

#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${GATEA_CONDA_ENV:-mobipi_gatea}"

experiment_root="${GATEA_ROOT:-$HOME/GateA_Experiment}"
repo_root="${GATEA_REPO:-$experiment_root/mobipi}"
code_root="${GATEA_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
pose_registry="${GATEA_POSE_REGISTRY:?Set GATEA_POSE_REGISTRY to the audited pose-registry JSON}"
geometry_config="${GATEA_GEOMETRY_CONFIG:?Set GATEA_GEOMETRY_CONFIG to reviewed contact geometry JSON}"
experiment_split="${GATEA_SPLIT:-test}"
if [[ "$experiment_split" == "calibration" ]]; then
  default_scenes="8,9"
elif [[ "$experiment_split" == "test" ]]; then
  default_scenes="0,1,2,3,4,5,6,7"
else
  echo "GATEA_SPLIT must be calibration or test." >&2
  exit 2
fi
experiment_id="${GATEA_EXPERIMENT_ID:-gate_a_v7_${experiment_split}}"
output_root="${GATEA_OUTPUT_ROOT:-$experiment_root/results/$experiment_id}"
log_root="${GATEA_LOG_ROOT:-$experiment_root/logs/$experiment_id}"
IFS=',' read -r -a gpu_ids <<< "${GATEA_GPUS:-0,1,2}"

mkdir -p "$output_root" "$log_root"
export MUJOCO_GL=egl
export PYTHONPATH="$code_root:$repo_root:${PYTHONPATH:-}"

IFS=',' read -r -a tasks <<< "${GATEA_TASKS:-TurnOnStove,TurnOnSinkFaucet,TurnOnMicrowave}"
IFS=',' read -r -a scenes <<< "${GATEA_SCENES:-$default_scenes}"
IFS=',' read -r -a policy_seeds <<< "${GATEA_POLICY_SEEDS:-1,2,3}"

run_shard() {
  local task="$1"
  local scene="$2"
  local policy_seed="$3"
  local gpu="$4"
  local shard="$output_root/shards/${task}_scene${scene}_seed${policy_seed}"
  local log="$log_root/${task}_scene${scene}_seed${policy_seed}.log"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0 \
    python "$code_root/eval_gate_a.py" \
      --asset-root "$repo_root/external/robocasa/robocasa/models/assets" \
      --ckpt-root "$repo_root/ckpts" \
      --data-root "$repo_root/data" \
      --clip-cache-root "$experiment_root/cache/clip" \
      --pose-registry "$pose_registry" \
      --geometry-config "$geometry_config" \
      --output-root "$shard" \
      --tasks "$task" \
      --scenes "$scene" \
      --policy-seeds "$policy_seed" \
      --split "$experiment_split" \
      --max-exact-filter-candidates 2500 \
      --global-grid-step 0.10 \
      --min-pixel-visibility 0.50 \
      --max-coarse-ik-position-residual 0.08 \
      --basin-candidates 64 \
      --basin-rollouts 10 \
      --basin-success-threshold 0.80 \
      --validation-rollouts 20 \
      > "$log" 2>&1
}

status=0
job_index=0
pids=()
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  pids=()
}
for task in "${tasks[@]}"; do
  for scene in "${scenes[@]}"; do
    for policy_seed in "${policy_seeds[@]}"; do
      gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
      run_shard "$task" "$scene" "$policy_seed" "$gpu" &
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
  echo "At least one Gate-A shard failed; inspect $log_root." >&2
  exit "$status"
fi

python "$code_root/analyze_gate_a.py" "$output_root" \
  --split "$experiment_split" \
  --minimum-effect 0.10 \
  --required-positive-tasks 2 \
  --minimum-oracle-success 0.20 \
  --expected-validation-rollouts 20 \
  --output "$output_root/aggregate.json"

#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mobipi_gatea

ROOT="$HOME/GateA_Experiment"
REPO="$ROOT/mobipi"

POSE_REGISTRY="${GATEA_POSE_REGISTRY:?Set GATEA_POSE_REGISTRY}"
GEOMETRY_CONFIG="${GATEA_GEOMETRY_CONFIG:?Set GATEA_GEOMETRY_CONFIG}"

OUT="${GATEA_SMOKE_OUTPUT:-$ROOT/results/gate_a_v7_smoke_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$ROOT:$REPO:${PYTHONPATH:-}"

python "$ROOT/eval_gate_a.py" \
  --asset-root "$REPO/external/robocasa/robocasa/models/assets" \
  --ckpt-root "$REPO/ckpts" \
  --data-root "$REPO/data" \
  --clip-cache-root "$ROOT/cache/clip" \
  --pose-registry "$POSE_REGISTRY" \
  --geometry-config "$GEOMETRY_CONFIG" \
  --output-root "$OUT" \
  --tasks TurnOnStove \
  --scenes 0 \
  --policy-seeds 1 \
  --split calibration \
  --max-exact-filter-candidates 250 \
  --global-grid-step 0.10 \
  --min-pixel-visibility 0.50 \
  --max-coarse-ik-position-residual 0.08 \
  --basin-candidates 8 \
  --basin-rollouts 1 \
  --basin-success-threshold 0.80 \
  --validation-rollouts 1 \
  --horizon 5

echo "Smoke output: $OUT"

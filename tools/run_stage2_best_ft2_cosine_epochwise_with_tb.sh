#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${ROOT_DIR}/projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage2_best_ft2_cosine.py"
WORK_DIR="${ROOT_DIR}/work_dirs/bevfusion_radar_weather_stage2_best_ft2_cosine_epochwise"
SOURCE_ANN_DIR="${ROOT_DIR}/work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1/weather_eval/ann_splits"
PYTHON_BIN="${PYTHON_BIN:-/home/vipuser/miniconda3/envs/bevfusion/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MAX_EPOCH="${MAX_EPOCH:-2}"
TB_PORT="${TB_PORT:-6011}"

mkdir -p "${WORK_DIR}/weather_eval/ann_splits" "${WORK_DIR}/epochwise_runner"

if [[ ! -f "${SOURCE_ANN_DIR}/nuscenes_infos_val_day.pkl" ]]; then
    echo "missing source ann splits: ${SOURCE_ANN_DIR}" >&2
    exit 1
fi

cp -f "${SOURCE_ANN_DIR}"/nuscenes_infos_val_*.pkl "${WORK_DIR}/weather_eval/ann_splits/"

pkill -f "run_stage2_epochwise_resume_with_weather_eval.sh .*bevfusion_lidar-cam-radar_geometry_enhanced_stage2_best_ft2_cosine.py.*${WORK_DIR}" >/dev/null 2>&1 || true
pkill -f "tensorboard.main --logdir ${WORK_DIR}" >/dev/null 2>&1 || true
fuser -k "${TB_PORT}"/tcp >/dev/null 2>&1 || true

nohup "${PYTHON_BIN}" -m tensorboard.main \
  --logdir "${WORK_DIR}" \
  --port "${TB_PORT}" \
  --host 0.0.0.0 \
  --load_fast=false \
  > "${WORK_DIR}/tensorboard.log" 2>&1 &
TB_PID=$!

nohup env \
  PYTHON_BIN="${PYTHON_BIN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  WEATHER_TEST_WORKERS="${WEATHER_TEST_WORKERS:-8}" \
  WEATHER_TEST_PERSISTENT_WORKERS="${WEATHER_TEST_PERSISTENT_WORKERS:-True}" \
  WEATHER_TEST_PIN_MEMORY="${WEATHER_TEST_PIN_MEMORY:-True}" \
  WEATHER_TEST_PREFETCH_FACTOR="${WEATHER_TEST_PREFETCH_FACTOR:-2}" \
  bash "${ROOT_DIR}/tools/run_stage2_epochwise_resume_with_weather_eval.sh" \
    "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${MAX_EPOCH}" \
  > "${WORK_DIR}/epochwise_runner/launcher.log" 2>&1 &
SUP_PID=$!

echo "supervisor_pid=${SUP_PID}"
echo "tensorboard_pid=${TB_PID}"
echo "work_dir=${WORK_DIR}"
echo "tensorboard=http://127.0.0.1:${TB_PORT}"
echo "launcher_log=${WORK_DIR}/epochwise_runner/launcher.log"
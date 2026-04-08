#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/root/mmdetection3d-main/work_dirs/bevfusion_lidar-cam-radar_geometry_enhanced_8e_2xa800_tb"
LOG_FILE="${WORK_DIR}/train_stdout.log"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "日志不存在: ${LOG_FILE}"
  exit 1
fi

echo "== 最近训练指标(一次性) =="
grep -E "Epoch\(train\)|Epoch\(val\)|NuScenes metric|NDS|mAP|lr:" "${LOG_FILE}" | tail -n 80 || true

echo
echo "== 实时追踪（Ctrl+C 退出） =="
tail -f "${LOG_FILE}" | grep --line-buffered -E "Epoch\(train\)|Epoch\(val\)|NuScenes metric|NDS|mAP|eta:|time:|data_time:|loss|lr:" || true

#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-/root/mmdetection3d-main/work_dirs/bevfusion_radar_weather_stage1_e6_depthsup_epoch7init_cameraaware}"
LOG_FILE="${WORK_DIR}/train_stdout.log"
OUT_FILE="${WORK_DIR}/monitor_status.log"
INTERVAL="${2:-60}"

touch "${OUT_FILE}"

echo "[$(date '+%F %T')] monitor start: WORK_DIR=${WORK_DIR}, interval=${INTERVAL}s" >> "${OUT_FILE}"

while true; do
  {
    echo "===== $(date '+%F %T') ====="

    TRAIN_PS=$(ps -ef | grep -E 'tools/train.py|dist_train.sh|torch.distributed.launch' | grep 'bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e6_depthsup_epoch7init.py' | grep -v grep || true)
    if [[ -n "${TRAIN_PS}" ]]; then
      echo "[process] alive"
      echo "${TRAIN_PS}"
    else
      echo "[process] missing"
    fi

    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

    if [[ -f "${LOG_FILE}" ]]; then
      echo "[recent-train]"
      grep -E 'Epoch\(train\)|Epoch\(val\)|NDS|mAP|The length of training dataset|FreezeModulesHook' "${LOG_FILE}" | tail -n 12 || true

      echo "[recent-errors]"
      grep -E 'Traceback|OutOfMemory|RuntimeError|ChildFailedError|Killed|SIGTERM|AssertionError' "${LOG_FILE}" | tail -n 12 || true
    else
      echo "[log] missing: ${LOG_FILE}"
    fi
  } >> "${OUT_FILE}"

  sleep "${INTERVAL}"
done
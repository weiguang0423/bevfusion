#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/mmdetection3d-main"
PY_BIN="/home/vipuser/miniconda3/envs/bevfusion/bin/python"
CONFIG="projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e12.py"
NPROC=2
MASTER_PORT=29605 
TB_PORT=6008
WORK_DIR="${ROOT_DIR}/work_dirs/bevfusion_radar_weather_stage1_e12"

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"

echo "[1/4] 清理旧训练进程..."
pkill -f "tools/train.py" >/dev/null 2>&1 || true
pkill -f "torch.distributed.launch" >/dev/null 2>&1 || true
fuser -k 29604/tcp >/dev/null 2>&1 || true
fuser -k 29605/tcp >/dev/null 2>&1 || true

echo "[2/4] 清理旧 TensorBoard 进程并释放端口..."
pkill -f "tensorboard" >/dev/null 2>&1 || true
fuser -k 6007/tcp >/dev/null 2>&1 || true
fuser -k 6008/tcp >/dev/null 2>&1 || true

sleep 1

echo "[3/4] 启动 Stage 1 训练 (后台)..."
nohup env PYTHONUNBUFFERED=1 "${PY_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" --launcher pytorch \
  > "${WORK_DIR}/train_stdout.log" 2>&1 &
TRAIN_PID=$!

sleep 2

echo "[4/4] 启动 TensorBoard (后台)..."
nohup "${PY_BIN}" -m tensorboard.main --logdir "${WORK_DIR}" --port "${TB_PORT}" --host 0.0.0.0 --load_fast=false > "${WORK_DIR}/tensorboard.log" 2>&1 &
TB_PID=$!

echo "=========================================================="
echo "✅ 训练已在后台启动 (PID=${TRAIN_PID})"
echo "✅ TensorBoard 已在后台启动 (PID=${TB_PID}, 访问端口 http://127.0.0.1:${TB_PORT})"
echo "=========================================================="
echo "👉 实时查看训练日志请运行:"
echo "tail -f ${WORK_DIR}/train_stdout.log"
echo "=========================================================="

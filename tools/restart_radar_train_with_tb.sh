#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/mmdetection3d-main"
PY_BIN="/home/vipuser/miniconda3/envs/bevfusion/bin/python"
CONFIG_CANDIDATES=(
  "projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced.py"
  "mmdetection3d-main/projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced.py"
  "mmdetection3d-main/mmdetection3d-main/projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced.py"
)
NPROC=2
MASTER_PORT=29604
TB_PORT=6007
FOLLOW_LOG=${FOLLOW_LOG:-1}
LOG_MODE=${LOG_MODE:-metrics}
TB_PATH_PREFIX=${TB_PATH_PREFIX:-}
WORK_DIR="${ROOT_DIR}/work_dirs/bevfusion_lidar-cam-radar_geometry_enhanced_8e_2xa800_tb"
PRETRAIN_CKPT="/root/mmdetection3d-main/work_dirs/bevfusion_lidar-cam_depth_supervision_2gpu/epoch_7.pth"

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"

CONFIG=""
for c in "${CONFIG_CANDIDATES[@]}"; do
  if [[ -f "${ROOT_DIR}/${c}" ]]; then
    CONFIG="${c}"
    break
  fi
done

if [[ -z "${CONFIG}" ]]; then
  echo "未找到配置文件，尝试过:"
  for c in "${CONFIG_CANDIDATES[@]}"; do
    echo "  - ${ROOT_DIR}/${c}"
  done
  exit 1
fi

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "预训练权重不存在: ${PRETRAIN_CKPT}"
  exit 1
fi

echo "使用配置: ${CONFIG}"
echo "使用预训练权重: ${PRETRAIN_CKPT}"

echo "[1/4] 停止旧训练进程..."
pkill -f "tools/train.py ${CONFIG}" >/dev/null 2>&1 || true
pkill -f "torch.distributed.launch.*${CONFIG}" >/dev/null 2>&1 || true
fuser -k ${MASTER_PORT}/tcp >/dev/null 2>&1 || true

echo "[2/4] 停止旧 TensorBoard 并释放端口 ${TB_PORT}..."
pkill -f "tensorboard.main --logdir ./work_dirs/bevfusion_lidar-cam-radar_geometry_enhanced_8e_2xa800_tb" >/dev/null 2>&1 || true
fuser -k ${TB_PORT}/tcp >/dev/null 2>&1 || true

sleep 1

echo "[3/4] 启动训练..."
nohup env PYTHONUNBUFFERED=1 "${PY_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" --launcher pytorch \
  --cfg-options \
  load_from="${PRETRAIN_CKPT}" \
  work_dir="${WORK_DIR}" \
  default_hooks.checkpoint.by_epoch=True \
  default_hooks.checkpoint.interval=1 \
  default_hooks.checkpoint.save_last=True \
  > "${WORK_DIR}/train_stdout.log" 2>&1 &
TRAIN_PID=$!

sleep 2

echo "[4/4] 启动 TensorBoard (固定端口 ${TB_PORT})..."
TB_ARGS=(
  -m tensorboard.main
  --logdir "${WORK_DIR}"
  --port "${TB_PORT}"
  --host 0.0.0.0
  --load_fast=false
)

if [[ -n "${TB_PATH_PREFIX}" ]]; then
  TB_ARGS+=(--path_prefix "${TB_PATH_PREFIX}")
fi

nohup "${PY_BIN}" "${TB_ARGS[@]}" > "${WORK_DIR}/tensorboard.log" 2>&1 &
TB_PID=$!

echo "训练已启动: PID=${TRAIN_PID}"
echo "TensorBoard已启动: PID=${TB_PID}"
echo "TensorBoard: http://127.0.0.1:${TB_PORT}"
if [[ -n "${TB_PATH_PREFIX}" ]]; then
  echo "TensorBoard path_prefix: ${TB_PATH_PREFIX}"
fi
echo "训练日志: ${WORK_DIR}/train_stdout.log"
echo "TB日志: ${WORK_DIR}/tensorboard.log"
echo "实时看训练: tail -f ${WORK_DIR}/train_stdout.log"
echo "只看迭代/epoch指标: grep -E \"Epoch\\(train\\)|Epoch\\(val\\)|NDS|mAP\" ${WORK_DIR}/train_stdout.log | tail -n 50"

if [[ "${FOLLOW_LOG}" == "1" ]]; then
  echo ""
  echo "进入实时日志模式 (FOLLOW_LOG=1, LOG_MODE=${LOG_MODE}) ..."
  if [[ "${LOG_MODE}" == "metrics" ]]; then
    exec bash -lc "stdbuf -oL -eL tail -n 200 -F '${WORK_DIR}/train_stdout.log' | stdbuf -oL -eL grep -E 'Epoch\\(train\\)|Epoch\\(val\\)|NDS|mAP|ERROR|Traceback|RuntimeError'"
  else
    exec tail -n 200 -F "${WORK_DIR}/train_stdout.log"
  fi
fi

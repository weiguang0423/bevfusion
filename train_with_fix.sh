#!/bin/bash
# 修复单核处理问题的训练启动脚本
# 硬件配置: NVIDIA A800-SXM4-40GB × 4, 48核/48GB

# ============ 关键：在 Python 启动前设置环境变量 ============
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# ============ CUDA 配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================================"
echo "训练启动脚本 - 已应用 GIL 锁修复"
echo "硬件: A800-SXM4-40GB × 4, 48核/48GB"
echo "============================================================"
echo "环境变量: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: bash train_with_fix.sh <config_file> [其他参数]"
    echo "示例: bash train_with_fix.sh projects/BEVFusion/configs/bevfusion_lidar-cam-radar_finetune.py"
    echo "多卡: bash train_with_fix.sh projects/BEVFusion/configs/bevfusion_lidar-cam-radar_finetune.py --launcher pytorch"
    exit 1
fi

CONFIG=$1
shift

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

echo "配置文件: $CONFIG"

# 检查是否使用分布式训练
USE_DIST=false
for arg in "$@"; do
    if [ "$arg" == "--launcher" ]; then
        USE_DIST=true
        break
    fi
done

# 检测 GPU 数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "检测到 GPU 数量: $NUM_GPUS"
echo "============================================================"

if [ "$USE_DIST" = true ]; then
    echo "启动分布式训练 (${NUM_GPUS} GPUs)..."
    torchrun --nproc_per_node=$NUM_GPUS tools/train.py "$CONFIG" "$@"
else
    echo "启动单卡训练..."
    python tools/train.py "$CONFIG" "$@"
fi

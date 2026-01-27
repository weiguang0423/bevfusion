#!/bin/bash
set -e

echo "=========================================="
echo "生成 nuScenes pkl 文件（包含历史帧）"
echo "=========================================="
echo ""

cd /root/mmdetection3d-main

echo "[1/3] 激活环境..."
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion

echo "[2/3] 开始生成（max_sweeps=10）..."
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --max-sweeps 10 \
    --workers 4

echo ""
echo "[3/3] 检查生成结果..."
ls -lh data/nuscenes/nuscenes_infos_*.pkl

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

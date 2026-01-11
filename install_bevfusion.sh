#!/bin/bash
# BEVFusion 安装脚本 (Ubuntu 22.04 + CUDA 12.1 + RTX 4090)
# 适用于 AutoDL 平台（已预装 PyTorch）

set -e

echo "=========================================="
echo "BEVFusion 环境安装 (Linux)"
echo "=========================================="

# ============ 检查 PyTorch ============
echo ""
echo "[检查] PyTorch 环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# ============ 安装依赖 ============
echo ""
echo "[安装] OpenMMLab 依赖..."
pip install mmengine>=0.8.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet>=3.1.0

echo ""
echo "[安装] 数据集相关依赖..."
pip install nuscenes-devkit lyft_dataset_sdk
pip install open3d trimesh plyfile
pip install networkx numba scikit-image
pip install tensorboard

echo ""
echo "[安装] mmdet3d..."
pip install -v -e .

# ============ 编译 BEVFusion（需要 GPU）============
echo ""
echo "[编译] BEVFusion CUDA 算子..."
python projects/BEVFusion/setup.py develop

# ============ 验证 ============
echo ""
echo "[验证] 检查安装..."
python -c "
import torch
import mmcv
import mmengine
import mmdet
import mmdet3d
print(f'PyTorch: {torch.__version__}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmengine: {mmengine.__version__}')
print(f'mmdet: {mmdet.__version__}')
print(f'mmdet3d: {mmdet3d.__version__}')

from projects.BEVFusion.bevfusion.ops.bev_pool import bev_pool_ext
from projects.BEVFusion.bevfusion.ops.voxel import voxel_layer
print('BEVFusion CUDA 算子: OK')
print('')
print('========================================')
print('安装完成 ✓')
print('========================================')
"

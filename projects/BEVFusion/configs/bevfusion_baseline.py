"""
BEVFusion Baseline (Ablation Study)

This configuration uses the standard DepthLSSTransform without any enhancements.
Used as baseline for ablation study to compare against enhanced versions.

Features:
- Standard DepthLSSTransform (no depth supervision, no camera-aware)
- Identical to original BEVFusion lidar-cam configuration
"""

_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# ============ 说明 ============
# 消融实验配置：基线模型
#
# 禁用功能：
# - 深度监督损失
# - Camera-Aware特征编码
#
# 用途：
# 作为消融实验的基线，与其他配置进行对比：
# 1. bevfusion_baseline.py (本配置)
# 2. bevfusion_depth_supervision_only.py (仅深度监督)
# 3. bevfusion_camera_aware_only.py (仅Camera-Aware)
# 4. bevfusion_depth_supervision.py (两者都启用)
#
# 使用方法：
# python tools/train.py projects/BEVFusion/configs/bevfusion_baseline.py
#
# 注意：
# 本配置实际上与bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# 完全相同，创建此文件仅为了消融实验的清晰性和一致性。

#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

print("=" * 80)
print("开始生成 nuScenes pkl 文件（包含历史帧）")
print("=" * 80)

print("\n[1/5] 导入模块...")
from tools.dataset_converters import nuscenes_converter

print("[2/5] 设置参数...")
root_path = './data/nuscenes'
info_prefix = 'nuscenes'
version = 'v1.0-trainval'
max_sweeps = 10

print(f"  - 数据路径: {root_path}")
print(f"  - 版本: {version}")
print(f"  - 历史帧数: {max_sweeps}")

print("\n[3/5] 初始化 NuScenes 数据集（这一步可能需要1-2分钟）...")
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

print("\n[4/5] 开始生成 pkl 文件...")
nuscenes_converter.create_nuscenes_infos(
    root_path=root_path,
    info_prefix=info_prefix,
    version=version,
    max_sweeps=max_sweeps
)

print("\n[5/5] 完成！")
print("=" * 80)
print("生成的文件:")
print("  - data/nuscenes/nuscenes_infos_train.pkl")
print("  - data/nuscenes/nuscenes_infos_val.pkl")
print("=" * 80)

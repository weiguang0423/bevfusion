#!/usr/bin/env python
"""验证多帧融合情况"""
import pickle

pkl_path = 'data/nuscenes/nuscenes_infos_train.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

sample = data['data_list'][0]

print("=" * 80)
print("检查第一个样本的多帧融合信息")
print("=" * 80)

# 1. 检查 LiDAR 多帧融合
print("\n【LiDAR 点云】")
print("-" * 80)
lidar_points = sample.get('lidar_points', {})
print(f"lidar_points 键: {list(lidar_points.keys())}")

if 'lidar_sweeps' in sample:
    print(f"✓ 有 lidar_sweeps 字段")
    print(f"  数量: {len(sample['lidar_sweeps'])}")
    if len(sample['lidar_sweeps']) > 0:
        print(f"  第一个 sweep 键: {list(sample['lidar_sweeps'][0].keys())}")
else:
    print(f"✗ 没有 lidar_sweeps 字段")

# 检查 lidar_points 内部是否有 sweeps
if 'sweeps' in lidar_points:
    print(f"✓ lidar_points 内有 sweeps")
    print(f"  数量: {len(lidar_points['sweeps'])}")
else:
    print(f"✗ lidar_points 内没有 sweeps")

# 2. 检查 Radar 多帧融合
print("\n【Radar 点云】")
print("-" * 80)

# 检查顶层 radar_sweeps
if 'radar_sweeps' in sample:
    print(f"✓ 有 radar_sweeps 字段（顶层）")
    print(f"  数量: {len(sample['radar_sweeps'])}")
    if len(sample['radar_sweeps']) > 0:
        print(f"  第一个 sweep 键: {list(sample['radar_sweeps'][0].keys())}")
else:
    print(f"✗ 没有 radar_sweeps 字段（顶层）")

# 检查 radars 字段
if 'radars' in sample:
    radars = sample['radars']
    print(f"\n✓ 有 radars 字段")
    print(f"  传感器: {list(radars.keys())}")
    
    # 检查每个传感器是否有 sweeps
    for sensor_name, sensor_data in radars.items():
        if 'sweeps' in sensor_data:
            print(f"  {sensor_name}: 有 sweeps ({len(sensor_data['sweeps'])} 个)")
        else:
            print(f"  {sensor_name}: 无 sweeps")
        break  # 只检查第一个

# 检查 radar_info 字段
if 'radar_info' in sample:
    radar_info = sample['radar_info']
    print(f"\n✓ 有 radar_info 字段")
    print(f"  传感器: {list(radar_info.keys())}")
    
    # 检查每个传感器是否有 sweeps
    for sensor_name, sensor_data in radar_info.items():
        if 'sweeps' in sensor_data:
            print(f"  {sensor_name}: 有 sweeps ({len(sensor_data['sweeps'])} 个)")
        else:
            print(f"  {sensor_name}: 无 sweeps")
        break  # 只检查第一个

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

# LiDAR 结论
has_lidar_sweeps = 'lidar_sweeps' in sample or 'sweeps' in sample.get('lidar_points', {})
print(f"\nLiDAR 多帧融合: {'✓ 有历史帧' if has_lidar_sweeps else '✗ 无历史帧（会复制当前帧）'}")

# Radar 结论
has_radar_sweeps = False
if 'radar_sweeps' in sample:
    has_radar_sweeps = True
elif 'radars' in sample:
    for sensor_data in sample['radars'].values():
        if 'sweeps' in sensor_data:
            has_radar_sweeps = True
            break
elif 'radar_info' in sample:
    for sensor_data in sample['radar_info'].values():
        if 'sweeps' in sensor_data:
            has_radar_sweeps = True
            break

print(f"Radar 多帧融合: {'✓ 有历史帧' if has_radar_sweeps else '✗ 无历史帧（会复制当前帧）'}")

if not has_radar_sweeps:
    print("\n⚠ 警告: Radar 没有历史帧数据")
    print("  - LoadRadarPointsFromMultiSweeps 会复制当前帧 5 次")
    print("  - 所有点的时间戳 dt = 0.0")
    print("  - 时间偏移量功能无法工作")
    print("  - 需要运行数据转换脚本添加 radar_sweeps")

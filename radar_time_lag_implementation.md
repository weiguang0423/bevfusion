# 雷达多帧融合时间偏移量（Time Lag / dt）实施方案

## 概述

时间偏移量（Time Lag / dt）是多帧融合中最关键的一步，它能把"物理上的对不齐"转化为"神经网络能理解的速度特征"。

## 核心原理

### 物理意义
- **位置校正逻辑**: `x_real ≈ x_radar + vx · dt`
- **速度一致性逻辑**: 如果一个点 `dt=0.5s`，且位置偏移量 `ΔP` 很大，网络会计算 `ΔP/dt`，从而反推出物体的真实速度

### 数据流
```
原始雷达点云 [N, 18维]
    ↓ LoadRadarPointsFromFile (use_dim=[0,1,2,5,8,9,16,17])
[N, 8维]: [x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms]
    ↓ LoadRadarPointsFromMultiSweeps (添加时间戳)
[N, 9维]: [x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms, dt]
    ↓ RadarGeometryEnhancer (添加几何特征)
[N, 11维]: [x, y, z, rcs, vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms, dt]
    ↓ 体素化 + PillarFeatureNet (in_channels=11)
[64维特征] → 语义融合路径
```

## 三步实施方案

### 第一步：确认数据加载器自动生成时间戳 ✓

**状态**: 需要修改 `LoadRadarPointsFromMultiSweeps`

**当前问题**:
- 当前实现没有添加时间戳维度
- 历史帧点云直接拼接，缺少时间信息

**修改方案**:
1. 在 `LoadRadarPointsFromMultiSweeps.transform()` 中添加时间戳计算
2. 当前帧点云: `dt = 0.0`
3. 历史帧点云: `dt = current_timestamp - sweep_timestamp`
4. 将 `dt` 作为最后一维拼接到点云特征中

### 第二步：修改 RadarGeometryEnhancer 保留时间戳 ✓

**状态**: 需要修改

**当前实现**:
- 输入: 8维 `[x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms]`
- 输出: 10维 `[x, y, z, rcs, vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms]`

**修改后**:
- 输入: 9维 `[x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms, dt]`
- 输出: 11维 `[x, y, z, rcs, vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms, dt]`

**关键代码**:
```python
# 提取时间戳 (最后一维，索引 8)
time_feats = points.tensor[:, 8:9]

# 拼接所有特征
enhanced_tensor = torch.cat([
    base_feats,   # 0-5: x, y, z, rcs, vx, vy
    geom_feats,   # 6-7: sin_theta, cos_theta
    rms_feats,    # 8-9: vx_rms, vy_rms
    time_feats    # 10: dt
], dim=1)
```

### 第三步：修改 Config 适配新维度 ✓

**状态**: 需要修改配置文件

**修改位置**: `bevfusion_lidar-cam-radar_geometry_enhanced.py`

**修改内容**:
```python
radar_voxel_encoder=dict(
    type='PillarFeatureNet',
    in_channels=11,  # 从 10 改为 11
    feat_channels=[64],
    # ...
),
```

## 网络如何利用时间戳

### 1. 位置校正
网络通过 MLP 学习:
- 当 `dt` 很大且 `vx` 很大时，特征应该向 `vx` 方向"偏移"
- 从而把历史点对齐到当前位置

### 2. 速度一致性
- 如果一个点 `dt=0.5s`，且位置偏移量 `ΔP` 很大
- 网络计算 `ΔP/dt`，反推物体的真实速度
- 比单纯依赖雷达测量的 `vr` 更鲁棒

### 3. 置信度加权
- 时间戳越大，位置不确定性越高
- 网络可以学习降低远历史帧的权重

## 实施检查清单

- [x] 修改 `LoadRadarPointsFromMultiSweeps` 添加时间戳
- [x] 修改 `RadarGeometryEnhancer` 保留时间戳
- [x] 修改配置文件 `in_channels=11`
- [ ] 测试数据加载管道
- [ ] 验证维度正确性
- [ ] 训练并观察效果

## 预期效果

添加时间戳后，之前担心的"1-2米位置误差"将不再是误差，而是网络判断物体速度的最强证据。

## 参考

- nuScenes 数据集时间戳精度: 微秒级
- 典型 sweep 间隔: 0.05s - 0.5s
- 速度范围: [-50, 50] m/s
- 位置偏移范围: `v * dt` ≈ [-25, 25] m (最大情况)

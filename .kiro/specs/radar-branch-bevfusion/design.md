# Design Document: BEVFusion Radar Branch

## Overview

本设计文档描述了在BEVFusion框架中添加毫米波雷达分支的技术方案。该方案遵循BEVFusion的设计理念，将雷达点云数据转换到统一的BEV空间，与图像和激光雷达特征进行融合。

### 设计目标

1. **模块化设计**: 雷达分支作为独立模块，可灵活启用/禁用
2. **代码复用**: 最大程度复用mmdetection3d现有模块
3. **兼容性**: 完全兼容nuScenes数据集格式
4. **一致性**: 与现有图像和激光雷达分支保持架构一致

### 数据流概览

```
雷达点云 -> 数据加载 -> 预处理/增强 -> 体素化 -> Pillar编码 -> BEV特征
                                                              ↓
图像 -> 骨干网络 -> Neck -> 视角变换 -> BEV特征 ─────────────→ 融合层 -> 检测头
                                                              ↑
激光雷达 -> 体素化 -> 稀疏卷积 -> BEV特征 ─────────────────────┘
```

## Architecture

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BEVFusion Model                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Image Branch  │  │  LiDAR Branch   │  │  Radar Branch   │         │
│  │                 │  │                 │  │   (NEW)         │         │
│  │  img_backbone   │  │ pts_voxel_layer │  │ radar_voxel_layer│        │
│  │  img_neck       │  │ pts_voxel_enc   │  │ radar_pillar_enc │        │
│  │  view_transform │  │ pts_middle_enc  │  │ radar_scatter    │        │
│  │       ↓         │  │       ↓         │  │       ↓         │         │
│  │  img_bev_feat   │  │  pts_bev_feat   │  │ radar_bev_feat  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ↓                                        │
│                    ┌───────────────────────┐                           │
│                    │     Fusion Layer      │                           │
│                    │  (ConvFuser Extended) │                           │
│                    └───────────┬───────────┘                           │
│                                ↓                                        │
│                    ┌───────────────────────┐                           │
│                    │    pts_backbone       │                           │
│                    │    pts_neck           │                           │
│                    │    bbox_head          │                           │
│                    └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 雷达分支详细架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Radar Branch                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  radar_points [N, 6]                                            │
│  (x, y, z, rcs, vx_comp, vy_comp)                               │
│         │                                                        │
│         ↓                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Radar Voxelization              │                    │
│  │  (reuse: Voxelization from mmdet3d)     │                    │
│  │  voxel_size: [0.5, 0.5, 8.0]           │                    │
│  │  point_cloud_range: [-54, -54, -5,      │                    │
│  │                       54, 54, 3]        │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ↓                                                        │
│  voxels [M, max_points, 6], coords [M, 3], num_points [M]       │
│         │                                                        │
│         ↓                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │      Radar Pillar Feature Net           │                    │
│  │  (reuse: PillarFeatureNet from mmdet3d) │                    │
│  │  in_channels: 6                         │                    │
│  │  feat_channels: [64]                    │                    │
│  │  with_distance: False                   │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ↓                                                        │
│  pillar_features [M, 64]                                        │
│         │                                                        │
│         ↓                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Point Pillars Scatter           │                    │
│  │  (reuse: PointPillarsScatter)           │                    │
│  │  output_shape: [216, 216]               │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ↓                                                        │
│  radar_bev_feat [B, 64, 216, 216]                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. LoadRadarPointsFromFile (数据加载模块)

```python
@TRANSFORMS.register_module()
class LoadRadarPointsFromFile(BaseTransform):
    """
    从文件加载雷达点云数据
    
    nuScenes雷达点云格式 (18维):
    - x, y, z: 3D坐标
    - dyn_prop: 动态属性
    - id: 点ID
    - rcs: 雷达散射截面
    - vx, vy: 速度分量
    - vx_comp, vy_comp: 补偿后的速度分量
    - is_quality_valid: 质量标志
    - ambig_state: 模糊状态
    - x_rms, y_rms: 位置误差
    - invalid_state: 无效状态
    - pdh0: 检测概率
    - vx_rms, vy_rms: 速度误差
    
    Args:
        coord_type (str): 坐标类型，默认'LIDAR'
        load_dim (int): 加载的特征维度，默认18
        use_dim (list[int]): 使用的特征维度索引，默认[0,1,2,5,8,9]
            对应 x, y, z, rcs, vx_comp, vy_comp
        backend_args (dict): 文件后端参数
    """
    
    def __init__(
        self,
        coord_type: str = 'LIDAR',
        load_dim: int = 18,
        use_dim: List[int] = [0, 1, 2, 5, 8, 9],
        backend_args: dict = None
    ):
        pass
    
    def transform(self, results: dict) -> dict:
        """
        加载雷达点云并添加到results字典
        
        Args:
            results: 包含radar_path的数据字典
            
        Returns:
            results: 添加了radar_points的数据字典
        """
        pass
```

### 2. LoadRadarPointsFromMultiSweeps (多帧雷达加载)

```python
@TRANSFORMS.register_module()
class LoadRadarPointsFromMultiSweeps(BaseTransform):
    """
    加载多帧雷达点云并合并
    
    将历史帧的雷达点云变换到当前帧坐标系后合并，
    增加雷达点云的密度。
    
    Args:
        sweeps_num (int): 加载的历史帧数量，默认5
        load_dim (int): 加载的特征维度
        use_dim (list[int]): 使用的特征维度
        pad_empty_sweeps (bool): 是否用当前帧填充空sweep
        remove_close (bool): 是否移除过近的点
        close_radius (float): 过近点的半径阈值
    """
    
    def transform(self, results: dict) -> dict:
        """
        加载多帧雷达点云
        
        处理流程:
        1. 获取当前帧雷达点云
        2. 遍历历史sweep
        3. 将历史帧点云变换到当前帧坐标系
        4. 合并所有点云
        """
        pass
```

### 3. RadarPointsRangeFilter (雷达点云范围过滤)

```python
@TRANSFORMS.register_module()
class RadarPointsRangeFilter(BaseTransform):
    """
    过滤超出范围的雷达点云
    
    Args:
        point_cloud_range (list): 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    
    def transform(self, results: dict) -> dict:
        """
        过滤雷达点云
        
        保留在point_cloud_range内的点
        """
        pass
```

### 4. BEVFusionWithRadar (扩展的BEVFusion模型)

```python
@MODELS.register_module()
class BEVFusionWithRadar(BEVFusion):
    """
    支持雷达分支的BEVFusion模型
    
    在原有BEVFusion基础上添加雷达处理分支:
    - radar_voxel_layer: 雷达点云体素化
    - radar_voxel_encoder: 雷达体素特征编码 (PillarFeatureNet)
    - radar_middle_encoder: 雷达中间编码器 (PointPillarsScatter)
    
    Args:
        radar_voxel_encoder (dict): 雷达体素编码器配置
        radar_middle_encoder (dict): 雷达中间编码器配置
        其他参数同BEVFusion
    """
    
    def __init__(
        self,
        radar_voxel_encoder: Optional[dict] = None,
        radar_middle_encoder: Optional[dict] = None,
        **kwargs
    ):
        pass
    
    def extract_radar_feat(self, batch_inputs_dict) -> torch.Tensor:
        """
        提取雷达BEV特征
        
        处理流程:
        1. 获取雷达点云
        2. 体素化
        3. Pillar特征编码
        4. Scatter到BEV空间
        
        Args:
            batch_inputs_dict: 包含radar_points的输入字典
            
        Returns:
            radar_bev_feat: 雷达BEV特征 [B, C, H, W]
        """
        pass
    
    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        """
        提取并融合多模态特征 (重写父类方法)
        
        在原有图像和激光雷达特征基础上，添加雷达特征
        """
        pass
```

### 5. 配置文件结构

```python
# bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

model = dict(
    type='BEVFusionWithRadar',
    
    # 雷达体素化配置 (在data_preprocessor中)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            # LiDAR体素化配置
            ...
        ),
        radar_voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.5, 0.5, 8.0],
            max_voxels=(30000, 40000),
        ),
    ),
    
    # 雷达编码器配置
    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=6,  # x, y, z, rcs, vx_comp, vy_comp
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.5, 0.5, 8.0],
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    ),
    
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[216, 216],  # 与LiDAR BEV尺寸一致
    ),
    
    # 更新融合层配置
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256, 64],  # [img, lidar, radar]
        out_channels=256,
    ),
)

# 数据管道配置
train_pipeline = [
    # ... 图像加载 ...
    # ... LiDAR加载 ...
    
    # 雷达数据加载
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9],
    ),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 5, 8, 9],
    ),
    
    # ... 数据增强 ...
    
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    ),
    
    # Pack时包含radar_points
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'radar_points', 'gt_bboxes_3d', 'gt_labels_3d'],
    ),
]

# 数据集模态配置
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=True,  # 新增
)
```

## Data Models

### 雷达点云数据结构

```python
# nuScenes雷达点云原始格式 (18维)
radar_points_raw = np.ndarray  # shape: [N, 18]
# 维度说明:
# 0: x - X坐标 (m)
# 1: y - Y坐标 (m)
# 2: z - Z坐标 (m)
# 3: dyn_prop - 动态属性
# 4: id - 点ID
# 5: rcs - 雷达散射截面 (dBsm)
# 6: vx - X方向速度 (m/s)
# 7: vy - Y方向速度 (m/s)
# 8: vx_comp - 补偿后X方向速度 (m/s)
# 9: vy_comp - 补偿后Y方向速度 (m/s)
# 10: is_quality_valid - 质量有效标志
# 11: ambig_state - 模糊状态
# 12: x_rms - X位置误差
# 13: y_rms - Y位置误差
# 14: invalid_state - 无效状态
# 15: pdh0 - 检测概率
# 16: vx_rms - X速度误差
# 17: vy_rms - Y速度误差

# 处理后的雷达点云 (6维)
radar_points = np.ndarray  # shape: [N, 6]
# 维度说明:
# 0: x - X坐标
# 1: y - Y坐标
# 2: z - Z坐标
# 3: rcs - 雷达散射截面
# 4: vx_comp - 补偿后X方向速度
# 5: vy_comp - 补偿后Y方向速度
```

### 体素化数据结构

```python
# 体素化输出
voxels = torch.Tensor  # shape: [M, max_points, 6]
coords = torch.Tensor  # shape: [M, 3], (z_idx, y_idx, x_idx)
num_points = torch.Tensor  # shape: [M]

# Pillar特征
pillar_features = torch.Tensor  # shape: [M, 64]

# BEV特征
radar_bev_feat = torch.Tensor  # shape: [B, 64, H, W]
```

### 融合特征数据结构

```python
# 各模态BEV特征
img_bev_feat = torch.Tensor  # shape: [B, 80, H, W]
pts_bev_feat = torch.Tensor  # shape: [B, 256, H, W]
radar_bev_feat = torch.Tensor  # shape: [B, 64, H, W]

# 融合后特征
fused_bev_feat = torch.Tensor  # shape: [B, 256, H, W]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Radar Data Loading Completeness

*For any* valid nuScenes sample with radar data, loading radar points from all 5 sensors SHALL produce a non-empty point cloud with correct feature dimensions matching the configured use_dim.

**Validates: Requirements 1.1, 1.3**

### Property 2: Coordinate Transformation Consistency

*For any* radar point, transforming from sensor coordinate to LiDAR coordinate and back to sensor coordinate SHALL produce coordinates within numerical tolerance of the original.

**Validates: Requirements 1.2**

### Property 3: Multi-Sweep Temporal Aggregation

*For any* valid nuScenes sample with historical sweeps, loading N sweeps SHALL produce a point cloud with point count greater than or equal to the single-frame point count.

**Validates: Requirements 1.4**

### Property 4: Range Filtering Correctness

*For any* radar point cloud and point cloud range configuration, all points in the filtered output SHALL have coordinates within the specified range bounds.

**Validates: Requirements 2.1, 2.4**

### Property 5: Augmentation Consistency

*For any* data augmentation applied to LiDAR points, the same transformation matrix SHALL be applied to radar points, maintaining geometric consistency between modalities.

**Validates: Requirements 2.2, 2.3**

### Property 6: Radar Encoder Output Shape

*For any* valid radar point cloud input, the Radar_Encoder SHALL output BEV features with shape [B, C, H, W] where C matches the configured output channels and H, W match the LiDAR BEV spatial dimensions.

**Validates: Requirements 3.1, 3.3**

### Property 7: Empty Input Handling

*For any* empty radar point cloud input, the Radar_Encoder SHALL return zero-filled BEV features with the correct output shape.

**Validates: Requirements 3.4**

### Property 8: Fusion Layer Channel Consistency

*For any* list of BEV features from N modalities, the Fusion_Layer output channel count SHALL equal the configured out_channels, regardless of the number of input modalities.

**Validates: Requirements 4.2, 4.3, 4.5**

### Property 9: Modality Flexibility

*For any* subset of modalities (camera, LiDAR, radar), the Fusion_Layer SHALL produce valid output when given the corresponding BEV features.

**Validates: Requirements 4.1, 4.4**

### Property 10: Configuration Backward Compatibility

*For any* BEVFusion configuration without radar branch, the BEVFusionWithRadar model SHALL produce identical results to the original BEVFusion model.

**Validates: Requirements 5.2, 5.3**

## Error Handling

### 数据加载错误

| 错误类型 | 处理策略 |
|---------|---------|
| 雷达文件不存在 | 返回空点云，记录警告日志 |
| 雷达文件损坏 | 返回空点云，记录错误日志 |
| 标定矩阵缺失 | 跳过该传感器，记录警告 |
| Sweep数据不足 | 用当前帧填充或减少sweep数量 |

### 特征编码错误

| 错误类型 | 处理策略 |
|---------|---------|
| 空点云输入 | 返回零填充的BEV特征 |
| 体素数量超限 | 截断到max_voxels |
| 内存不足 | 减少batch_size或voxel数量 |

### 融合错误

| 错误类型 | 处理策略 |
|---------|---------|
| 特征尺寸不匹配 | 使用插值对齐到目标尺寸 |
| 模态缺失 | 跳过该模态，使用可用模态融合 |

## Testing Strategy

### 单元测试

1. **数据加载测试**
   - 测试单帧雷达加载
   - 测试多帧sweep加载
   - 测试坐标变换正确性
   - 测试错误处理

2. **预处理测试**
   - 测试范围过滤
   - 测试数据增强一致性

3. **编码器测试**
   - 测试体素化输出形状
   - 测试Pillar编码输出
   - 测试空输入处理

4. **融合层测试**
   - 测试多模态输入
   - 测试单模态输入
   - 测试输出形状

### Property-Based Tests

使用hypothesis库进行属性测试：

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=6, max_size=6))
def test_range_filter_property(point):
    """Property 4: 范围过滤正确性"""
    # 生成随机点，验证过滤后的点都在范围内
    pass

@given(st.integers(min_value=0, max_value=1000))
def test_encoder_output_shape_property(num_points):
    """Property 6: 编码器输出形状"""
    # 生成随机数量的点，验证输出形状正确
    pass
```

### 集成测试

1. **端到端推理测试**
   - 使用nuScenes mini数据集
   - 验证三模态融合推理正确性

2. **训练测试**
   - 验证梯度传播
   - 验证损失收敛

### 测试配置

- Property-based tests: 最少100次迭代
- 使用pytest + hypothesis框架
- 测试覆盖率目标: >80%


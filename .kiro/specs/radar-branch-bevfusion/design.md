# 设计文档：几何感知型多路径雷达融合架构

## 概述

本设计文档描述了在BEVFusion框架中重新设计毫米波雷达分支的技术方案。该方案基于**几何感知型多路径雷达融合架构**，核心理念是：

1. **几何感知 (Geometry-Awareness)**：通过引入方位角特征（sin θ, cos θ），使模型能够理解径向速度的物理置信度
2. **任务解耦 (Task Decoupling)**：
   - **语义特征路径**负责"物体在哪里"（分类与定位）
   - **物理速度路径**负责"物体开多快"（运动估计）

### 设计目标

1. **几何增强**：将6维雷达点云扩展为10维，引入方位角和速度不确定度特征
2. **双路径解耦**：语义路径用于融合，速度路径直连检测头
3. **代码复用**：最大程度复用mmdetection3d现有模块
4. **向后兼容**：不配置雷达分支时与原BEVFusion行为一致

### 数据流概览

```
[原始雷达18维] --> [几何增强: 引入 sinθ/cosθ/RMS] --> [10维增强点云]
                                                          |
                            +-----------------------------+-----------------------------+
                            |                                                           |
                            v                                                           v
                  [路径A：语义特征路径]                                      [路径B：物理速度路径]
                  (体素化 -> PillarNet -> Scatter)                        (MLP -> VelocityBEVEncoder)
                            |                                                           |
                            v                                                           |
[Lidar/Image] --> [多模态语义融合 ConvFuser]                                             |
                            |                                                           |
                            v                                                           v
                  [检测头 TransFusionHead] <---(速度查询与几何校验)--- [物理速度图4ch]
                            |
                            v
                  [最终结果：位置、类别、高精度速度、航向角]
```

## 架构设计

### 整体架构

雷达分支采用双路径设计：
- **路径A（语义特征路径）**：体素化 -> PillarFeatureNet -> PointPillarsScatter -> 64通道BEV特征
- **路径B（物理速度路径）**：MLP融合 -> BEV投影 -> 4通道速度图

两条路径并行处理，语义特征参与多模态融合，速度图直接供检测头查询。

### 雷达分支详细设计

#### 几何增强层

输入：原始雷达点云 [N, 18]（nuScenes格式）
输出：增强点云 [N, 10]

增强过程：
1. 提取核心特征：x, y, z, rcs, vx_comp, vy_comp
2. 计算方位角：norm = sqrt(x² + y²), sin_θ = y/norm, cos_θ = x/norm
3. 提取速度不确定度：vx_rms, vy_rms
4. 组合输出：[x, y, z, rcs, vx_comp, vy_comp, sin_θ, cos_θ, vx_rms, vy_rms]

#### 语义特征路径

- 体素化：voxel_size = [0.6, 0.6, 8.0]
- PillarFeatureNet：in_channels=10, feat_channels=[64]
- PointPillarsScatter：output_shape=[180, 180]
- 输出：radar_bev_feat [B, 64, H, W]

#### 物理速度路径

- MLP：6 -> 32 -> 16 -> 4
- 输入：[vx_comp, vy_comp, sin_θ, cos_θ, vx_rms, vy_rms]
- BEV投影：scatter_add聚合
- 输出：velocity_bev [B, 4, H, W]（vx, vy, rms, confidence）

## 组件与接口

### 1. RadarGeometryEnhancer

```python
@TRANSFORMS.register_module()
class RadarGeometryEnhancer(BaseTransform):
    """
    雷达点云几何增强预处理
    
    Args:
        eps (float): 避免除零的小量，默认1e-6
    """
    def transform(self, results: dict) -> dict:
        # 计算方位角特征并扩展点云维度
        pass
```

### 2. GeometryAwareVelocityEncoder

```python
@MODELS.register_module()
class GeometryAwareVelocityEncoder(nn.Module):
    """
    几何感知的雷达速度BEV编码器
    
    Args:
        point_cloud_range (list): 点云范围
        bev_size (tuple): BEV网格大小
        hidden_channels (list): MLP隐藏层通道数
    """
    def forward(self, radar_points, batch_size) -> torch.Tensor:
        # 返回 velocity_bev [B, 4, H, W]
        pass
```

### 3. VelocityRefinementModule

```python
@MODELS.register_module()
class VelocityRefinementModule(nn.Module):
    """
    速度校准模块
    
    Args:
        hidden_channel (int): 隐藏层通道数
    """
    def forward(self, pred_velocity, velocity_bev, query_pos) -> torch.Tensor:
        # 返回校准后的速度
        pass
```

## 数据模型

### 雷达点云格式

原始格式（18维）：
- 0-2: x, y, z
- 5: rcs
- 8-9: vx_comp, vy_comp
- 16-17: vx_rms, vy_rms

增强格式（10维）：
- 0-2: x, y, z
- 3: rcs
- 4-5: vx_comp, vy_comp
- 6-7: sin_theta, cos_theta
- 8-9: vx_rms, vy_rms

### 输出特征

- 语义BEV特征：[B, 64, 180, 180]
- 速度BEV图：[B, 4, 180, 180]
- 融合BEV特征：[B, 256, 180, 180]

## 正确性属性

*正确性属性是系统在所有有效执行中都应保持为真的特征或行为。*

### Property 1: 几何增强输出正确性

*For any* 雷达点云输入，几何增强后的输出应当满足：输出维度为10，且sin²θ + cos²θ = 1（在数值误差范围内），当norm接近零时sin θ = cos θ = 0。

**Validates: Requirements 1.2, 1.4, 1.5**

### Property 2: 语义路径输出形状

*For any* 有效的雷达点云输入（包括空输入），语义特征路径应当输出形状为[B, 64, H, W]的BEV特征。

**Validates: Requirements 2.3, 2.4, 2.5**

### Property 3: 速度路径输出形状

*For any* 有效的雷达点云输入（包括空输入），物理速度路径应当输出形状为[B, 4, H, W]的速度图。

**Validates: Requirements 3.3, 3.5**

### Property 4: 速度值保留

*For any* 非空雷达点云，对于BEV网格中只有一个雷达点的位置，输出的vx, vy应当等于输入的vx_comp, vy_comp。

**Validates: Requirements 3.4**

### Property 5: 融合层输出形状

*For any* 有效的多模态BEV特征输入，融合层应当输出形状为[B, 256, H, W]的融合特征。

**Validates: Requirements 4.2, 4.3**

### Property 6: 模态灵活性

*For any* 模态子集，融合层应当能够正确处理并产生有效输出。

**Validates: Requirements 4.4**

### Property 7: 速度采样正确性

*For any* 预测的物体中心点位置，从物理速度图中采样的速度值应当与该位置的雷达观测值一致。

**Validates: Requirements 5.1**

### Property 8: 速度校准回退

*For any* 物理速度图中置信度为零的位置，速度校准模块的输出应当等于网络预测的速度值。

**Validates: Requirements 5.4**

### Property 9: 范围过滤正确性

*For any* 雷达点云和点云范围配置，过滤后的所有点都应当在指定范围内。

**Validates: Requirements 6.4**

### Property 10: 增强一致性

*For any* 数据增强操作，应用于LiDAR点云的变换矩阵应当与应用于雷达点云的变换矩阵相同。

**Validates: Requirements 6.5**

### Property 11: 后向兼容性

*For any* 不配置雷达分支的BEVFusionWithRadar模型，其行为应当与原BEVFusion模型完全一致。

**Validates: Requirements 7.3**

## 错误处理

| 错误类型 | 处理策略 |
|---------|---------|
| 雷达文件不存在 | 返回空点云，记录警告日志 |
| norm接近零 | 将sin θ和cos θ设置为0 |
| 空点云输入 | 返回零填充的BEV特征 |
| 特征尺寸不匹配 | 使用插值对齐到目标尺寸 |

## 测试策略

### 单元测试

1. 几何增强测试：输出维度、三角函数恒等式、边界条件
2. 语义路径测试：体素化输出、Pillar编码、空输入处理
3. 速度路径测试：MLP输出、BEV投影、速度值保留
4. 融合层测试：多模态输入、输出形状
5. 速度校准测试：采样正确性、回退逻辑

### Property-Based Tests

使用hypothesis库进行属性测试，最少100次迭代。

### 集成测试

使用nuScenes mini数据集验证端到端推理和训练。

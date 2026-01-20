# 设计文档：LiDAR深度监督与Camera-Aware深度估计

## 概述

本设计文档描述了在BEVFusion框架中增强图像分支的技术方案，包括两个核心特性：

1. **LiDAR深度监督 (Depth Supervision)**：利用LiDAR点云投影生成稀疏深度GT，通过损失函数显式监督深度估计网络
2. **Camera-Aware深度估计**：将相机内参、外参和数据增强矩阵编码为特征，使深度网络能够感知不同相机的几何差异

### 设计目标

1. **显式深度监督**：通过BCE/Focal Loss监督深度分布预测，提高深度估计准确性
2. **相机感知**：让网络理解"这是前视还是后视"以及"图像是否被缩放/裁剪过"
3. **数据管道优化**：将深度GT生成放在CPU数据加载阶段，不阻塞GPU训练
4. **向后兼容**：不配置增强功能时与原BEVFusion行为一致

### 数据流概览

```
[训练数据管道 - CPU阶段]
LiDAR点云 --> LoadDepthFromPoints --> 稀疏深度GT (depth_gt, valid_mask)
                    |
                    +-- 投影到图像平面
                    +-- 下采样到特征图尺度 (stride=16)
                    +-- Min Pooling (多点取最近)
                    +-- 形态学膨胀 (增加覆盖)
                    +-- 深度离散化 (Uniform/SID)

[模型前向传播 - GPU阶段]
图像特征 [B*N, C, fH, fW]
    |
    +-- Camera-Aware编码 --> SE调制
    |       |
    |       +-- 内参 (fx, fy, cx, cy)
    |       +-- 外参 (R, t)
    |       +-- IDA矩阵
    |
    v
调制后特征 --> DepthNet --> 深度分布 [B*N, D, fH, fW]
                              |
                              +-- 与深度GT计算损失 (BCE/Focal)
                              +-- 与语义特征外积 --> BEV池化
```

## 架构设计

### 整体架构

增强的图像分支采用模块化设计：
- **LoadDepthFromPoints**：数据管道类，在CPU阶段生成深度GT
- **CameraAwareDepthNet**：Camera-Aware深度估计网络
- **DepthSupervisionLoss**：深度监督损失模块
- **CameraAwareDepthLSSTransform**：集成所有功能的视角变换模块

### 模块详细设计

#### 1. LoadDepthFromPoints (数据管道)

```python
@TRANSFORMS.register_module()
class LoadDepthFromPoints(BaseTransform):
    """
    从LiDAR点云生成深度监督GT
    
    在CPU数据加载阶段完成计算，不阻塞GPU训练。
    
    Args:
        feature_size (tuple): 特征图尺寸 (fH, fW)
        dbound (tuple): 深度范围 (min, max, interval)
        discretization (str): 深度分桶策略 'uniform' 或 'sid'
        dilation_kernel (int): 形态学膨胀核大小，默认3
    """
```

处理流程：
1. 获取LiDAR点云和相机参数
2. 将点云投影到各相机图像平面
3. 应用图像数据增强变换
4. 下采样到特征图尺度（stride=16）
5. Min Pooling处理重叠点
6. 形态学膨胀增加覆盖
7. 深度离散化为bin索引
8. 生成valid_mask

#### 2. CameraAwareDepthNet (深度网络)

```python
@MODELS.register_module()
class CameraAwareDepthNet(nn.Module):
    """
    Camera-Aware深度估计网络
    
    将相机参数编码为特征，通过SE机制调制图像特征。
    
    Args:
        in_channels (int): 输入特征通道数
        mid_channels (int): 中间层通道数
        depth_channels (int): 深度bin数量 D
        context_channels (int): 语义特征通道数 C
        embed_dim (int): 相机参数编码维度，默认256
    """
```

相机参数编码：
- 内参：fx, fy, cx, cy (4维)
- 外参：旋转矩阵展平 (9维) + 平移向量 (3维) = 12维
- IDA矩阵：展平 (9维)
- 总计：25维 → MLP → embed_dim维

SE调制机制：
```
cam_embed = MLP(cam_params)  # [B*N, embed_dim]
scale = Sigmoid(FC(cam_embed))  # [B*N, C, 1, 1]
x = x * scale  # 通道级调制
```

#### 3. DepthSupervisionLoss (损失模块)

```python
@MODELS.register_module()
class DepthSupervisionLoss(nn.Module):
    """
    深度监督损失模块
    
    支持BCE Loss和Focal Loss。
    
    Args:
        loss_type (str): 'bce' 或 'focal'
        loss_weight (float): 损失权重
        focal_gamma (float): Focal Loss的gamma参数，默认2.0
    """
```

损失计算：
```python
def forward(self, depth_pred, depth_gt_indices, valid_mask):
    """
    Args:
        depth_pred: 预测的深度分布 [B*N, D, fH, fW]
        depth_gt_indices: 深度GT索引 [B*N, fH, fW]，无效位置为-1
        valid_mask: 有效掩码 [B*N, fH, fW]
    """
    if valid_mask.sum() == 0:
        return depth_pred.sum() * 0  # 返回零损失
    
    # 使用CrossEntropyLoss，ignore_index=-1自动忽略无效位置
    if self.loss_type == 'ce':
        loss = F.cross_entropy(
            depth_pred, depth_gt_indices, 
            ignore_index=-1, reduction='mean'
        )
    else:  # focal loss
        # 在GPU上按需转换为概率分布计算Focal Loss
        loss = self.focal_loss(depth_pred, depth_gt_indices, valid_mask)
    
    return loss * self.loss_weight
```

#### 4. CameraAwareDepthLSSTransform (视角变换)

```python
@MODELS.register_module()
class CameraAwareDepthLSSTransform(DepthLSSTransform):
    """
    集成深度监督和Camera-Aware的视角变换模块
    
    Args:
        use_depth_supervision (bool): 是否启用深度监督
        use_camera_aware (bool): 是否启用Camera-Aware
        depth_loss_cfg (dict): 深度损失配置
        camera_aware_cfg (dict): Camera-Aware配置
        其他参数同DepthLSSTransform
    """
```

## 组件与接口

### 1. LoadDepthFromPoints

```python
@TRANSFORMS.register_module()
class LoadDepthFromPoints(BaseTransform):
    """
    从LiDAR点云生成深度监督GT的数据管道
    
    Required Keys:
    - points (BasePoints): LiDAR点云
    - lidar2img (np.ndarray): LiDAR到图像的投影矩阵
    - img_aug_matrix (np.ndarray): 图像数据增强矩阵
    
    Added Keys:
    - depth_gt_indices (np.ndarray): 深度GT索引 [N, fH, fW]，无效位置为-1
    - depth_valid_mask (np.ndarray): 有效掩码 [N, fH, fW]
    """
    def __init__(
        self,
        feature_size: Tuple[int, int],
        dbound: Tuple[float, float, float],
        discretization: str = 'uniform',
        dilation_kernel: int = 3,
    ) -> None:
        pass
    
    def transform(self, results: dict) -> dict:
        pass
```

### 2. CameraAwareDepthNet

```python
@MODELS.register_module()
class CameraAwareDepthNet(nn.Module):
    """
    Camera-Aware深度估计网络
    
    Args:
        in_channels (int): 输入特征通道数
        mid_channels (int): 中间层通道数
        depth_channels (int): 深度bin数量
        context_channels (int): 语义特征通道数
        embed_dim (int): 相机参数编码维度
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        depth_channels: int,
        context_channels: int,
        embed_dim: int = 256,
    ) -> None:
        pass
    
    def forward(
        self,
        x: Tensor,
        intrinsics: Tensor,
        cam2ego: Tensor,
        ida_matrix: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: 图像特征 [B*N, C, fH, fW]
            intrinsics: 增强后相机内参 [B*N, 3, 3]，必须是Post-Augmentation后的
            cam2ego: Camera-to-Ego外参 [B*N, 4, 4]
            ida_matrix: IDA矩阵 [B*N, 4, 4]
        
        Returns:
            depth: 深度分布 [B*N, D, fH, fW]
            context: 语义特征 [B*N, C, fH, fW]
        """
        pass
```

### 3. DepthSupervisionLoss

```python
@MODELS.register_module()
class DepthSupervisionLoss(nn.Module):
    """
    深度监督损失模块
    
    支持两种损失类型：
    - 'ce': CrossEntropyLoss，直接使用索引格式的GT
    - 'focal': Focal Loss，处理类别不平衡
    """
    def __init__(
        self,
        loss_type: str = 'ce',  # 'ce' 或 'focal'
        loss_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        pass
    
    def forward(
        self,
        depth_pred: Tensor,  # [B*N, D, fH, fW]
        depth_gt_indices: Tensor,  # [B*N, fH, fW]，索引格式
        valid_mask: Tensor,  # [B*N, fH, fW]
    ) -> Tensor:
        """
        使用索引格式的GT计算损失，避免One-Hot转换的显存开销
        """
        pass
```

### 4. CameraAwareDepthLSSTransform

```python
@MODELS.register_module()
class CameraAwareDepthLSSTransform(DepthLSSTransform):
    """
    集成深度监督和Camera-Aware的视角变换模块
    """
    def __init__(
        self,
        use_depth_supervision: bool = False,
        use_camera_aware: bool = False,
        depth_loss_cfg: Optional[dict] = None,
        camera_aware_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        pass
    
    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        depth_gt_indices=None,
        depth_valid_mask=None,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        """
        Returns:
            训练时: (bev_feat, {'depth_pred': ..., 'depth_gt_indices': ..., 'valid_mask': ...})
            推理时: bev_feat
        """
        pass
```

## 数据模型

### 深度GT格式

为了优化CPU到GPU的数据传输效率，深度GT使用索引格式而非One-Hot编码：

```python
# 稀疏深度图（连续值，中间结果）
sparse_depth: [N, 1, fH, fW]  # N=相机数

# 深度GT（离散化后，索引格式）
# 使用索引而非One-Hot，显存占用减少约D倍（~100倍）
depth_gt_indices: [N, fH, fW]  # int64，值为0~D-1的bin索引，无效位置为-1

# 有效掩码
depth_valid_mask: [N, fH, fW]  # bool，True表示有深度值
```

**设计理由**：
- One-Hot格式 [N, D, fH, fW] 在 D=118 时占用大量显存
- 索引格式 [N, fH, fW] 显存占用减少约100倍
- CrossEntropyLoss 可直接使用索引格式
- 如需BCEWithLogitsLoss，在GPU上按需转换为One-Hot

### 相机参数编码

**坐标系定义**：
- 外参使用 **Camera-to-Ego** 变换，而非 Camera-to-LiDAR
- 这能最大程度帮助网络理解"自车坐标系下的空间位置"

**内参定义**：
- 内参必须是**应用了数据增强（Post-Augmentation）后的内参**
- 即 Resize/Crop 后更新过的 (fx, fy, cx, cy)
- IDA矩阵仍然保留，因为它包含旋转/剪切等内参无法体现的信息

```python
# 增强后内参 (4维)
# 注意：必须是应用了Resize/Crop后的内参
intrinsic_params = [fx_aug, fy_aug, cx_aug, cy_aug]

# Camera-to-Ego外参 (12维)
# 使用Camera-to-Ego而非Camera-to-LiDAR
cam2ego_rot = cam2ego[:3, :3].flatten()  # 9维
cam2ego_trans = cam2ego[:3, 3]  # 3维
extrinsic_params = concat([cam2ego_rot, cam2ego_trans])

# IDA矩阵 (9维)
# 保留IDA矩阵，包含旋转/剪切等内参无法体现的信息
ida_params = ida_matrix[:3, :3].flatten()

# 总计 25维
cam_params = concat([intrinsic_params, extrinsic_params, ida_params])
```

## 正确性属性

*正确性属性是系统在所有有效执行中都应保持为真的特征或行为。*

### Property 1: 深度GT生成正确性

*For any* LiDAR点云和相机参数，生成的深度GT应当满足：
- 输出尺寸为 [N, D, fH, fW]，其中 fH, fW 为特征图尺寸
- 对于同一网格内的多个点，保留的深度值为最小值（Min Pooling）
- 形态学膨胀后，非零像素数量 >= 膨胀前的数量
- 所有深度值都在 dbound 范围内

**Validates: Requirements 1.1, 1.2, 1.3, 1.6**

### Property 2: 深度离散化正确性

*For any* 连续深度值 d 在 [dmin, dmax] 范围内，离散化后的bin索引应当满足：
- Uniform策略：bin_idx = floor((d - dmin) / interval)
- SID策略：bin_idx 随深度增加而增加，但间隔逐渐变大
- 离散化是可逆的：从bin索引可以恢复到近似的深度值

**Validates: Requirements 1.4, 6.4**

### Property 3: 数据增强一致性

*For any* 图像数据增强变换（Resize, Crop, Flip, Rotate），深度GT应当与增强后的图像保持几何一致：
- 翻转后，深度图也应当翻转
- 缩放后，深度图的有效像素位置应当相应缩放
- 裁剪后，深度图应当裁剪到相同区域

**Validates: Requirements 1.5**

### Property 4: 深度损失计算正确性

*For any* 深度预测和GT，损失计算应当满足：
- 只在 valid_mask=True 的位置计算损失
- 当 valid_mask 全为 False 时，返回零损失
- 损失值被 loss_weight 正确缩放
- BCE和Focal Loss的计算结果符合数学定义

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 5: Camera-Aware编码正确性

*For any* 相机参数（内参、外参、IDA），编码后的特征应当满足：
- 不同相机产生不同的编码（除非参数完全相同）
- 编码维度为配置的 embed_dim
- SE调制后的特征与输入特征形状相同

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 6: 后向兼容性

*For any* 禁用所有增强功能的 CameraAwareDepthLSSTransform，其输出应当与原 DepthLSSTransform 完全一致。

**Validates: Requirements 4.5**

### Property 7: 训练流程正确性

*For any* 启用深度监督的训练，损失字典应当包含 'loss_depth' 键，且梯度能够正确反向传播到深度网络。

**Validates: Requirements 5.1, 5.2, 5.4, 5.5**

## 错误处理

| 错误类型 | 处理策略 |
|---------|---------|
| LiDAR点云为空 | 返回全零的深度GT和全False的valid_mask |
| 所有点都超出深度范围 | 返回全零的深度GT和全False的valid_mask |
| 相机参数缺失 | 跳过该相机，记录警告日志 |
| valid_mask全为False | 深度损失返回零 |

## 测试策略

### 单元测试

1. **LoadDepthFromPoints测试**：
   - 投影正确性
   - Min Pooling正确性
   - 形态学膨胀正确性
   - 深度离散化正确性

2. **CameraAwareDepthNet测试**：
   - 编码维度正确性
   - SE调制正确性
   - 不同相机产生不同编码

3. **DepthSupervisionLoss测试**：
   - BCE Loss计算正确性
   - Focal Loss计算正确性
   - 掩码处理正确性
   - 空输入处理

4. **CameraAwareDepthLSSTransform测试**：
   - 后向兼容性
   - 深度预测返回
   - 配置开关正确性

### Property-Based Tests

使用hypothesis库进行属性测试，最少100次迭代。

### 集成测试

使用nuScenes mini数据集验证端到端训练和推理。


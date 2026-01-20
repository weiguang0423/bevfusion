# 实现计划：LiDAR深度监督与Camera-Aware深度估计

## 概述

本实现计划将LiDAR深度监督和Camera-Aware深度估计集成到BEVFusion框架中。实现遵循增量开发原则，每个任务都在前一个任务的基础上构建，确保代码始终处于可运行状态。

## 任务列表

- [x] 1. 实现深度GT生成数据管道
  - [x] 1.1 创建LoadDepthFromPoints类
    - 在`projects/BEVFusion/bevfusion/loading.py`中添加
    - 实现LiDAR点云到图像平面的投影
    - 实现下采样到特征图尺度（stride=16）
    - 实现Min Pooling处理重叠点
    - 实现形态学膨胀增加覆盖
    - 实现深度离散化（Uniform/SID策略）
    - 输出索引格式的depth_gt_indices和valid_mask
    - 添加中文注释
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.2 编写深度GT生成属性测试
    - **Property 1: 深度GT生成正确性**
    - **Property 2: 深度离散化正确性**
    - **Property 3: 数据增强一致性**
    - 测试Min Pooling、形态学膨胀、深度范围过滤
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**

- [x] 2. 实现深度监督损失模块
  - [x] 2.1 创建DepthSupervisionLoss类
    - 在`projects/BEVFusion/bevfusion/depth_lss.py`中添加
    - 实现CrossEntropyLoss（使用ignore_index=-1）
    - 实现Focal Loss
    - 支持可配置的损失权重
    - 处理空输入返回零损失
    - 添加中文注释
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 编写深度损失属性测试
    - **Property 4: 深度损失计算正确性**
    - 测试CE Loss、Focal Loss、掩码处理、空输入
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [x] 3. Checkpoint - 验证数据管道和损失模块
  - 确保所有测试通过
  - 验证深度GT生成和损失计算可以正确工作

- [x] 4. 实现Camera-Aware深度网络
  - [x] 4.1 创建CameraAwareDepthNet类
    - 在`projects/BEVFusion/bevfusion/depth_lss.py`中添加
    - 实现相机参数编码（内参+外参+IDA，共25维）
    - 实现MLP映射到embed_dim维
    - 实现SE机制调制图像特征
    - 支持可配置的编码维度
    - 添加中文注释
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 编写Camera-Aware属性测试
    - **Property 5: Camera-Aware编码正确性**
    - 测试编码维度、SE调制、不同相机产生不同编码
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 5. 实现增强的视角变换模块
  - [x] 5.1 创建CameraAwareDepthLSSTransform类
    - 在`projects/BEVFusion/bevfusion/depth_lss.py`中添加
    - 继承自DepthLSSTransform
    - 集成CameraAwareDepthNet（可选）
    - 训练时返回深度预测用于计算损失
    - 支持配置开关（use_depth_supervision, use_camera_aware）
    - 添加中文注释
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 5.2 编写后向兼容性测试
    - **Property 6: 后向兼容性**
    - 验证禁用所有功能时与原模块输出一致
    - **Validates: Requirements 4.5**


- [x] 6. Checkpoint - 验证Camera-Aware和视角变换模块
  - 确保所有测试通过
  - 验证Camera-Aware编码和视角变换可以正确工作

- [x] 7. 集成到BEVFusion训练流程
  - [x] 7.1 更新BEVFusion模型的loss方法
    - 修改`projects/BEVFusion/bevfusion/bevfusion.py`
    - 在extract_img_feat中获取深度预测
    - 在loss方法中计算深度监督损失
    - 将loss_depth添加到损失字典
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 7.2 编写训练流程属性测试
    - **Property 7: 训练流程正确性**
    - 验证损失字典包含loss_depth
    - 验证梯度正确反向传播
    - **Validates: Requirements 5.1, 5.2, 5.4, 5.5**

- [x] 8. 更新配置文件
  - [x] 8.1 创建深度监督配置文件
    - 创建`projects/BEVFusion/configs/bevfusion_depth_supervision.py`
    - 添加LoadDepthFromPoints到数据管道
    - 配置CameraAwareDepthLSSTransform
    - 配置深度损失参数
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 8.2 创建消融实验配置
    - 创建仅深度监督配置（无Camera-Aware）
    - 创建仅Camera-Aware配置（无深度监督）
    - 创建完整配置（两者都启用）
    - _Requirements: 7.1, 7.2_

- [x] 9. 更新模块注册
  - [x] 9.1 更新__init__.py
    - 导出LoadDepthFromPoints
    - 导出DepthSupervisionLoss
    - 导出CameraAwareDepthNet
    - 导出CameraAwareDepthLSSTransform
    - _Requirements: 8.4_

- [x] 10. 编写集成测试
  - [x] 10.1 编写端到端训练测试
    - 使用mini数据集验证训练流程
    - 验证损失下降
    - 验证梯度流动
    - **Validates: Requirements 5.1, 5.2**

  - [x] 10.2 编写推理测试
    - 验证推理时不需要深度GT
    - 验证推理输出与原模型兼容
    - **Validates: Requirements 4.5**

- [x] 11. Final Checkpoint - 完整功能验证
  - 确保所有测试通过
  - 验证配置文件正确
  - 验证模型可以正确构建和训练
  - 验证推理正常工作

## 注意事项

- 所有任务都是必需的，包括测试任务
- 每个任务都引用了具体的需求以便追溯
- Checkpoint任务用于增量验证
- 所有新增代码应包含中文注释以保持与现有代码风格一致
- Property测试验证通用正确性属性
- 单元测试验证具体示例和边界情况

## 关键实现细节

### 深度GT索引格式
- 使用索引格式 `[N, fH, fW]` 而非One-Hot `[N, D, fH, fW]`
- 无效位置使用 -1 标记
- CrossEntropyLoss使用 `ignore_index=-1` 自动忽略

### Camera-Aware参数编码
- 内参：增强后的 (fx, fy, cx, cy)，4维
- 外参：Camera-to-Ego的旋转(9维)+平移(3维)，12维
- IDA矩阵：3x3展平，9维
- 总计25维 → MLP → embed_dim维

### 坐标系定义
- 外参使用 Camera-to-Ego 变换
- 内参必须是 Post-Augmentation 后的值


# 实现计划：几何感知型多路径雷达融合架构

## 概述

本实现计划将几何感知型多路径雷达分支集成到BEVFusion框架中。实现遵循增量开发原则，每个任务都在前一个任务的基础上构建，确保代码始终处于可运行状态。

## 任务列表

- [x] 1. 实现几何增强预处理模块
  - [x] 1.1 创建RadarGeometryEnhancer类
    - 在`projects/BEVFusion/bevfusion/transforms_3d.py`中添加
    - 实现方位角计算：sin_theta = y/norm, cos_theta = x/norm
    - 处理norm接近零的边界情况
    - 添加中文注释
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 更新LoadRadarPointsFromFile支持扩展维度
    - 修改`projects/BEVFusion/bevfusion/loading.py`
    - 支持加载vx_rms, vy_rms（索引16, 17）
    - 更新use_dim默认值为[0, 1, 2, 5, 8, 9, 16, 17]
    - _Requirements: 1.1, 1.3_

  - [x] 1.3 编写几何增强属性测试
    - **Property 1: 几何增强输出正确性**
    - 测试输出维度、三角函数恒等式、边界条件
    - **Validates: Requirements 1.2, 1.4, 1.5**

- [x] 2. 实现几何感知速度编码器
  - [x] 2.1 创建GeometryAwareVelocityEncoder类
    - 在`projects/BEVFusion/bevfusion/radar_velocity_encoder.py`中添加
    - 实现MLP融合：6 -> 32 -> 16 -> 4
    - 实现BEV投影和聚合
    - 输出4通道速度图：[vx, vy, rms, confidence]
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 2.2 实现空输入处理
    - 当雷达点云为空时返回零填充速度图
    - _Requirements: 3.5_

  - [x] 2.3 编写速度路径属性测试
    - **Property 3: 速度路径输出形状**
    - **Property 4: 速度值保留**
    - **Validates: Requirements 3.3, 3.4, 3.5**
    - _CUDA测试通过 ✅_

- [x] 3. Checkpoint - 验证预处理和速度编码器
  - 确保所有测试通过
  - 验证几何增强和速度编码器可以正确工作
  - _所有测试通过 ✅_

- [x] 4. 更新语义特征路径
  - [x] 4.1 更新PillarFeatureNet配置
    - 修改in_channels从6改为10
    - 更新配置文件中的雷达编码器参数
    - _Requirements: 2.1, 2.2_

  - [x] 4.2 验证语义路径输出
    - 确保输出64通道BEV特征
    - 测试空输入处理
    - _Requirements: 2.3, 2.4, 2.5_
    - _验证通过 ✅_

  - [x] 4.3 编写语义路径属性测试
    - **Property 2: 语义路径输出形状**
    - **Validates: Requirements 2.3, 2.4, 2.5**
    - _测试通过 ✅_

- [x] 5. 实现速度校准模块
  - [x] 5.1 创建VelocityRefinementModule类
    - 在`projects/BEVFusion/bevfusion/radar_velocity_encoder.py`中添加
    - 实现从速度图采样
    - 实现基于置信度的加权融合
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 编写速度校准属性测试
    - **Property 7: 速度采样正确性**
    - **Property 8: 速度校准回退**
    - **Validates: Requirements 5.1, 5.4**
    - _CUDA测试通过 ✅_

- [x] 6. 更新BEVFusionWithRadar模型
  - [x] 6.1 集成几何增强预处理
    - 在extract_radar_feat中调用几何增强
    - _Requirements: 1.1, 1.2_
    - _Note: 几何增强在数据管道中完成，模型直接接收增强后的点云_

  - [x] 6.2 实现双路径特征提取
    - 同时提取语义BEV特征和速度BEV图
    - 返回两个输出供后续使用
    - _Requirements: 2.1, 3.1_

  - [x] 6.3 集成速度校准到检测头
    - 修改TransFusionHead支持速度校准
    - 传递速度BEV图给检测头
    - _Requirements: 5.1, 5.2, 5.3_
    - _Note: 添加velocity_refinement模块配置_

- [x] 7. Checkpoint - 验证模型构建和前向传播
  - 确保所有测试通过
  - 验证模型可以正确构建
  - 验证前向传播正常工作
  - _模型构建和前向传播测试通过 ✅_

- [x] 8. 更新配置文件
  - [x] 8.1 更新雷达数据加载配置
    - 修改use_dim包含vx_rms, vy_rms
    - 添加RadarGeometryEnhancer到数据管道
    - _Requirements: 6.1, 6.3, 6.4_

  - [x] 8.2 更新雷达编码器配置
    - 更新PillarFeatureNet的in_channels为10
    - 添加GeometryAwareVelocityEncoder配置
    - _Requirements: 7.1, 7.4_

  - [x] 8.3 更新融合层配置
    - 确保in_channels正确：[80, 256, 64]
    - _Requirements: 4.1, 4.2, 4.3_

- [x] 9. 更新模块注册
  - [x] 9.1 更新__init__.py
    - 导出RadarGeometryEnhancer
    - 导出GeometryAwareVelocityEncoder
    - 导出VelocityRefinementModule
    - _Requirements: 8.3, 8.4_

- [x] 10. 编写集成测试
  - [x] 10.1 编写融合层属性测试
    - **Property 5: 融合层输出形状**
    - **Property 6: 模态灵活性**
    - **Validates: Requirements 4.2, 4.3, 4.4**
    - _测试通过 ✅_

  - [x] 10.2 编写增强一致性测试
    - **Property 10: 增强一致性**
    - **Validates: Requirements 6.5**
    - _测试通过 ✅_

  - [x] 10.3 编写后向兼容性测试
    - **Property 11: 后向兼容性**
    - **Validates: Requirements 7.3**
    - _测试通过 ✅_

- [x] 11. Final Checkpoint - 完整功能验证
  - 确保所有测试通过 ✅ (24/24 tests passed)
  - 验证配置文件正确 ✅
  - 验证模型可以正确构建和前向传播 ✅
  - _所有验证通过_

- [x] 12. 清理测试文件
  - [x] 12.1 删除测试文件
    - 删除`tests/`目录下为本功能创建的测试文件
    - 保留核心实现代码
    - 保持代码库整洁
    - _Note: 仅在所有测试通过后执行_

## 注意事项

- 所有任务都是必需的，包括测试任务
- 每个任务都引用了具体的需求以便追溯
- Checkpoint任务用于增量验证
- 所有新增代码应包含中文注释以保持与现有代码风格一致
- Property测试验证通用正确性属性
- 单元测试验证具体示例和边界情况
- **任务12在所有功能验证通过后执行，清理测试文件保持代码库整洁**

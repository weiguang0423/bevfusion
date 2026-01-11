# Implementation Plan: BEVFusion Radar Branch

## Overview

本实现计划将毫米波雷达分支集成到BEVFusion框架中。实现遵循增量开发原则，每个任务都在前一个任务的基础上构建，确保代码始终处于可运行状态。

## Tasks

- [x] 1. 实现雷达数据加载模块
  - [x] 1.1 创建LoadRadarPointsFromFile类
    - 在`projects/BEVFusion/bevfusion/loading.py`中添加
    - 支持nuScenes雷达点云格式（18维）
    - 支持可配置的use_dim参数
    - 添加中文注释
    - _Requirements: 1.1, 1.3_

  - [x] 1.2 创建LoadRadarPointsFromMultiSweeps类
    - 支持加载多帧历史雷达数据
    - 实现坐标变换到当前帧
    - 支持pad_empty_sweeps和remove_close选项
    - _Requirements: 1.4_

  - [x] 1.3 编写雷达数据加载单元测试
    - 测试单帧加载
    - 测试多帧sweep加载
    - 测试错误处理
    - _Requirements: 1.5_

- [x] 2. 实现雷达点云预处理模块
  - [x] 2.1 创建RadarPointsRangeFilter类
    - 在`projects/BEVFusion/bevfusion/transforms_3d.py`中添加
    - 过滤超出范围的雷达点
    - _Requirements: 2.1, 2.4_

  - [x] 2.2 更新数据增强模块支持雷达点云
    - 修改BEVFusionRandomFlip3D支持radar_points
    - 修改BEVFusionGlobalRotScaleTrans支持radar_points
    - 确保雷达点云与LiDAR点云使用相同的增强矩阵
    - _Requirements: 2.2, 2.3_

  - [x] 2.3 编写预处理属性测试
    - **Property 4: Range Filtering Correctness**
    - **Property 5: Augmentation Consistency**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. Checkpoint - 验证数据加载和预处理
  - 确保所有测试通过
  - 验证雷达数据可以正确加载和预处理
  - 如有问题请询问用户

- [x] 4. 实现雷达特征编码模块
  - [x] 4.1 创建BEVFusionWithRadar模型类
    - 在`projects/BEVFusion/bevfusion/bevfusion.py`中添加
    - 继承自BEVFusion
    - 添加radar_voxel_layer、radar_voxel_encoder、radar_middle_encoder
    - _Requirements: 5.1, 5.2_

  - [x] 4.2 实现extract_radar_feat方法
    - 实现雷达点云体素化
    - 复用Voxelization模块
    - 复用PillarFeatureNet和PointPillarsScatter
    - _Requirements: 3.1, 3.2, 3.3, 7.1, 7.2_

  - [x] 4.3 实现空输入处理
    - 当雷达点云为空时返回零填充BEV特征
    - _Requirements: 3.4_

  - [x] 4.4 编写编码器属性测试
    - **Property 6: Radar Encoder Output Shape**
    - **Property 7: Empty Input Handling**
    - **Validates: Requirements 3.1, 3.3, 3.4**

- [x] 5. 实现三模态特征融合
  - [x] 5.1 重写extract_feat方法
    - 在BEVFusionWithRadar中重写
    - 添加雷达特征提取逻辑
    - 将雷达BEV特征加入融合列表
    - _Requirements: 4.1, 5.2_

  - [x] 5.2 验证ConvFuser兼容性
    - 确保ConvFuser支持可变数量的输入
    - 测试2模态和3模态融合
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

  - [x] 5.3 编写融合层属性测试
    - **Property 8: Fusion Layer Channel Consistency**
    - **Property 9: Modality Flexibility**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

- [x] 6. Checkpoint - 验证模型构建和前向传播
  - 确保所有测试通过
  - 验证模型可以正确构建
  - 验证前向传播正常工作
  - 如有问题请询问用户

- [x] 7. 创建配置文件
  - [x] 7.1 创建三模态BEVFusion配置文件
    - 创建`bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
    - 配置雷达体素化参数
    - 配置雷达编码器参数
    - 更新融合层in_channels
    - _Requirements: 5.4, 5.5_

  - [x] 7.2 更新数据管道配置
    - 添加雷达数据加载transform
    - 添加雷达点云预处理transform
    - 更新Pack3DDetInputs包含radar_points
    - _Requirements: 6.1, 6.3, 6.4_

  - [x] 7.3 更新模态配置
    - 添加use_radar标志
    - _Requirements: 6.3_

- [x] 8. 更新模块注册
  - [x] 8.1 更新__init__.py
    - 在`projects/BEVFusion/bevfusion/__init__.py`中注册新模块
    - 导出LoadRadarPointsFromFile
    - 导出LoadRadarPointsFromMultiSweeps
    - 导出RadarPointsRangeFilter
    - 导出BEVFusionWithRadar
    - _Requirements: 7.3, 7.4_

- [x] 9. 编写集成测试
  - [x] 9.1 编写配置加载测试
    - 测试配置文件可以正确解析
    - 测试模型可以从配置构建
    - _Requirements: 5.1, 5.4_

  - [x] 9.2 编写后向兼容性测试
    - **Property 10: Configuration Backward Compatibility**
    - 验证不带雷达配置时与原BEVFusion行为一致
    - **Validates: Requirements 5.3**

- [x] 10. Final Checkpoint - 完整功能验证
  - 确保所有测试通过
  - 验证配置文件正确
  - 验证模型可以正常训练和推理
  - 如有问题请询问用户

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- 所有新增代码应包含中文注释以保持与现有代码风格一致


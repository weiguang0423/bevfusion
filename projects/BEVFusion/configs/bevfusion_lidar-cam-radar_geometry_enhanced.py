"""
BEVFusion 三模态配置文件 - 几何增强版本

本配置文件实现了几何感知型多路径雷达融合架构：
1. 几何增强预处理：将6维雷达点云扩展为10维
2. 双路径解耦：语义路径用于融合，速度路径直连检测头
3. 速度校准：基于置信度的加权融合

核心改进：
- RadarGeometryEnhancer: 引入方位角特征（sin θ, cos θ）
- GeometryAwareVelocityEncoder: 物理速度路径
- VelocityRefinementModule: 速度校准模块

数据流：
[原始雷达6维] --> [几何增强: 引入 sinθ/cosθ/RMS] --> [10维增强点云]
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

Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.4, 5.1-5.5, 6.1-6.5
"""

_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ============ 点云范围配置 ============
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ============ 雷达体素化配置 ============
# 雷达点云比LiDAR更稀疏，使用较大的体素尺寸
# 点云范围 108m / 180 格子 = 0.6m 每格
radar_voxel_size = [0.6, 0.6, 8.0]

# ============ 模态配置 ============
# 启用三模态：LiDAR + Camera + Radar
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=True,
)

backend_args = None

# ============ 模型配置 ============
model = dict(
    type='BEVFusionWithRadar',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        # 雷达体素化配置
        radar_voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=radar_voxel_size,
            max_voxels=(30000, 40000),
            voxelize_reduce=True,
        ),
    ),
    # 雷达体素编码器配置 (PillarFeatureNet)
    # 输入：11维增强点云 [x, y, z, rcs, vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms, dt]
    # 其中 dt 是时间偏移量，是多帧融合的关键特征
    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=11,  # 11维增强点云（包含时间戳 dt）
        feat_channels=[64],
        with_distance=False,
        voxel_size=radar_voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    # 雷达中间编码器配置 (PointPillarsScatter)
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[180, 180],
    ),
    # 几何感知速度编码器配置
    radar_velocity_encoder=dict(
        type='GeometryAwareVelocityEncoder',
        point_cloud_range=point_cloud_range,
        bev_size=(180, 180),
        hidden_channels=[32, 16],
        output_channels=4,  # [vx, vy, rms, confidence]
        vel_scale=10.0,  # 速度归一化：[-50, 50] m/s -> [-5, 5]
        rms_scale=1.0,   # RMS归一化
    ),
    # 速度校准模块配置
    velocity_refinement=dict(
        type='VelocityRefinementModule',
        hidden_channel=128,
        confidence_threshold=0.1,
    ),
    # 更新融合层配置：支持三模态输入，使用SE注意力防止Radar被淹没
    fusion_layer=dict(
        type='SEConvFuser',
        in_channels=[80, 256, 64],  # [img_bev, lidar_bev, radar_bev]
        out_channels=256,
        reduction=4,  # SE模块通道压缩比
        use_se=[False, False, True],  # 只对Radar分支使用SE（通道数少，容易被淹没）
        modality_dropout_rate=0.0,   # 默认关闭，Stage2中启用
        uncertainty_gate=False,      # 默认关闭
        fusion_warmup_iters=0,       # 默认不做融合渐进，Stage2中启用
        residual_radar=True,         # 残差雷达融合模式
        radar_gate_init=0.0,         # radar门控初始值
        radar_warmup_epochs=2,       # warmup epoch数
    ),
    # 检测头配置：增大速度损失权重
    bbox_head=dict(
        train_cfg=dict(
            # code_weights: [x, y, z, w, l, h, sin, cos, vx, vy]
            # 增大速度权重从0.2到0.5，让模型更关注速度预测
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        ),
    ),
)

# ============ 训练数据管道配置 ============
train_pipeline = [
    # 图像加载
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    # LiDAR点云加载
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # 雷达点云加载（包含vx_rms, vy_rms）
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9, 16, 17],  # x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=5,
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9, 16, 17],
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=1.0,
        backend_args=backend_args),
    # 标注加载
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    # 图像数据增强
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    # 3D数据增强（同时应用于LiDAR和雷达点云）
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    # 雷达几何增强：在旋转后计算方位角特征（sin θ, cos θ）
    # 确保方位角是基于增强后的坐标计算的，避免时序陷阱
    dict(type='RadarGeometryEnhancer'),
    # 点云范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # 雷达点云范围过滤
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    # 目标范围过滤
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # GridMask增强
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    # 数据打包
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'radar_points',
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

# ============ 测试数据管道配置 ============
test_pipeline = [
    # 图像加载
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    # LiDAR点云加载
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # 雷达点云加载（包含vx_rms, vy_rms）
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9, 16, 17],
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=5,
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9, 16, 17],
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=1.0,
        test_mode=True,
        backend_args=backend_args),
    # 图像数据增强（测试模式）
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    # 雷达几何增强：测试时也需要计算方位角特征
    dict(type='RadarGeometryEnhancer'),
    # 点云范围过滤
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    # 雷达点云范围过滤
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=point_cloud_range),
    # 数据打包
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'radar_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

# ============ 数据加载器配置 ============
# 硬件参考: A800-SXM4-40GB × 4, 48核/48GB
train_dataloader = dict(
    batch_size=6,          # 每GPU的batch size
    num_workers=4,         # 每GPU的数据加载进程数
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=test_pipeline, modality=input_modality))

test_dataloader = val_dataloader

# ============ 第一阶段训练配置：冻结LiDAR/Camera，训练Radar分支 ============
# 
# 训练策略：
#   - Backbones (LiDAR/Camera): lr_mult=0，冻结不动，保持已学到的空间表征
#   - Radar Branch: lr_mult=1.0，主力训练，学习速度特征提取
#   - Fusion Layer: lr_mult=1.0，学习如何平衡三模态信息
#   - BBox Head: lr_mult=0.1，微调检测头，适应新增的雷达特征
#
# 使用方法：
#   python tools/train.py <config> --cfg-options load_from=<LiDAR-Camera预训练权重>
#
# 设计理念：
#   保证LiDAR建立的"空间秩序"不被破坏（位置、形状依然准确），
#   同时给雷达分支足够机会证明自己在"速度预测"上的价值。
#
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 开启混合精度训练，降低显存占用约30%
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=0.0006, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # ====== 冻结：已训练好的模块 ======
            # 图像分支 (lr_mult=0)
            'img_backbone': dict(lr_mult=0, decay_mult=0),
            'img_neck': dict(lr_mult=0, decay_mult=0),
            'view_transform': dict(lr_mult=0, decay_mult=0),
            # 激光雷达分支 (lr_mult=0)
            'pts_voxel_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_middle_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_backbone': dict(lr_mult=0, decay_mult=0),
            'pts_neck': dict(lr_mult=0, decay_mult=0),
            
            # ====== 训练：新增模块 (lr_mult=1.0，使用默认值) ======
            # radar_voxel_encoder, radar_middle_encoder, 
            # radar_velocity_encoder, fusion_layer, velocity_refinement
            # 这些模块未在custom_keys中指定，将使用基础学习率
            
            # ====== 微调：检测头 ======
            'bbox_head': dict(lr_mult=0.1),
        }
    )
)

# ============ 学习率调度器 ============
param_scheduler = [
    # 线性预热：前500 iter从 lr*0.1 增加到 lr
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500),
    # 余弦退火：从 lr 降到 lr*1e-4
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=10,
        end=10,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

# ============ 训练配置 ============
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)

# ============ 模型包装配置（分布式训练）============
# velocity_refinement 模块在第一阶段可能未参与loss计算，需要启用此选项
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

# ============ 日志和检查点 ============
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))  # NDS越大越好


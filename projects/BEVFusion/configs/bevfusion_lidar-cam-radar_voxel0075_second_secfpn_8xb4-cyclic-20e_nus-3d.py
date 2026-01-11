"""
BEVFusion 三模态配置文件 (LiDAR + Camera + Radar)

本配置文件实现了相机-激光雷达-雷达三模态融合的BEVFusion模型。
在原有双模态BEVFusion基础上，添加了雷达分支，将雷达点云转换为BEV特征
并与图像和激光雷达特征进行融合。

主要特点：
1. 支持nuScenes数据集的5个雷达传感器
2. 雷达分支使用基于Pillar的编码方式
3. 三模态特征在BEV空间进行融合
4. 雷达点云与LiDAR点云使用相同的数据增强

Requirements: 5.4, 5.5, 6.1, 6.3, 6.4
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
    use_radar=True,  # 新增：启用雷达模态
)

backend_args = None

# ============ 模型配置 ============
model = dict(
    type='BEVFusionWithRadar',  # 使用支持雷达的BEVFusion模型
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        # 雷达体素化配置
        radar_voxelize_cfg=dict(
            max_num_points=10,  # 每个体素最大点数
            point_cloud_range=point_cloud_range,
            voxel_size=radar_voxel_size,
            max_voxels=(30000, 40000),  # (训练, 测试) 最大体素数
            voxelize_reduce=True,  # 对体素内的点进行平均
        ),
    ),
    # 雷达体素编码器配置 (PillarFeatureNet)
    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=6,  # x, y, z, rcs, vx_comp, vy_comp
        feat_channels=[64],  # 输出通道数
        with_distance=False,  # 不使用距离特征
        voxel_size=radar_voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    # 雷达中间编码器配置 (PointPillarsScatter)
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,  # 与PillarFeatureNet输出通道一致
        output_shape=[180, 180],  # BEV特征图尺寸，与LiDAR BEV一致
    ),
    # 更新融合层配置：支持三模态输入
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256, 64],  # [img_bev, lidar_bev, radar_bev]
        out_channels=256,
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
    # 雷达点云加载（新增）
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,  # nuScenes雷达点云18维
        use_dim=[0, 1, 2, 5, 8, 9],  # x, y, z, rcs, vx_comp, vy_comp
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=5,  # 加载5帧历史雷达数据
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9],
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
    # 点云范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # 雷达点云范围过滤（新增）
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    # 目标范围过滤
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # GridMask增强（实际未启用，prob=0.0）
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
    # 数据打包（包含雷达点云）
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'radar_points',  # 新增radar_points
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
    # 雷达点云加载（新增）
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9],
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=5,
        load_dim=18,
        use_dim=[0, 1, 2, 5, 8, 9],
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=1.0,
        test_mode=True,  # 测试模式：选择最近的帧
        backend_args=backend_args),
    # 图像数据增强（测试模式：固定参数）
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    # 点云范围过滤
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    # 雷达点云范围过滤（新增）
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=point_cloud_range),
    # 数据打包（包含雷达点云）
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
train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

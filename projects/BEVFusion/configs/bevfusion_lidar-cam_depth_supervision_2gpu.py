"""
BEVFusion 激光雷达-图像双分支训练配置 (带深度监督)

配置要点：
1. 分层学习率：骨干网络0.1x，新模块1.0x
2. 梯度裁剪：max_norm=35，防止深度监督初期梯度爆炸
3. 2x A800 GPU训练，相应调整学习率
4. 数据增强对齐：图像增强先于点云增强，LoadDepthFromPoints会读取增强矩阵
5. 10 epochs训练
6. 深度监督损失：focal loss + LiDAR投影监督

硬件：2x A800 GPU, batch_size=4 per GPU = 8 total
原始配置：8 GPU x 4 = 32 total batch size
学习率线性缩放：lr = 0.0002 * (8/32) = 0.00005
"""

_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ============ 基础配置 ============
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ============ 模型配置（带深度监督） ============
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    # 图像骨干网络 - Swin Transformer (lr_mult=0.1)
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    # 图像Neck (lr_mult=0.1)
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    # 视角变换模块（带深度监督）- 新模块，lr_mult=1.0
    view_transform=dict(
        type='CameraAwareDepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        # 启用Camera-Aware和深度监督
        use_camera_aware=True,
        use_depth_supervision=True,
        # Camera-Aware DepthNet配置
        camera_aware_cfg=dict(
            type='CameraAwareDepthNet',
            in_channels=320,  # 256 (image feat) + 64 (camera embed)
            mid_channels=320,
            context_channels=80,
            depth_channels=118,  # D = (60-1)/0.5 = 118
            embed_dim=256,
        ),
        # 深度监督损失配置
        depth_loss_cfg=dict(
            type='DepthSupervisionLoss',
            loss_type='focal',
            focal_gamma=2.0,
            loss_weight=1.0,  # 深度监督损失权重
        ),
    ),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256))

# ============ 训练数据管道（带深度监督GT生成） ============
# ⚠️ 关键：数据增强顺序确保深度-图像对齐
# 
# 正确顺序：
# 1. 加载原始数据
# 2. ImageAug3D: 图像增强，生成 img_aug_matrix
# 3. LoadDepthFromPoints: 使用【原始点云】+ ori_lidar2img + img_aug_matrix 生成深度GT
#    - 原始点云通过 ori_lidar2img 投影到原始图像坐标
#    - 再通过 img_aug_matrix 变换到增强后的图像坐标
#    - 这样深度GT与增强后的图像完美对齐！
# 4. 点云几何增强：之后点云被旋转/缩放/翻转，但深度GT已经生成，不受影响
#
train_pipeline = [
    # 1. 加载数据
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
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
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    # 2. 图像数据增强（生成 img_aug_matrix）
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    # 3. ⚠️ 深度监督GT生成 - 必须在点云增强之前！
    # 此时点云还是原始坐标，可以正确投影到增强后的图像
    # 使用 ori_lidar2img（原始投影）+ img_aug_matrix（图像增强）
    dict(
        type='LoadDepthFromPoints',
        feature_size=(32, 88),  # 与 view_transform.feature_size 一致
        dbound=(1.0, 60.0, 0.5),  # 与 view_transform.dbound 一致
        discretization='uniform',
        dilation_kernel=3,  # 形态学膨胀，增加监督信号密度
    ),
    # 4. 点云数据增强（记录 lidar_aug_matrix）
    # 深度GT已经生成，这些增强不会影响它
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    # 5. 范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # 6. 其他增强
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=8,  # 更新为8 epochs
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    # 7. 打包输入（包含深度GT）
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels', 'depth_gt_indices', 'depth_valid_mask'  # 深度监督GT
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

# ============ 测试数据管道（无增强） ============
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
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
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

# ============ 数据加载器配置（2 GPU） ============
train_dataloader = dict(
    batch_size=6,  # per GPU
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

# ============ 学习率调度器（10 epochs） ============
param_scheduler = [
    # 线性预热
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    # 余弦退火学习率
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=8,  # 8 epochs
        end=8,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # 动量调度器
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=3.2,  # 调整为8 epochs的40%
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=3.2,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True)
]

# ============ 训练配置（10 epochs） ============
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ============ 优化器配置（分层学习率 + 梯度裁剪） ============
# 原始配置: 8 GPU x 4 = 32 batch size, lr = 0.0002
# 当前配置: 2 GPU x 6 = 12 batch size
# 线性缩放: lr = 0.0002 * (12/32) = 0.000075
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 使用混合精度训练
    optimizer=dict(type='AdamW', lr=0.000075, weight_decay=0.01),
    # 梯度裁剪：深度监督初期梯度极大，必须限制，防止 NaN
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
    # 分层学习率配置
    paramwise_cfg=dict(
        custom_keys={
            # 1. 图像骨干 (Swin-T): 老专家，已有知识，轻轻微调
            'img_backbone': dict(lr_mult=0.1),
            # 2. 激光骨干 (VoxelNet): 老专家，轻轻微调
            'pts_backbone': dict(lr_mult=0.1),
            # 3. 图像颈部 (FPN): 通常也视为特征提取的一部分，建议微调
            'img_neck': dict(lr_mult=0.1),
            # 4. 点云编码器：微调
            'pts_voxel_encoder': dict(lr_mult=0.1),
            'pts_middle_encoder': dict(lr_mult=0.1),
            'pts_neck': dict(lr_mult=0.1),
            # ====== 以下模块不在列表中，默认 lr_mult=1.0 ======
            # - view_transform (CameraAwareDepthLSSTransform): 新模块，全速学习
            # - fusion_layer (ConvFuser): 融合层，全速学习
            # - bbox_head: 检测头，全速学习
        },
        # BN层不使用权重衰减
        norm_decay_mult=0.0
    ),
)

# ============ 分布式训练配置 ============
# find_unused_parameters=True: 深度监督可能有未使用的参数
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')

# ============ 学习率自动缩放（已手动设置，禁用） ============
# base_batch_size = 12 (2 GPUs x 6 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=12)

# ============ 日志和检查点 ============
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
del _base_.custom_hooks

# ============ 预训练权重 ============
# 从已有的 lidar-cam 预训练权重加载
# 深度监督模块会自动随机初始化
load_from = 'checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

# ============ 工作目录 ============
work_dir = './work_dirs/bevfusion_lidar-cam_depth_supervision_2gpu'

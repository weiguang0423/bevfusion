"""
BEVFusion 三模态微调配置文件

基于双模态预训练权重进行微调，只需训练：
1. 雷达分支 (radar_voxel_encoder, radar_middle_encoder)
2. 融合层 (fusion_layer) - 因为输入通道变化

其他模块完全冻结。

硬件配置：4x NVIDIA A800-SXM4-40GB, 48核 CPU, 48GB 内存
"""

# ============ 多卡训练配置 ============
opencv_num_threads = 0
mp_start_method = 'spawn'  # 多卡必须用 spawn

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

_base_ = [
    './bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

load_from = 'checkpoints/bevfusion_lidar_cam_converted.pth'

# DDP 配置 - 允许未使用的参数（因为冻结了部分模块）
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=3,
        end=3,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

# ============ 4x A800 分布式训练配置 ============
# 每卡 batch=4，4卡总 batch=16
# 48核/4卡 = 12核/卡，num_workers=6
train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5,
    ),
    logger=dict(type='LoggerHook', interval=50),
)

# 优化器配置 - 4卡分布式
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,  # 总 batch=16
        weight_decay=0.01,
    ),
    dtype='float16',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0, decay_mult=0),
            'img_neck': dict(lr_mult=0, decay_mult=0),
            'view_transform': dict(lr_mult=0, decay_mult=0),
            'pts_voxel_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_middle_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_backbone': dict(lr_mult=0, decay_mult=0),
            'pts_neck': dict(lr_mult=0, decay_mult=0),
            'bbox_head': dict(lr_mult=0, decay_mult=0),
            'radar_voxel_encoder': dict(lr_mult=1.0),
            'radar_middle_encoder': dict(lr_mult=1.0),
            'fusion_layer': dict(lr_mult=1.0),
        }
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

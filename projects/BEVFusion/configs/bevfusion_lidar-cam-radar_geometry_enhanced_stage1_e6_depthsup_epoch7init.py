_base_ = ['./bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e12.py']

load_from = 'checkpoints/bevfusion_lidar_cam_radar_stage1_init_depthsup_epoch7_cameraaware.pth'

custom_hooks = [
    dict(
        type='FreezeModulesHook',
        module_names=[
            'img_backbone',
            'img_neck',
            'pts_backbone',
            'pts_middle_encoder',
            'pts_neck',
            'pts_voxel_encoder',
            'view_transform',
        ],
        set_eval=True,
    ),
]

model = dict(
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
        use_camera_aware=True,
        use_depth_supervision=False,
        camera_aware_cfg=dict(
            type='CameraAwareDepthNet',
            in_channels=320,
            mid_channels=320,
            context_channels=80,
            depth_channels=118,
            embed_dim=256,
        ),
    ),
)

train_dataloader = dict(batch_size=6)

optim_wrapper = dict(accumulative_counts=1)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=6,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=6,
        eta_min_ratio=0.0001,
        type='CosineAnnealingLR'),
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

work_dir = 'work_dirs/bevfusion_radar_weather_stage1_e6_depthsup_epoch7init_cameraaware'
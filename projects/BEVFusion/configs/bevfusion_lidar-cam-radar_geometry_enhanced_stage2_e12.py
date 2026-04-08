from copy import deepcopy

_base_ = ['./bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e6_depthsup_epoch7init.py']

stage2_max_epochs = 12
stage2_load_from = '/root/mmdetection3d-main/work_dirs/bevfusion_radar_weather_stage1_e6_depthsup_epoch7init_cameraaware/epoch_1.pth'

train_pipeline = deepcopy(_base_.train_pipeline)
for transform in train_pipeline:
    if transform.get('type') == 'GridMask':
        transform['max_epoch'] = stage2_max_epochs
        break

default_hooks = deepcopy(_base_.default_hooks)
default_hooks.logger.interval = 25

env_cfg = deepcopy(_base_.env_cfg)
env_cfg.cudnn_benchmark = True

model_wrapper_cfg = deepcopy(_base_.model_wrapper_cfg)
model_wrapper_cfg.find_unused_parameters = True

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader.batch_size = 7
train_dataloader.num_workers = 8
train_dataloader.persistent_workers = True
train_dataloader.pin_memory = True
train_dataloader.prefetch_factor = 2
train_dataloader.dataset.dataset.pipeline = train_pipeline

val_dataloader = deepcopy(_base_.val_dataloader)
val_dataloader.num_workers = 6
val_dataloader.persistent_workers = True
val_dataloader.pin_memory = True
val_dataloader.prefetch_factor = 2

test_dataloader = deepcopy(_base_.test_dataloader)
test_dataloader.num_workers = 6
test_dataloader.persistent_workers = True
test_dataloader.pin_memory = True
test_dataloader.prefetch_factor = 2

train_cfg = dict(by_epoch=True, max_epochs=stage2_max_epochs, val_interval=1)

custom_hooks = [
    dict(type='EvidenceEpochHook'),
    dict(
        type='FreezeModulesHook',
        module_names=['img_backbone'],
        set_eval=True,
    ),
    dict(
        type='WeatherEvalHook',
        enabled=False,
        interval=1,
        adverse_version='v1.0-trainval',
        output_dir='weather_eval',
        test_num_workers=8,
        test_persistent_workers=True,
        test_pin_memory=True,
        test_prefetch_factor=2,
        groups=dict(
            day=[],
            night=['night'],
            rain=['rain'],
        ),
    ),
]

model = dict(
    img_backbone=dict(init_cfg=None),
    enable_evidence_loss=True,
    evi_loss_weight=0.05,
    evi_lambda_bg=0.05,
    evi_pos_weight=10.0,
    evi_kl_weight=0.05,
    evi_warmup_iters=4000,
    evi_kl_annealing_epochs=8,
    fusion_layer=dict(
        type='SEConvFuser',
        in_channels=[80, 256, 64],
        out_channels=256,
        reduction=4,
        use_se=[False, False, True],
        enable_evidence_fusion=True,
        enable_evidence_heads=True,
        return_evidence=True,
        branch_scale=[1.0, 1.0, 1.0],
        modality_dropout_rate=0.05,
        uncertainty_gate=False,
        fusion_warmup_iters=8000,
        residual_radar=True,
        radar_gate_init=0.0,
        radar_warmup_epochs=1,
    ),
)

load_from = stage2_load_from
resume = False

optim_wrapper = deepcopy(_base_.optim_wrapper)
optim_wrapper.optimizer.lr = 0.0001
optim_wrapper.optimizer.weight_decay = 0.01
optim_wrapper.paramwise_cfg.custom_keys = {
    'bbox_head': dict(lr_mult=0.3),
    'img_backbone': dict(decay_mult=0, lr_mult=0),
    'img_neck': dict(decay_mult=0, lr_mult=0.05),
    'pts_backbone': dict(decay_mult=1.0, lr_mult=0.1),
    'pts_middle_encoder': dict(decay_mult=0, lr_mult=0.05),
    'pts_neck': dict(decay_mult=1.0, lr_mult=0.1),
    'pts_voxel_encoder': dict(decay_mult=0, lr_mult=0.05),
    'view_transform': dict(decay_mult=0, lr_mult=0.1),
    'fusion_layer.primary_conv': dict(decay_mult=0, lr_mult=0.15),
    'fusion_layer.bn': dict(decay_mult=0, lr_mult=0.15),
    'fusion_layer.radar_gate': dict(decay_mult=0),
}

param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=stage2_max_epochs,
        end=stage2_max_epochs,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

work_dir = 'work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1'

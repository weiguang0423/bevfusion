from copy import deepcopy

_base_ = ['./bevfusion_lidar-cam-radar_geometry_enhanced_stage2_e12.py']

finetune_max_epochs = 2
finetune_load_from = '/root/mmdetection3d-main/work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_10.pth'

train_cfg = dict(by_epoch=True, max_epochs=finetune_max_epochs, val_interval=1)

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader.dataset.weather_resampling_ratios = dict(
    day=1,
    night=3,
    rain=4,
)

model = dict(
    # 从 best 权重做短程 finetune 时，直接保持 KL 项满权重，
    # 避免 _evi_epoch 从 0 重新计数后把证据正则短暂放松。
    evi_kl_annealing_epochs=0,
)

optim_wrapper = deepcopy(_base_.optim_wrapper)
optim_wrapper.optimizer.lr = 2e-5

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=finetune_max_epochs,
        end=finetune_max_epochs,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

load_from = finetune_load_from
resume = False

work_dir = 'work_dirs/bevfusion_radar_weather_stage2_best_ft2_cosine_epochwise'
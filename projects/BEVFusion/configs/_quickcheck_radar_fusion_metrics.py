_base_ = ['./bevfusion_lidar-cam-radar_geometry_enhanced.py']

# Disable external checkpoint download / heavy custom hooks for fast local check.
load_from = None
custom_hooks = []

# Quick smoke run settings (iter-based to finish fast)
train_cfg = dict(_delete_=True, by_epoch=False, max_iters=20, val_interval=100)

# Keep model build compatible with current SEConvFuser signature.
model = dict(
    fusion_layer=dict(
    _delete_=True,
        type='SEConvFuser',
        in_channels=[80, 256, 64],
        out_channels=256,
        reduction=4,
    use_se=[False, False, True],
    enable_evidence_fusion=True,
    enable_evidence_heads=True,
    return_evidence=True),
)

# Make dataloader lightweight for quick verification.
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False)

# Faster logging/checkpoint policy for quick run.
default_hooks = dict(
    logger=dict(interval=5),
    checkpoint=dict(interval=10, save_best=None))

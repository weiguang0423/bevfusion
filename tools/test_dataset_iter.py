import sys
import os

# Set cwd context correctly
print("Importing modules...")
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS
from mmengine.utils import import_modules_from_strings

cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e12.py')
init_default_scope(cfg.get('default_scope', 'mmdet3d'))

if 'custom_imports' in cfg:
    import_modules_from_strings(**cfg['custom_imports'])

# Let's fix the path relative issue by enforcing the base data root config which might be overwritten by default initialization logic
if 'val_dataloader' in cfg and 'dataset' in cfg.val_dataloader:
    cfg.val_dataloader.dataset.data_root = 'data/nuscenes/'

try:
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print("Successfully built TRAIN dataset!")
    data = dataset[0]
    print("Successfully fetched first item!")
    print("Radar points shape:", data['inputs']['radar_points'].shape)
except Exception as e:
    import traceback
    traceback.print_exc()

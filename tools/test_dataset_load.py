from mmengine.config import Config
from mmdet3d.registry import DATASETS
import projects.BEVFusion.bevfusion
cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e12.py')
# remove extra dataset pipeline steps to speed up just checking registry
try:
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    print("Successfully built dataset!")
except Exception as e:
    import traceback
    traceback.print_exc()

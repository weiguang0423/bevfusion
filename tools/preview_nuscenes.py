import projects.BEVFusion.bevfusion
import mmdet3d
from mmdet3d.datasets import NuScenesDataset
dataset = NuScenesDataset(
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_train.pkl',
    pipeline=[],
    modality=dict(use_camera=True, use_lidar=True, use_radar=True))
print(dataset.data_list[0].keys())
print("Token:", dataset.data_list[0]['token'])
print("Sample idx:", dataset.data_list[0].get('sample_idx', None))

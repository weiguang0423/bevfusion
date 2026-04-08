import os
import numpy as np
from tqdm import tqdm
import argparse
from mmengine.config import Config
from mmdet3d.registry import DATASETS, TRANSFORMS
import projects.BEVFusion.bevfusion

def process_split(cfg_path, split, out_dir):
    cfg = Config.fromfile(cfg_path)
    
    loader_cfg = getattr(cfg, f'{split}_dataloader')
    dataset_cfg = loader_cfg.dataset
    if dataset_cfg.type == 'CBGSDataset':
        original_pipeline = dataset_cfg.dataset.pipeline
        dataset_cfg = dataset_cfg.dataset
    else:
        original_pipeline = dataset_cfg.pipeline
        
    radar_pipeline_cfg = []
    for step in original_pipeline:
        if step['type'] in ['LoadRadarPointsFromFile', 'LoadRadarPointsFromMultiSweeps']:
            radar_pipeline_cfg.append(step)
            
    print(f"[{split}] Extracted radar pipeline:", radar_pipeline_cfg)
    
    transforms = [TRANSFORMS.build(cfg) for cfg in radar_pipeline_cfg]
    
    dataset_cfg.pipeline = []
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"[{split}] Total samples: {len(dataset)}")
    os.makedirs(out_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        info = dataset.get_data_info(i)
        results = info
        try:
            for t in transforms:
                results = t(results)
                if results is None: break
        except Exception as e:
            print(f"Error processing {info['token']}: {e}")
            continue
            
        if results is None or 'radar_points' not in results:
            continue
            
        points = results['radar_points'].tensor.numpy()
        token = info['token']
        np.save(os.path.join(out_dir, f"{token}.npy"), points)

if __name__ == '__main__':
    cfg_path = 'projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage1_e12.py'
    out_dir = 'data/nuscenes/radar_multisweep_preprocessed'
    process_split(cfg_path, 'val', out_dir)
    process_split(cfg_path, 'train', out_dir)

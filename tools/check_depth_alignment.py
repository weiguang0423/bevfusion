"""
检查Depth GT与增强后图像的对齐情况

对比展示：
1. 原始图像 + 原始投影点
2. 增强后图像 + 增强后投影点
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


def project_points_to_image(points, lidar2img, img_aug_matrix, img_shape):
    """
    投影点云到图像（应用增强变换）
    
    Args:
        points: [N, 3+] LiDAR点云
        lidar2img: [4, 4] 原始投影矩阵
        img_aug_matrix: [4, 4] 图像增强矩阵
        img_shape: (H, W) 图像尺寸
        
    Returns:
        u, v: 像素坐标
        depth: 深度值
    """
    pts_xyz = points[:, :3]
    pts_homo = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1))], axis=1)
    
    # 投影到原始图像坐标
    img_coords = lidar2img @ pts_homo.T
    depth = img_coords[2, :]
    
    # 过滤相机后方的点
    valid_depth = depth > 0
    
    # 归一化得到原始像素坐标
    u_orig = img_coords[0, :] / (depth + 1e-6)
    v_orig = img_coords[1, :] / (depth + 1e-6)
    
    # 应用图像增强变换
    uv_homo = np.stack([u_orig, v_orig, np.ones_like(u_orig)], axis=0)
    aug_coords = img_aug_matrix[:3, :3] @ uv_homo + img_aug_matrix[:3, 3:4]
    u_aug = aug_coords[0, :]
    v_aug = aug_coords[1, :]
    
    # 过滤图像范围外的点
    H, W = img_shape
    valid_u = (u_aug >= 0) & (u_aug < W)
    valid_v = (v_aug >= 0) & (v_aug < H)
    valid_mask = valid_depth & valid_u & valid_v
    
    return u_orig[valid_mask], v_orig[valid_mask], u_aug[valid_mask], v_aug[valid_mask], depth[valid_mask]


def overlay_points(img, u, v, color=(0, 255, 0), radius=2):
    """在图像上叠加点"""
    img_overlay = img.copy()
    for i in range(len(u)):
        x, y = int(u[i]), int(v[i])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_overlay, (x, y), radius, color, -1)
    return img_overlay


def visualize_alignment(ori_img, aug_img, points, ori_lidar2img, img_aug_matrix, 
                       ori_shape, aug_shape, save_path, idx):
    """
    可视化原始和增强后的对齐情况
    """
    print(f"\n样本 {idx}:")
    print(f"  原始图像: {ori_img.shape}, 范围: [{ori_img.min():.1f}, {ori_img.max():.1f}]")
    print(f"  增强图像: {aug_img.shape}, 范围: [{aug_img.min():.1f}, {aug_img.max():.1f}]")
    
    # 确保图像是uint8格式
    if ori_img.dtype != np.uint8:
        if ori_img.max() <= 1.0:
            ori_img = (ori_img * 255).astype(np.uint8)
        else:
            ori_img = np.clip(ori_img, 0, 255).astype(np.uint8)
    
    if aug_img.dtype != np.uint8:
        if aug_img.max() <= 1.0:
            aug_img = (aug_img * 255).astype(np.uint8)
        else:
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
    
    # 确保是HWC格式
    if ori_img.ndim == 3 and ori_img.shape[0] == 3:
        ori_img = ori_img.transpose(1, 2, 0)
    if aug_img.ndim == 3 and aug_img.shape[0] == 3:
        aug_img = aug_img.transpose(1, 2, 0)
    
    # 投影点云
    u_orig, v_orig, u_aug, v_aug, depth = project_points_to_image(
        points, ori_lidar2img, img_aug_matrix, aug_shape
    )
    
    print(f"  投影点数: {len(u_orig)}")
    print(f"  img_aug_matrix:\n{img_aug_matrix}")
    
    # 叠加点到图像
    ori_with_pts = overlay_points(ori_img, u_orig, v_orig, (0, 255, 0), 2)
    aug_with_pts = overlay_points(aug_img, u_aug, v_aug, (0, 255, 0), 2)
    
    # 绘图
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    axs[0, 0].imshow(ori_img)
    axs[0, 0].set_title("原始图像 (未增强)", fontsize=14, fontweight='bold')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(ori_with_pts)
    axs[0, 1].set_title(f"原始图像 + 投影点 ({len(u_orig)} 点)", fontsize=14, fontweight='bold')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(aug_img)
    axs[1, 0].set_title("增强后图像 (旋转/裁剪/翻转)", fontsize=14, fontweight='bold')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(aug_with_pts)
    axs[1, 1].set_title(f"增强后图像 + 变换后的点 ({len(u_aug)} 点)", fontsize=14, fontweight='bold')
    axs[1, 1].axis('off')
    
    fig.text(0.5, 0.02, 
             "检查要点：\n"
             "1. 看灰色三角形（旋转填充）：下方绿点不应该进入灰色区域\n"
             "2. 看车身：图像旋转了，绿点也应该跟着旋转\n"
             "3. 如果下方绿点形成矩形且覆盖灰边 = 未对齐！",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_file = os.path.join(save_path, f'alignment_{idx}.png')
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    print(f"  保存到: {output_file}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default='debug_vis_output')
    parser.add_argument('--num-samples', type=int, default=2)
    args = parser.parse_args()
    
    print("加载配置...")
    cfg = Config.fromfile(args.config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    register_all_modules()
    
    # 构建两个pipeline：一个不含ImageAug3D，一个包含
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, 'dataset'):
        dataset_cfg = dataset_cfg.dataset
    
    from mmengine.dataset import Compose
    
    # 分离pipeline（排除Pack3DDetInputs）
    ori_transforms = []
    aug_transforms = []
    found_aug = False
    
    for t in dataset_cfg.pipeline:
        if t['type'] == 'ImageAug3D':
            found_aug = True
            aug_transforms.append(t)
        elif t['type'] in ['LoadDepthFromPoints', 'Pack3DDetInputs']:
            continue  # 跳过这些transform
        elif not found_aug:
            ori_transforms.append(t)
            aug_transforms.append(t)
        else:
            aug_transforms.append(t)
    
    print("创建数据集...")
    dataset = DATASETS.build(dataset_cfg)
    
    ori_pipeline = Compose(ori_transforms)
    aug_pipeline = Compose(aug_transforms)
    
    print(f"\n开始可视化 {args.num_samples} 个样本...\n")
    
    for idx in range(min(args.num_samples, len(dataset))):
        print(f"{'='*60}")
        print(f"处理样本 {idx}")
        print(f"{'='*60}")
        
        try:
            # 获取原始数据
            data_info = dataset.get_data_info(idx)
            ori_results = ori_pipeline(data_info)
            
            # 获取增强数据
            data_info2 = dataset.get_data_info(idx)
            aug_results = aug_pipeline(data_info2)
            
            # 提取数据
            points = ori_results['points'].tensor.numpy()
            ori_img = ori_results['img'][0]  # 第一个相机
            aug_img = aug_results['img'][0]
            
            # 使用lidar2img（原始的投影矩阵）
            ori_lidar2img = ori_results['lidar2img'][0]
            img_aug_matrix = aug_results['img_aug_matrix'][0]
            
            ori_shape = ori_img.shape[:2]
            aug_shape = aug_img.shape[:2]
            
            # 可视化
            visualize_alignment(
                ori_img, aug_img, points,
                ori_lidar2img, img_aug_matrix,
                ori_shape, aug_shape,
                str(output_dir), idx
            )
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"完成！结果保存在: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

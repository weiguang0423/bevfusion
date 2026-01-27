"""
可视化数据增强后的激光雷达、毫米波雷达、GT框对齐情况

该脚本将数据增强后的LiDAR点云、Radar点云和GT框绘制在同一张图中，
并标注GT框和雷达点云的速度方向。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

# 注册所有模块
register_all_modules()


def draw_box_with_velocity(ax, center, dims, yaw, velocity=None, color='green', label=None):
    """
    绘制3D边界框的BEV投影和速度箭头
    
    Args:
        ax: matplotlib轴对象
        center: 中心点 [x, y]
        dims: 尺寸 [w, l, h]
        yaw: 偏航角（弧度）
        velocity: 速度 [vx, vy]，如果为None则不绘制速度箭头
        color: 颜色
        label: 标签
    """
    w, l, h = dims
    
    # 计算四个角点（BEV视图）
    # 车辆坐标系：x向前，y向左
    corners = np.array([
        [l/2, w/2],   # 前左
        [l/2, -w/2],  # 前右
        [-l/2, -w/2], # 后右
        [-l/2, w/2],  # 后左
    ])
    
    # 旋转
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    corners = corners @ rot_mat.T
    
    # 平移
    corners[:, 0] += center[0]
    corners[:, 1] += center[1]
    
    # 绘制边界框
    corners_closed = np.vstack([corners, corners[0]])
    ax.plot(corners_closed[:, 0], corners_closed[:, 1], 
            color=color, linewidth=2.0, label=label, zorder=3)
    
    # 绘制车头方向（从中心到前边中点）
    front_center = (corners[0] + corners[1]) / 2
    ax.arrow(center[0], center[1], 
             front_center[0] - center[0], 
             front_center[1] - center[1],
             head_width=0.8, head_length=0.6, 
             fc=color, ec=color, linewidth=1.5, zorder=3)
    
    # 绘制速度箭头（如果提供）
    if velocity is not None and len(velocity) >= 2:
        vx, vy = velocity[0], velocity[1]
        v_mag = np.sqrt(vx**2 + vy**2)
        
        if v_mag > 0.1:  # 只绘制有意义的速度
            # 固定箭头长度，只显示方向
            arrow_length = 5.0  # 固定长度5米
            vx_norm = vx / v_mag * arrow_length
            vy_norm = vy / v_mag * arrow_length
            
            ax.arrow(center[0], center[1], 
                     vx_norm, vy_norm,
                     head_width=0.8, head_length=0.6, 
                     fc='red', ec='red', linewidth=1.5, 
                     alpha=0.9, zorder=4)
            
            # 标注速度大小
            text_x = center[0] + vx_norm * 0.5
            text_y = center[1] + vy_norm * 0.5
            ax.text(text_x, text_y, f'{v_mag:.1f}m/s', 
                   fontsize=8, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='red', alpha=0.9),
                   zorder=5)


def visualize_sample(dataset, sample_idx, save_path):
    """
    可视化单个样本的数据增强后的对齐情况
    
    Args:
        dataset: 数据集对象
        sample_idx: 样本索引
        save_path: 保存路径
    """
    # 获取数据
    print(f"\n{'='*70}")
    print(f"正在处理样本 {sample_idx}")
    print('='*70)
    
    # 直接使用dataset的__getitem__获取处理后的数据
    results = dataset[sample_idx]
    
    # 提取数据
    inputs = results['inputs']
    data_samples = results['data_samples']
    
    # 检查数据
    if 'points' not in inputs:
        print("❌ 没有LiDAR点云数据")
        return
    
    if 'radar_points' not in inputs:
        print("❌ 没有雷达点云数据")
        return
    
    # 获取点云
    lidar_pts = inputs['points'].numpy()
    radar_pts = inputs['radar_points'].numpy()
    
    print(f"  LiDAR 点数: {len(lidar_pts)}")
    print(f"  Radar 点数: {len(radar_pts)}")
    print(f"  Radar 特征维度: {radar_pts.shape[1]}")
    
    # 获取GT框
    if not hasattr(data_samples, 'gt_instances_3d'):
        print("❌ 没有GT标注")
        return
    
    gt_instances = data_samples.gt_instances_3d
    if not hasattr(gt_instances, 'bboxes_3d'):
        print("❌ 没有GT边界框")
        return
    
    gt_bboxes = gt_instances.bboxes_3d
    gt_labels = gt_instances.labels_3d.numpy()
    
    # 获取速度标注（如果有）
    gt_velocities = None
    if hasattr(gt_instances, 'velocities'):
        gt_velocities = gt_instances.velocities.numpy()
    
    print(f"  GT 框数量: {len(gt_bboxes)}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 1. 绘制LiDAR点云（灰色，半透明）
    lidar_step = max(1, len(lidar_pts) // 5000)  # 降采样以提高性能
    ax.scatter(lidar_pts[::lidar_step, 0], lidar_pts[::lidar_step, 1], 
              s=0.5, c='gray', alpha=0.3, label='LiDAR Points', zorder=1)
    
    # 2. 绘制雷达点云（红色叉号）
    ax.scatter(radar_pts[:, 0], radar_pts[:, 1], 
              s=80, c='red', marker='x', linewidths=2.0, 
              label='Radar Points', zorder=4, alpha=0.9)
    
    # 3. 绘制雷达速度箭头（使用与GT相同的样式和长度）
    radar_velocity_drawn = False
    if radar_pts.shape[1] >= 6:
        vx = radar_pts[:, 4]
        vy = radar_pts[:, 5]
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # 只绘制速度大于阈值的点（使用合理的阈值）
        moving_mask = v_mag > 0.5  # 0.5 m/s，过滤掉噪声
        
        if np.sum(moving_mask) > 0:
            print(f"  运动的雷达点 (>0.5m/s): {np.sum(moving_mask)} / {len(radar_pts)}")
            
            # 为每个运动的雷达点绘制速度箭头（与GT速度箭头样式和长度一致）
            for idx in np.where(moving_mask)[0]:
                rx, ry = radar_pts[idx, 0], radar_pts[idx, 1]
                vx_i, vy_i = vx[idx], vy[idx]
                v_mag_i = v_mag[idx]
                
                # 使用与GT相同的缩放因子
                scale = 2.0
                label_text = 'Radar Velocity' if not radar_velocity_drawn else None
                ax.arrow(rx, ry, 
                        vx_i * scale, vy_i * scale,
                        head_width=0.6, head_length=0.5, 
                        fc='orange', ec='orange', linewidth=1.2, 
                        alpha=0.9, zorder=4, label=label_text)
                radar_velocity_drawn = True
    
    # 4. 绘制GT边界框和速度
    centers = gt_bboxes.gravity_center.numpy()
    dims = gt_bboxes.tensor.numpy()[:, 3:6]  # w, l, h
    yaws = gt_bboxes.tensor.numpy()[:, 6]
    
    # 类别名称映射
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    for i in range(len(gt_bboxes)):
        center = centers[i, :2]
        dim = dims[i]
        yaw = yaws[i]
        label_idx = gt_labels[i]
        
        # 获取类别名称
        class_name = class_names[label_idx] if label_idx < len(class_names) else f'class_{label_idx}'
        
        # 获取速度
        velocity = None
        if gt_velocities is not None and i < len(gt_velocities):
            velocity = gt_velocities[i, :2]
        
        # 绘制边界框
        label_text = f'GT: {class_name}' if i == 0 else None
        draw_box_with_velocity(ax, center, dim, yaw, velocity, 
                              color='lime', label=label_text)
        
        # 标注类别
        ax.text(center[0], center[1] + dim[1]/2 + 1.5, class_name,
               fontsize=9, color='lime', fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                        edgecolor='lime', alpha=0.7),
               zorder=5)
    
    # 5. 绘制自车位置（原点）
    ax.plot(0, 0, 'b*', markersize=20, label='Ego Vehicle', zorder=6)
    ax.text(0, -2, 'EGO', fontsize=12, color='blue', fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='blue', alpha=0.9),
           zorder=6)
    
    # 6. 统计信息
    # 计算GT框内的点数
    radar_in_boxes = 0
    lidar_in_boxes = 0
    
    for i in range(len(gt_bboxes)):
        center = centers[i, :2]
        dim = dims[i]
        max_dim = max(dim[0], dim[1]) * 1.2  # 扩大20%
        
        # 雷达点
        radar_dist = np.sqrt((radar_pts[:, 0] - center[0])**2 + 
                            (radar_pts[:, 1] - center[1])**2)
        radar_in_boxes += np.sum(radar_dist < max_dim)
        
        # LiDAR点
        lidar_dist = np.sqrt((lidar_pts[:, 0] - center[0])**2 + 
                            (lidar_pts[:, 1] - center[1])**2)
        lidar_in_boxes += np.sum(lidar_dist < max_dim)
    
    # 添加统计信息文本框
    stats_text = (
        f"Sample #{sample_idx}\n"
        f"LiDAR: {len(lidar_pts)} points\n"
        f"Radar: {len(radar_pts)} points\n"
        f"GT Boxes: {len(gt_bboxes)}\n"
        f"Radar in boxes: {radar_in_boxes} ({100*radar_in_boxes/max(1,len(radar_pts)):.1f}%)\n"
        f"LiDAR in boxes: {lidar_in_boxes} ({100*lidar_in_boxes/len(lidar_pts):.1f}%)"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.set_title('Data Augmentation Alignment Check\n'
                'LiDAR (gray) + Radar (red X) + GT Boxes (green) + Velocities',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 添加坐标轴方向标注
    ax.annotate('', xy=(55, 0), xytext=(50, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(57, 0, 'X (Forward)', fontsize=10, va='center', fontweight='bold')
    
    ax.annotate('', xy=(0, 55), xytext=(0, 50),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(0, 57, 'Y (Left)', fontsize=10, ha='center', fontweight='bold')
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 可视化结果已保存至: {save_path}")
    print(f"  GT框附近的雷达点: {radar_in_boxes} / {len(radar_pts)} ({100*radar_in_boxes/max(1,len(radar_pts)):.1f}%)")
    print(f"  GT框附近的LiDAR点: {lidar_in_boxes} / {len(lidar_pts)} ({100*lidar_in_boxes/len(lidar_pts):.1f}%)")


def main():
    """主函数"""
    print("="*70)
    print("数据增强后的对齐可视化")
    print("="*70)
    
    # 加载配置
    config_path = 'projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced.py'
    print(f"\n正在加载配置: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 构建数据集
    print("正在构建数据集...")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"数据集大小: {len(dataset)}")
    
    # 可视化多个样本
    num_samples = 5
    print(f"\n将可视化 {num_samples} 个样本")
    
    for i in range(num_samples):
        save_path = f'alignment_check_{i}.png'
        try:
            visualize_sample(dataset, i, save_path)
        except Exception as e:
            print(f"❌ 样本 {i} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("可视化完成！")
    print("="*70)
    print("\n检查要点:")
    print("  1. 绿色框（GT）内应该有红色叉号（雷达点）")
    print("  2. 绿色框（GT）内应该有灰色点（LiDAR点）")
    print("  3. 红色虚线箭头（雷达速度）应该与绿色实线箭头（GT速度）方向一致")
    print("  4. 如果对齐正确，说明数据增强后多模态数据保持几何一致性")
    print("="*70)


if __name__ == '__main__':
    main()

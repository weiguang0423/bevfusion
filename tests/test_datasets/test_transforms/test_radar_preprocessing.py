# Copyright (c) OpenMMLab. All rights reserved.
"""
雷达点云预处理模块属性测试

测试 RadarPointsRangeFilter 和数据增强模块对雷达点云的支持。

Property 4: Range Filtering Correctness
Property 5: Augmentation Consistency

Validates: Requirements 2.1, 2.2, 2.3, 2.4

注意：本测试文件可以在没有CUDA编译的环境下运行。
BEVFusion包的__init__.py已经实现了延迟导入机制，
非CUDA依赖的模块（如transforms_3d）可以直接导入。
"""
import os
import sys
import unittest

import numpy as np

from mmdet3d.structures import LiDARPoints, LiDARInstance3DBoxes

# 将项目根目录添加到路径，以便导入projects下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 直接从bevfusion包导入非CUDA依赖的变换类
# 由于__init__.py实现了延迟导入，这不会触发CUDA扩展的加载
from projects.BEVFusion.bevfusion import (
    RadarPointsRangeFilter,
    BEVFusionRandomFlip3D,
    BEVFusionGlobalRotScaleTrans,
)


def generate_random_radar_points(num_points, coord_range=(-100, 100)):
    """
    生成随机雷达点云数据
    
    Args:
        num_points: 点数
        coord_range: 坐标范围 (min, max)
        
    Returns:
        LiDARPoints: 雷达点云对象
    """
    # 生成6维雷达点云: x, y, z, rcs, vx_comp, vy_comp
    points = np.random.uniform(
        coord_range[0], coord_range[1], 
        size=(num_points, 6)
    ).astype(np.float32)
    return LiDARPoints(points, points_dim=6)


def generate_random_lidar_points(num_points, coord_range=(-100, 100)):
    """
    生成随机LiDAR点云数据
    
    Args:
        num_points: 点数
        coord_range: 坐标范围 (min, max)
        
    Returns:
        LiDARPoints: LiDAR点云对象
    """
    # 生成4维LiDAR点云: x, y, z, intensity
    points = np.random.uniform(
        coord_range[0], coord_range[1], 
        size=(num_points, 4)
    ).astype(np.float32)
    return LiDARPoints(points, points_dim=4)


def generate_random_bboxes(num_boxes):
    """
    生成随机3D边界框
    
    Args:
        num_boxes: 边界框数量
        
    Returns:
        LiDARInstance3DBoxes: 3D边界框对象
    """
    # 生成7维边界框: x, y, z, l, w, h, yaw
    bboxes = np.random.uniform(-50, 50, size=(num_boxes, 7)).astype(np.float32)
    # 确保尺寸为正
    bboxes[:, 3:6] = np.abs(bboxes[:, 3:6]) + 0.1
    return LiDARInstance3DBoxes(bboxes)


class TestRadarPointsRangeFilter(unittest.TestCase):
    """
    测试 RadarPointsRangeFilter 类
    
    Property 4: Range Filtering Correctness
    *For any* radar point cloud and point cloud range configuration, 
    all points in the filtered output SHALL have coordinates within 
    the specified range bounds.
    
    Validates: Requirements 2.1, 2.4
    """

    def test_property_range_filtering_correctness(self):
        """
        Property 4: Range Filtering Correctness
        
        *For any* radar point cloud and point cloud range configuration, 
        all points in the filtered output SHALL have coordinates within 
        the specified range bounds.
        
        **Feature: radar-branch-bevfusion, Property 4: Range Filtering Correctness**
        **Validates: Requirements 2.1, 2.4**
        """
        # 运行100次迭代以验证属性
        for _ in range(100):
            # 生成随机点云范围
            x_min = np.random.uniform(-100, 0)
            x_max = np.random.uniform(0, 100)
            y_min = np.random.uniform(-100, 0)
            y_max = np.random.uniform(0, 100)
            z_min = np.random.uniform(-10, 0)
            z_max = np.random.uniform(0, 10)
            
            point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]
            
            # 生成随机点云（范围更大，确保有些点在范围外）
            num_points = np.random.randint(10, 200)
            radar_points = generate_random_radar_points(num_points, coord_range=(-150, 150))
            
            # 创建输入数据
            input_dict = {'radar_points': radar_points}
            
            # 应用范围过滤
            filter_transform = RadarPointsRangeFilter(
                point_cloud_range=point_cloud_range
            )
            output_dict = filter_transform(input_dict)
            
            # 验证属性：所有输出点都在范围内
            filtered_points = output_dict['radar_points'].tensor.numpy()
            
            if len(filtered_points) > 0:
                # 检查x坐标
                self.assertTrue(
                    np.all(filtered_points[:, 0] >= x_min),
                    f"Found points with x < {x_min}"
                )
                self.assertTrue(
                    np.all(filtered_points[:, 0] <= x_max),
                    f"Found points with x > {x_max}"
                )
                
                # 检查y坐标
                self.assertTrue(
                    np.all(filtered_points[:, 1] >= y_min),
                    f"Found points with y < {y_min}"
                )
                self.assertTrue(
                    np.all(filtered_points[:, 1] <= y_max),
                    f"Found points with y > {y_max}"
                )
                
                # 检查z坐标
                self.assertTrue(
                    np.all(filtered_points[:, 2] >= z_min),
                    f"Found points with z < {z_min}"
                )
                self.assertTrue(
                    np.all(filtered_points[:, 2] <= z_max),
                    f"Found points with z > {z_max}"
                )

    def test_empty_radar_points(self):
        """测试空雷达点云的处理"""
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        
        # 创建空点云
        empty_points = LiDARPoints(
            np.zeros((0, 6), dtype=np.float32),
            points_dim=6
        )
        
        input_dict = {'radar_points': empty_points}
        
        filter_transform = RadarPointsRangeFilter(
            point_cloud_range=point_cloud_range
        )
        output_dict = filter_transform(input_dict)
        
        # 应该返回空点云
        self.assertEqual(output_dict['radar_points'].shape[0], 0)

    def test_no_radar_points_key(self):
        """测试没有radar_points键时的处理"""
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        
        input_dict = {'points': generate_random_lidar_points(100)}
        
        filter_transform = RadarPointsRangeFilter(
            point_cloud_range=point_cloud_range
        )
        output_dict = filter_transform(input_dict)
        
        # 应该直接返回，不添加radar_points
        self.assertNotIn('radar_points', output_dict)

    def test_repr(self):
        """测试__repr__方法"""
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        filter_transform = RadarPointsRangeFilter(
            point_cloud_range=point_cloud_range
        )
        
        repr_str = repr(filter_transform)
        self.assertIn('RadarPointsRangeFilter', repr_str)
        self.assertIn('point_cloud_range', repr_str)


class TestAugmentationConsistency(unittest.TestCase):
    """
    测试数据增强的一致性
    
    Property 5: Augmentation Consistency
    *For any* data augmentation applied to LiDAR points, the same 
    transformation matrix SHALL be applied to radar points, 
    maintaining geometric consistency between modalities.
    
    Validates: Requirements 2.2, 2.3
    """

    def test_property_flip_augmentation_consistency(self):
        """
        Property 5: Augmentation Consistency (Flip)
        
        *For any* flip augmentation applied to LiDAR points, the same 
        flip SHALL be applied to radar points.
        
        **Feature: radar-branch-bevfusion, Property 5: Augmentation Consistency**
        **Validates: Requirements 2.2, 2.3**
        """
        flip_transform = BEVFusionRandomFlip3D()
        
        # 运行100次迭代以验证属性
        for _ in range(100):
            # 生成随机点云
            num_lidar_points = np.random.randint(10, 100)
            num_radar_points = np.random.randint(10, 100)
            
            lidar_points = generate_random_lidar_points(num_lidar_points)
            radar_points = generate_random_radar_points(num_radar_points)
            
            # 保存原始坐标
            original_lidar = lidar_points.tensor.numpy().copy()
            original_radar = radar_points.tensor.numpy().copy()
            
            # 创建输入数据
            input_dict = {
                'points': lidar_points,
                'radar_points': radar_points,
            }
            
            # 应用翻转
            output_dict = flip_transform(input_dict)
            
            # 获取变换后的坐标
            transformed_lidar = output_dict['points'].tensor.numpy()
            transformed_radar = output_dict['radar_points'].tensor.numpy()
            
            # 计算LiDAR点云的变换
            lidar_x_flipped = np.allclose(transformed_lidar[:, 0], -original_lidar[:, 0])
            lidar_y_flipped = np.allclose(transformed_lidar[:, 1], -original_lidar[:, 1])
            
            # 计算雷达点云的变换
            radar_x_flipped = np.allclose(transformed_radar[:, 0], -original_radar[:, 0])
            radar_y_flipped = np.allclose(transformed_radar[:, 1], -original_radar[:, 1])
            
            # 验证属性：LiDAR和雷达应该应用相同的翻转
            self.assertEqual(
                lidar_x_flipped, radar_x_flipped,
                "X-axis flip inconsistency between LiDAR and radar"
            )
            self.assertEqual(
                lidar_y_flipped, radar_y_flipped,
                "Y-axis flip inconsistency between LiDAR and radar"
            )

    def test_property_rot_scale_trans_augmentation_consistency(self):
        """
        Property 5: Augmentation Consistency (Rotation, Scale, Translation)
        
        *For any* rotation/scale/translation augmentation applied to LiDAR points, 
        the same transformation SHALL be applied to radar points.
        
        **Feature: radar-branch-bevfusion, Property 5: Augmentation Consistency**
        **Validates: Requirements 2.2, 2.3**
        """
        # 运行100次迭代以验证属性
        for _ in range(100):
            # 创建变换器（每次迭代使用新的随机参数）
            rot_scale_trans = BEVFusionGlobalRotScaleTrans(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0.5, 0.5, 0.5]
            )
            
            # 生成随机点云
            num_lidar_points = np.random.randint(10, 100)
            num_radar_points = np.random.randint(10, 100)
            
            lidar_points = generate_random_lidar_points(num_lidar_points)
            radar_points = generate_random_radar_points(num_radar_points)
            
            # 生成随机边界框
            num_boxes = np.random.randint(1, 10)
            gt_bboxes_3d = generate_random_bboxes(num_boxes)
            
            # 保存原始坐标
            original_lidar = lidar_points.tensor.numpy().copy()
            original_radar = radar_points.tensor.numpy().copy()
            
            # 创建输入数据
            input_dict = {
                'points': lidar_points,
                'radar_points': radar_points,
                'gt_bboxes_3d': gt_bboxes_3d,
            }
            
            # 应用变换
            output_dict = rot_scale_trans(input_dict)
            
            # 获取变换参数（转换为numpy）
            rotation = output_dict['pcd_rotation']
            if hasattr(rotation, 'numpy'):
                rotation = rotation.numpy()
            translation = output_dict['pcd_trans']
            if hasattr(translation, 'numpy'):
                translation = translation.numpy()
            scale = output_dict['pcd_scale_factor']
            
            # 获取变换后的坐标
            transformed_lidar = output_dict['points'].tensor.numpy()
            transformed_radar = output_dict['radar_points'].tensor.numpy()
            
            # 手动计算预期的雷达点云变换
            # 变换顺序: R -> T -> S
            expected_radar = original_radar.copy()
            # 旋转
            expected_radar[:, :3] = expected_radar[:, :3] @ rotation
            # 平移
            expected_radar[:, :3] += translation
            # 缩放
            expected_radar[:, :3] *= scale
            
            # 验证属性：雷达点云应该应用相同的变换
            np.testing.assert_array_almost_equal(
                transformed_radar[:, :3],
                expected_radar[:, :3],
                decimal=4,
                err_msg="Radar points transformation inconsistent with LiDAR"
            )

    def test_lidar_aug_matrix_updated(self):
        """测试lidar_aug_matrix是否正确更新"""
        rot_scale_trans = BEVFusionGlobalRotScaleTrans(
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0.5, 0.5, 0.5]
        )
        
        lidar_points = generate_random_lidar_points(50)
        radar_points = generate_random_radar_points(30)
        gt_bboxes_3d = generate_random_bboxes(5)
        
        input_dict = {
            'points': lidar_points,
            'radar_points': radar_points,
            'gt_bboxes_3d': gt_bboxes_3d,
        }
        
        output_dict = rot_scale_trans(input_dict)
        
        # 验证lidar_aug_matrix存在
        self.assertIn('lidar_aug_matrix', output_dict)
        
        # 验证矩阵形状
        self.assertEqual(output_dict['lidar_aug_matrix'].shape, (4, 4))

    def test_flip_with_no_radar_points(self):
        """测试没有雷达点云时翻转不报错"""
        flip_transform = BEVFusionRandomFlip3D()
        
        lidar_points = generate_random_lidar_points(50)
        
        input_dict = {
            'points': lidar_points,
        }
        
        # 应该不报错
        output_dict = flip_transform(input_dict)
        
        self.assertIn('points', output_dict)
        self.assertNotIn('radar_points', output_dict)

    def test_rot_scale_trans_with_no_radar_points(self):
        """测试没有雷达点云时旋转缩放平移不报错"""
        rot_scale_trans = BEVFusionGlobalRotScaleTrans(
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0.5, 0.5, 0.5]
        )
        
        lidar_points = generate_random_lidar_points(50)
        gt_bboxes_3d = generate_random_bboxes(5)
        
        input_dict = {
            'points': lidar_points,
            'gt_bboxes_3d': gt_bboxes_3d,
        }
        
        # 应该不报错
        output_dict = rot_scale_trans(input_dict)
        
        self.assertIn('points', output_dict)
        self.assertNotIn('radar_points', output_dict)


if __name__ == '__main__':
    unittest.main()

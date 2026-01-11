# Copyright (c) OpenMMLab. All rights reserved.
"""
雷达数据加载模块单元测试

测试 LoadRadarPointsFromFile 和 LoadRadarPointsFromMultiSweeps 类的功能。

注意：本测试文件可以在没有CUDA编译的环境下运行。
BEVFusion包的__init__.py已经实现了延迟导入机制，
非CUDA依赖的模块（如loading）可以直接导入。
"""
import os
import sys
import tempfile
import unittest

import numpy as np

from mmdet3d.structures import LiDARPoints

# 将项目根目录添加到路径，以便导入projects下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 直接从bevfusion包导入非CUDA依赖的加载类
# 由于__init__.py实现了延迟导入，这不会触发CUDA扩展的加载
from projects.BEVFusion.bevfusion import (
    LoadRadarPointsFromFile,
    LoadRadarPointsFromMultiSweeps,
)


class TestLoadRadarPointsFromFile(unittest.TestCase):
    """测试 LoadRadarPointsFromFile 类"""

    @classmethod
    def setUpClass(cls):
        """创建测试用的临时雷达点云文件"""
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟的nuScenes雷达点云数据（18维）
        # 维度: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
        #       is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
        #       pdh0, vx_rms, vy_rms
        cls.num_points = 50
        cls.radar_data = np.random.randn(cls.num_points, 18).astype(np.float32)
        # 设置合理的坐标范围
        cls.radar_data[:, 0] = np.random.uniform(-50, 50, cls.num_points)  # x
        cls.radar_data[:, 1] = np.random.uniform(-50, 50, cls.num_points)  # y
        cls.radar_data[:, 2] = np.random.uniform(-2, 2, cls.num_points)    # z
        cls.radar_data[:, 5] = np.random.uniform(-20, 20, cls.num_points)  # rcs
        cls.radar_data[:, 8] = np.random.uniform(-10, 10, cls.num_points)  # vx_comp
        cls.radar_data[:, 9] = np.random.uniform(-10, 10, cls.num_points)  # vy_comp
        
        # 保存到临时文件
        cls.radar_file_path = os.path.join(cls.temp_dir, 'radar_front.pcd.bin')
        cls.radar_data.tofile(cls.radar_file_path)
        
        # 创建空的雷达点云文件
        cls.empty_radar_path = os.path.join(cls.temp_dir, 'radar_empty.pcd.bin')
        np.array([], dtype=np.float32).tofile(cls.empty_radar_path)

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        import shutil
        shutil.rmtree(cls.temp_dir)

    def test_load_single_radar(self):
        """测试单帧雷达加载"""
        # 创建变换矩阵（单位矩阵，不做变换）
        radar2lidar = np.eye(4, dtype=np.float32)
        
        # 创建输入数据
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': self.radar_file_path,
                    'radar2lidar': radar2lidar
                }
            }
        }
        
        # 创建加载器
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        # 执行加载
        results = loader(results)
        
        # 验证结果
        self.assertIn('radar_points', results)
        self.assertIsInstance(results['radar_points'], LiDARPoints)
        self.assertEqual(results['radar_points'].shape[0], self.num_points)
        self.assertEqual(results['radar_points'].shape[1], 6)  # use_dim长度

    def test_load_multiple_radars(self):
        """测试多个雷达传感器加载"""
        # 创建第二个雷达文件
        radar_data_2 = np.random.randn(30, 18).astype(np.float32)
        radar_file_2 = os.path.join(self.temp_dir, 'radar_back.pcd.bin')
        radar_data_2.tofile(radar_file_2)
        
        radar2lidar = np.eye(4, dtype=np.float32)
        
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': self.radar_file_path,
                    'radar2lidar': radar2lidar
                },
                'RADAR_BACK': {
                    'radar_path': radar_file_2,
                    'radar2lidar': radar2lidar
                }
            }
        }
        
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        results = loader(results)
        
        # 验证合并后的点数
        self.assertEqual(results['radar_points'].shape[0], self.num_points + 30)

    def test_coordinate_transformation(self):
        """测试坐标变换"""
        # 创建一个简单的旋转矩阵（绕z轴旋转90度）
        angle = np.pi / 2
        radar2lidar = np.array([
            [np.cos(angle), -np.sin(angle), 0, 1.0],
            [np.sin(angle), np.cos(angle), 0, 2.0],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 创建简单的测试点
        test_point = np.zeros((1, 18), dtype=np.float32)
        test_point[0, 0] = 1.0  # x = 1
        test_point[0, 1] = 0.0  # y = 0
        test_point[0, 2] = 0.0  # z = 0
        
        test_file = os.path.join(self.temp_dir, 'radar_test.pcd.bin')
        test_point.tofile(test_file)
        
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': test_file,
                    'radar2lidar': radar2lidar
                }
            }
        }
        
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2]
        )
        
        results = loader(results)
        
        # 验证变换后的坐标
        # 原点 (1, 0, 0) 经过90度旋转后应该变成 (0, 1, 0)
        # 再加上平移 (1, 2, 0.5) 应该变成 (1, 3, 0.5)
        transformed = results['radar_points'].tensor.numpy()
        np.testing.assert_array_almost_equal(
            transformed[0, :3], [1.0, 3.0, 0.5], decimal=5)

    def test_missing_radar_info(self):
        """测试缺少雷达信息时的处理"""
        results = {}  # 没有radar_info
        
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        # 应该返回空的雷达点云而不是报错
        results = loader(results)
        
        self.assertIn('radar_points', results)
        self.assertEqual(results['radar_points'].shape[0], 0)

    def test_empty_radar_file(self):
        """测试空雷达文件的处理"""
        radar2lidar = np.eye(4, dtype=np.float32)
        
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': self.empty_radar_path,
                    'radar2lidar': radar2lidar
                }
            }
        }
        
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        results = loader(results)
        
        # 应该返回空的雷达点云
        self.assertEqual(results['radar_points'].shape[0], 0)

    def test_repr(self):
        """测试__repr__方法"""
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        repr_str = repr(loader)
        self.assertIn('LoadRadarPointsFromFile', repr_str)
        self.assertIn('coord_type=LIDAR', repr_str)
        self.assertIn('load_dim=18', repr_str)


class TestLoadRadarPointsFromMultiSweeps(unittest.TestCase):
    """测试 LoadRadarPointsFromMultiSweeps 类"""

    @classmethod
    def setUpClass(cls):
        """创建测试用的临时雷达点云文件"""
        cls.temp_dir = tempfile.mkdtemp()
        
        # 创建当前帧雷达数据
        cls.num_points = 50
        cls.radar_data = np.random.randn(cls.num_points, 18).astype(np.float32)
        cls.radar_file_path = os.path.join(cls.temp_dir, 'radar_current.pcd.bin')
        cls.radar_data.tofile(cls.radar_file_path)
        
        # 创建历史帧雷达数据
        cls.sweep_files = []
        cls.sweep_num_points = [30, 40, 35]
        for i, num_pts in enumerate(cls.sweep_num_points):
            sweep_data = np.random.randn(num_pts, 18).astype(np.float32)
            sweep_file = os.path.join(cls.temp_dir, f'radar_sweep_{i}.pcd.bin')
            sweep_data.tofile(sweep_file)
            cls.sweep_files.append(sweep_file)

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        import shutil
        shutil.rmtree(cls.temp_dir)

    def _create_results_with_sweeps(self):
        """创建包含sweep信息的results字典"""
        radar2lidar = np.eye(4, dtype=np.float32)
        
        # 首先加载当前帧
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': self.radar_file_path,
                    'radar2lidar': radar2lidar
                }
            },
            'timestamp': 1000.0
        }
        
        # 加载当前帧雷达点云
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        results = loader(results)
        
        # 添加sweep信息
        results['radar_sweeps'] = []
        for i, sweep_file in enumerate(self.sweep_files):
            sweep = {
                'RADAR_FRONT': {
                    'radar_path': sweep_file,
                    'radar2lidar': radar2lidar
                },
                'timestamp': 1000.0 - (i + 1) * 0.1
            }
            results['radar_sweeps'].append(sweep)
        
        return results

    def test_load_multi_sweeps(self):
        """测试多帧sweep加载"""
        results = self._create_results_with_sweeps()
        
        sweep_loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=3,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9],
            test_mode=True
        )
        
        results = sweep_loader(results)
        
        # 验证点数增加（当前帧 + 所有sweep）
        expected_points = self.num_points + sum(self.sweep_num_points)
        self.assertEqual(results['radar_points'].shape[0], expected_points)

    def test_sweeps_num_limit(self):
        """测试sweep数量限制"""
        results = self._create_results_with_sweeps()
        
        # 只加载2个sweep
        sweep_loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9],
            test_mode=True
        )
        
        results = sweep_loader(results)
        
        # 验证点数（当前帧 + 2个sweep）
        expected_points = self.num_points + sum(self.sweep_num_points[:2])
        self.assertEqual(results['radar_points'].shape[0], expected_points)

    def test_pad_empty_sweeps(self):
        """测试空sweep填充"""
        radar2lidar = np.eye(4, dtype=np.float32)
        
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': self.radar_file_path,
                    'radar2lidar': radar2lidar
                }
            },
            'timestamp': 1000.0
        }
        
        # 加载当前帧
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        results = loader(results)
        
        # 没有sweep信息，但启用pad_empty_sweeps
        sweep_loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=3,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9],
            pad_empty_sweeps=True
        )
        
        results = sweep_loader(results)
        
        # 验证点数（当前帧复制4次：1个当前帧 + 3个填充）
        expected_points = self.num_points * 4
        self.assertEqual(results['radar_points'].shape[0], expected_points)

    def test_remove_close_points(self):
        """测试移除近点功能"""
        # 创建包含近点的数据
        close_data = np.zeros((10, 18), dtype=np.float32)
        close_data[:5, 0] = 0.5  # 5个点在x=0.5处（在close_radius内）
        close_data[:5, 1] = 0.5  # y=0.5
        close_data[5:, 0] = 5.0  # 5个点在x=5.0处（在close_radius外）
        close_data[5:, 1] = 5.0
        
        close_file = os.path.join(self.temp_dir, 'radar_close.pcd.bin')
        close_data.tofile(close_file)
        
        radar2lidar = np.eye(4, dtype=np.float32)
        
        results = {
            'radar_info': {
                'RADAR_FRONT': {
                    'radar_path': close_file,
                    'radar2lidar': radar2lidar
                }
            },
            'timestamp': 1000.0,
            'radar_sweeps': [{
                'RADAR_FRONT': {
                    'radar_path': close_file,
                    'radar2lidar': radar2lidar
                },
                'timestamp': 999.9
            }]
        }
        
        # 加载当前帧
        loader = LoadRadarPointsFromFile(
            coord_type='LIDAR',
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        results = loader(results)
        
        # 加载sweep并移除近点
        sweep_loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=1,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9],
            remove_close=True,
            close_radius=1.0,
            test_mode=True
        )
        
        results = sweep_loader(results)
        
        # 当前帧10个点 + sweep中5个远点（近点被移除）
        self.assertEqual(results['radar_points'].shape[0], 15)

    def test_no_radar_points(self):
        """测试没有雷达点云时的处理"""
        results = {'timestamp': 1000.0}
        
        sweep_loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=3,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9]
        )
        
        # 应该直接返回，不报错
        results = sweep_loader(results)
        self.assertNotIn('radar_points', results)

    def test_repr(self):
        """测试__repr__方法"""
        loader = LoadRadarPointsFromMultiSweeps(
            sweeps_num=5,
            load_dim=18,
            use_dim=[0, 1, 2, 5, 8, 9],
            pad_empty_sweeps=True,
            remove_close=True,
            close_radius=1.5
        )
        
        repr_str = repr(loader)
        self.assertIn('LoadRadarPointsFromMultiSweeps', repr_str)
        self.assertIn('sweeps_num=5', repr_str)
        self.assertIn('pad_empty_sweeps=True', repr_str)
        self.assertIn('remove_close=True', repr_str)


if __name__ == '__main__':
    unittest.main()

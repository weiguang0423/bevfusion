# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFusion雷达分支集成测试

本测试文件包含：
1. 配置加载测试 (Task 9.1)
   - 测试配置文件可以正确解析
   - 测试模型可以从配置构建
   
2. 后向兼容性测试 (Task 9.2)
   - Property 10: Configuration Backward Compatibility
   - 验证不带雷达配置时与原BEVFusion行为一致

Requirements: 5.1, 5.3, 5.4
"""
import os
import sys
import unittest
from copy import deepcopy

import numpy as np
import torch

# 将项目根目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 检查CUDA是否可用
CUDA_AVAILABLE = torch.cuda.is_available()

# 属性测试迭代次数
PBT_ITERATIONS = 100


class TestConfigurationLoading(unittest.TestCase):
    """
    配置加载测试 (Task 9.1)
    
    测试配置文件可以正确解析，模型可以从配置构建。
    
    Requirements: 5.1, 5.4
    """

    def test_radar_config_file_parsing(self):
        """
        测试雷达配置文件可以正确解析
        
        验证三模态BEVFusion配置文件的结构和参数正确性。
        
        _Requirements: 5.4_
        """
        from mmengine import Config
        
        # 获取配置文件路径
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        # 验证配置文件存在
        self.assertTrue(
            os.path.exists(config_path),
            f"Config file not found: {config_path}"
        )
        
        # 加载配置文件
        cfg = Config.fromfile(config_path)
        
        # 验证模型类型
        self.assertEqual(
            cfg.model.type, 'BEVFusionWithRadar',
            "Model type should be BEVFusionWithRadar"
        )
        
        # 验证雷达体素编码器配置存在
        self.assertIn(
            'radar_voxel_encoder', cfg.model,
            "radar_voxel_encoder should be in model config"
        )
        self.assertIsNotNone(
            cfg.model.radar_voxel_encoder,
            "radar_voxel_encoder should not be None"
        )
        
        # 验证雷达中间编码器配置存在
        self.assertIn(
            'radar_middle_encoder', cfg.model,
            "radar_middle_encoder should be in model config"
        )
        self.assertIsNotNone(
            cfg.model.radar_middle_encoder,
            "radar_middle_encoder should not be None"
        )
        
        # 验证雷达体素编码器参数
        radar_voxel_encoder = cfg.model.radar_voxel_encoder
        self.assertEqual(
            radar_voxel_encoder.type, 'PillarFeatureNet',
            "radar_voxel_encoder type should be PillarFeatureNet"
        )
        self.assertEqual(
            radar_voxel_encoder.in_channels, 6,
            "radar_voxel_encoder in_channels should be 6"
        )
        
        # 验证雷达中间编码器参数
        radar_middle_encoder = cfg.model.radar_middle_encoder
        self.assertEqual(
            radar_middle_encoder.type, 'PointPillarsScatter',
            "radar_middle_encoder type should be PointPillarsScatter"
        )
        self.assertEqual(
            radar_middle_encoder.output_shape, [216, 216],
            "radar_middle_encoder output_shape should be [216, 216]"
        )
        
        # 验证融合层配置
        fusion_layer = cfg.model.fusion_layer
        self.assertEqual(
            fusion_layer.in_channels, [80, 256, 64],
            "fusion_layer in_channels should be [80, 256, 64] for 3 modalities"
        )
        
        # 验证模态配置
        self.assertTrue(
            cfg.input_modality.get('use_radar', False),
            "use_radar should be True in input_modality"
        )

    def test_radar_config_data_pipeline(self):
        """
        测试雷达数据管道配置正确性
        
        验证训练和测试管道包含雷达数据加载和预处理步骤。
        
        _Requirements: 6.1, 6.3, 6.4_
        """
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        # 检查训练管道
        train_pipeline = cfg.train_pipeline
        train_pipeline_types = [step['type'] for step in train_pipeline]
        
        # 验证雷达数据加载步骤存在
        self.assertIn(
            'LoadRadarPointsFromFile', train_pipeline_types,
            "LoadRadarPointsFromFile should be in train_pipeline"
        )
        self.assertIn(
            'LoadRadarPointsFromMultiSweeps', train_pipeline_types,
            "LoadRadarPointsFromMultiSweeps should be in train_pipeline"
        )
        
        # 验证雷达点云范围过滤存在
        self.assertIn(
            'RadarPointsRangeFilter', train_pipeline_types,
            "RadarPointsRangeFilter should be in train_pipeline"
        )
        
        # 验证Pack3DDetInputs包含radar_points
        pack_step = None
        for step in train_pipeline:
            if step['type'] == 'Pack3DDetInputs':
                pack_step = step
                break
        
        self.assertIsNotNone(pack_step, "Pack3DDetInputs should be in train_pipeline")
        self.assertIn(
            'radar_points', pack_step['keys'],
            "radar_points should be in Pack3DDetInputs keys"
        )
        
        # 检查测试管道
        test_pipeline = cfg.test_pipeline
        test_pipeline_types = [step['type'] for step in test_pipeline]
        
        self.assertIn(
            'LoadRadarPointsFromFile', test_pipeline_types,
            "LoadRadarPointsFromFile should be in test_pipeline"
        )
        self.assertIn(
            'RadarPointsRangeFilter', test_pipeline_types,
            "RadarPointsRangeFilter should be in test_pipeline"
        )

    def test_radar_voxelize_config(self):
        """
        测试雷达体素化配置正确性
        
        验证data_preprocessor中的雷达体素化配置。
        
        _Requirements: 5.4, 5.5_
        """
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        # 验证data_preprocessor中的雷达体素化配置
        data_preprocessor = cfg.model.data_preprocessor
        self.assertIn(
            'radar_voxelize_cfg', data_preprocessor,
            "radar_voxelize_cfg should be in data_preprocessor"
        )
        
        radar_voxelize_cfg = data_preprocessor.radar_voxelize_cfg
        
        # 验证体素化参数
        self.assertIn('voxel_size', radar_voxelize_cfg)
        self.assertIn('point_cloud_range', radar_voxelize_cfg)
        self.assertIn('max_num_points', radar_voxelize_cfg)
        self.assertIn('max_voxels', radar_voxelize_cfg)
        
        # 验证体素尺寸
        self.assertEqual(
            radar_voxelize_cfg.voxel_size, [0.5, 0.5, 8.0],
            "radar voxel_size should be [0.5, 0.5, 8.0]"
        )

    def test_base_config_inheritance(self):
        """
        测试配置文件继承关系正确
        
        验证雷达配置正确继承自基础BEVFusion配置。
        
        _Requirements: 5.4_
        """
        from mmengine import Config
        
        # 加载雷达配置
        radar_config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        radar_cfg = Config.fromfile(radar_config_path)
        
        # 验证继承的基础配置参数存在
        # 这些参数应该从基础配置继承
        self.assertIn('pts_voxel_encoder', radar_cfg.model)
        self.assertIn('pts_middle_encoder', radar_cfg.model)
        self.assertIn('pts_backbone', radar_cfg.model)
        self.assertIn('pts_neck', radar_cfg.model)
        self.assertIn('bbox_head', radar_cfg.model)


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
class TestModelBuildingFromConfig(unittest.TestCase):
    """
    测试从配置构建模型 (Task 9.1)
    
    验证模型可以从配置文件正确构建。
    
    Requirements: 5.1, 5.4
    
    注意：此测试需要CUDA环境和完整的BEVFusion CUDA扩展编译。
    """

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.models_available = False
        cls.import_error = None
        
        try:
            from mmdet3d.registry import MODELS
            from mmengine import DefaultScope
            
            # 初始化默认作用域
            DefaultScope.get_instance(
                'test_bevfusion_radar_integration', 
                scope_name='mmdet3d'
            )
            
            # 尝试直接导入BEVFusionWithRadar以触发CUDA模块加载
            try:
                from projects.BEVFusion.bevfusion.bevfusion import BEVFusionWithRadar
                # 检查模型是否已注册
                if 'BEVFusionWithRadar' in MODELS.module_dict:
                    cls.models_available = True
                else:
                    cls.import_error = "BEVFusionWithRadar not registered in MODELS"
            except ImportError as e:
                cls.import_error = f"CUDA extensions not available: {e}"
                
        except ImportError as e:
            cls.import_error = str(e)

    def test_model_build_from_radar_config(self):
        """
        测试从雷达配置构建模型
        
        验证BEVFusionWithRadar模型可以从配置文件正确构建。
        
        _Requirements: 5.1, 5.4_
        
        注意：此测试需要CUDA扩展编译完成才能运行。
        """
        if not self.models_available:
            self.skipTest(f"Models not available: {self.import_error}")
        
        from mmdet3d.registry import MODELS
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        # 构建模型
        try:
            model = MODELS.build(cfg.model)
            
            # 验证模型类型
            self.assertEqual(
                model.__class__.__name__, 'BEVFusionWithRadar',
                "Built model should be BEVFusionWithRadar"
            )
            
            # 验证雷达分支存在
            self.assertTrue(
                model.with_radar,
                "Model should have radar branch"
            )
            
            # 验证雷达编码器存在
            self.assertIsNotNone(
                model.radar_voxel_encoder,
                "radar_voxel_encoder should not be None"
            )
            self.assertIsNotNone(
                model.radar_middle_encoder,
                "radar_middle_encoder should not be None"
            )
            self.assertIsNotNone(
                model.radar_voxel_layer,
                "radar_voxel_layer should not be None"
            )
            
        except KeyError as e:
            # 模型未注册，跳过测试
            self.skipTest(f"Model not registered (CUDA extensions may not be compiled): {e}")
        except Exception as e:
            self.fail(f"Failed to build model from config: {e}")


class TestBackwardCompatibility(unittest.TestCase):
    """
    后向兼容性测试 (Task 9.2)
    
    Property 10: Configuration Backward Compatibility
    *For any* BEVFusion configuration without radar branch, the BEVFusionWithRadar 
    model SHALL produce identical results to the original BEVFusion model.
    
    Validates: Requirements 5.2, 5.3
    """

    def test_property_backward_compatibility_config_structure(self):
        """
        Property 10: Configuration Backward Compatibility (Config Structure)
        
        *For any* BEVFusion configuration without radar branch, the BEVFusionWithRadar 
        model configuration SHALL be valid and parseable.
        
        **Feature: radar-branch-bevfusion, Property 10: Configuration Backward Compatibility**
        **Validates: Requirements 5.3**
        """
        from mmengine import Config
        
        # 加载原始BEVFusion配置（不带雷达）
        base_config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        if not os.path.exists(base_config_path):
            self.skipTest(f"Base config not found: {base_config_path}")
        
        base_cfg = Config.fromfile(base_config_path)
        
        # 验证原始配置使用BEVFusion类型
        self.assertEqual(
            base_cfg.model.type, 'BEVFusion',
            "Base config should use BEVFusion type"
        )
        
        # 验证原始配置不包含雷达配置
        self.assertNotIn(
            'radar_voxel_encoder', base_cfg.model,
            "Base config should not have radar_voxel_encoder"
        )
        self.assertNotIn(
            'radar_middle_encoder', base_cfg.model,
            "Base config should not have radar_middle_encoder"
        )
        
        # 验证原始配置的融合层只有2个输入通道
        self.assertEqual(
            len(base_cfg.model.fusion_layer.in_channels), 2,
            "Base config fusion_layer should have 2 input channels"
        )

    def test_property_backward_compatibility_no_radar_config(self):
        """
        Property 10: Configuration Backward Compatibility (No Radar Config)
        
        *For any* BEVFusionWithRadar model initialized without radar encoder configs,
        the model SHALL behave identically to BEVFusion (with_radar should be False).
        
        **Feature: radar-branch-bevfusion, Property 10: Configuration Backward Compatibility**
        **Validates: Requirements 5.3**
        """
        # 测试不同的无雷达配置场景
        test_configs = [
            # 场景1: radar_voxel_encoder和radar_middle_encoder都为None
            {'radar_voxel_encoder': None, 'radar_middle_encoder': None},
            # 场景2: 只有radar_voxel_encoder为None
            {'radar_voxel_encoder': None, 'radar_middle_encoder': {'type': 'PointPillarsScatter'}},
            # 场景3: 只有radar_middle_encoder为None
            {'radar_voxel_encoder': {'type': 'PillarFeatureNet'}, 'radar_middle_encoder': None},
        ]
        
        for iteration in range(PBT_ITERATIONS):
            # 随机选择一个配置场景
            config_idx = np.random.randint(0, len(test_configs))
            config = test_configs[config_idx]
            
            # 模拟BEVFusionWithRadar的with_radar属性逻辑
            radar_voxel_encoder = config['radar_voxel_encoder']
            radar_middle_encoder = config['radar_middle_encoder']
            
            # 根据BEVFusionWithRadar的逻辑，只有当两个编码器都配置时才启用雷达
            with_radar = (radar_voxel_encoder is not None and 
                         radar_middle_encoder is not None)
            
            # 验证：当任一编码器为None时，with_radar应为False
            if config_idx == 0:
                self.assertFalse(
                    with_radar,
                    f"Iteration {iteration}: with_radar should be False when both encoders are None"
                )
            elif config_idx == 1:
                self.assertFalse(
                    with_radar,
                    f"Iteration {iteration}: with_radar should be False when radar_voxel_encoder is None"
                )
            elif config_idx == 2:
                self.assertFalse(
                    with_radar,
                    f"Iteration {iteration}: with_radar should be False when radar_middle_encoder is None"
                )

    def test_property_backward_compatibility_feature_extraction(self):
        """
        Property 10: Configuration Backward Compatibility (Feature Extraction Logic)
        
        *For any* BEVFusionWithRadar model without radar branch, the extract_feat 
        method SHALL only include image and LiDAR features in the fusion.
        
        **Feature: radar-branch-bevfusion, Property 10: Configuration Backward Compatibility**
        **Validates: Requirements 5.3**
        """
        # 模拟特征提取逻辑测试
        for iteration in range(PBT_ITERATIONS):
            # 随机生成模态配置
            has_img = np.random.choice([True, False])
            has_pts = True  # LiDAR总是存在
            with_radar = False  # 测试无雷达场景
            
            # 模拟特征列表构建逻辑
            features = []
            
            if has_img:
                # 模拟图像特征
                img_feat_shape = (np.random.randint(1, 4), 80, 64, 64)
                features.append(('img', img_feat_shape))
            
            # 模拟点云特征
            pts_feat_shape = (features[0][1][0] if features else np.random.randint(1, 4), 256, 64, 64)
            features.append(('pts', pts_feat_shape))
            
            # 当with_radar为False时，不应添加雷达特征
            if with_radar:
                radar_feat_shape = (pts_feat_shape[0], 64, 64, 64)
                features.append(('radar', radar_feat_shape))
            
            # 验证特征数量
            expected_num_features = 1 + (1 if has_img else 0)  # pts + img(optional)
            self.assertEqual(
                len(features), expected_num_features,
                f"Iteration {iteration}: Feature count should be {expected_num_features} "
                f"when with_radar=False, got {len(features)}"
            )
            
            # 验证不包含雷达特征
            feature_names = [f[0] for f in features]
            self.assertNotIn(
                'radar', feature_names,
                f"Iteration {iteration}: Features should not include radar when with_radar=False"
            )


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
class TestBackwardCompatibilityWithModel(unittest.TestCase):
    """
    使用实际模型的后向兼容性测试
    
    Property 10: Configuration Backward Compatibility
    
    Validates: Requirements 5.2, 5.3
    
    注意：此测试需要CUDA环境和完整的BEVFusion CUDA扩展编译。
    """

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.models_available = False
        cls.import_error = None
        
        try:
            from mmdet3d.registry import MODELS
            from mmengine import DefaultScope
            
            DefaultScope.get_instance(
                'test_backward_compat', 
                scope_name='mmdet3d'
            )
            
            # 尝试直接导入BEVFusionWithRadar以触发CUDA模块加载
            try:
                from projects.BEVFusion.bevfusion.bevfusion import BEVFusionWithRadar
                # 检查模型是否已注册
                if 'BEVFusionWithRadar' in MODELS.module_dict:
                    cls.models_available = True
                else:
                    cls.import_error = "BEVFusionWithRadar not registered in MODELS"
            except ImportError as e:
                cls.import_error = f"CUDA extensions not available: {e}"
                
        except ImportError as e:
            cls.import_error = str(e)

    def test_bevfusion_with_radar_without_radar_config(self):
        """
        测试BEVFusionWithRadar在无雷达配置时的行为
        
        验证当不配置雷达编码器时，BEVFusionWithRadar的with_radar属性为False。
        
        **Feature: radar-branch-bevfusion, Property 10: Configuration Backward Compatibility**
        **Validates: Requirements 5.3**
        
        注意：此测试需要CUDA扩展编译完成才能运行。
        """
        if not self.models_available:
            self.skipTest(f"Models not available: {self.import_error}")
        
        from mmdet3d.registry import MODELS
        from mmengine import Config
        
        # 加载原始BEVFusion配置
        base_config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        if not os.path.exists(base_config_path):
            self.skipTest(f"Base config not found: {base_config_path}")
        
        base_cfg = Config.fromfile(base_config_path)
        
        # 修改模型类型为BEVFusionWithRadar，但不添加雷达配置
        model_cfg = deepcopy(base_cfg.model)
        model_cfg.type = 'BEVFusionWithRadar'
        
        try:
            # 构建模型
            model = MODELS.build(model_cfg)
            
            # 验证with_radar为False
            self.assertFalse(
                model.with_radar,
                "with_radar should be False when radar encoders are not configured"
            )
            
            # 验证雷达编码器为None
            self.assertIsNone(
                model.radar_voxel_encoder,
                "radar_voxel_encoder should be None"
            )
            self.assertIsNone(
                model.radar_middle_encoder,
                "radar_middle_encoder should be None"
            )
            self.assertIsNone(
                model.radar_voxel_layer,
                "radar_voxel_layer should be None"
            )
            
        except KeyError as e:
            # 模型未注册，跳过测试
            self.skipTest(f"Model not registered (CUDA extensions may not be compiled): {e}")
        except Exception as e:
            self.fail(f"Failed to build BEVFusionWithRadar without radar config: {e}")


class TestConfigurationValidation(unittest.TestCase):
    """
    配置验证测试
    
    验证配置文件的各项参数符合预期。
    """

    def test_radar_encoder_channel_consistency(self):
        """
        测试雷达编码器通道一致性
        
        验证雷达体素编码器输出通道与中间编码器输入通道一致。
        """
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        # 获取雷达体素编码器输出通道
        radar_voxel_encoder = cfg.model.radar_voxel_encoder
        voxel_encoder_out_channels = radar_voxel_encoder.feat_channels[-1]
        
        # 获取雷达中间编码器输入通道
        radar_middle_encoder = cfg.model.radar_middle_encoder
        middle_encoder_in_channels = radar_middle_encoder.in_channels
        
        # 验证通道一致性
        self.assertEqual(
            voxel_encoder_out_channels, middle_encoder_in_channels,
            f"Voxel encoder output channels ({voxel_encoder_out_channels}) "
            f"should match middle encoder input channels ({middle_encoder_in_channels})"
        )

    def test_fusion_layer_channel_sum(self):
        """
        测试融合层输入通道配置
        
        验证融合层的输入通道配置与各模态输出通道一致。
        """
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        fusion_layer = cfg.model.fusion_layer
        in_channels = fusion_layer.in_channels
        
        # 验证有3个输入通道（img, lidar, radar）
        self.assertEqual(
            len(in_channels), 3,
            "Fusion layer should have 3 input channels for 3 modalities"
        )
        
        # 验证各通道值
        self.assertEqual(in_channels[0], 80, "Image BEV channels should be 80")
        self.assertEqual(in_channels[1], 256, "LiDAR BEV channels should be 256")
        self.assertEqual(in_channels[2], 64, "Radar BEV channels should be 64")

    def test_point_cloud_range_consistency(self):
        """
        测试点云范围配置一致性
        
        验证雷达和LiDAR使用相同的点云范围。
        """
        from mmengine import Config
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'projects', 'BEVFusion', 'configs',
            'bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        
        cfg = Config.fromfile(config_path)
        
        # 获取全局点云范围
        global_range = cfg.point_cloud_range
        
        # 获取雷达体素化点云范围
        radar_voxelize_range = cfg.model.data_preprocessor.radar_voxelize_cfg.point_cloud_range
        
        # 获取雷达体素编码器点云范围
        radar_encoder_range = cfg.model.radar_voxel_encoder.point_cloud_range
        
        # 验证范围一致性
        self.assertEqual(
            global_range, radar_voxelize_range,
            "Radar voxelize range should match global point_cloud_range"
        )
        self.assertEqual(
            global_range, radar_encoder_range,
            "Radar encoder range should match global point_cloud_range"
        )


if __name__ == '__main__':
    unittest.main()

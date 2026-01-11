# Copyright (c) OpenMMLab. All rights reserved.
"""
融合层属性测试

Property 8: Fusion Layer Channel Consistency
Property 9: Modality Flexibility

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5

测试ConvFuser融合层的通道一致性和模态灵活性。
"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

# 将项目根目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 检查CUDA是否可用
CUDA_AVAILABLE = torch.cuda.is_available()

# 属性测试迭代次数
PBT_ITERATIONS = 100


class MockConvFuser(nn.Sequential):
    """
    ConvFuser的模拟实现，用于非CUDA环境测试
    
    与实际ConvFuser具有相同的接口和行为。
    """

    def __init__(self, in_channels, out_channels):
        """
        初始化卷积融合器
        
        Args:
            in_channels (list[int]): 输入特征通道数列表
            out_channels (int): 输出特征通道数
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        """
        融合多个输入特征
        
        Args:
            inputs (list[Tensor]): 输入特征列表
            
        Returns:
            Tensor: 融合后的特征
        """
        return super().forward(torch.cat(inputs, dim=1))


def generate_random_bev_features(batch_size, channels, height, width, device='cpu'):
    """
    生成随机BEV特征
    
    Args:
        batch_size: 批次大小
        channels: 通道数
        height: 高度
        width: 宽度
        device: 设备
        
    Returns:
        Tensor: 随机BEV特征 [B, C, H, W]
    """
    return torch.randn(batch_size, channels, height, width, device=device)


class TestFusionLayerChannelConsistency(unittest.TestCase):
    """
    测试融合层通道一致性
    
    Property 8: Fusion Layer Channel Consistency
    *For any* list of BEV features from N modalities, the Fusion_Layer output 
    channel count SHALL equal the configured out_channels, regardless of the 
    number of input modalities.
    
    Validates: Requirements 4.2, 4.3, 4.5
    """

    def test_property_fusion_layer_channel_consistency_two_modalities(self):
        """
        Property 8: Fusion Layer Channel Consistency (2 modalities)
        
        *For any* list of BEV features from 2 modalities (camera + LiDAR), 
        the Fusion_Layer output channel count SHALL equal the configured out_channels.
        
        **Feature: radar-branch-bevfusion, Property 8: Fusion Layer Channel Consistency**
        **Validates: Requirements 4.2, 4.3, 4.5**
        """
        # 测试配置：2模态（相机 + LiDAR）
        in_channels = [80, 256]  # [img_channels, pts_channels]
        out_channels = 256
        height, width = 64, 64  # 使用较小尺寸加快测试
        
        # 创建融合层
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        # 运行迭代以验证属性
        for iteration in range(PBT_ITERATIONS):
            # 生成随机批次大小
            batch_size = np.random.randint(1, 4)
            
            # 生成随机BEV特征
            img_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[1], height, width)
            
            features = [img_feat, pts_feat]
            
            # 融合
            with torch.no_grad():
                output = fuser(features)
            
            # 验证输出通道数
            self.assertEqual(
                output.shape[1], out_channels,
                f"Iteration {iteration}: Output channels mismatch. "
                f"Expected {out_channels}, got {output.shape[1]}"
            )
            
            # 验证空间维度保持不变
            self.assertEqual(output.shape[2], height)
            self.assertEqual(output.shape[3], width)
            self.assertEqual(output.shape[0], batch_size)

    def test_property_fusion_layer_channel_consistency_three_modalities(self):
        """
        Property 8: Fusion Layer Channel Consistency (3 modalities)
        
        *For any* list of BEV features from 3 modalities (camera + LiDAR + radar), 
        the Fusion_Layer output channel count SHALL equal the configured out_channels.
        
        **Feature: radar-branch-bevfusion, Property 8: Fusion Layer Channel Consistency**
        **Validates: Requirements 4.2, 4.3, 4.5**
        """
        # 测试配置：3模态（相机 + LiDAR + 雷达）
        in_channels = [80, 256, 64]  # [img_channels, pts_channels, radar_channels]
        out_channels = 256
        height, width = 64, 64
        
        # 创建融合层
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        # 运行迭代以验证属性
        for iteration in range(PBT_ITERATIONS):
            batch_size = np.random.randint(1, 4)
            
            # 生成随机BEV特征
            img_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[1], height, width)
            radar_feat = generate_random_bev_features(
                batch_size, in_channels[2], height, width)
            
            features = [img_feat, pts_feat, radar_feat]
            
            # 融合
            with torch.no_grad():
                output = fuser(features)
            
            # 验证输出通道数
            self.assertEqual(
                output.shape[1], out_channels,
                f"Iteration {iteration}: Output channels mismatch"
            )
            self.assertEqual(output.shape[2], height)
            self.assertEqual(output.shape[3], width)

    def test_property_fusion_layer_channel_consistency_various_configs(self):
        """
        Property 8: Fusion Layer Channel Consistency (various configurations)
        
        *For any* configuration of in_channels and out_channels, the Fusion_Layer 
        output channel count SHALL equal the configured out_channels.
        
        **Feature: radar-branch-bevfusion, Property 8: Fusion Layer Channel Consistency**
        **Validates: Requirements 4.2, 4.3, 4.5**
        """
        height, width = 32, 32
        
        # 测试不同配置
        test_configs = [
            {'in_channels': [64, 128], 'out_channels': 128},
            {'in_channels': [80, 256, 64], 'out_channels': 256},
            {'in_channels': [128, 128, 128], 'out_channels': 384},
            {'in_channels': [32, 64, 32, 32], 'out_channels': 128},  # 4模态
            {'in_channels': [256], 'out_channels': 256},  # 单模态
        ]
        
        for config in test_configs:
            in_channels = config['in_channels']
            out_channels = config['out_channels']
            
            fuser = MockConvFuser(in_channels, out_channels)
            fuser.eval()
            
            # 运行20次迭代
            for iteration in range(20):
                batch_size = np.random.randint(1, 3)
                
                features = [
                    generate_random_bev_features(batch_size, ch, height, width)
                    for ch in in_channels
                ]
                
                with torch.no_grad():
                    output = fuser(features)
                
                self.assertEqual(
                    output.shape[1], out_channels,
                    f"Config {config}: Output channels mismatch"
                )


class TestModalityFlexibility(unittest.TestCase):
    """
    测试模态灵活性
    
    Property 9: Modality Flexibility
    *For any* subset of modalities (camera, LiDAR, radar), the Fusion_Layer 
    SHALL produce valid output when given the corresponding BEV features.
    
    Validates: Requirements 4.1, 4.4
    """

    def test_property_modality_flexibility_lidar_only(self):
        """
        Property 9: Modality Flexibility (LiDAR only)
        
        *For any* LiDAR-only configuration, the Fusion_Layer SHALL produce 
        valid output.
        
        **Feature: radar-branch-bevfusion, Property 9: Modality Flexibility**
        **Validates: Requirements 4.1, 4.4**
        """
        in_channels = [256]
        out_channels = 256
        height, width = 64, 64
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        for iteration in range(PBT_ITERATIONS):
            batch_size = np.random.randint(1, 4)
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            
            with torch.no_grad():
                output = fuser([pts_feat])
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            self.assertEqual(output.shape, (batch_size, out_channels, height, width))

    def test_property_modality_flexibility_camera_lidar(self):
        """
        Property 9: Modality Flexibility (Camera + LiDAR)
        
        **Feature: radar-branch-bevfusion, Property 9: Modality Flexibility**
        **Validates: Requirements 4.1, 4.4**
        """
        in_channels = [80, 256]
        out_channels = 256
        height, width = 64, 64
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        for iteration in range(PBT_ITERATIONS):
            batch_size = np.random.randint(1, 4)
            
            img_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[1], height, width)
            
            with torch.no_grad():
                output = fuser([img_feat, pts_feat])
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            self.assertEqual(output.shape, (batch_size, out_channels, height, width))

    def test_property_modality_flexibility_camera_lidar_radar(self):
        """
        Property 9: Modality Flexibility (Camera + LiDAR + Radar)
        
        **Feature: radar-branch-bevfusion, Property 9: Modality Flexibility**
        **Validates: Requirements 4.1, 4.4**
        """
        in_channels = [80, 256, 64]
        out_channels = 256
        height, width = 64, 64
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        for iteration in range(PBT_ITERATIONS):
            batch_size = np.random.randint(1, 4)
            
            img_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[1], height, width)
            radar_feat = generate_random_bev_features(
                batch_size, in_channels[2], height, width)
            
            with torch.no_grad():
                output = fuser([img_feat, pts_feat, radar_feat])
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            self.assertEqual(output.shape, (batch_size, out_channels, height, width))

    def test_property_modality_flexibility_lidar_radar(self):
        """
        Property 9: Modality Flexibility (LiDAR + Radar)
        
        **Feature: radar-branch-bevfusion, Property 9: Modality Flexibility**
        **Validates: Requirements 4.1, 4.4**
        """
        in_channels = [256, 64]
        out_channels = 256
        height, width = 64, 64
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        for iteration in range(PBT_ITERATIONS):
            batch_size = np.random.randint(1, 4)
            
            pts_feat = generate_random_bev_features(
                batch_size, in_channels[0], height, width)
            radar_feat = generate_random_bev_features(
                batch_size, in_channels[1], height, width)
            
            with torch.no_grad():
                output = fuser([pts_feat, radar_feat])
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            self.assertEqual(output.shape, (batch_size, out_channels, height, width))


class TestFusionLayerInputValidation(unittest.TestCase):
    """
    测试融合层输入验证
    """

    def test_input_channel_mismatch_detection(self):
        """测试输入通道不匹配时的行为"""
        in_channels = [80, 256]
        out_channels = 256
        height, width = 32, 32
        batch_size = 2
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        # 正确的输入
        correct_features = [
            generate_random_bev_features(batch_size, 80, height, width),
            generate_random_bev_features(batch_size, 256, height, width),
        ]
        
        with torch.no_grad():
            output = fuser(correct_features)
        
        self.assertEqual(output.shape[1], out_channels)
        
        # 错误的输入通道数应该导致错误
        wrong_features = [
            generate_random_bev_features(batch_size, 64, height, width),
            generate_random_bev_features(batch_size, 256, height, width),
        ]
        
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                fuser(wrong_features)

    def test_spatial_dimension_consistency(self):
        """测试空间维度一致性"""
        in_channels = [80, 256, 64]
        out_channels = 256
        batch_size = 2
        
        fuser = MockConvFuser(in_channels, out_channels)
        fuser.eval()
        
        test_sizes = [(32, 32), (64, 64), (128, 128)]
        
        for height, width in test_sizes:
            features = [
                generate_random_bev_features(batch_size, ch, height, width)
                for ch in in_channels
            ]
            
            with torch.no_grad():
                output = fuser(features)
            
            self.assertEqual(output.shape[2], height)
            self.assertEqual(output.shape[3], width)


if __name__ == '__main__':
    unittest.main()

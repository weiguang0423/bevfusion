"""
雷达速度编码器模块

本模块实现了几何感知的雷达速度BEV编码器，用于从雷达点云中提取
物理速度特征并投影到BEV空间。

主要组件：
- GeometryAwareVelocityEncoder: 几何感知的速度BEV编码器
- VelocityRefinementModule: 速度校准模块
- RadarVelocityBEVEncoder: GeometryAwareVelocityEncoder的别名（向后兼容）
- VelocityAwareFuser: VelocityRefinementModule的别名（向后兼容）

设计理念：
- 物理速度路径负责"物体开多快"（运动估计）
- 与语义特征路径解耦，直接供检测头查询
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeometryAwareVelocityEncoder(nn.Module):
    """
    几何感知的雷达速度BEV编码器
    
    Geometry-aware radar velocity BEV encoder.
    
    该模块直接从原始雷达点云中提取速度相关特征（不经过体素化），
    通过MLP融合后投影到BEV空间，生成4通道速度图。
    
    与语义路径（Voxelization -> PillarNet）不同，速度路径直接处理原始点，
    避免体素化平均导致的速度信息损失。
    
    特征归一化：
    - vx, vy: 除以 vel_scale (默认10.0)，将 [-50, 50] m/s 映射到 [-5, 5]
    - vx_rms, vy_rms: 除以 rms_scale (默认1.0)
    - sin_theta, cos_theta: 已经在 [-1, 1] 范围内，无需归一化
    
    聚合策略：基于RMS的置信度加权聚合
    - 置信度 = 1 / (vx_rms + vy_rms + eps)
    - 同一网格内多个点按置信度加权平均
    - 这样RMS小（测量更准确）的点权重更大
    
    输入特征：[vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms]
    输出：4通道速度图 [vx, vy, rms, confidence]
    
    Args:
        point_cloud_range (list): 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_size (tuple): BEV网格大小 (H, W)
        hidden_channels (list): MLP隐藏层通道数，默认 [32, 16]
        output_channels (int): 输出通道数，默认4 (vx, vy, rms, confidence)
        eps (float): 避免除零的小量，默认1e-6
        aggregation (str): 聚合方式，'weighted'(置信度加权) 或 'max'(取最大置信度)
        vel_scale (float): 速度归一化因子，默认10.0
        rms_scale (float): RMS归一化因子，默认1.0
    """

    def __init__(
        self,
        point_cloud_range: List[float],
        bev_size: Tuple[int, int] = (180, 180),
        hidden_channels: List[int] = [32, 16],
        output_channels: int = 4,
        eps: float = 1e-6,
        aggregation: str = 'weighted',
        vel_scale: float = 10.0,
        rms_scale: float = 1.0
    ) -> None:
        super().__init__()
        
        self.point_cloud_range = point_cloud_range
        self.bev_size = bev_size
        self.output_channels = output_channels
        self.eps = eps
        self.aggregation = aggregation
        self.vel_scale = vel_scale
        self.rms_scale = rms_scale
        
        # 计算BEV网格参数
        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]
        self.x_max = point_cloud_range[3]
        self.y_max = point_cloud_range[4]
        
        # 计算体素大小
        self.voxel_size_x = (self.x_max - self.x_min) / bev_size[1]  # W
        self.voxel_size_y = (self.y_max - self.y_min) / bev_size[0]  # H
        
        # 构建MLP网络
        # 输入：6维 [vx_norm, vy_norm, sin_theta, cos_theta, vx_rms_norm, vy_rms_norm]
        # 输出：4维 [vx, vy, rms, confidence]
        in_channels = 6
        layers = []
        prev_channels = in_channels
        
        for hidden_ch in hidden_channels:
            layers.append(nn.Linear(prev_channels, hidden_ch))
            layers.append(nn.BatchNorm1d(hidden_ch))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = hidden_ch
        
        # 输出层
        layers.append(nn.Linear(prev_channels, output_channels))
        
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        radar_points: List[torch.Tensor],
        batch_size: int
    ) -> torch.Tensor:
        """
        前向传播
        
        Forward pass.
        
        处理流程：
        1. 从雷达点云中提取坐标和速度
        2. 实时计算几何特征 (sin_theta, cos_theta)
        3. 通过MLP融合特征
        4. 计算BEV网格索引
        5. 使用scatter_add聚合到BEV空间
        6. 归一化置信度
        
        Args:
            radar_points (List[torch.Tensor]): 雷达点云列表
                支持两种格式：
                - 8维原始点云：[x, y, z, rcs, vx_comp, vy_comp, vx_rms, vy_rms]
                - 10维增强点云：[x, y, z, rcs, vx_comp, vy_comp, sin_theta, cos_theta, vx_rms, vy_rms]
                几何特征会在模型内部实时计算，不依赖预处理
            batch_size (int): 批次大小
            
        Returns:
            torch.Tensor: 速度BEV图，形状为 [B, 4, H, W]
                通道含义：[vx, vy, rms, confidence]
        """
        # 确定设备：遍历所有点云找到第一个非空的，否则使用MLP参数的设备
        device = None
        for points in radar_points:
            if len(points) > 0:
                device = points.device
                break
        
        if device is None:
            # 如果所有点云都为空，使用MLP参数的设备
            device = next(self.mlp.parameters()).device
        
        # 初始化输出
        H, W = self.bev_size
        velocity_bev = torch.zeros(
            batch_size, self.output_channels, H, W,
            device=device, dtype=torch.float32
        )
        
        # 用于置信度加权聚合的权重累加器
        weight_bev = torch.zeros(
            batch_size, 1, H, W,
            device=device, dtype=torch.float32
        )
        
        # 处理每个批次
        for batch_idx, points in enumerate(radar_points):
            if len(points) == 0:
                # 空点云，跳过
                continue
            
            # 确保点云在正确的设备上
            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, device=device, dtype=torch.float32)
            elif points.device != device:
                points = points.to(device)
            
            # 提取坐标
            x = points[:, 0]
            y = points[:, 1]
            
            # ============ 过滤异常点（防止NaN）============
            # 过滤掉坐标为(0,0)或包含NaN/Inf的点
            valid_points_mask = (
                torch.isfinite(x) & torch.isfinite(y) &
                ((x.abs() > 0.1) | (y.abs() > 0.1))  # 排除原点附近的点
            )
            
            if not valid_points_mask.any():
                continue
            
            # 只保留有效点
            points = points[valid_points_mask]
            x = points[:, 0]
            y = points[:, 1]
            
            # 实时计算几何特征 (sin_theta, cos_theta)
            # 这样就不受数据增强顺序影响
            # 使用更大的eps防止除零
            norm = torch.sqrt(x ** 2 + y ** 2).clamp(min=0.1)
            sin_theta = (y / norm).clamp(-1.0, 1.0)
            cos_theta = (x / norm).clamp(-1.0, 1.0)
            
            # 提取速度分量
            vx_comp = points[:, 4]
            vy_comp = points[:, 5]
            
            # 提取速度不确定度
            # 支持多种输入格式：
            # - 11维（RadarGeometryEnhancer增强后）：
            #   [x,y,z,rcs,vx_comp,vy_comp,sin_theta,cos_theta,vx_rms,vy_rms,dt]
            # - 9维（未增强，预处理后原始格式）：
            #   [x,y,z,rcs,vx_comp,vy_comp,vx_rms,vy_rms,dt]
            # - 8维：[x,y,z,rcs,vx_comp,vy_comp,vx_rms,vy_rms]
            dim = points.shape[1]
            if dim == 11:
                # 11维增强格式：sin_theta在6,cos_theta在7,vx_rms在8,vy_rms在9,dt在10
                vx_rms = points[:, 8]
                vy_rms = points[:, 9]
            elif dim == 10:
                # 10维旧格式：vx_rms在索引8，vy_rms在索引9
                vx_rms = points[:, 8]
                vy_rms = points[:, 9]
            elif dim >= 8:
                # 8维或9维未增强格式：vx_rms在索引6，vy_rms在索引7
                vx_rms = points[:, 6]
                vy_rms = points[:, 7]
            else:
                # 6维格式：使用默认值
                vx_rms = torch.full_like(vx_comp, 0.1)
                vy_rms = torch.full_like(vy_comp, 0.1)
            
            # 组合速度特征并归一化
            # vx, vy: 除以 vel_scale，将大范围速度映射到较小范围
            # vx_rms, vy_rms: 除以 rms_scale
            # sin_theta, cos_theta: 已在 [-1, 1]，无需归一化
            
            # 对速度值进行clamp，防止极端值
            vx_norm = (vx_comp / self.vel_scale).clamp(-10.0, 10.0)
            vy_norm = (vy_comp / self.vel_scale).clamp(-10.0, 10.0)
            vx_rms_norm = vx_rms.clamp(min=0.01) / self.rms_scale
            vy_rms_norm = vy_rms.clamp(min=0.01) / self.rms_scale
            
            velocity_features = torch.stack([
                vx_norm, vy_norm, sin_theta, cos_theta, vx_rms_norm, vy_rms_norm
            ], dim=-1)  # [N, 6]
            
            # 计算每个点的置信度权重（基于RMS，RMS越小置信度越高）
            # 使用clamp后的rms值，确保分母不为零
            point_confidence = 1.0 / (vx_rms_norm + vy_rms_norm + 0.1)  # [N]
            point_confidence = point_confidence.clamp(max=10.0)  # 限制最大置信度
            
            # 通过MLP融合
            # 输出：[vx, vy, rms, confidence]
            mlp_output = self.mlp(velocity_features)  # [N, 4]
            
            # 转换到网格坐标
            grid_x = ((x - self.x_min) / self.voxel_size_x).long()
            grid_y = ((y - self.y_min) / self.voxel_size_y).long()
            
            # 过滤超出范围的点
            valid_mask = (
                (grid_x >= 0) & (grid_x < W) &
                (grid_y >= 0) & (grid_y < H)
            )
            
            if not valid_mask.any():
                continue
            
            grid_x = grid_x[valid_mask]
            grid_y = grid_y[valid_mask]
            mlp_output = mlp_output[valid_mask]
            point_confidence = point_confidence[valid_mask]
            
            # 计算线性索引
            linear_idx = grid_y * W + grid_x  # [N_valid]
            
            # 使用scatter_add聚合到BEV空间（更稳定的实现）
            # 先在CPU上创建索引扩展，避免CUDA scatter问题
            linear_idx_expanded = linear_idx.unsqueeze(0).expand(self.output_channels, -1)  # [4, N_valid]
            
            # 加权特征
            weighted_features = mlp_output.t() * point_confidence.unsqueeze(0)  # [4, N_valid]
            
            # 聚合到BEV
            velocity_bev_flat = velocity_bev[batch_idx].view(self.output_channels, -1)  # [4, H*W]
            velocity_bev_flat.scatter_add_(1, linear_idx_expanded, weighted_features)
            
            # 聚合权重
            weight_bev_flat = weight_bev[batch_idx].view(1, -1)  # [1, H*W]
            weight_bev_flat.scatter_add_(1, linear_idx.unsqueeze(0), point_confidence.unsqueeze(0))
            
            # 更新回原始形状
            velocity_bev[batch_idx] = velocity_bev_flat.view(self.output_channels, H, W)
            weight_bev[batch_idx] = weight_bev_flat.view(1, H, W)
        
        # 归一化：除以权重和（加权平均）
        weight_bev = weight_bev.clamp(min=self.eps)
        velocity_bev = velocity_bev / weight_bev
        
        # 对于没有点的位置，置信度设为0
        no_points_mask = (weight_bev < self.eps * 2).squeeze(1)  # [B, H, W]
        velocity_bev[:, 3, :, :][no_points_mask] = 0.0
        
        return velocity_bev

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.point_cloud_range}, '
        repr_str += f'bev_size={self.bev_size}, '
        repr_str += f'output_channels={self.output_channels})'
        return repr_str



@MODELS.register_module()
class VelocityRefinementModule(nn.Module):
    """
    速度校准模块
    
    Velocity refinement module.
    
    该模块从物理速度图中采样速度值，并与网络预测的速度进行
    基于置信度的加权融合，实现速度校准。
    
    处理流程：
    1. 根据预测的物体中心点位置，从速度图中采样
    2. 获取采样位置的置信度
    3. 基于置信度进行加权融合：
       refined_vel = confidence * sampled_vel + (1 - confidence) * pred_vel
    
    Args:
        hidden_channel (int): 隐藏层通道数（未使用，保留接口兼容性）
        confidence_threshold (float): 置信度阈值，低于此值使用网络预测，默认0.1
    """

    def __init__(
        self,
        hidden_channel: int = 128,
        confidence_threshold: float = 0.1
    ) -> None:
        """
        初始化速度校准模块
        
        Args:
            hidden_channel: 隐藏层通道数
            confidence_threshold: 置信度阈值
        """
        super().__init__()
        self.hidden_channel = hidden_channel
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        pred_velocity: torch.Tensor,
        velocity_bev: torch.Tensor,
        query_pos: torch.Tensor,
        point_cloud_range: List[float],
        bev_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        前向传播：速度校准
        
        Forward pass: velocity refinement.
        
        处理流程：
        1. 将查询位置转换为BEV网格坐标
        2. 使用双线性插值从速度图中采样
        3. 基于置信度进行加权融合
        
        Args:
            pred_velocity (torch.Tensor): 网络预测的速度 [B, N, 2] 或 [N, 2]
                N是查询点数量，2是vx和vy
            velocity_bev (torch.Tensor): 物理速度图 [B, 4, H, W]
                通道含义：[vx, vy, rms, confidence]
            query_pos (torch.Tensor): 查询位置 [B, N, 2] 或 [N, 2]
                2是x和y坐标
            point_cloud_range (list): 点云范围
            bev_size (tuple): BEV网格大小 (H, W)
            
        Returns:
            torch.Tensor: 校准后的速度 [B, N, 2] 或 [N, 2]
        """
        # 处理输入维度
        input_is_2d = pred_velocity.dim() == 2
        if input_is_2d:
            pred_velocity = pred_velocity.unsqueeze(0)  # [1, N, 2]
            query_pos = query_pos.unsqueeze(0)  # [1, N, 2]
        
        batch_size, num_queries, _ = pred_velocity.shape
        device = pred_velocity.device
        
        # 提取点云范围参数
        x_min, y_min = point_cloud_range[0], point_cloud_range[1]
        x_max, y_max = point_cloud_range[3], point_cloud_range[4]
        H, W = bev_size
        
        # 将查询位置转换为归一化网格坐标 [-1, 1]
        # grid_sample 使用的坐标系：(-1, -1) 是左上角，(1, 1) 是右下角
        query_x = query_pos[:, :, 0]  # [B, N]
        query_y = query_pos[:, :, 1]  # [B, N]
        
        # 归一化到 [-1, 1]
        norm_x = 2.0 * (query_x - x_min) / (x_max - x_min) - 1.0
        norm_y = 2.0 * (query_y - y_min) / (y_max - y_min) - 1.0
        
        # 构建采样网格 [B, N, 1, 2]
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
        
        # 使用双线性插值从速度图中采样
        # velocity_bev: [B, 4, H, W]
        # grid: [B, N, 1, 2]
        # 输出: [B, 4, N, 1]
        sampled = F.grid_sample(
            velocity_bev,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # 调整形状 [B, 4, N, 1] -> [B, N, 4]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, 4]
        
        # 提取采样的速度和置信度
        sampled_vx = sampled[:, :, 0]  # [B, N]
        sampled_vy = sampled[:, :, 1]  # [B, N]
        # sampled_rms = sampled[:, :, 2]  # [B, N] - 未使用
        sampled_confidence = sampled[:, :, 3]  # [B, N]
        
        # 将置信度限制在 [0, 1] 范围内
        sampled_confidence = sampled_confidence.clamp(0.0, 1.0)
        
        # 对于置信度低于阈值的位置，使用网络预测
        low_confidence_mask = sampled_confidence < self.confidence_threshold
        sampled_confidence[low_confidence_mask] = 0.0
        
        # 基于置信度进行加权融合
        # refined_vel = confidence * sampled_vel + (1 - confidence) * pred_vel
        pred_vx = pred_velocity[:, :, 0]  # [B, N]
        pred_vy = pred_velocity[:, :, 1]  # [B, N]
        
        refined_vx = sampled_confidence * sampled_vx + (1.0 - sampled_confidence) * pred_vx
        refined_vy = sampled_confidence * sampled_vy + (1.0 - sampled_confidence) * pred_vy
        
        # 组合输出
        refined_velocity = torch.stack([refined_vx, refined_vy], dim=-1)  # [B, N, 2]
        
        # 恢复原始维度
        if input_is_2d:
            refined_velocity = refined_velocity.squeeze(0)  # [N, 2]
        
        return refined_velocity

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(hidden_channel={self.hidden_channel}, '
        repr_str += f'confidence_threshold={self.confidence_threshold})'
        return repr_str



# ============ 向后兼容别名 ============
# 为了保持与现有配置文件的兼容性，提供以下别名

@MODELS.register_module()
class RadarVelocityBEVEncoder(GeometryAwareVelocityEncoder):
    """
    RadarVelocityBEVEncoder 是 GeometryAwareVelocityEncoder 的别名
    
    保留此类以保持与现有配置文件的向后兼容性。
    新代码应使用 GeometryAwareVelocityEncoder。
    """
    pass


@MODELS.register_module()
class VelocityAwareFuser(VelocityRefinementModule):
    """
    VelocityAwareFuser 是 VelocityRefinementModule 的别名
    
    保留此类以保持与现有配置文件的向后兼容性。
    新代码应使用 VelocityRefinementModule。
    """
    pass

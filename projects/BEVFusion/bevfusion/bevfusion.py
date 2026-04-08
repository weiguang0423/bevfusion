"""
BEVFusion: 多模态3D目标检测模型

BEVFusion 是一种多传感器融合框架，将相机图像和激光雷达点云在鸟瞰图(BEV)空间中进行融合。
该方法的核心思想是：
1. 将多视角相机图像通过视角变换(View Transform)投影到BEV空间
2. 将激光雷达点云通过体素化(Voxelization)和稀疏卷积编码到BEV空间
3. 在BEV空间中融合两种模态的特征
4. 使用统一的检测头进行3D目标检测

主要组件：
- img_backbone: 图像特征提取骨干网络 (如 Swin Transformer)
- img_neck: 图像特征金字塔网络 (如 FPN)
- view_transform: 视角变换模块，将2D图像特征转换到BEV空间 (如 LSS)
- pts_voxel_encoder: 点云体素编码器
- pts_middle_encoder: 点云中间编码器 (稀疏卷积)
- fusion_layer: 多模态特征融合层
- pts_backbone: BEV特征骨干网络
- pts_neck: BEV特征金字塔网络
- bbox_head: 3D检测头 (如 TransFusion Head)

扩展模块 (BEVFusionWithRadar):
- radar_voxel_layer: 雷达点云体素化层
- radar_voxel_encoder: 雷达体素特征编码器 (PillarFeatureNet)
- radar_middle_encoder: 雷达中间编码器 (PointPillarsScatter)
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):
    """
    BEVFusion 多模态3D目标检测器
    
    该类实现了基于BEV空间的相机-激光雷达融合检测框架。
    继承自 Base3DDetector，遵循 MMDetection3D 的模型设计规范。
    
    数据流程：
    1. 图像分支: imgs -> img_backbone -> img_neck -> view_transform -> img_bev_feat
    2. 点云分支: points -> voxelize -> pts_voxel_encoder -> pts_middle_encoder -> pts_bev_feat
    3. 融合: [img_bev_feat, pts_bev_feat] -> fusion_layer -> fused_bev_feat
    4. 检测: fused_bev_feat -> pts_backbone -> pts_neck -> bbox_head -> predictions
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,  # 数据预处理配置
        pts_voxel_encoder: Optional[dict] = None,  # 点云体素编码器配置
        pts_middle_encoder: Optional[dict] = None,  # 点云中间编码器配置（稀疏卷积）
        fusion_layer: Optional[dict] = None,  # 多模态融合层配置
        img_backbone: Optional[dict] = None,  # 图像骨干网络配置
        pts_backbone: Optional[dict] = None,  # BEV骨干网络配置
        view_transform: Optional[dict] = None,  # 视角变换模块配置（2D->BEV）
        img_neck: Optional[dict] = None,  # 图像特征金字塔配置
        pts_neck: Optional[dict] = None,  # BEV特征金字塔配置
        bbox_head: Optional[dict] = None,  # 3D检测头配置
        init_cfg: OptMultiConfig = None,  # 权重初始化配置
        seg_head: Optional[dict] = None,  # 分割头配置（可选）
        **kwargs,
    ) -> None:
        """
        初始化 BEVFusion 模型
        
        构建流程：
        1. 从数据预处理配置中提取体素化配置
        2. 构建点云处理分支（体素化层 + 体素编码器 + 中间编码器）
        3. 构建图像处理分支（骨干网络 + Neck + 视角变换）
        4. 构建融合层和检测头
        """
        # 从数据预处理配置中分离出体素化配置
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ============ 点云体素化配置 ============
        # voxelize_reduce: 是否对体素内的点进行平均（减少计算量）
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        # 体素化层：将无序点云转换为规则的体素网格
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        # ============ 点云编码器 ============
        # 体素特征编码器：对每个体素内的点进行特征编码
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        # ============ 图像处理分支 ============
        # 图像骨干网络：提取多尺度图像特征（如 Swin Transformer, ResNet）
        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        # 图像Neck：特征金字塔网络，融合多尺度特征
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        # 视角变换：将2D图像特征投影到3D BEV空间（核心模块，如 LSS, BEVDepth）
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        
        # ============ 点云中间编码器 ============
        # 稀疏卷积编码器：处理体素化后的点云，输出BEV特征
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        # ============ 多模态融合层 ============
        # 融合图像BEV特征和点云BEV特征（如 ConvFuser）
        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        # ============ BEV特征处理 ============
        # BEV骨干网络：进一步提取融合后的BEV特征
        self.pts_backbone = MODELS.build(pts_backbone)
        # BEV Neck：BEV特征金字塔
        self.pts_neck = MODELS.build(pts_neck)

        # ============ 检测头 ============
        # 3D目标检测头（如 TransFusion Head, CenterHead）
        self.bbox_head = MODELS.build(bbox_head)

        # 初始化模型权重
        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """
        网络前向传播过程（抽象方法实现）
        
        通常包括骨干网络、Neck和检测头的前向传播，不包含后处理。
        此方法主要用于 ONNX 导出等场景。
        
        Args:
            batch_inputs: 批量输入张量
            batch_data_samples: 批量数据样本
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        解析网络输出的损失值
        
        将网络输出的原始损失字典转换为：
        1. 用于优化器的总损失张量（加权求和）
        2. 用于日志记录的损失字典
        
        Args:
            losses (dict): 网络输出的原始损失字典，通常包含多个损失项
                          如 {'loss_cls': tensor, 'loss_bbox': tensor, ...}

        Returns:
            tuple[Tensor, dict]: 
                - loss: 总损失张量，传递给优化器进行反向传播
                - log_vars: 日志变量字典，用于 TensorBoard 等可视化工具
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                # 单个张量损失：取平均值
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                # 张量列表损失：对所有张量求和后取平均
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # 计算总损失：只对名称中包含 'loss' 的项求和
        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        # 分布式训练时，对所有进程的损失进行平均
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                # 所有进程的损失求和后除以进程数，得到平均损失
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        """
        初始化模型权重
        
        仅对图像骨干网络进行权重初始化，其他模块使用默认初始化或预训练权重。
        这是因为图像骨干网络通常使用 ImageNet 预训练权重。
        """
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: 检测器是否有3D检测头"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: 检测器是否有分割头"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        depth_gt_indices=None,
        depth_valid_mask=None,
    ):
        """
        提取图像特征并转换到BEV空间
        
        这是图像分支的核心处理流程：
        1. 将多视角图像通过骨干网络提取特征
        2. 通过Neck进行多尺度特征融合
        3. 通过视角变换模块将2D特征投影到3D BEV空间
        
        Args:
            x: 多视角图像张量，形状为 [B, N, C, H, W]
               B=batch_size, N=相机数量, C=通道数, H=高度, W=宽度
            points: 点云数据，用于视角变换中的深度估计参考
            lidar2image: LiDAR到图像的变换矩阵
            camera_intrinsics: 相机内参矩阵
            camera2lidar: 相机到LiDAR的变换矩阵
            img_aug_matrix: 图像数据增强矩阵
            lidar_aug_matrix: LiDAR数据增强矩阵
            img_metas: 图像元信息
            depth_gt_indices: 深度GT索引 [B, N, fH, fW]（可选，训练时使用）
            depth_valid_mask: 有效掩码 [B, N, fH, fW]（可选，训练时使用）
            
        Returns:
            训练时（如果提供了depth_gt）: (bev_feat, aux_dict)
            推理时: bev_feat
        """
        # 获取输入维度: B=batch, N=相机数, C=通道, H=高, W=宽
        B, N, C, H, W = x.size()
        # 将batch和相机维度合并，便于批量处理: [B*N, C, H, W]
        x = x.view(B * N, C, H, W).contiguous()

        # 通过图像骨干网络提取特征（如 Swin Transformer）
        x = self.img_backbone(x)
        # 通过Neck进行多尺度特征融合（如 FPN）
        x = self.img_neck(x)

        # 如果Neck输出是元组/列表，取第一个特征图
        if not isinstance(x, torch.Tensor):
            x = x[0]

        # 恢复batch和相机维度: [B, N, C, H, W]
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # 视角变换：将2D图像特征投影到3D BEV空间
        # 使用float32精度以保证数值稳定性（深度估计对精度敏感）
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            result = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                depth_gt_indices=depth_gt_indices,
                depth_valid_mask=depth_valid_mask,
            )
        return result

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        """
        提取点云特征并转换到BEV空间
        
        这是点云分支的核心处理流程：
        1. 将点云体素化（Voxelization）
        2. 通过体素编码器对每个体素进行特征编码
        3. 通过中间编码器（稀疏卷积）生成BEV特征
        
        Args:
            batch_inputs_dict: 包含点云数据的字典
                - 'points': 点云列表，每个元素是一帧点云 [N, 4+]
                  其中 N 是点数，4+ 表示至少包含 (x, y, z, intensity)
        
        Returns:
            torch.Tensor: BEV空间的点云特征，形状为 [B, C, H_bev, W_bev]
        """
        points = batch_inputs_dict['points']
        # 关闭自动混合精度，使用float32保证体素化的数值精度
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            # 体素化：将点云转换为体素网格
            # feats: 体素特征, coords: 体素坐标, sizes: 每个体素内的点数
            feats, coords, sizes = self.voxelize(points)
            # 从坐标中获取batch_size（坐标第一列是batch索引）
            batch_size = coords[-1, 0] + 1
        # 通过中间编码器（稀疏卷积）处理体素特征，输出BEV特征
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        """
        点云体素化
        
        将无序的点云数据转换为规则的3D体素网格。这是点云处理的关键步骤，
        使得后续可以使用3D卷积或稀疏卷积进行特征提取。
        
        体素化过程：
        1. 根据点的坐标将其分配到对应的体素中
        2. 对每个体素内的点进行特征聚合（如平均、最大值等）
        3. 记录每个体素的坐标和包含的点数
        
        Args:
            points: 点云列表，每个元素是一帧点云张量 [N, 4+]
        
        Returns:
            tuple:
                - feats: 体素特征张量 [num_voxels, max_points, C] 或 [num_voxels, C]
                - coords: 体素坐标张量 [num_voxels, 4]，格式为 (batch_idx, z, y, x)
                - sizes: 每个体素内的点数 [num_voxels]
        
        Note:
            使用 @torch.no_grad() 装饰器，因为体素化是不可微的操作，
            不需要计算梯度，可以节省显存。
        """
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            # 对每帧点云进行体素化
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # 硬体素化 (hard voxelize): 返回特征、坐标、点数
                # 每个体素最多保留固定数量的点
                f, c, n = ret
            else:
                # 动态体素化: 只返回特征和坐标
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            # 在坐标前面添加batch索引，用于区分不同样本
            # F.pad(c, (1, 0)) 在坐标的最前面添加一列
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        # 将所有样本的体素特征和坐标拼接
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                # 体素特征聚合：对每个体素内的点特征求平均
                # feats.sum(dim=1): 对体素内所有点的特征求和
                # / sizes: 除以点数得到平均特征
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """
        推理/测试阶段的前向传播
        
        完整的推理流程：
        1. 提取多模态特征并融合
        2. 通过检测头进行预测
        3. 将预测结果添加到数据样本中
        
        Args:
            batch_inputs_dict (dict): 模型输入字典，包含：
                - 'points' (list[torch.Tensor]): 每个样本的点云数据
                - 'imgs' (torch.Tensor): 多视角图像，可选
            batch_data_samples (List[Det3DDataSample]): 数据样本列表，
                通常包含 gt_instance_3d 等标注信息

        Returns:
            list[Det3DDataSample]: 检测结果列表，每个样本包含 'pred_instances_3d'：
                - scores_3d (Tensor): 分类置信度，形状 (num_instances,)
                - labels_3d (Tensor): 类别标签，形状 (num_instances,)
                - bbox_3d (BaseInstance3DBoxes): 3D边界框预测，
                  包含形状为 (num_instances, 7) 的张量
                  7个参数: (x, y, z, w, l, h, yaw)
        """
        # 提取元信息（如相机参数、图像尺寸等）
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # 提取并融合多模态特征
        result = self.extract_feat(batch_inputs_dict, batch_input_metas)
        
        # 处理返回值：推理时应该只返回feats，但为了兼容性处理tuple情况
        if isinstance(result, tuple):
            feats = result[0]
        else:
            feats = result

        # 通过检测头进行预测
        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        # 将预测结果添加到数据样本中
        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        """
        提取并融合多模态特征（核心方法）
        
        这是 BEVFusion 的核心特征提取流程：
        1. 图像分支：提取图像特征并转换到BEV空间
        2. 点云分支：提取点云特征并转换到BEV空间
        3. 特征融合：在BEV空间融合两种模态的特征
        4. 特征增强：通过骨干网络和Neck进一步处理融合特征
        
        Args:
            batch_inputs_dict: 输入数据字典
                - 'imgs': 多视角图像 [B, N, C, H, W]
                - 'points': 点云列表
            batch_input_metas: 元信息列表，包含相机参数等
        
        Returns:
            训练时（如果有深度GT）: (融合后的BEV特征, aux_dict)
            推理时: 融合后的BEV特征
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        aux_dict = None
        
        # ============ 图像分支处理 ============
        if imgs is not None:
            imgs = imgs.contiguous()
            
            # 收集相机参数和数据增强矩阵
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                # LiDAR到图像的投影矩阵
                lidar2image.append(meta['lidar2img'])
                # 相机内参矩阵 (焦距、主点等)
                camera_intrinsics.append(meta['cam2img'])
                # 相机到LiDAR的变换矩阵
                camera2lidar.append(meta['cam2lidar'])
                # 图像数据增强矩阵（如翻转、缩放等）
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                # LiDAR数据增强矩阵
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            # 将numpy数组转换为与图像相同设备和类型的张量
            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            
            # 获取深度GT（如果存在）
            depth_gt_indices = batch_inputs_dict.get('depth_gt_indices', None)
            depth_valid_mask = batch_inputs_dict.get('depth_valid_mask', None)
            
            # 提取图像BEV特征
            # 使用 deepcopy(points) 避免修改原始点云数据
            img_result = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                depth_gt_indices=depth_gt_indices,
                depth_valid_mask=depth_valid_mask,
            )
            
            # 处理返回值：可能是 (bev_feat, aux_dict) 或 bev_feat
            if isinstance(img_result, tuple):
                img_feature, aux_dict = img_result
            else:
                img_feature = img_result
            
            features.append(img_feature)
        
        # ============ 点云分支处理 ============
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # ============ 多模态特征融合 ============
        if self.fusion_layer is not None:
            # 使用融合层融合图像和点云的BEV特征
            # 常见的融合方式：拼接后卷积、注意力机制等
            fusion_result = self.fusion_layer(features)
            if isinstance(fusion_result, tuple):
                x, evidence_aux = fusion_result
                if aux_dict is None:
                    aux_dict = {}
                aux_dict['evidence_aux'] = evidence_aux
            else:
                x = fusion_result
        else:
            # 如果没有融合层，只使用单一模态特征
            assert len(features) == 1, features
            x = features[0]

        # ============ BEV特征增强 ============
        # 通过BEV骨干网络进一步提取特征
        x = self.pts_backbone(x)
        # 通过BEV Neck进行多尺度特征融合
        x = self.pts_neck(x)

        # 如果有辅助字典，返回元组
        if aux_dict is not None:
            return x, aux_dict
        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        训练阶段的损失计算
        
        完整的训练流程：
        1. 提取多模态特征并融合
        2. 通过检测头计算损失
        3. 如果启用深度监督，计算深度监督损失
        
        Args:
            batch_inputs_dict (dict): 模型输入字典，包含：
                - 'points': 点云数据列表
                - 'imgs': 多视角图像（可选）
                - 'depth_gt_indices': 深度GT索引（可选）
                - 'depth_valid_mask': 有效掩码（可选）
            batch_data_samples (List[Det3DDataSample]): 数据样本列表，
                包含 gt_instance_3d 等标注信息用于计算损失
        
        Returns:
            dict: 损失字典，包含各项损失值
                - 'loss_cls': 分类损失
                - 'loss_bbox': 边界框回归损失
                - 'loss_heatmap': 热力图损失（如果使用 CenterHead）
                - 'loss_depth': 深度监督损失（如果启用深度监督）
                - 其他检测头特定的损失项
        """
        # 提取元信息
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # 提取并融合多模态特征
        result = self.extract_feat(batch_inputs_dict, batch_input_metas)
        
        # 处理返回值：可能是 (feats, aux_dict) 或 feats
        if isinstance(result, tuple):
            feats, aux_dict = result
        else:
            feats = result
            aux_dict = None

        losses = dict()
        # 通过检测头计算损失
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
            losses.update(bbox_loss)
        
        # 计算深度监督损失
        if aux_dict is not None and hasattr(self.view_transform, 'depth_loss'):
            depth_loss_module = self.view_transform.depth_loss
            if depth_loss_module is not None:
                depth_pred = aux_dict['depth_pred']
                depth_gt_indices = aux_dict['depth_gt_indices']
                valid_mask = aux_dict['valid_mask']
                
                # 计算深度损失
                loss_depth = depth_loss_module(
                    depth_pred, depth_gt_indices, valid_mask
                )
                losses['loss_depth'] = loss_depth

        return losses


@MODELS.register_module()
class BEVFusionWithRadar(BEVFusion):
    """
    支持雷达分支的BEVFusion模型
    
    BEVFusion with Radar Branch for Multi-Modal 3D Object Detection.
    
    在原有BEVFusion基础上添加雷达处理分支，实现相机-激光雷达-雷达三模态融合。
    雷达分支使用基于Pillar的编码方式，将稀疏的雷达点云转换为BEV特征。
    
    数据流程：
    1. 图像分支: imgs -> img_backbone -> img_neck -> view_transform -> img_bev_feat
    2. 点云分支: points -> voxelize -> pts_voxel_encoder -> pts_middle_encoder -> pts_bev_feat
    3. 雷达分支: radar_points -> radar_voxelize -> radar_voxel_encoder -> radar_middle_encoder -> radar_bev_feat
    4. 雷达速度分支: radar_points -> radar_velocity_bev (直接编码多普勒速度)
    5. 融合: [img_bev_feat, pts_bev_feat, radar_bev_feat] -> fusion_layer -> fused_bev_feat
    6. 检测: fused_bev_feat + radar_velocity_bev -> bbox_head -> predictions
    
    雷达分支组件：
    - radar_voxel_layer: 雷达点云体素化层 (复用Voxelization)
    - radar_voxel_encoder: 雷达体素特征编码器 (复用PillarFeatureNet)
    - radar_middle_encoder: 雷达中间编码器 (复用PointPillarsScatter)
    - radar_velocity_encoder: 雷达速度BEV编码器 (直接编码多普勒速度)
    
    Args:
        radar_voxel_encoder (dict, optional): 雷达体素编码器配置
            Configuration for radar voxel encoder (PillarFeatureNet).
            Defaults to None.
        radar_middle_encoder (dict, optional): 雷达中间编码器配置
            Configuration for radar middle encoder (PointPillarsScatter).
            Defaults to None.
        radar_velocity_encoder (dict, optional): 雷达速度BEV编码器配置
            Configuration for radar velocity BEV encoder.
            Defaults to None.
        其他参数同BEVFusion
    
    Note:
        - 当radar_voxel_encoder和radar_middle_encoder都为None时，
          模型行为与原BEVFusion完全一致（后向兼容）
        - 雷达点云为空时，返回零填充的BEV特征，保证模型正常运行
    """

    def __init__(
        self,
        radar_voxel_encoder: Optional[dict] = None,
        radar_middle_encoder: Optional[dict] = None,
        radar_velocity_encoder: Optional[dict] = None,
        velocity_refinement: Optional[dict] = None,
        enable_evidence_loss: bool = True,
        evi_loss_weight: float = 0.1,
        evi_lambda_bg: float = 0.05,
        evi_pos_weight: float = 10.0,
        evi_kl_weight: float = 0.1,
        evi_warmup_iters: int = 1000,
        evi_kl_annealing_epochs: int = 6,
        **kwargs,
    ) -> None:
        """
        初始化 BEVFusionWithRadar 模型
        
        Initialize BEVFusionWithRadar model.
        
        构建流程：
        1. 从data_preprocessor中提取雷达体素化配置（如果存在）
        2. 调用父类初始化（构建图像和点云分支）
        3. 构建雷达处理分支（体素化层 + 体素编码器 + 中间编码器）
        4. 构建雷达速度BEV编码器（直接编码多普勒速度）
        5. 构建速度校准模块（基于置信度的加权融合）
        
        Args:
            radar_voxel_encoder (dict, optional): 雷达体素编码器配置
            radar_middle_encoder (dict, optional): 雷达中间编码器配置
            radar_velocity_encoder (dict, optional): 雷达速度BEV编码器配置
            velocity_refinement (dict, optional): 速度校准模块配置
            **kwargs: 传递给父类BEVFusion的其他参数
        """
        # 从data_preprocessor中提取雷达体素化配置
        radar_voxelize_cfg = None
        if 'data_preprocessor' in kwargs and kwargs['data_preprocessor'] is not None:
            data_preprocessor = kwargs['data_preprocessor']
            if 'radar_voxelize_cfg' in data_preprocessor:
                radar_voxelize_cfg = data_preprocessor.pop('radar_voxelize_cfg')
        
        # 调用父类初始化
        super().__init__(**kwargs)

        # ============ 证据监督配置（分支级） ============
        self.enable_evidence_loss = enable_evidence_loss
        self.evi_loss_weight = evi_loss_weight
        self.evi_lambda_bg = evi_lambda_bg
        self.evi_pos_weight = evi_pos_weight
        self.evi_kl_weight = evi_kl_weight
        self.evi_warmup_iters = evi_warmup_iters
        self.evi_kl_annealing_epochs = evi_kl_annealing_epochs
        # 持久化 evidence warmup 计数，避免每次 resume 都重新 warmup。
        self.register_buffer(
            '_evi_step_buffer', torch.zeros((), dtype=torch.long), persistent=True)
        self._evi_epoch = 0
        self._evidence_aux = None
        
        # ============ 雷达分支初始化 ============
        self.radar_voxel_encoder = None
        self.radar_middle_encoder = None
        self.radar_voxel_layer = None
        self.radar_velocity_encoder = None
        self.velocity_refinement = None
        
        # 只有当配置了雷达编码器时才构建雷达分支
        if radar_voxel_encoder is not None and radar_middle_encoder is not None:
            # 构建雷达体素化层
            if radar_voxelize_cfg is not None:
                # 雷达体素化配置中可能包含voxelize_reduce参数
                self.radar_voxelize_reduce = radar_voxelize_cfg.pop('voxelize_reduce', True)
                self.radar_voxel_layer = Voxelization(**radar_voxelize_cfg)
            else:
                # 如果没有单独的雷达体素化配置，使用默认配置
                self.radar_voxelize_reduce = True
                self.radar_voxel_layer = Voxelization(
                    voxel_size=[0.5, 0.5, 8.0],
                    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                    max_num_points=10,
                    max_voxels=(30000, 40000),
                )
            
            # 构建雷达体素编码器 (PillarFeatureNet)
            self.radar_voxel_encoder = MODELS.build(radar_voxel_encoder)
            
            # 构建雷达中间编码器 (PointPillarsScatter)
            self.radar_middle_encoder = MODELS.build(radar_middle_encoder)
        
        # 构建雷达速度BEV编码器（可选，用于直接利用多普勒速度）
        if radar_velocity_encoder is not None:
            self.radar_velocity_encoder = MODELS.build(radar_velocity_encoder)
        
        # 构建速度校准模块（可选，用于基于置信度的速度融合）
        if velocity_refinement is not None:
            self.velocity_refinement = MODELS.build(velocity_refinement)
    
    @property
    def with_radar(self) -> bool:
        """bool: 检测器是否有雷达分支"""
        return (self.radar_voxel_encoder is not None and 
                self.radar_middle_encoder is not None and
                self.radar_voxel_layer is not None)
    
    @property
    def with_radar_velocity(self) -> bool:
        """bool: 检测器是否有雷达速度BEV编码器"""
        return self.radar_velocity_encoder is not None
    
    @property
    def with_velocity_refinement(self) -> bool:
        """bool: 检测器是否有速度校准模块"""
        return self.velocity_refinement is not None

    @property
    def _evi_step(self) -> int:
        return int(self._evi_step_buffer.item())

    @_evi_step.setter
    def _evi_step(self, value: int) -> None:
        self._evi_step_buffer.fill_(int(value))

    def sync_resume_state(self, runner_iter: int) -> Dict[str, int]:
        """在旧 checkpoint 缺失自定义 step 状态时，用全局 iter 回填。"""
        synced = {}
        runner_iter = max(int(runner_iter), 0)
        if runner_iter <= 0:
            return synced

        if self.evi_warmup_iters > 0 and self._evi_step < runner_iter:
            self._evi_step = runner_iter
            synced['evi_step'] = self._evi_step

        fusion_layer = getattr(self, 'fusion_layer', None)
        if fusion_layer is not None and hasattr(fusion_layer, 'sync_resume_state'):
            synced.update(fusion_layer.sync_resume_state(runner_iter))

        return synced

    def _infer_batch_size(self, batch_inputs_dict: Dict) -> int:
        """根据输入字典推断 batch_size。"""
        if 'points' in batch_inputs_dict:
            return len(batch_inputs_dict['points'])
        if 'imgs' in batch_inputs_dict:
            return batch_inputs_dict['imgs'].shape[0]
        return 1

    def _build_empty_radar_velocity_bev(self, batch_size: int, device) -> Tensor:
        """构建空雷达速度 BEV 图，保持输出形状稳定。"""
        out_channels = getattr(self.radar_velocity_encoder, 'output_channels', 4)
        return torch.zeros(
            batch_size, out_channels, 180, 180,
            dtype=torch.float32, device=device
        )

    def _build_empty_radar_feat(self, batch_size: int, device) -> Tensor:
        """构建空雷达语义 BEV 特征，保持融合输入维度稳定。"""
        out_channels = self.radar_middle_encoder.in_channels
        ny = self.radar_middle_encoder.ny
        nx = self.radar_middle_encoder.nx
        return torch.zeros(
            batch_size, out_channels, ny, nx,
            dtype=torch.float32, device=device
        )

    def _inject_bbox_head_runtime_context(self) -> None:
        """向检测头注入雷达运行时上下文。"""
        if hasattr(self.bbox_head, 'set_radar_velocity_bev'):
            self.bbox_head.set_radar_velocity_bev(self._radar_velocity_bev)

        if hasattr(self.bbox_head, 'set_bev_world_meta'):
            pcr = self._get_point_cloud_range()
            cfg = self.bbox_head.train_cfg if self.bbox_head.train_cfg is not None else self.bbox_head.test_cfg
            voxel_size = cfg.get('voxel_size', [0.075, 0.075, 0.2]) if cfg is not None else [0.075, 0.075, 0.2]
            out_size_factor = cfg.get('out_size_factor', 8) if cfg is not None else 8
            self.bbox_head.set_bev_world_meta(pcr, voxel_size, out_size_factor)

        if hasattr(self.bbox_head, 'set_velocity_refinement_module'):
            self.bbox_head.set_velocity_refinement_module(
                self.velocity_refinement if self.with_velocity_refinement else None)
    
    def extract_radar_velocity_bev(self, batch_inputs_dict: Dict) -> Optional[Tensor]:
        """
        提取雷达速度BEV图
        
        直接将雷达点云的多普勒速度编码为BEV速度图，
        供检测头直接查询使用。
        
        Args:
            batch_inputs_dict (dict): 包含雷达点云的输入字典
        
        Returns:
            Tensor or None: 雷达速度BEV图 [B, 3, H, W]
                - [:, 0:2, :, :]: vx, vy 速度分量
                - [:, 2, :, :]: 置信度
        """
        if not self.with_radar_velocity:
            return None
        
        radar_points = batch_inputs_dict.get('radar_points', None)
        
        batch_size = self._infer_batch_size(batch_inputs_dict)
        
        if radar_points is None or len(radar_points) == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return self._build_empty_radar_velocity_bev(batch_size, device)
        
        return self.radar_velocity_encoder(radar_points, batch_size)
    
    @torch.no_grad()
    def radar_voxelize(self, radar_points: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        雷达点云体素化
        
        Voxelize radar point clouds.
        
        将无序的雷达点云数据转换为规则的3D体素网格（Pillar）。
        由于雷达点云比激光雷达更稀疏，使用较大的体素尺寸。
        
        Args:
            radar_points (List[Tensor]): 雷达点云列表，每个元素是一帧雷达点云
                List of radar point clouds, each element is a frame of radar points.
                Shape: [N, C] where N is number of points, C is feature dimension.
        
        Returns:
            tuple:
                - feats (Tensor): 体素特征张量
                    Voxel features. Shape: [num_voxels, max_points, C] or [num_voxels, C]
                - coords (Tensor): 体素坐标张量
                    Voxel coordinates. Shape: [num_voxels, 4], format: (batch_idx, z, y, x)
                - sizes (Tensor): 每个体素内的点数
                    Number of points in each voxel. Shape: [num_voxels]
        
        Note:
            使用 @torch.no_grad() 装饰器，因为体素化是不可微的操作，
            不需要计算梯度，可以节省显存。
        """
        feats, coords, sizes = [], [], []
        for k, res in enumerate(radar_points):
            # 对每帧雷达点云进行体素化
            ret = self.radar_voxel_layer(res)
            if len(ret) == 3:
                # 硬体素化 (hard voxelize): 返回特征、坐标、点数
                f, c, n = ret
            else:
                # 动态体素化: 只返回特征和坐标
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            # 在坐标前面添加batch索引
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        # 将所有样本的体素特征和坐标拼接
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.radar_voxelize_reduce:
                # 体素特征聚合：对每个体素内的点特征求平均
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes
    
    def extract_radar_feat(self, batch_inputs_dict: Dict) -> Optional[Tensor]:
        """
        提取雷达BEV特征
        
        Extract radar BEV features from radar point clouds.
        
        处理流程：
        1. 获取雷达点云数据
        2. 体素化（Pillar化）
        3. 通过PillarFeatureNet编码体素特征
        4. 通过PointPillarsScatter散布到BEV空间
        
        Args:
            batch_inputs_dict (dict): 包含雷达点云的输入字典
                Input dict containing radar points.
                - 'radar_points' (list[Tensor]): 雷达点云列表
        
        Returns:
            Tensor or None: 雷达BEV特征
                Radar BEV features. Shape: [B, C, H, W]
                如果没有雷达分支或雷达点云为空，返回零填充的BEV特征
        
        Note:
            - 当雷达点云为空时，返回形状正确的零填充BEV特征，保证模型正常运行
            - 复用mmdetection3d中的PillarFeatureNet和PointPillarsScatter模块
        """
        if not self.with_radar:
            return None
        
        radar_points = batch_inputs_dict.get('radar_points', None)
        
        batch_size = self._infer_batch_size(batch_inputs_dict)
        
        if radar_points is None or len(radar_points) == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return self._build_empty_radar_feat(batch_size, device)
        
        # 关闭自动混合精度，使用float32保证体素化的数值精度
        with torch.autocast('cuda', enabled=False):
            radar_points = [point.float() for point in radar_points]
            
            # 检查是否所有雷达点云都为空
            total_points = sum(p.shape[0] for p in radar_points)
            
            if total_points == 0:
                batch_size = len(radar_points)
                device = radar_points[0].device if len(radar_points) > 0 else 'cuda'
                return self._build_empty_radar_feat(batch_size, device)
            
            # 体素化雷达点云
            feats, coords, sizes = self.radar_voxelize(radar_points)
            
            # 使用前面推断的batch_size，不要从coords获取，防止最后一个点云为空时出错
            # batch_size = coords[-1, 0].item() + 1
        
        # 通过PillarFeatureNet编码体素特征
        # PillarFeatureNet输入: (voxels, num_points, coors)
        # 输出: pillar_features [num_voxels, out_channels]
        pillar_features = self.radar_voxel_encoder(feats, sizes, coords)
        
        # 通过PointPillarsScatter散布到BEV空间
        # 输入: (voxel_features, coors, batch_size)
        # 输出: BEV特征 [B, C, H, W]
        radar_bev_feat = self.radar_middle_encoder(pillar_features, coords, batch_size)
        
        # === 雷达区域概率扩张 (Radius Dilation) ===
        # 使用 3x3 avg_pool 把单像素扩张为面特征
        # avg_pool 保留激活幅度的相对大小，避免 max_pool 将所有非零区域拉到同一峰值
        # kernel=3 对应 0.6m×3=1.8m，匹配雷达典型测距误差
        import torch.nn.functional as F
        if radar_bev_feat is not None:
            radar_bev_feat = F.avg_pool2d(
                radar_bev_feat, 
                kernel_size=3,
                stride=1, 
                padding=1
            )
        # ============================================
        
        return radar_bev_feat
    
    def extract_feat(
        self,
        batch_inputs_dict: Dict,
        batch_input_metas: List[Dict],
        **kwargs,
    ) -> Tensor:
        """
        提取并融合多模态特征（重写父类方法）
        
        Extract and fuse multi-modal features (override parent method).
        
        在原有图像和激光雷达特征基础上，添加雷达特征提取和融合。
        
        处理流程：
        1. 图像分支：提取图像特征并转换到BEV空间
        2. 点云分支：提取点云特征并转换到BEV空间
        3. 雷达分支：提取雷达特征并转换到BEV空间（新增）
        4. 特征融合：在BEV空间融合所有模态的特征
        5. 特征增强：通过骨干网络和Neck进一步处理融合特征
        
        Args:
            batch_inputs_dict (dict): 输入数据字典
                - 'imgs': 多视角图像 [B, N, C, H, W]
                - 'points': 点云列表
                - 'radar_points': 雷达点云列表（新增）
            batch_input_metas (list[dict]): 元信息列表，包含相机参数等
        
        Returns:
            Tensor: 融合后的BEV特征，用于检测头
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        aux_dict = {}
        
        # ============ 图像分支处理 ============
        if imgs is not None:
            imgs = imgs.contiguous()
            
            # 收集相机参数和数据增强矩阵
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            # 将numpy数组转换为与图像相同设备和类型的张量
            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            
            # 提取图像BEV特征
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            if isinstance(img_feature, tuple):
                img_feature, img_aux = img_feature
                if img_aux is not None:
                    aux_dict.update(img_aux)
            features.append(img_feature)
        
        # ============ 点云分支处理 ============
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)
        
        # ============ 雷达分支处理（新增）============
        if self.with_radar:
            radar_feature = self.extract_radar_feat(batch_inputs_dict)
            # 即使雷达特征为空（零填充），也要添加到features中
            # 这样融合层的输入通道数才能匹配
            features.append(radar_feature)
        
        # ============ 雷达速度BEV图（新增）============
        # 直接编码雷达多普勒速度，供检测头查询
        if self.with_radar_velocity:
            self._radar_velocity_bev = self.extract_radar_velocity_bev(batch_inputs_dict)
        else:
            self._radar_velocity_bev = None

        # ============ 多模态特征融合 ============
        if self.fusion_layer is not None:
            fusion_result = self.fusion_layer(features)
            if isinstance(fusion_result, tuple):
                x, evidence_aux = fusion_result
                aux_dict['evidence_aux'] = evidence_aux
            else:
                x = fusion_result
        else:
            assert len(features) == 1, features
            x = features[0]

        # ============ BEV特征增强 ============
        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        if aux_dict:
            return x, aux_dict
        return x

    def _get_point_cloud_range(self) -> List[float]:
        """获取 BEV 目标构建所需的点云范围。"""
        default_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        if hasattr(self.bbox_head, 'train_cfg') and self.bbox_head.train_cfg is not None:
            pcr = self.bbox_head.train_cfg.get('point_cloud_range', None)
            if pcr is not None:
                return list(pcr)

        if hasattr(self.pts_voxel_layer, 'point_cloud_range'):
            pcr = self.pts_voxel_layer.point_cloud_range
            if isinstance(pcr, (list, tuple)):
                return list(pcr)
            if hasattr(pcr, 'tolist'):
                return list(pcr.tolist())

        return default_range

    def _build_bev_occ_target(self, batch_data_samples: List[Det3DDataSample],
                              H: int, W: int, device: torch.device,
                              dtype: torch.dtype) -> Tensor:
        """将 GT 3D 框投影到 BEV 网格，生成占用监督目标。"""
        point_cloud_range = self._get_point_cloud_range()
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)

        target = torch.zeros(
            (len(batch_data_samples), 1, H, W),
            device=device,
            dtype=dtype,
        )

        for b, sample in enumerate(batch_data_samples):
            if not hasattr(sample, 'gt_instances_3d'):
                continue
            gt_instances_3d = sample.gt_instances_3d
            if not hasattr(gt_instances_3d, 'bboxes_3d'):
                continue

            boxes_tensor = gt_instances_3d.bboxes_3d.tensor
            if boxes_tensor.numel() == 0:
                continue

            centers_x = boxes_tensor[:, 0]
            centers_y = boxes_tensor[:, 1]
            widths = boxes_tensor[:, 3]
            lengths = boxes_tensor[:, 4]

            x_center = ((centers_x - x_min) / x_span * W).long()
            y_center = ((centers_y - y_min) / y_span * H).long()
            half_w = torch.clamp((widths / x_span * W * 0.5).long(), min=1)
            half_l = torch.clamp((lengths / y_span * H * 0.5).long(), min=1)

            for i in range(boxes_tensor.shape[0]):
                xc = x_center[i].item()
                yc = y_center[i].item()
                if xc < 0 or xc >= W or yc < 0 or yc >= H:
                    continue
                x0 = max(0, xc - half_w[i].item())
                x1 = min(W, xc + half_w[i].item() + 1)
                y0 = max(0, yc - half_l[i].item())
                y1 = min(H, yc + half_l[i].item() + 1)
                target[b, 0, y0:y1, x0:x1] = 1.0

        return target

    def _compute_evidential_losses(self, evidence_aux: Dict,
                                   batch_data_samples: List[Det3DDataSample]) -> Dict[str, Tensor]:
        """计算三分支证据损失与统计量。
        
        相比原始实现的改进：
        1. 前景/背景平衡 MSE：用 evi_pos_weight 对前景像素加权，
           解决 97:3 的极端类别不平衡问题
        2. KL 散度正则化：惩罚"预测错误但证据很高"的情况，
           迫使网络在不确定时输出低证据值
        3. KL Annealing：KL 项权重随训练进度逐渐增大，
           前期让网络自由学习特征，后期收紧约束
        4. Modality Dropout 感知：被 dropout 的分支不计算拟合损失，
           但仍计算 KL 损失（强迫被致盲的分支输出低证据）
        """
        losses = {}
        evidence_maps = evidence_aux.get('evidence_maps', [])
        belief_maps = evidence_aux.get('belief_maps', [])
        uncertainty_maps = evidence_aux.get('uncertainty_maps', [])
        fusion_weights = evidence_aux.get('fusion_weights', [])
        dropped_branch = evidence_aux.get('dropped_branch', None)

        branch_names = ['img', 'lidar', 'radar']
        if len(evidence_maps) < 3 or len(belief_maps) < 3:
            return losses

        H, W = evidence_maps[0].shape[-2:]
        target_occ = self._build_bev_occ_target(
            batch_data_samples, H, W,
            evidence_maps[0].device, evidence_maps[0].dtype,
        )
        losses['mean_occ_ratio'] = target_occ.detach().mean()

        # ---- 全局 Warmup（控制整体证据损失的权重） ----
        self._evi_step += 1
        if self.evi_warmup_iters > 0:
            warmup = min(1.0, float(self._evi_step) / float(self.evi_warmup_iters))
        else:
            warmup = 1.0

        # ---- KL Annealing（KL 项逐 epoch 增强） ----
        if self.evi_kl_annealing_epochs > 0:
            kl_anneal = min(1.0, float(self._evi_epoch) / float(self.evi_kl_annealing_epochs))
        else:
            kl_anneal = 1.0

        # ---- 预计算前景/背景 mask ----
        pos_mask = (target_occ > 0.5).float()
        neg_mask = 1.0 - pos_mask
        pos_count = pos_mask.sum().clamp(min=1.0)
        neg_count = neg_mask.sum().clamp(min=1.0)

        evi_total = torch.tensor(0.0, device=evidence_maps[0].device,
                                 dtype=evidence_maps[0].dtype)

        for idx, name in enumerate(branch_names):
            e_map = evidence_maps[idx]
            b_map = belief_maps[idx]

            is_dropped = (dropped_branch is not None and dropped_branch == idx)

            # ==== 1. 前景/背景平衡 MSE Loss ====
            # 前景像素少（~3%），需要更大的权重才能与背景平衡
            if is_dropped:
                # 被 dropout 的分支：不计算拟合损失
                # 其特征全为零，evidence 应该自然趋近于零
                loss_fit = torch.tensor(0.0, device=e_map.device, dtype=e_map.dtype)
            else:
                sq_err = (b_map - target_occ) ** 2
                loss_pos = (sq_err * pos_mask).sum() / pos_count
                loss_neg = (sq_err * neg_mask).sum() / neg_count
                loss_fit = loss_pos * self.evi_pos_weight + loss_neg

            # ==== 2. 背景证据抑制 ====
            # 在背景区域应抑制证据值，防止虚假自信
            loss_bg = (e_map * neg_mask).sum() / neg_count

            # ==== 3. KL 散度正则化 ====
            # 当 GT=0（背景）但 evidence 高时惩罚
            # 当 GT=1（前景）但 evidence 低时由 MSE 处理
            # 公式简化：KL[Dir(alpha_tilde) || Dir(1)]
            #   对于被 dropout 的分支，额外强制其 evidence→0
            if is_dropped:
                # 被致盲分支的全部 evidence 应为 0
                loss_kl = e_map.mean()
            else:
                # 只惩罚背景区域的错误自信
                loss_kl = (e_map * neg_mask).sum() / neg_count

            branch_loss = (loss_fit
                           + self.evi_lambda_bg * loss_bg
                           + self.evi_kl_weight * kl_anneal * loss_kl)

            # 注意：键名不能包含 'loss'，否则 parse_losses 会将其计入总损失，
            # 导致与 loss_evi_total 双重计数。这里仅用于日志记录。
            losses[f'evi_raw_{name}'] = branch_loss.detach()
            evi_total = evi_total + branch_loss

            # 可解释统计：不参与梯度
            losses[f'mean_e_{name}'] = e_map.detach().mean()
            losses[f'mean_b_{name}'] = b_map.detach().mean()
            if idx < len(uncertainty_maps):
                losses[f'mean_u_{name}'] = uncertainty_maps[idx].detach().mean()
            if idx < len(fusion_weights):
                losses[f'mean_w_{name}'] = fusion_weights[idx].detach().mean()

        if len(uncertainty_maps) > 0:
            all_u = torch.stack([u.detach().mean() for u in uncertainty_maps])
            losses['mean_u_all'] = all_u.mean()

        losses['loss_evi_total'] = self.evi_loss_weight * warmup * evi_total
        losses['evi_warmup'] = torch.tensor(
            warmup, device=evidence_maps[0].device, dtype=evidence_maps[0].dtype)
        losses['evi_kl_anneal'] = torch.tensor(
            kl_anneal, device=evidence_maps[0].device, dtype=evidence_maps[0].dtype)

        # 记录融合渐进恢复的混合比例
        fusion_blend = evidence_aux.get('fusion_blend', 1.0)
        losses['fusion_blend'] = torch.tensor(
            fusion_blend, device=evidence_maps[0].device,
            dtype=evidence_maps[0].dtype)
        for stat_name in ['radar_gate', 'radar_warmup', 'radar_effective_gate']:
            if stat_name in evidence_aux:
                losses[stat_name] = torch.tensor(
                    float(evidence_aux[stat_name]),
                    device=evidence_maps[0].device,
                    dtype=evidence_maps[0].dtype)
        return losses
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """
        推理/测试阶段的前向传播（重写父类方法）
        
        Args:
            batch_inputs_dict (dict): 模型输入字典
            batch_data_samples (List[Det3DDataSample]): 数据样本列表

        Returns:
            list[Det3DDataSample]: 检测结果列表
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        result = self.extract_feat(batch_inputs_dict, batch_input_metas)
        if isinstance(result, tuple):
            feats = result[0]
        else:
            feats = result

        if self.with_bbox_head:
            self._inject_bbox_head_runtime_context()
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res
    
    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        训练阶段的损失计算（重写父类方法）
        
        Args:
            batch_inputs_dict (dict): 模型输入字典
            batch_data_samples (List[Det3DDataSample]): 数据样本列表
        
        Returns:
            dict: 损失字典
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        result = self.extract_feat(batch_inputs_dict, batch_input_metas)
        if isinstance(result, tuple):
            feats, aux_dict = result
        else:
            feats = result
            aux_dict = {}

        losses = dict()
        if self.with_bbox_head:
            self._inject_bbox_head_runtime_context()
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        if self.enable_evidence_loss and 'evidence_aux' in aux_dict:
            evidence_loss = self._compute_evidential_losses(
                aux_dict['evidence_aux'], batch_data_samples)
            losses.update(evidence_loss)

        return losses

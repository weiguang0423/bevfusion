"""
TransFusion 检测头

TransFusion 是一种基于 Transformer 的3D目标检测头，
结合了基于中心点的检测和基于query的检测方法。

核心思想：
1. 从BEV特征图预测密集热力图
2. 从热力图中选择top-K个候选位置作为初始query
3. 通过Transformer解码器迭代细化query
4. 每层解码器输出检测结果

主要组件：
- ConvFuser: 多模态特征融合
- TransFusionHead: 主检测头
  - heatmap_head: 热力图预测
  - decoder: Transformer解码器
  - prediction_heads: 多层预测头

参考论文：
TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with 
Transformers (CVPR 2022)

modify from https://github.com/mit-han-lab/bevfusion
"""
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr


def clip_sigmoid(x, eps=1e-4):
    """
    带裁剪的sigmoid函数
    
    将sigmoid输出限制在 [eps, 1-eps] 范围内，
    避免数值不稳定和梯度消失。
    
    Args:
        x: 输入张量
        eps: 裁剪阈值
        
    Returns:
        裁剪后的sigmoid输出
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@MODELS.register_module()
class ConvFuser(nn.Sequential):
    """
    卷积融合器
    
    通过简单的卷积操作融合多模态特征。
    
    融合策略：拼接 -> 3x3卷积 -> BN -> ReLU
    
    Args:
        in_channels (list[int]): 输入特征通道数列表
        out_channels (int): 输出特征通道数
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化卷积融合器
        
        Args:
            in_channels: 输入通道数列表（将被求和）
            out_channels: 输出通道数
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多个输入特征
        
        Args:
            inputs: 输入特征列表
            
        Returns:
            融合后的特征
        """
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class TransFusionHead(nn.Module):
    """
    TransFusion 检测头
    
    基于Transformer的3D目标检测头，结合了：
    - 密集预测：热力图预测目标中心
    - 稀疏细化：Transformer解码器细化检测结果
    
    工作流程：
    1. 通过热力图预测目标中心位置
    2. 选择top-K个候选作为初始object query
    3. 通过多层Transformer解码器细化query
    4. 每层输出检测结果（支持辅助监督）
    
    Args:
        num_proposals: object query数量，默认128
        auxiliary: 是否使用辅助监督，默认True
        in_channels: 输入特征通道数
        hidden_channel: 隐藏层通道数
        num_classes: 类别数量
        num_decoder_layers: 解码器层数
        decoder_layer: 解码器层配置
        num_heads: 注意力头数
        nms_kernel_size: NMS核大小
        bn_momentum: BN动量
        common_heads: 公共预测头配置
        num_heatmap_convs: 热力图卷积层数
        conv_cfg: 卷积配置
        norm_cfg: 归一化配置
        bias: 偏置配置
        loss_cls: 分类损失配置
        loss_bbox: 边界框损失配置
        loss_heatmap: 热力图损失配置
        train_cfg: 训练配置
        test_cfg: 测试配置
        bbox_coder: 边界框编码器配置
    """

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        """
        初始化TransFusion检测头
        
        构建组件：
        1. 共享卷积层：降维
        2. 热力图预测头：预测目标中心
        3. 类别编码层：将类别one-hot编码为特征
        4. Transformer解码器：多层细化
        5. 预测头：每层输出检测结果
        """
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # 共享卷积：降维
        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        # 热力图预测头：预测目标中心位置
        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        
        # 类别编码：将类别one-hot向量编码为特征
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # Transformer解码器：多层迭代细化
        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # 预测头：每层解码器对应一个预测头
        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))

        self.init_weights()
        self._init_assigner_sampler()

        # BEV位置编码：用于交叉注意力，训练时重用
        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        """
        创建2D网格位置编码
        
        生成BEV空间的网格坐标，用于位置编码。
        
        Args:
            x_size: x方向网格数
            y_size: y方向网格数
            
        Returns:
            网格坐标 [1, x_size*y_size, 2]
        """
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5  # 网格中心
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        """初始化权重"""
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        """初始化BN层的动量"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """初始化目标分配器和采样器"""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, metas):
        """
        单层前向传播（TransFusion的核心）
        
        处理流程：
        1. 通过共享卷积降维
        2. 预测密集热力图
        3. 从热力图中选择top-K候选作为初始query
        4. 添加类别编码到query
        5. 通过多层Transformer解码器迭代细化
        6. 每层输出检测结果
        
        Args:
            inputs (torch.Tensor): 输入BEV特征图
                形状 [B, 512, 128(H), 128(W)]
        
        Returns:
            list[dict]: 输出结果字典列表
        
        Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        # 共享卷积降维
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        # 展平BEV特征: [BS, C, H*W]
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        # BEV位置编码
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization (查询初始化)
        #################################
        # 预测密集热力图
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        
        # NMS：抑制非极大值
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        
        # 特殊处理：对行人和交通锥使用1x1核（保留所有候选）
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        # 只保留局部最大值
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # 选择top num_proposals个候选
        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]  # 类别
        top_proposals_index = top_proposals % heatmap.shape[-1]  # 位置索引
        
        # 从BEV特征中提取对应位置的特征作为初始query
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # 添加类别编码
        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        # 提取query的位置
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        
        #################################
        # transformer decoder layer (Fusion feature as K,V)
        # Transformer解码器层（融合特征作为K和V）
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # 预测
            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            # 中心位置 = 预测偏移 + query位置
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # 更新下一层的位置编码
            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        # 记录query的热力图分数
        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # 只返回最后一层的结果
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # 返回所有层的结果用于辅助监督
        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                # 拼接所有层的预测
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        """
        前向传播
        
        Args:
            feats (list[torch.Tensor]): 多尺度特征（通常只有一层）
        
        Returns:
            tuple(list[dict]): 输出结果
        
        Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, batch_input_metas):
        """
        预测接口
        
        Args:
            batch_feats: 批量特征
            batch_input_metas: 批量元信息
            
        Returns:
            预测结果
        """
        preds_dicts = self(batch_feats, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """
        根据预测特征生成边界框
        
        处理流程：
        1. 解码预测结果为3D边界框
        2. 应用NMS抑制重复检测
        3. 封装为InstanceData格式
        
        Args:
            preds_dicts (tuple[list[dict]]): 预测结果
            metas: 元信息
            img: 图像（未使用）
            rescale: 是否缩放到原始尺寸
            for_roi: 是否用于ROI
        
        Returns:
            list[list[dict]]: 每层每个batch的解码后的边界框、分数和标签
        
        Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            # 获取分类分数
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            
            # 应用类别掩码
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            # 提取边界框参数
            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            # 解码为真实世界的边界框
            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            # 配置不同数据集的NMS任务
            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,  # 不使用NMS
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,  # 行人使用小半径NMS
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,  # 交通锥使用小半径NMS
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                
                # 对不同类别采用circle NMS
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        # 为每个任务创建类别掩码
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        
                        if task['radius'] > 0:
                            # 应用NMS
                            if self.test_cfg['nms_type'] == 'circle':
                                # Circle NMS：基于中心点距离
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                # BEV NMS：基于BEV IoU
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            # 不使用NMS，保留所有检测
                            task_keep_indices = torch.arange(task_mask.sum())
                        
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # 不使用NMS
                    # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                # 封装为InstanceData
                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """
        生成训练目标
        
        处理流程：
        1. 将预测字典按batch重组
        2. 对每个样本调用get_targets_single
        3. 汇总所有样本的目标
        
        Args:
            batch_gt_instances_3d (List[InstanceData]): 真值实例列表
            preds_dict (list[dict]): 预测结果字典，索引为层ID
                内部字典包含一个mini-batch的预测：
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: 输入transformer解码器前的原始热力图
                    形状 (bs, 10, h, w)
        
        Returns:
            tuple[torch.Tensor]: 目标元组，按顺序包含：
                - torch.Tensor: 分类目标 [BS, num_proposals]
                - torch.Tensor: 分类权重（掩码）[BS, num_proposals]
                - torch.Tensor: 回归目标 [BS, num_proposals, 8]
                - torch.Tensor: 回归权重 [BS, num_proposals, 8]
                - torch.Tensor: IoU目标 [BS, num_proposals]
                - int: 正样本数量
                - float: 平均匹配IoU
                - torch.Tensor: 热力图目标
        
        Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # 将预测字典按batch_id重组
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        # 对每个样本生成目标
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        # 汇总结果
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """
        为单个样本生成训练目标
        
        处理流程：
        1. 目标分配：将预测框与真值框匹配
        2. 采样：选择正负样本
        3. 创建目标：生成分类和回归目标
        4. 生成密集热力图目标
        
        Args:
            gt_instances_3d (:obj:`InstanceData`): 真值实例
            preds_dict (dict): 单个样本的预测结果字典
            batch_idx: batch索引
        
        Returns:
            tuple[torch.Tensor]: 目标元组，按顺序包含：
                - torch.Tensor: 分类目标 [1, num_proposals]
                - torch.Tensor: 分类权重（掩码）[1, num_proposals]
                - torch.Tensor: 回归目标 [1, num_proposals, 8]
                - torch.Tensor: 回归权重 [1, num_proposals, 8]
                - torch.Tensor: IoU目标 [1, num_proposals]
                - int: 正样本数量
                - float: 平均匹配IoU
                - torch.Tensor: 热力图目标
        
        Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # ============ 1. 目标分配 ============
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # 获取预测框，注意不要改变网络输出
        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        # 解码预测为真实世界的边界框
        # decode the prediction to real world metric bbox
        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        
        # 每层应该单独进行标签分配
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            # 提取当前层的预测
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            # 根据分配器类型进行目标分配
            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                # 匈牙利分配：最优二分匹配
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                # 启发式分配：基于距离和IoU
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # 合并所有层的分配结果
        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # ============ 2. 采样 ============
        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds  # 正样本索引
        neg_inds = sampling_result.neg_inds  # 负样本索引
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # ============ 3. 创建损失计算目标 ============
        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # 默认标签是-1
            # default label is -1
            labels += self.num_classes

        # 正负样本都有分类损失，只有正样本有回归和IoU损失
        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            # 编码正样本的真值边界框
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            
            # 设置正样本权重
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # ============ 计算密集热力图目标 ============
        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        # 初始化热力图
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        
        # 为每个真值框绘制高斯热力图
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            # 转换到特征图尺度
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            
            if width > 0 and length > 0:
                # 计算高斯半径
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                # 转换到特征图坐标
                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # 绘制高斯热力图
                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix - 修正坐标顺序
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        # 计算平均IoU
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, batch_data_samples):
        """
        计算损失
        
        Args:
            batch_feats: 批量特征
            batch_data_samples (List[:obj:`Det3DDataSample`]): 数据样本
                通常包含 gt_instance_3d 等标注信息
        
        Returns:
            dict[str:torch.Tensor]: 各项损失
        
        Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        """
        根据预测特征计算损失
        
        损失包括：
        1. loss_heatmap: 密集热力图损失（Focal Loss）
        2. loss_cls: 分类损失（Focal Loss）
        3. loss_bbox: 边界框回归损失（L1 Loss）
        
        Args:
            preds_dicts: 预测结果字典
            batch_gt_instances_3d: 真值实例列表
            
        Returns:
            dict: 损失字典
        """
        # 获取训练目标
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        
        # 如果有图像掩码，应用到权重上
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # 计算热力图损失
        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # 计算每层的损失
        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            # 确定层前缀
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            # 提取当前层的标签和预测
            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            
            # 分类损失
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            # 提取边界框预测
            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            # 拼接预测: [center, height, dim, rot]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            
            # 如果有速度预测，也拼接上
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            
            # 边界框回归损失
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        # 记录匹配的IoU
        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

"""
BEVFusion Neck 模块

本模块实现了用于BEV特征金字塔的Neck网络。

GeneralizedLSSFPN 是一个通用的特征金字塔网络，
采用自顶向下的特征融合策略，将多尺度特征融合为统一的BEV表示。

modify from https://github.com/mit-han-lab/bevfusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule):
    """
    通用LSS特征金字塔网络
    
    采用自顶向下的特征融合策略，将多尺度特征逐层融合。
    
    融合流程：
    1. 从最高层（最小分辨率）开始
    2. 上采样高层特征
    3. 与当前层特征拼接
    4. 通过1x1卷积降维
    5. 通过3x3卷积细化特征
    6. 重复直到最低层（最大分辨率）
    
    与标准FPN的区别：
    - 使用拼接(concat)而非相加(add)进行特征融合
    - 先拼接再卷积，保留更多信息
    
    Args:
        in_channels (list[int]): 输入特征的通道数列表
        out_channels (int): 输出特征的通道数
        num_outs (int): 输出特征层数
        start_level (int): 起始层索引
        end_level (int): 结束层索引，-1表示使用所有层
        no_norm_on_lateral (bool): 侧边卷积是否使用归一化
        conv_cfg (dict): 卷积层配置
        norm_cfg (dict): 归一化层配置
        act_cfg (dict): 激活函数配置
        upsample_cfg (dict): 上采样配置
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
    ) -> None:
        """
        初始化GeneralizedLSSFPN
        
        构建侧边卷积(lateral_convs)和FPN卷积(fpn_convs)。
        """
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        # 侧边卷积：用于特征融合
        self.lateral_convs = nn.ModuleList()
        # FPN卷积：用于特征细化
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # 侧边卷积输入通道数 = 当前层通道 + 上层通道(或out_channels)
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,  # 1x1卷积降维
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            # FPN卷积：3x3卷积细化特征
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """
        前向传播：自顶向下的特征融合
        
        融合策略：
        1. 上采样高层特征到当前层分辨率
        2. 拼接高层特征和当前层特征
        3. 通过1x1卷积融合
        4. 通过3x3卷积细化
        
        Args:
            inputs (list[Tensor]): 多尺度输入特征
            
        Returns:
            tuple[Tensor]: 融合后的多尺度特征
        """
        # 提取需要的层级
        assert len(inputs) == len(self.in_channels)

        # 构建侧边特征
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # 自顶向下融合
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            # 上采样高层特征
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            # 拼接当前层和上采样的高层特征
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            # 1x1卷积融合
            laterals[i] = self.lateral_convs[i](laterals[i])
            # 3x3卷积细化
            laterals[i] = self.fpn_convs[i](laterals[i])

        # 构建输出
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)

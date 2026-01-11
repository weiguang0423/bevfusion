"""
BEVFusion 稀疏卷积编码器

本模块实现了用于点云处理的稀疏卷积编码器。
与标准的 SparseEncoder 的主要区别是3D卷积的形状顺序：
- BEVFusionSparseEncoder: (H, W, D) 
- SparseEncoder: (D, H, W)

这个差异来自于体素化实现的不同。

稀疏卷积的优势：
- 只对非空体素进行计算，大幅减少计算量
- 保留点云的稀疏性，内存效率高
- 适合处理大范围3D场景

Copyright (c) OpenMMLab. All rights reserved.
"""
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    """
    BEVFusion 稀疏编码器
    
    用于将体素化的点云特征编码为BEV特征图。
    
    网络结构：
    1. conv_input: 输入卷积层，将体素特征编码为base_channels
    2. encoder_layers: 多层稀疏卷积编码器，逐步提取特征
    3. conv_out: 输出卷积层，沿Z轴下采样并输出BEV特征
    
    与标准 SparseEncoder 的区别：
    - 3D卷积的形状顺序为 (H, W, D) 而非 (D, H, W)
    - 这个差异源于 voxelization 的实现方式
    
    Args:
        in_channels: 输入体素特征通道数
        sparse_shape: 稀疏张量的形状 [H, W, D]
        order: 卷积模块的顺序，如 ('conv', 'norm', 'act')
        norm_cfg: 归一化层配置
        base_channels: 基础通道数
        output_channels: 输出通道数
        encoder_channels: 每个编码块的通道数配置
        encoder_paddings: 每个编码块的padding配置
        block_type: 块类型 ('conv_module' 或 'basicblock')
        return_middle_feats: 是否返回中间特征
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 return_middle_feats=False):
        super(SparseEncoder, self).__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    def forward(self, voxel_features, coors, batch_size):
        """
        稀疏编码器前向传播
        
        处理流程：
        1. 创建稀疏张量
        2. 通过输入卷积层
        3. 通过多层编码器
        4. 通过输出卷积层并沿Z轴下采样
        5. 转换为密集BEV特征图
        
        Args:
            voxel_features (torch.Tensor): 体素特征，形状 (N, C)
                N 是非空体素数量，C 是特征维度
            coors (torch.Tensor): 体素坐标，形状 (N, 4)
                每行格式为 (batch_idx, z_idx, y_idx, x_idx)
            batch_size (int): 批量大小

        Returns:
            torch.Tensor | tuple: 返回空间特征
                - spatial_features (torch.Tensor): BEV空间特征
                  形状为 [B, C*D, H, W]，其中D是Z方向压缩后的层数
                - encode_features (List[SparseConvTensor], optional): 
                  中间层特征（当 return_middle_feats=True 时）
        """
        # 坐标转换为整数类型
        coors = coors.int()
        # 创建稀疏张量
        # sparse_shape: [H, W, D] - BEVFusion的坐标顺序
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        # 输入卷积
        x = self.conv_input(input_sp_tensor)

        # 多层稀疏卷积编码
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # 输出卷积：沿Z轴下采样
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        # 转换为密集张量
        spatial_features = out.dense()

        # 调整维度顺序：[N, C, H, W, D] -> [N, C, D, H, W] -> [N, C*D, H, W]
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        # 沿Z轴展平，得到BEV特征
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features

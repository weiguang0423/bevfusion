"""
BEVFusion Transformer 解码器

本模块实现了用于 TransFusion Head 的 Transformer 解码器层。

主要特点：
- 使用学习的位置编码
- 自注意力和交叉注意力机制
- 与标准DETR解码器的区别：value编码了位置信息

Copyright (c) OpenMMLab. All rights reserved.
"""
from mmdet.models import DetrTransformerDecoderLayer
from torch import Tensor, nn

from mmdet3d.registry import MODELS


class PositionEncodingLearned(nn.Module):
    """
    学习的绝对位置编码
    
    通过可学习的卷积网络将位置坐标编码为高维特征。
    
    Args:
        input_channel: 输入坐标维度（如2D位置为2，3D位置为3）
        num_pos_feats: 位置编码的特征维度
    
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        """
        初始化位置编码网络
        
        网络结构：Conv1d -> BN -> ReLU -> Conv1d
        """
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """
        前向传播
        
        Args:
            xyz: 位置坐标 [B, N, C]
            
        Returns:
            位置编码 [B, num_pos_feats, N]
        """
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@MODELS.register_module()
class TransformerDecoderLayer(DetrTransformerDecoderLayer):
    """
    Transformer 解码器层
    
    用于 TransFusion Head 的解码器层，包含：
    1. 自注意力 (Self-Attention): query之间的交互
    2. 交叉注意力 (Cross-Attention): query与BEV特征的交互
    3. 前馈网络 (FFN): 特征变换
    
    与标准DETR解码器的区别：
    - value被编码了位置信息 (value = feature + pos_encoding)
    - 这使得注意力机制能够同时关注特征和位置
    
    Args:
        pos_encoding_cfg: 位置编码配置
            - input_channel: 输入坐标维度
            - num_pos_feats: 位置编码维度
        **kwargs: 其他DETR解码器参数
    """

    def __init__(self,
                 pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 **kwargs):
        """
        初始化Transformer解码器层
        
        Args:
            pos_encoding_cfg: 位置编码配置
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(**kwargs)
        # 自注意力的位置编码
        self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        # 交叉注意力的位置编码
        self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Transformer解码器层前向传播
        
        处理流程：
        1. 自注意力：query之间相互交互
        2. 交叉注意力：query与BEV特征交互
        3. 前馈网络：特征变换
        
        关键设计：
        - 自注意力的value = query + query_pos（编码位置信息）
        - 交叉注意力的value = key + key_pos（编码位置信息）
        - 这与标准DETR不同，使注意力能同时关注特征和位置
        
        Args:
            query (Tensor): 输入query，形状 (bs, num_queries, dim)
                在TransFusion中是object query
            key (Tensor, optional): 输入key，形状 (bs, num_keys, dim)
                在TransFusion中是BEV特征。如果为None，使用query
            value (Tensor, optional): 输入value，与key形状相同
                如果为None，使用key
            query_pos (Tensor, optional): query的位置编码
                在TransFusion中是query的BEV位置
            key_pos (Tensor, optional): key的位置编码
                在TransFusion中是BEV网格位置
            self_attn_mask (Tensor, optional): 自注意力掩码
            cross_attn_mask (Tensor, optional): 交叉注意力掩码
            key_padding_mask (Tensor, optional): key的padding掩码

        Returns:
            Tensor: 输出特征，形状 (bs, num_queries, dim)
        
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # 编码位置信息
        if self.self_posembed is not None and query_pos is not None:
            query_pos = self.self_posembed(query_pos).transpose(1, 2)
        else:
            query_pos = None
        if self.cross_posembed is not None and key_pos is not None:
            key_pos = self.cross_posembed(key_pos).transpose(1, 2)
        else:
            key_pos = None
        
        # 调整维度顺序以适配注意力模块
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        
        # 自注意力：query之间的交互
        # Note: value被编码了位置信息 (query + query_pos)
        # 这与标准DETR解码器层不同
        # Note that the `value` (equal to `query`) is encoded with `query_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.self_attn(
            query=query,
            key=query,
            value=query + query_pos,  # 关键：value包含位置信息
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        
        # 交叉注意力：query与BEV特征的交互
        # Note: value被编码了位置信息 (key + key_pos)
        # 这与标准DETR解码器层不同
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.cross_attn(
            query=query,
            key=key,
            value=key + key_pos,  # 关键：value包含位置信息
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        
        # 前馈网络
        query = self.ffn(query)
        query = self.norms[2](query)

        # 恢复维度顺序
        query = query.transpose(1, 2)
        return query

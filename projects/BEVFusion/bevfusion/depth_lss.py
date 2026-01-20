"""
Depth LSS (Lift-Splat-Shoot) 视角变换模块

本模块实现了将2D图像特征转换到3D BEV空间的核心算法。
LSS 是一种经典的视角变换方法，其核心思想是：
1. Lift: 将2D图像特征"提升"到3D空间，通过预测每个像素的深度分布
2. Splat: 将3D特征"溅射"到BEV网格中，通过体素池化操作
3. Shoot: 在BEV空间中进行后续处理（如检测、分割）

主要类：
- BaseViewTransform: 视角变换基类，定义了基本的几何计算和BEV池化
- LSSTransform: 标准LSS实现，使用网络预测深度分布
- BaseDepthTransform: 带深度监督的视角变换基类
- DepthLSSTransform: 使用LiDAR稀疏深度增强的LSS实现

参考论文：
- Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly 
  Unprojecting to 3D (ECCV 2020)
- BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View 
  Representation (ICRA 2023)

modify from https://github.com/mit-han-lab/bevfusion
"""
from typing import Tuple

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    """
    生成BEV网格的参数
    
    根据x、y、z三个方向的边界配置，计算BEV网格的：
    - 体素大小 (dx)
    - 体素中心起始位置 (bx)
    - 各方向的体素数量 (nx)
    
    Args:
        xbound: x方向边界配置 (min, max, resolution)
        ybound: y方向边界配置 (min, max, resolution)
        zbound: z方向边界配置 (min, max, resolution)
        
    Returns:
        dx: 各方向的体素大小 [dx, dy, dz]
        bx: 各方向第一个体素的中心坐标 [bx, by, bz]
        nx: 各方向的体素数量 [nx, ny, nz]
        
    Example:
        xbound = (-50, 50, 0.5) 表示x方向从-50到50米，分辨率0.5米
        则 dx=0.5, bx=-49.75, nx=200
    """
    # 各方向的体素大小（分辨率）
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    # 各方向第一个体素的中心坐标 = min + resolution/2
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    # 各方向的体素数量 = (max - min) / resolution
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):
    """
    视角变换基类
    
    实现了从2D图像特征到3D BEV特征的基本转换流程：
    1. 创建视锥体 (frustum): 定义图像空间中的3D采样点
    2. 几何变换 (get_geometry): 将视锥体点从图像坐标系转换到LiDAR坐标系
    3. BEV池化 (bev_pool): 将3D特征聚合到BEV网格中
    
    视锥体的概念：
    - 对于每个图像像素，在不同深度位置采样，形成一个3D点集
    - 这些点构成了相机的视锥体 (frustum)
    - 通过深度分布加权，可以将2D特征"提升"到3D空间
    """

    def __init__(
        self,
        in_channels: int,  # 输入特征通道数
        out_channels: int,  # 输出特征通道数
        image_size: Tuple[int, int],  # 原始图像尺寸 (H, W)
        feature_size: Tuple[int, int],  # 特征图尺寸 (fH, fW)
        xbound: Tuple[float, float, float],  # x方向BEV范围 (min, max, res)
        ybound: Tuple[float, float, float],  # y方向BEV范围 (min, max, res)
        zbound: Tuple[float, float, float],  # z方向BEV范围 (min, max, res)
        dbound: Tuple[float, float, float],  # 深度范围 (min, max, res)
    ) -> None:
        """
        初始化视角变换模块
        
        Args:
            in_channels: 输入图像特征的通道数
            out_channels: 输出BEV特征的通道数
            image_size: 原始输入图像的尺寸 (高度, 宽度)
            feature_size: 骨干网络输出特征图的尺寸 (高度, 宽度)
            xbound: BEV空间x方向的范围配置 (最小值, 最大值, 分辨率)
            ybound: BEV空间y方向的范围配置 (最小值, 最大值, 分辨率)
            zbound: BEV空间z方向的范围配置 (最小值, 最大值, 分辨率)
            dbound: 深度估计的范围配置 (最小深度, 最大深度, 深度间隔)
        """
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # 生成BEV网格参数
        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        # 注册为不需要梯度的参数（常量）
        self.dx = nn.Parameter(dx, requires_grad=False)  # 体素大小
        self.bx = nn.Parameter(bx, requires_grad=False)  # 起始位置
        self.nx = nn.Parameter(nx, requires_grad=False)  # 体素数量

        self.C = out_channels  # 输出通道数
        self.frustum = self.create_frustum()  # 创建视锥体
        self.D = self.frustum.shape[0]  # 深度离散化的数量
        self.fp16_enabled = False

    def create_frustum(self):
        """
        创建视锥体 (Frustum)
        
        视锥体是LSS方法的核心数据结构，它定义了图像空间中的3D采样点。
        对于特征图上的每个位置 (u, v)，在不同深度 d 处创建采样点。
        
        视锥体的形状: [D, fH, fW, 3]
        - D: 深度离散化的数量
        - fH, fW: 特征图的高度和宽度
        - 3: 每个点的坐标 (x, y, d)，其中 (x, y) 是图像坐标，d 是深度
        
        Returns:
            frustum: 视锥体张量，形状为 [D, fH, fW, 3]
        """
        iH, iW = self.image_size  # 原始图像尺寸
        fH, fW = self.feature_size  # 特征图尺寸

        # 创建深度采样点: 从 dbound[0] 到 dbound[1]，间隔 dbound[2]
        # 形状: [D, 1, 1] -> [D, fH, fW]
        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        # 创建x方向（宽度）的采样点
        # 从0到iW-1均匀采样fW个点，对应特征图到原图的映射
        # 形状: [1, 1, fW] -> [D, fH, fW]
        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        # 创建y方向（高度）的采样点
        # 形状: [1, fH, 1] -> [D, fH, fW]
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        # 堆叠成视锥体: [D, fH, fW, 3]，最后一维是 (x, y, depth)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        """
        计算视锥体点在LiDAR坐标系下的3D位置
        
        坐标变换流程：
        1. 撤销图像数据增强变换 (post_rots, post_trans)
        2. 将图像坐标转换为相机坐标系下的3D点
        3. 将相机坐标系转换为LiDAR坐标系
        4. 应用额外的LiDAR数据增强变换（如果有）
        
        数学原理：
        - 图像坐标 (u, v, d) -> 相机坐标 (X, Y, Z)
        - X = (u - cx) * d / fx
        - Y = (v - cy) * d / fy
        - Z = d
        - 然后通过 camera2lidar 变换到LiDAR坐标系
        
        Args:
            camera2lidar_rots: 相机到LiDAR的旋转矩阵 [B, N, 3, 3]
            camera2lidar_trans: 相机到LiDAR的平移向量 [B, N, 3]
            intrins: 相机内参矩阵 [B, N, 3, 3]
            post_rots: 图像增强的旋转矩阵 [B, N, 3, 3]
            post_trans: 图像增强的平移向量 [B, N, 3]
            **kwargs: 可选的额外变换 (extra_rots, extra_trans)
            
        Returns:
            points: LiDAR坐标系下的3D点 [B, N, D, H, W, 3]
        """
        B, N, _ = camera2lidar_trans.shape

        # 第一步：撤销图像数据增强变换
        # points: [B, N, D, H, W, 3]
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        
        # 第二步：将图像坐标转换为相机坐标系
        # 使用针孔相机模型的逆变换: (u*d, v*d, d) -> (X, Y, Z)
        # 这里 points[:,:,:,:,:,:2] 是 (u, v)，points[:,:,:,:,:,2:3] 是深度 d
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        
        # 第三步：从相机坐标系转换到LiDAR坐标系
        # combine = R_cam2lidar @ K^(-1)，其中 K 是相机内参
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 加上平移向量
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # 第四步：应用额外的LiDAR数据增强变换（可选）
        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        """
        获取相机特征（抽象方法）
        
        子类需要实现此方法，将2D图像特征转换为带深度分布的3D特征。
        
        Args:
            x: 图像特征 [B, N, C, fH, fW]
            
        Returns:
            带深度分布的特征 [B, N, D, fH, fW, C]
        """
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        """
        BEV池化：将3D特征聚合到BEV网格中
        
        这是LSS方法中的"Splat"步骤，将视锥体中的3D特征点
        根据其几何位置聚合到对应的BEV体素中。
        
        处理流程：
        1. 将特征和几何坐标展平
        2. 将连续坐标离散化为体素索引
        3. 过滤掉超出BEV范围的点
        4. 使用高效的BEV池化操作聚合特征
        5. 沿Z轴压缩得到最终的BEV特征
        
        Args:
            geom_feats: 几何坐标 [B, N, D, H, W, 3]，LiDAR坐标系下的3D位置
            x: 图像特征 [B, N, D, H, W, C]，带深度分布的特征
            
        Returns:
            final: BEV特征 [B, C*Z, H_bev, W_bev]
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W  # 总点数

        # 展平特征: [Nprime, C]
        x = x.reshape(Nprime, C)

        # 将连续坐标转换为离散的体素索引
        # (coord - (bx - dx/2)) / dx 将坐标映射到体素索引
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        
        # 添加batch索引作为第四维
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # [Nprime, 4]

        # 过滤掉超出BEV范围的点
        # 只保留在有效范围内的点: 0 <= x < nx[0], 0 <= y < nx[1], 0 <= z < nx[2]
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 调用高效的BEV池化CUDA核函数
        # 将落入同一体素的特征进行求和聚合
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # 沿Z轴压缩: 将 [B, C, Z, X, Y] 变为 [B, C*Z, X, Y]
        # 这样可以保留不同高度的信息
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        """
        前向传播：完整的视角变换流程
        
        将多视角2D图像特征转换为统一的BEV特征表示。
        
        Args:
            img: 多视角图像特征 [B, N, C, fH, fW]
            points: 点云数据（本方法未使用，子类可能使用）
            lidar2image: LiDAR到图像的投影矩阵
            camera_intrinsics: 相机内参矩阵 [B, N, 4, 4]
            camera2lidar: 相机到LiDAR的变换矩阵 [B, N, 4, 4]
            img_aug_matrix: 图像数据增强矩阵 [B, N, 4, 4]
            lidar_aug_matrix: LiDAR数据增强矩阵 [B, 4, 4]
            metas: 元信息
            
        Returns:
            BEV特征 [B, C, H_bev, W_bev]
        """
        # 提取3x3旋转矩阵和3维平移向量
        intrins = camera_intrinsics[..., :3, :3]  # 相机内参
        post_rots = img_aug_matrix[..., :3, :3]  # 图像增强旋转
        post_trans = img_aug_matrix[..., :3, 3]  # 图像增强平移
        camera2lidar_rots = camera2lidar[..., :3, :3]  # 相机到LiDAR旋转
        camera2lidar_trans = camera2lidar[..., :3, 3]  # 相机到LiDAR平移

        # LiDAR数据增强变换
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # 计算视锥体点在LiDAR坐标系下的几何位置
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # 获取带深度分布的图像特征
        x = self.get_cam_feats(img)
        # BEV池化：将3D特征聚合到BEV网格
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):
    """
    标准LSS (Lift-Splat-Shoot) 视角变换
    
    使用神经网络预测每个像素的深度分布，然后将2D特征"提升"到3D空间。
    
    核心思想：
    - 对于每个像素，预测一个深度概率分布 (softmax over D bins)
    - 同时预测该像素的语义特征
    - 将深度分布与语义特征相乘，得到3D空间中的特征分布
    
    网络结构：
    - depthnet: 1x1卷积，输出 D+C 通道
      - 前D个通道: 深度分布 (经过softmax)
      - 后C个通道: 语义特征
    - downsample: 可选的下采样模块，降低BEV分辨率
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,  # BEV特征下采样倍数
    ) -> None:
        """
        初始化LSS视角变换模块
        
        Args:
            downsample: BEV特征的下采样倍数，用于降低计算量
                       1表示不下采样，2表示2倍下采样
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        # 深度预测网络：1x1卷积，输出深度分布(D通道) + 语义特征(C通道)
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        
        # BEV特征下采样模块
        if downsample > 1:
            assert downsample == 2, downsample
            # 使用3层卷积进行2倍下采样
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,  # stride=2 实现下采样
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()  # 不下采样

    def get_cam_feats(self, x):
        """
        获取带深度分布的相机特征
        
        LSS的核心操作：将2D特征"提升"到3D空间
        
        处理流程：
        1. 通过depthnet预测深度分布和语义特征
        2. 对深度通道应用softmax得到概率分布
        3. 将深度分布与语义特征外积，得到3D特征
        
        Args:
            x: 图像特征 [B, N, C, fH, fW]
               B=batch, N=相机数, C=通道, fH/fW=特征图尺寸
               
        Returns:
            x: 3D特征 [B, N, D, fH, fW, C]
               D=深度离散化数量
        """
        B, N, C, fH, fW = x.shape

        # 合并batch和相机维度: [B*N, C, fH, fW]
        x = x.view(B * N, C, fH, fW)

        # 通过深度网络预测: [B*N, D+C, fH, fW]
        x = self.depthnet(x)
        
        # 分离深度分布和语义特征
        # depth: [B*N, D, fH, fW]，经过softmax归一化为概率分布
        depth = x[:, :self.D].softmax(dim=1)
        # 语义特征: [B*N, C, fH, fW]
        # 外积操作: depth.unsqueeze(1) * feat.unsqueeze(2)
        # [B*N, 1, D, fH, fW] * [B*N, C, 1, fH, fW] -> [B*N, C, D, fH, fW]
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        # 恢复维度并调整顺序: [B, N, C, D, fH, fW] -> [B, N, D, fH, fW, C]
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        """
        前向传播
        
        调用父类的forward方法完成视角变换，然后进行可选的下采样。
        """
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):
    """
    带深度监督的视角变换基类
    
    与BaseViewTransform的区别：
    - 利用LiDAR点云生成稀疏深度图作为深度估计的监督信号
    - 将稀疏深度信息融入特征提取过程
    
    这种方法可以提高深度估计的准确性，从而改善视角变换的质量。
    """

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        """
        带深度监督的前向传播
        
        与父类的主要区别：利用LiDAR点云生成稀疏深度图，
        作为深度估计网络的额外输入或监督信号。
        
        处理流程：
        1. 将LiDAR点云投影到各相机图像平面，生成稀疏深度图
        2. 计算视锥体的几何位置
        3. 获取融合了深度信息的相机特征
        4. BEV池化生成最终特征
        
        Args:
            img: 图像特征 [B, N, C, fH, fW]
            points: LiDAR点云列表，用于生成稀疏深度图
            其他参数同父类
            
        Returns:
            BEV特征
        """
        # 提取变换矩阵
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        # 初始化稀疏深度图: [B, N, 1, H, W]
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        # ============ 生成稀疏深度图 ============
        # 将LiDAR点云投影到每个相机的图像平面
        for b in range(batch_size):
            # 获取当前样本的点云坐标 [N_points, 3]
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # 第一步：撤销LiDAR数据增强
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            
            # 第二步：LiDAR坐标系 -> 图像坐标系
            # 应用外参旋转
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            # 应用外参平移
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            
            # 第三步：透视除法，得到2D图像坐标
            dist = cur_coords[:, 2, :]  # 保存深度值
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]  # (u, v) = (X/Z, Y/Z)

            # 第四步：应用图像数据增强变换
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)  # [N_cam, N_points, 2]

            # 交换坐标顺序以匹配图像索引 (y, x)
            cur_coords = cur_coords[..., [1, 0]]

            # 过滤掉超出图像范围的点
            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))
            
            # 将有效点的深度值填入稀疏深度图
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist

        # ============ 视角变换 ============
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        
        # 计算视锥体几何位置
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # 获取融合深度信息的相机特征
        x = self.get_cam_feats(img, depth)
        # BEV池化
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    """
    深度增强的LSS视角变换 (Depth-aware LSS)
    
    与标准LSSTransform的区别：
    - 利用LiDAR点云生成稀疏深度图
    - 将稀疏深度图编码后与图像特征拼接
    - 深度网络同时接收图像特征和深度特征
    
    这种设计的优势：
    1. LiDAR提供准确的深度参考，减少深度估计的歧义性
    2. 稀疏深度可以指导网络学习更准确的深度分布
    3. 多模态信息融合发生在早期阶段，提高特征质量
    
    网络结构：
    - dtransform: 稀疏深度编码网络，将1通道深度图编码为64通道特征
    - depthnet: 深度预测网络，输入为图像特征+深度特征的拼接
    - downsample: 可选的BEV下采样模块
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """
        初始化深度增强LSS模块
        
        与LSSTransform相比，增加了稀疏深度编码网络(dtransform)，
        将LiDAR生成的稀疏深度图编码为特征，与图像特征拼接后
        送入深度预测网络。
        
        Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        # 稀疏深度编码网络
        # 将1通道的稀疏深度图逐步编码为64通道的深度特征
        # 同时进行8倍下采样以匹配图像特征的分辨率
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),  # 1->8通道
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),  # 4倍下采样
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 2倍下采样，共8倍
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # 深度预测网络
        # 输入: 图像特征(in_channels) + 深度特征(64) = in_channels+64
        # 输出: 深度分布(D) + 语义特征(C) = D+C
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),  # 最终预测
        )
        # BEV下采样模块
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        """
        获取融合深度信息的相机特征
        
        与LSSTransform.get_cam_feats的区别：
        - 额外接收稀疏深度图作为输入
        - 将深度特征与图像特征拼接后送入深度网络
        
        处理流程：
        1. 编码稀疏深度图为深度特征
        2. 拼接深度特征和图像特征
        3. 通过深度网络预测深度分布和语义特征
        4. 外积操作生成3D特征
        
        Args:
            x: 图像特征 [B, N, C, fH, fW]
            d: 稀疏深度图 [B, N, 1, H, W]，由LiDAR点云投影生成
            
        Returns:
            3D特征 [B, N, D, fH, fW, C]
        """
        B, N, C, fH, fW = x.shape

        # 展平batch和相机维度
        d = d.view(B * N, *d.shape[2:])  # [B*N, 1, H, W]
        x = x.view(B * N, C, fH, fW)  # [B*N, C, fH, fW]

        # 编码稀疏深度图: [B*N, 1, H, W] -> [B*N, 64, fH, fW]
        # dtransform包含8倍下采样，使深度特征与图像特征尺寸匹配
        d = self.dtransform(d)
        
        # 拼接深度特征和图像特征: [B*N, C+64, fH, fW]
        x = torch.cat([d, x], dim=1)
        
        # 通过深度网络预测: [B*N, D+C, fH, fW]
        x = self.depthnet(x)

        # 分离深度分布和语义特征，进行外积
        # depth: [B*N, D, fH, fW] 经过softmax归一化
        depth = x[:, :self.D].softmax(dim=1)
        # 外积: [B*N, 1, D, fH, fW] * [B*N, C, 1, fH, fW] -> [B*N, C, D, fH, fW]
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        # 恢复维度: [B, N, C, D, fH, fW] -> [B, N, D, fH, fW, C]
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        """
        前向传播
        
        调用父类(BaseDepthTransform)的forward方法完成：
        1. 生成稀疏深度图
        2. 视角变换
        3. BEV池化
        
        然后进行可选的下采样。
        """
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x



@MODELS.register_module()
class DepthSupervisionLoss(nn.Module):
    """
    深度监督损失模块
    
    支持两种损失类型：
    - 'ce': CrossEntropyLoss，直接使用索引格式的GT
    - 'focal': Focal Loss，处理类别不平衡
    
    Args:
        loss_type (str): 损失类型，'ce' 或 'focal'
        loss_weight (float): 损失权重
        focal_gamma (float): Focal Loss的gamma参数，默认2.0
    """
    
    def __init__(
        self,
        loss_type: str = 'ce',
        loss_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        depth_pred: torch.Tensor,  # [B*N, D, fH, fW]
        depth_gt_indices: torch.Tensor,  # [B*N, fH, fW]，索引格式
        valid_mask: torch.Tensor,  # [B*N, fH, fW]
    ) -> torch.Tensor:
        """
        计算深度监督损失
        
        使用索引格式的GT计算损失，避免One-Hot转换的显存开销
        
        Args:
            depth_pred: 预测的深度分布 [B*N, D, fH, fW]
            depth_gt_indices: 深度GT索引 [B*N, fH, fW]，无效位置为-1
            valid_mask: 有效掩码 [B*N, fH, fW]
            
        Returns:
            loss: 深度监督损失
        """
        # 如果没有有效像素，返回零损失
        if valid_mask.sum() == 0:
            return depth_pred.sum() * 0
        
        # 将无效位置标记为-1（CrossEntropyLoss会自动忽略）
        depth_gt_indices = depth_gt_indices.clone()
        depth_gt_indices[~valid_mask] = -1
        
        if self.loss_type == 'ce':
            # 使用CrossEntropyLoss，ignore_index=-1自动忽略无效位置
            loss = torch.nn.functional.cross_entropy(
                depth_pred,
                depth_gt_indices,
                ignore_index=-1,
                reduction='mean'
            )
        else:  # focal loss
            # Focal Loss实现
            # 先计算log_softmax
            log_probs = torch.nn.functional.log_softmax(depth_pred, dim=1)
            
            # 获取GT对应的log概率
            # depth_gt_indices: [B*N, fH, fW]
            # log_probs: [B*N, D, fH, fW]
            B, D, H, W = log_probs.shape
            
            # 只在有效位置计算损失
            valid_indices = depth_gt_indices[valid_mask]  # [N_valid]
            valid_log_probs = log_probs.permute(0, 2, 3, 1)[valid_mask]  # [N_valid, D]
            
            # 获取GT对应的log概率
            gt_log_probs = valid_log_probs[torch.arange(valid_indices.shape[0]), valid_indices]
            
            # 计算概率（用于focal weight）
            gt_probs = torch.exp(gt_log_probs)
            
            # Focal Loss: -(1-p)^gamma * log(p)
            focal_weight = (1 - gt_probs) ** self.focal_gamma
            loss = -(focal_weight * gt_log_probs).mean()
        
        return loss * self.loss_weight


@MODELS.register_module()
class CameraAwareDepthNet(nn.Module):
    """
    Camera-Aware深度估计网络
    
    将相机参数编码为特征，通过SE机制调制图像特征。
    
    Args:
        in_channels (int): 输入特征通道数
        mid_channels (int): 中间层通道数
        depth_channels (int): 深度bin数量
        context_channels (int): 语义特征通道数
        embed_dim (int): 相机参数编码维度，默认256
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        depth_channels: int,
        context_channels: int,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.context_channels = context_channels
        self.embed_dim = embed_dim
        
        # 相机参数编码MLP
        # 输入: 25维 (内参4 + 外参12 + IDA9)
        # 输出: embed_dim维
        self.cam_encoder = nn.Sequential(
            nn.Linear(25, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
        )
        
        # SE调制模块
        # 将相机编码映射到通道注意力权重
        self.se_layer = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Sigmoid(),
        )
        
        # 深度预测网络
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, depth_channels + context_channels, 1),
        )
    
    def encode_camera_params(
        self,
        intrinsics: torch.Tensor,  # [B*N, 3, 3]
        cam2ego: torch.Tensor,  # [B*N, 4, 4]
        ida_matrix: torch.Tensor,  # [B*N, 4, 4]
    ) -> torch.Tensor:
        """
        编码相机参数为特征向量
        
        Args:
            intrinsics: 增强后相机内参 [B*N, 3, 3]
            cam2ego: Camera-to-Ego外参 [B*N, 4, 4]
            ida_matrix: IDA矩阵 [B*N, 4, 4]
        
        Returns:
            cam_embed: 相机参数编码 [B*N, embed_dim]
        """
        BN = intrinsics.shape[0]
        
        # 内参: fx, fy, cx, cy (4维)
        intrinsic_params = torch.stack([
            intrinsics[:, 0, 0],  # fx
            intrinsics[:, 1, 1],  # fy
            intrinsics[:, 0, 2],  # cx
            intrinsics[:, 1, 2],  # cy
        ], dim=1)  # [B*N, 4]
        
        # 外参: 旋转矩阵展平(9维) + 平移向量(3维) = 12维
        extrinsic_params = torch.cat([
            cam2ego[:, :3, :3].reshape(BN, 9),  # 旋转
            cam2ego[:, :3, 3],  # 平移
        ], dim=1)  # [B*N, 12]
        
        # IDA矩阵: 展平(9维)
        ida_params = ida_matrix[:, :3, :3].reshape(BN, 9)  # [B*N, 9]
        
        # 拼接所有参数: 4 + 12 + 9 = 25维
        cam_params = torch.cat([
            intrinsic_params,
            extrinsic_params,
            ida_params,
        ], dim=1)  # [B*N, 25]
        
        # 编码相机参数
        cam_embed = self.cam_encoder(cam_params)  # [B*N, embed_dim]
        
        return cam_embed
    
    def forward(
        self,
        x: torch.Tensor,  # [B*N, C, fH, fW]
        intrinsics: torch.Tensor,  # [B*N, 3, 3]
        cam2ego: torch.Tensor,  # [B*N, 4, 4]
        ida_matrix: torch.Tensor,  # [B*N, 4, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 图像特征 [B*N, C, fH, fW]
            intrinsics: 增强后相机内参 [B*N, 3, 3]
            cam2ego: Camera-to-Ego外参 [B*N, 4, 4]
            ida_matrix: IDA矩阵 [B*N, 4, 4]
        
        Returns:
            depth: 深度分布 [B*N, D, fH, fW]
            context: 语义特征 [B*N, C, fH, fW]
        """
        # 编码相机参数
        cam_embed = self.encode_camera_params(intrinsics, cam2ego, ida_matrix)
        
        # SE调制
        scale = self.se_layer(cam_embed)  # [B*N, C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B*N, C, 1, 1]
        x = x * scale  # 通道级调制
        
        # 深度预测
        x = self.depthnet(x)  # [B*N, D+C, fH, fW]
        
        # 分离深度和语义特征
        depth = x[:, :self.depth_channels]  # [B*N, D, fH, fW]
        context = x[:, self.depth_channels:]  # [B*N, C, fH, fW]
        
        return depth, context


@MODELS.register_module()
class CameraAwareDepthLSSTransform(DepthLSSTransform):
    """
    集成深度监督和Camera-Aware的视角变换模块
    
    Args:
        use_depth_supervision (bool): 是否启用深度监督
        use_camera_aware (bool): 是否启用Camera-Aware
        depth_loss_cfg (dict): 深度损失配置
        camera_aware_cfg (dict): Camera-Aware配置
        其他参数同DepthLSSTransform
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        use_depth_supervision: bool = False,
        use_camera_aware: bool = False,
        depth_loss_cfg: dict = None,
        camera_aware_cfg: dict = None,
    ) -> None:
        """
        初始化Camera-Aware深度LSS模块
        
        Args:
            use_depth_supervision: 是否启用深度监督
            use_camera_aware: 是否启用Camera-Aware
            depth_loss_cfg: 深度损失配置字典
            camera_aware_cfg: Camera-Aware配置字典
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            downsample=downsample,
        )
        
        self.use_depth_supervision = use_depth_supervision
        self.use_camera_aware = use_camera_aware
        
        # 初始化深度损失模块
        if use_depth_supervision and depth_loss_cfg is not None:
            self.depth_loss = MODELS.build(depth_loss_cfg)
        else:
            self.depth_loss = None
        
        # 初始化Camera-Aware深度网络
        if use_camera_aware:
            if camera_aware_cfg is None:
                camera_aware_cfg = dict(
                    type='CameraAwareDepthNet',
                    in_channels=in_channels + 64,  # 图像特征 + 深度特征
                    mid_channels=in_channels + 64,
                    depth_channels=self.D,
                    context_channels=self.C,
                    embed_dim=256,
                )
            
            # 替换原来的depthnet为Camera-Aware版本
            self.camera_aware_depthnet = MODELS.build(camera_aware_cfg)
    
    def get_cam_feats(self, x, d, camera_intrinsics=None, camera2lidar=None, img_aug_matrix=None):
        """
        获取融合深度信息和相机感知的特征
        
        Args:
            x: 图像特征 [B, N, C, fH, fW]
            d: 稀疏深度图 [B, N, 1, H, W]
            camera_intrinsics: 相机内参 [B, N, 4, 4]
            camera2lidar: 相机到LiDAR变换 [B, N, 4, 4]
            img_aug_matrix: 图像增强矩阵 [B, N, 4, 4]
            
        Returns:
            3D特征 [B, N, D, fH, fW, C]
        """
        B, N, C, fH, fW = x.shape
        
        # 展平batch和相机维度
        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)
        
        # 编码稀疏深度图
        d = self.dtransform(d)
        
        # 拼接深度特征和图像特征
        x = torch.cat([d, x], dim=1)
        
        # 如果启用Camera-Aware
        if self.use_camera_aware and camera_intrinsics is not None:
            # 准备相机参数
            intrinsics = camera_intrinsics[..., :3, :3].reshape(B * N, 3, 3)
            
            # 计算Camera-to-Ego变换
            # camera2lidar是Camera-to-LiDAR，我们需要Camera-to-Ego
            # 这里假设LiDAR坐标系就是Ego坐标系
            cam2ego = camera2lidar.reshape(B * N, 4, 4)
            
            # IDA矩阵
            ida_matrix = img_aug_matrix.reshape(B * N, 4, 4)
            
            # 使用Camera-Aware深度网络
            depth_logits, context = self.camera_aware_depthnet(
                x, intrinsics, cam2ego, ida_matrix
            )
            
            # 应用softmax得到深度分布
            depth = depth_logits.softmax(dim=1)
            
            # 外积操作
            x = depth.unsqueeze(1) * context.unsqueeze(2)
        else:
            # 使用原始depthnet
            x = self.depthnet(x)
            
            # 分离深度分布和语义特征
            depth = x[:, :self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        
        # 恢复维度
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x
    
    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depth_gt_indices=None,
        depth_valid_mask=None,
        **kwargs,
    ):
        """
        前向传播
        
        Args:
            img: 图像特征 [B, N, C, fH, fW]
            points: LiDAR点云
            lidar2image: LiDAR到图像投影矩阵
            cam_intrinsic: 相机内参 [B, N, 4, 4]
            camera2lidar: 相机到LiDAR变换 [B, N, 4, 4]
            img_aug_matrix: 图像增强矩阵 [B, N, 4, 4]
            lidar_aug_matrix: LiDAR增强矩阵 [B, 4, 4]
            metas: 元信息
            depth_gt_indices: 深度GT索引 [B, N, fH, fW]（可选）
            depth_valid_mask: 有效掩码 [B, N, fH, fW]（可选）
            
        Returns:
            训练时: (bev_feat, aux_dict) 如果提供了depth_gt
            推理时: bev_feat
        """
        # 提取变换矩阵
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        
        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                           *self.image_size).to(points[0].device)
        
        # 生成稀疏深度图（与父类相同的逻辑）
        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            
            # 撤销LiDAR增强
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            
            # LiDAR -> 图像
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            
            # 透视除法
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            
            # 应用图像增强
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            cur_coords = cur_coords[..., [1, 0]]
            
            # 过滤
            on_img = ((cur_coords[..., 0] < self.image_size[0])
                     & (cur_coords[..., 0] >= 0)
                     & (cur_coords[..., 1] < self.image_size[1])
                     & (cur_coords[..., 1] >= 0))
            
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist
        
        # 计算几何
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        
        # 获取相机特征
        x = self.get_cam_feats(
            img, depth,
            camera_intrinsics=cam_intrinsic if self.use_camera_aware else None,
            camera2lidar=camera2lidar if self.use_camera_aware else None,
            img_aug_matrix=img_aug_matrix if self.use_camera_aware else None,
        )
        
        # BEV池化
        bev_feat = self.bev_pool(geom, x)
        bev_feat = self.downsample(bev_feat)
        
        # 如果是训练模式且提供了深度GT，返回辅助字典
        if self.training and self.use_depth_supervision and depth_gt_indices is not None:
            # 这里我们需要保存深度预测用于损失计算
            # 但是get_cam_feats已经应用了softmax，我们需要logits
            # 为了简化，我们在这里重新计算一次（仅用于训练）
            
            B, N, C, fH, fW = img.shape
            d_flat = depth.view(B * N, *depth.shape[2:])
            x_flat = img.view(B * N, C, fH, fW)
            d_feat = self.dtransform(d_flat)
            x_cat = torch.cat([d_feat, x_flat], dim=1)
            
            if self.use_camera_aware and cam_intrinsic is not None:
                intrinsics_flat = cam_intrinsic[..., :3, :3].reshape(B * N, 3, 3)
                cam2ego = camera2lidar.reshape(B * N, 4, 4)
                ida_matrix_flat = img_aug_matrix.reshape(B * N, 4, 4)
                depth_pred, _ = self.camera_aware_depthnet(
                    x_cat, intrinsics_flat, cam2ego, ida_matrix_flat
                )
            else:
                depth_pred = self.depthnet(x_cat)[:, :self.D]
            
            aux_dict = {
                'depth_pred': depth_pred,
                'depth_gt_indices': depth_gt_indices.reshape(B * N, fH, fW),
                'valid_mask': depth_valid_mask.reshape(B * N, fH, fW),
            }
            return bev_feat, aux_dict
        
        return bev_feat

"""
BEVFusion 数据增强模块

本模块实现了用于3D目标检测的数据增强变换，包括：
- ImageAug3D: 图像数据增强（缩放、裁剪、翻转、旋转）
- BEVFusionRandomFlip3D: 3D随机翻转（水平/垂直）
- BEVFusionGlobalRotScaleTrans: 全局旋转、缩放、平移
- GridMask: 网格遮挡增强
- RadarPointsRangeFilter: 雷达点云范围过滤

这些增强操作会同步更新图像和对应的相机参数、3D标注，
确保数据的几何一致性。

modify from https://github.com/mit-han-lab/bevfusion
"""
from typing import Any, Dict, List

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):
    """
    3D检测的图像数据增强
    
    对多视角图像进行数据增强，包括：
    - 随机缩放 (resize)
    - 随机裁剪 (crop)
    - 随机水平翻转 (flip)
    - 随机旋转 (rotate)
    
    同时计算并记录增强变换矩阵，用于后续的坐标变换。
    
    增强流程：
    1. 缩放图像到指定尺寸
    2. 从底部裁剪固定大小区域（保留地面信息）
    3. 可选的水平翻转
    4. 可选的旋转
    
    Args:
        final_dim: 最终输出图像尺寸 (H, W)
        resize_lim: 缩放比例范围 (min, max)
        bot_pct_lim: 底部裁剪比例范围 (min, max)
        rot_lim: 旋转角度范围 (min_deg, max_deg)
        rand_flip: 是否随机水平翻转
        is_train: 是否为训练模式
    """

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        """
        初始化图像增强参数
        
        Args:
            final_dim: 最终输出尺寸 (H, W)
            resize_lim: 缩放范围 (min_scale, max_scale)
            bot_pct_lim: 底部裁剪比例范围 (min_pct, max_pct)
            rot_lim: 旋转角度范围 (min_deg, max_deg)
            rand_flip: 是否随机翻转
            is_train: 训练/测试模式
        """
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        """
        采样数据增强参数
        
        训练模式：随机采样所有增强参数
        测试模式：使用固定的增强参数（取范围中值）
        
        Args:
            results: 包含原始图像信息的字典
            
        Returns:
            tuple: (resize, resize_dims, crop, flip, rotate)
                - resize: 缩放比例
                - resize_dims: 缩放后的尺寸 (W, H)
                - crop: 裁剪区域 (x1, y1, x2, y2)
                - flip: 是否翻转
                - rotate: 旋转角度
        """
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            # 训练模式：随机采样
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # 从底部裁剪：保留地面和近处物体
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            # 测试模式：使用固定参数
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        """
        应用图像变换并更新相机内参
        
        对图像进行几何变换，同时计算对应的变换矩阵，
        用于更新相机内参和坐标映射关系。
        
        变换顺序：
        1. 缩放 (resize)
        2. 裁剪 (crop)
        3. 翻转 (flip)
        4. 旋转 (rotate)
        
        Args:
            img: 输入图像
            rotation: 初始旋转矩阵 (2x2)
            translation: 初始平移向量 (2,)
            resize: 缩放比例
            resize_dims: 缩放后尺寸
            crop: 裁剪区域
            flip: 是否翻转
            rotate: 旋转角度
            
        Returns:
            tuple: (变换后的图像, 更新后的旋转矩阵, 更新后的平移向量)
        """
        # 对图像进行变换
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # 计算后处理的齐次变换矩阵
        # 用于更新相机内参，使其与变换后的图像对应
        
        # 1. 缩放变换
        rotation *= resize
        # 2. 裁剪变换（平移）
        translation -= torch.Tensor(crop[:2])
        # 3. 翻转变换
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])  # 水平翻转矩阵
            b = torch.Tensor([crop[2] - crop[0], 0])  # 翻转中心偏移
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        # 4. 旋转变换
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b  # 绕图像中心旋转
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对所有视角的图像应用数据增强
        
        为每个相机视角独立采样增强参数并应用变换，
        记录增强变换矩阵到 'img_aug_matrix'。
        
        Args:
            data: 包含多视角图像的数据字典
            
        Returns:
            更新后的数据字典，包含：
                - img: 增强后的图像列表
                - img_aug_matrix: 增强变换矩阵列表 (4x4)
        """
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            # 为每个视角采样增强参数
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            # 初始化单位变换
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 应用图像变换并获取变换矩阵
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            # 构建4x4齐次变换矩阵
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # 更新标定矩阵：记录图像增强变换
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """
    3D随机翻转数据增强
    
    与标准的 RandomFlip3D 相比，该类直接在数据中记录 LiDAR 增强矩阵。
    
    支持水平和垂直两个方向的翻转，同时更新：
    - 点云坐标 (points)
    - 雷达点云坐标 (radar_points)
    - 3D边界框
    - BEV分割掩码
    - LiDAR增强矩阵
    
    雷达点云与LiDAR点云使用相同的翻转变换，确保几何一致性。
    
    Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行随机翻转
        
        随机决定是否进行水平和垂直翻转，
        并更新所有相关的数据（点云、雷达点云、边界框、掩码、变换矩阵）。
        
        雷达点云与LiDAR点云使用相同的翻转变换矩阵，
        确保多模态数据的几何一致性。
        
        Args:
            data: 输入数据字典
            
        Returns:
            更新后的数据字典
        """
        # 随机决定翻转方向
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        
        # 水平翻转（沿Y轴）
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            # 对雷达点云应用相同的翻转变换
            if 'radar_points' in data:
                data['radar_points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        # 垂直翻转（沿X轴）
        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            # 对雷达点云应用相同的翻转变换
            if 'radar_points' in data:
                data['radar_points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        # 更新LiDAR增强矩阵
        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """
    全局旋转、缩放、平移增强
    
    与标准的 GlobalRotScaleTrans 相比，该类的增强顺序是：
    旋转 (R) -> 平移 (T) -> 缩放 (S)
    
    而标准版本的顺序是：旋转 -> 缩放 -> 平移
    
    这种顺序更符合某些坐标系统的变换习惯。
    
    同时支持雷达点云的同步变换，确保雷达点云与LiDAR点云
    使用相同的增强矩阵，保持几何一致性。
    
    Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS).
    """

    def _trans_bbox_points(self, input_dict: dict) -> None:
        """
        平移边界框和点云（包括雷达点云）
        
        Private function to translate bounding boxes and points.
        
        对LiDAR点云和雷达点云应用相同的平移变换，
        确保多模态数据的几何一致性。
        
        Args:
            input_dict (dict): 输入数据字典
                Result dict from loading pipeline.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        # 平移LiDAR点云
        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        
        # 平移雷达点云（使用相同的平移向量）
        if 'radar_points' in input_dict:
            input_dict['radar_points'].translate(trans_factor)
        
        # 平移边界框
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)

    def _rot_bbox_points(self, input_dict: dict) -> None:
        """
        旋转边界框和点云（包括雷达点云）
        
        Private function to rotate bounding boxes and points.
        
        对LiDAR点云和雷达点云应用相同的旋转变换，
        确保多模态数据的几何一致性。
        
        Args:
            input_dict (dict): 输入数据字典
                Result dict from loading pipeline.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            # 旋转点云和边界框
            # rotate points with bboxes
            points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                noise_rotation, input_dict['points'])
            input_dict['points'] = points
        else:
            # 如果没有边界框，只旋转点云
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict['points'].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation
        
        # 旋转雷达点云（使用相同的旋转角度）
        if 'radar_points' in input_dict:
            input_dict['radar_points'].rotate(noise_rotation)

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """
        缩放边界框和点云（包括雷达点云）
        
        Private function to scale bounding boxes and points.
        
        对LiDAR点云和雷达点云应用相同的缩放变换，
        确保多模态数据的几何一致性。
        
        Args:
            input_dict (dict): 输入数据字典
                Result dict from loading pipeline.
        """
        scale = input_dict['pcd_scale_factor']
        
        # 缩放LiDAR点云
        points = input_dict['points']
        points.scale(scale)
        
        # 缩放雷达点云（使用相同的缩放因子）
        if 'radar_points' in input_dict:
            input_dict['radar_points'].scale(scale)
        
        # 缩放边界框
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)

    def transform(self, input_dict: dict) -> dict:
        """
        应用旋转、平移、缩放变换
        
        变换顺序：R -> T -> S (旋转 -> 平移 -> 缩放)
        
        对LiDAR点云和雷达点云应用相同的变换，
        确保多模态数据的几何一致性。
        
        Args:
            input_dict: 输入数据字典
            
        Returns:
            更新后的数据字典，包含：
                - points: 变换后的点云
                - radar_points: 变换后的雷达点云（如果存在）
                - gt_bboxes_3d: 变换后的3D边界框
                - pcd_rotation: 旋转矩阵
                - pcd_trans: 平移向量
                - pcd_scale_factor: 缩放因子
                - lidar_aug_matrix: LiDAR增强矩阵 (4x4)
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # 1. 旋转变换
        self._rot_bbox_points(input_dict)

        # 2. 缩放变换（如果未设置则随机采样）
        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        
        # 3. 平移变换
        self._trans_bbox_points(input_dict)
        
        # 4. 应用缩放到边界框和点云
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        # 构建LiDAR增强矩阵
        # 组合旋转、平移、缩放为单个4x4矩阵
        lidar_augs = np.eye(4)
        # 旋转矩阵的转置 × 缩放因子
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        # 平移向量 × 缩放因子
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        # 累积到已有的增强矩阵
        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):
    """
    GridMask 数据增强
    
    在图像上应用网格状遮挡，增强模型对遮挡的鲁棒性。
    
    GridMask 通过在图像上叠加规则的网格遮挡模式，
    模拟物体被部分遮挡的情况，提高模型的泛化能力。
    
    参考论文：
    GridMask Data Augmentation (https://arxiv.org/abs/2001.04086)
    
    Args:
        use_h: 是否在高度方向应用网格
        use_w: 是否在宽度方向应用网格
        max_epoch: 最大训练轮数
        rotate: 网格旋转角度范围
        offset: 是否添加随机偏移
        ratio: 网格遮挡比例
        mode: 遮挡模式 (0: 保留网格内, 1: 保留网格外)
        prob: 应用概率
        fixed_prob: 是否使用固定概率
    """

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        """
        初始化GridMask参数
        
        Args:
            use_h: 是否在高度方向应用网格
            use_w: 是否在宽度方向应用网格
            max_epoch: 最大训练轮数（用于动态调整概率）
            rotate: 网格旋转角度
            offset: 是否添加随机偏移
            ratio: 遮挡比例 (0-1)
            mode: 0=保留网格内，1=保留网格外
            prob: 初始应用概率
            fixed_prob: 是否使用固定概率
        """
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        """
        设置当前训练轮数
        
        根据训练进度动态调整应用概率（如果未固定）
        """
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        """动态调整应用概率：随训练进度线性增加"""
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        """
        应用GridMask增强
        
        生成网格遮挡模式并应用到所有视角的图像上。
        
        Args:
            results: 包含图像的数据字典
            
        Returns:
            更新后的数据字典
        """
        # 根据概率决定是否应用
        if np.random.rand() > self.prob:
            return results
        
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        
        # 网格参数
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        
        # 随机网格大小
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        
        # 创建网格遮挡
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        # 在高度方向应用网格
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        
        # 在宽度方向应用网格
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        # 旋转网格
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        
        # 裁剪到原始尺寸
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        
        # 反转模式
        if self.mode == 1:
            mask = 1 - mask

        # 应用遮挡到所有图像
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results


@TRANSFORMS.register_module()
class RadarPointsRangeFilter(BaseTransform):
    """
    过滤超出范围的雷达点云
    
    Filter radar points by the specified point cloud range.
    
    该类用于过滤超出指定范围的雷达点云，确保只有在有效范围内的
    雷达点被用于后续处理。雷达点云范围可以与LiDAR点云范围不同，
    以适应不同传感器的特性。
    
    Required Keys:
    - radar_points (BasePoints): 雷达点云数据
    
    Modified Keys:
    - radar_points (BasePoints): 过滤后的雷达点云数据
    
    Args:
        point_cloud_range (list[float]): 点云范围
            Point cloud range in format [x_min, y_min, z_min, x_max, y_max, z_max].
            Points outside this range will be filtered out.
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        """
        初始化雷达点云范围过滤器
        
        Args:
            point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """
        过滤超出范围的雷达点云
        
        Transform function to filter radar points by the range.
        
        处理流程：
        1. 获取雷达点云
        2. 使用 in_range_3d 方法计算范围内的点掩码
        3. 应用掩码过滤点云
        4. 更新结果字典
        
        Args:
            input_dict (dict): 包含雷达点云的结果字典
                Result dict from loading pipeline.

        Returns:
            dict: 过滤后的结果字典，'radar_points' 键被更新
                Results after filtering, 'radar_points' key is updated.
        """
        # 检查是否存在雷达点云
        if 'radar_points' not in input_dict:
            return input_dict
        
        radar_points = input_dict['radar_points']
        
        # 如果雷达点云为空，直接返回
        if len(radar_points) == 0:
            return input_dict
        
        # 使用 in_range_3d 方法计算范围内的点掩码
        # in_range_3d 检查点的 x, y, z 坐标是否在指定范围内
        points_mask = radar_points.in_range_3d(self.pcd_range)
        
        # 应用掩码过滤点云
        clean_radar_points = radar_points[points_mask]
        
        # 更新结果字典
        input_dict['radar_points'] = clean_radar_points
        
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

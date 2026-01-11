"""
BEVFusion 多视角图像加载模块

本模块实现了多视角图像的加载和预处理，
为 BEVFusion 提供必要的相机参数和图像数据。

主要功能：
- 加载多视角图像
- 计算相机到LiDAR的变换矩阵
- 处理多帧时序数据
- 支持不同尺寸的图像
- 加载雷达点云数据（nuScenes格式）
- 加载多帧雷达sweep数据

Copyright (c) OpenMMLab. All rights reserved.
"""
import copy
import warnings
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type, BasePoints


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """
    从文件加载多视角图像
    
    相比标准的 LoadMultiViewImageFromFiles，该类额外添加了
    用于视角变换的关键信息：
    - 'cam2lidar': 相机到LiDAR的变换矩阵
    - 'lidar2img': LiDAR到图像的投影矩阵
    
    这些矩阵对于 BEVFusion 的视角变换至关重要。
    
    Args:
        to_float32 (bool): 是否转换为float32，默认False
        color_type (str): 颜色类型，默认'unchanged'
        backend_args (dict, optional): 后端参数，默认None
        num_views (int): 每帧的视角数量，默认5
        num_ref_frames (int): 参考帧数量，默认-1（不使用）
        test_mode (bool): 是否为测试模式，默认False
        set_default_scale (bool): 是否设置默认缩放，默认True
    
    Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """
        加载多视角图像并计算相机参数
        
        处理流程：
        1. 处理多帧时序数据（如果启用）
        2. 加载所有视角的图像
        3. 计算相机变换矩阵（cam2lidar, lidar2img等）
        4. 处理不同尺寸的图像（padding）
        5. 设置元信息
        
        Args:
            results (dict): 包含图像文件名的结果字典

        Returns:
            dict: 包含多视角图像数据的结果字典，添加的键值：
                - filename (str): 多视角图像文件名
                - img (np.ndarray): 多视角图像数组
                - img_shape (tuple[int]): 图像形状
                - ori_shape (tuple[int]): 原始图像形状
                - pad_shape (tuple[int]): padding后的形状
                - scale_factor (float): 缩放因子
                - img_norm_cfg (dict): 图像归一化配置
                - cam2lidar: 相机到LiDAR变换矩阵
                - lidar2img: LiDAR到图像投影矩阵
        
        Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # ============ 处理多帧时序数据 ============
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # 初始化选择：当前帧
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            
            # 根据可用帧数选择参考帧
            if num_frames == 0:  # 没有历史帧，复制当前帧
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # 有足够的历史帧
                # NOTE: 假设信息按从最新到最早的顺序保存
                if self.test_mode:
                    # 测试模式：选择最近的帧
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 是为了选择历史帧
                else:
                    # 训练模式：随机选择
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                # 历史帧不足，需要重复采样
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            
            # 合并当前帧和参考帧
            choices = np.concatenate([init_choice, choices])
            
            # 选择对应的文件名和相机参数
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            
            # 将 lidar2cam 变换为 [cur_lidar]2[prev_img] 和 [cur_lidar]2[prev_cam]
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # 只修改历史帧的矩阵
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        # 构建4x4齐次变换矩阵
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        # 计算当前帧到历史帧的变换
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        # 更新历史帧的 lidar2cam
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        
        # ============ 加载图像和计算相机参数 ============
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            # 计算 cam2lidar 变换矩阵
            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]  # 旋转矩阵
            lidar2cam_trans = lidar2cam_array[:3, 3:4]  # 平移向量
            # cam2lidar = inv(lidar2cam)
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T  # R^(-1) = R^T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))  # -R^T * t
            cam2lidar.append(camera2lidar)

            # 构建相机内参矩阵（4x4齐次形式）
            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            # lidar2img = cam2img @ lidar2cam
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # ============ 加载图像数据 ============
        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        
        # ============ 处理不同尺寸的图像 ============
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]  # 通道数必须相同
        if not np.all(img_shape_max == img_shape_min):
            # 如果尺寸不同，padding到最大尺寸
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        
        # 堆叠所有视角的图像
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # 展开为列表，参见 DefaultFormatBundle
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # 设置默认元信息
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class LoadRadarPointsFromFile(BaseTransform):
    """
    从文件加载雷达点云数据
    
    Load radar points from file for nuScenes dataset.
    
    nuScenes雷达点云格式 (18维):
    - 0: x - X坐标 (m)
    - 1: y - Y坐标 (m)
    - 2: z - Z坐标 (m)
    - 3: dyn_prop - 动态属性
    - 4: id - 点ID
    - 5: rcs - 雷达散射截面 (dBsm)
    - 6: vx - X方向速度 (m/s)
    - 7: vy - Y方向速度 (m/s)
    - 8: vx_comp - 补偿后X方向速度 (m/s)
    - 9: vy_comp - 补偿后Y方向速度 (m/s)
    - 10: is_quality_valid - 质量有效标志
    - 11: ambig_state - 模糊状态
    - 12: x_rms - X位置误差
    - 13: y_rms - Y位置误差
    - 14: invalid_state - 无效状态
    - 15: pdh0 - 检测概率
    - 16: vx_rms - X速度误差
    - 17: vy_rms - Y速度误差
    
    默认使用维度 [0, 1, 2, 5, 8, 9] 对应 x, y, z, rcs, vx_comp, vy_comp
    
    Required Keys:
    - radar_info (dict): 包含雷达传感器信息的字典
        - radar_path (str): 雷达点云文件路径
        - radar2lidar (np.ndarray): 雷达到LiDAR的变换矩阵
    
    Added Keys:
    - radar_points (BasePoints): 雷达点云数据
    
    Args:
        coord_type (str): 坐标类型，默认'LIDAR'，表示输出点云在LiDAR坐标系下
            The type of coordinates of points cloud.
            Defaults to 'LIDAR'.
        load_dim (int): 加载的特征维度，nuScenes雷达为18维
            The dimension of the loaded points. Defaults to 18.
        use_dim (list[int]): 使用的特征维度索引
            Which dimensions of the points to use.
            Defaults to [0, 1, 2, 5, 8, 9] for x, y, z, rcs, vx_comp, vy_comp.
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend.
            Defaults to None.
    """

    def __init__(
        self,
        coord_type: str = 'LIDAR',
        load_dim: int = 18,
        use_dim: List[int] = [0, 1, 2, 5, 8, 9],
        backend_args: Optional[dict] = None
    ) -> None:
        # 验证参数
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH'], \
            f'coord_type must be CAMERA, LIDAR or DEPTH, got {coord_type}'
        
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """
        加载点云数据的私有函数
        
        Private function to load point clouds data.
        
        支持从远程后端或本地文件加载。
        nuScenes雷达点云存储为 .pcd 格式，需要使用 RadarPointCloud 类加载。
        
        Args:
            pts_filename (str): 点云文件路径
                Filename of point clouds data.

        Returns:
            np.ndarray: 点云数据数组，形状为 [N, 18]
                An array containing point clouds data.
        """
        from nuscenes.utils.data_classes import RadarPointCloud
        
        try:
            # 使用 nuScenes 的 RadarPointCloud 类加载 PCD 文件
            pc = RadarPointCloud.from_file(pts_filename)
            # RadarPointCloud.points 形状为 [18, N]，需要转置为 [N, 18]
            points = pc.points.T.astype(np.float32)
            return points
        except Exception as e:
            # 如果加载失败，返回空数组
            warnings.warn(f'Failed to load radar points from {pts_filename}: {e}')
            return np.zeros((0, self.load_dim), dtype=np.float32)

    def transform(self, results: dict) -> dict:
        """
        加载雷达点云并添加到results字典
        
        Load radar points from file and add to results dict.
        
        处理流程：
        1. 检查是否存在雷达信息
        2. 从文件加载雷达点云
        3. 使用标定矩阵将雷达点从传感器坐标系转换到LiDAR坐标系
        4. 选择需要的特征维度
        5. 创建点云对象并添加到results
        
        Args:
            results (dict): 包含radar_info的数据字典
                Result dict containing radar info.
                
        Returns:
            dict: 添加了radar_points的数据字典
                Result dict with radar_points added.
        """
        # 检查是否存在雷达信息
        if 'radar_info' not in results:
            # 如果没有雷达信息，返回空的雷达点云
            warnings.warn('radar_info not found in results, '
                         'returning empty radar points')
            points_class = get_points_type(self.coord_type)
            empty_points = np.zeros((0, len(self.use_dim)), dtype=np.float32)
            results['radar_points'] = points_class(
                empty_points,
                points_dim=len(self.use_dim),
                attribute_dims=None)
            return results
        
        radar_info = results['radar_info']
        
        # 收集所有雷达传感器的点云
        all_radar_points = []
        
        # 遍历所有雷达传感器
        for sensor_name, sensor_info in radar_info.items():
            if 'radar_path' not in sensor_info:
                continue
                
            radar_path = sensor_info['radar_path']
            
            try:
                # 加载雷达点云（已经是 [N, 18] 形状）
                points = self._load_points(radar_path)
                
                # 如果点云为空，跳过
                if points.shape[0] == 0:
                    continue
                
                # 获取雷达到LiDAR的变换矩阵
                if 'radar2lidar' in sensor_info:
                    radar2lidar = np.array(sensor_info['radar2lidar'])
                    # 应用坐标变换：将雷达点从传感器坐标系转换到LiDAR坐标系
                    # 变换公式: p_lidar = R @ p_radar + t
                    points[:, :3] = points[:, :3] @ radar2lidar[:3, :3].T
                    points[:, :3] += radar2lidar[:3, 3]
                    
                    # 如果有速度信息，也需要变换（只旋转，不平移）
                    # vx, vy 在索引 6, 7；vx_comp, vy_comp 在索引 8, 9
                    if self.load_dim >= 10:
                        # 变换原始速度
                        velocity = np.zeros((points.shape[0], 3))
                        velocity[:, :2] = points[:, 6:8]
                        velocity = velocity @ radar2lidar[:3, :3].T
                        points[:, 6:8] = velocity[:, :2]
                        
                        # 变换补偿后速度
                        velocity_comp = np.zeros((points.shape[0], 3))
                        velocity_comp[:, :2] = points[:, 8:10]
                        velocity_comp = velocity_comp @ radar2lidar[:3, :3].T
                        points[:, 8:10] = velocity_comp[:, :2]
                
                all_radar_points.append(points)
                
            except Exception as e:
                # 文件加载失败，记录警告并继续
                warnings.warn(f'Failed to load radar points from {radar_path}: {e}')
                continue
        
        # 合并所有雷达传感器的点云
        if len(all_radar_points) > 0:
            radar_points = np.concatenate(all_radar_points, axis=0)
        else:
            # 如果没有有效的雷达点云，返回空数组
            radar_points = np.zeros((0, self.load_dim), dtype=np.float32)
        
        # 选择需要的特征维度
        radar_points = radar_points[:, self.use_dim]
        
        # 创建点云对象
        points_class = get_points_type(self.coord_type)
        radar_points = points_class(
            radar_points,
            points_dim=radar_points.shape[-1],
            attribute_dims=None)
        
        results['radar_points'] = radar_points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'coord_type={self.coord_type}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim}, '
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadRadarPointsFromMultiSweeps(BaseTransform):
    """
    加载多帧雷达点云并合并
    
    Load radar points from multiple sweeps and merge them.
    
    将历史帧的雷达点云变换到当前帧坐标系后合并，
    增加雷达点云的密度。这对于稀疏的雷达点云特别有用。
    
    Required Keys:
    - radar_points (BasePoints): 当前帧雷达点云（由LoadRadarPointsFromFile加载）
    - radar_sweeps (list[dict]): 历史帧雷达sweep信息列表
    - timestamp (float): 当前帧时间戳
    
    Modified Keys:
    - radar_points (BasePoints): 合并后的多帧雷达点云
    
    Args:
        sweeps_num (int): 加载的历史帧数量
            Number of sweeps to load. Defaults to 5.
        load_dim (int): 加载的特征维度
            The dimension of the loaded points. Defaults to 18.
        use_dim (list[int]): 使用的特征维度
            Which dimensions of the points to use.
            Defaults to [0, 1, 2, 5, 8, 9].
        pad_empty_sweeps (bool): 当没有历史帧时是否复制当前帧
            Whether to repeat keyframe when sweeps is empty.
            Defaults to False.
        remove_close (bool): 是否移除原点附近的点（避免自车点）
            Whether to remove close points. Defaults to False.
        close_radius (float): 过近点的半径阈值
            Radius below which points are removed. Defaults to 1.0.
        test_mode (bool): 测试模式下选择最近的N帧，否则随机选择
            If test_mode=True, select the nearest N frames.
            Otherwise, randomly select. Defaults to False.
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend.
            Defaults to None.
    """

    def __init__(
        self,
        sweeps_num: int = 5,
        load_dim: int = 18,
        use_dim: List[int] = [0, 1, 2, 5, 8, 9],
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        close_radius: float = 1.0,
        test_mode: bool = False,
        backend_args: Optional[dict] = None
    ) -> None:
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.close_radius = close_radius
        self.test_mode = test_mode
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """
        加载点云数据的私有函数
        
        Private function to load point clouds data.
        
        Args:
            pts_filename (str): 点云文件路径
                Filename of point clouds data.

        Returns:
            np.ndarray: 点云数据数组
                An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            # np.frombuffer返回只读数组，需要复制一份可写的
            points = np.copy(points)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(
        self,
        points: Union[np.ndarray, BasePoints],
        radius: float = 1.0
    ) -> Union[np.ndarray, BasePoints]:
        """
        移除原点附近的点（通常是自车上的点）
        
        Remove points too close within a certain radius from origin.
        
        Args:
            points (np.ndarray | BasePoints): 点云数据
                Sweep points.
            radius (float): 移除半径，小于此距离的点将被移除
                Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | BasePoints: 移除后的点云
                Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        
        # 计算 x 和 y 方向上是否在半径内
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        # 保留不在原点附近的点
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """
        加载多帧雷达点云并融合
        
        Load multi-sweep radar points and merge them.
        
        处理流程：
        1. 获取当前帧雷达点云
        2. 选择历史帧（随机或顺序）
        3. 加载历史帧雷达点云并变换到当前帧坐标系
        4. 添加时间戳差作为特征（如果需要）
        5. 拼接所有帧的点云
        
        Args:
            results (dict): 包含雷达点云和sweep信息的结果字典
                Result dict containing radar points and sweep info.

        Returns:
            dict: 包含融合后雷达点云的结果字典
                The result dict containing the multi-sweep radar points.
        """
        # 获取当前帧雷达点云
        if 'radar_points' not in results:
            warnings.warn('radar_points not found in results, skipping multi-sweep loading')
            return results
        
        points = results['radar_points']
        sweep_points_list = [points]
        
        # 获取当前帧时间戳
        ts = results.get('timestamp', 0)
        
        # 如果没有历史帧信息
        if 'radar_sweeps' not in results or len(results['radar_sweeps']) == 0:
            if self.pad_empty_sweeps:
                # 复制当前帧填充
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(
                            self._remove_close(points, self.close_radius))
                    else:
                        sweep_points_list.append(points)
        else:
            radar_sweeps = results['radar_sweeps']
            
            # 选择历史帧
            if len(radar_sweeps) <= self.sweeps_num:
                # 历史帧不足，全部使用
                choices = np.arange(len(radar_sweeps))
            elif self.test_mode:
                # 测试模式：选择最近的帧
                choices = np.arange(self.sweeps_num)
            else:
                # 训练模式：随机选择
                choices = np.random.choice(
                    len(radar_sweeps),
                    self.sweeps_num,
                    replace=False)
            
            # 加载并处理每一帧历史点云
            for idx in choices:
                sweep = radar_sweeps[idx]
                
                # 收集该sweep中所有雷达传感器的点云
                sweep_all_points = []
                
                for sensor_name, sensor_info in sweep.items():
                    if sensor_name == 'timestamp':
                        continue
                    if not isinstance(sensor_info, dict):
                        continue
                    if 'radar_path' not in sensor_info:
                        continue
                    
                    try:
                        # 加载雷达点云
                        points_sweep = self._load_points(sensor_info['radar_path'])
                        points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                        
                        if points_sweep.shape[0] == 0:
                            continue
                        
                        # 获取历史帧到当前帧的变换矩阵
                        # radar2lidar: 雷达传感器到当前帧LiDAR的变换
                        if 'radar2lidar' in sensor_info:
                            radar2lidar = np.array(sensor_info['radar2lidar'])
                            # 应用坐标变换
                            points_sweep[:, :3] = points_sweep[:, :3] @ radar2lidar[:3, :3].T
                            points_sweep[:, :3] += radar2lidar[:3, 3]
                            
                            # 变换速度（只旋转）
                            if self.load_dim >= 10:
                                velocity = np.zeros((points_sweep.shape[0], 3))
                                velocity[:, :2] = points_sweep[:, 6:8]
                                velocity = velocity @ radar2lidar[:3, :3].T
                                points_sweep[:, 6:8] = velocity[:, :2]
                                
                                velocity_comp = np.zeros((points_sweep.shape[0], 3))
                                velocity_comp[:, :2] = points_sweep[:, 8:10]
                                velocity_comp = velocity_comp @ radar2lidar[:3, :3].T
                                points_sweep[:, 8:10] = velocity_comp[:, :2]
                        
                        sweep_all_points.append(points_sweep)
                        
                    except Exception as e:
                        warnings.warn(f'Failed to load radar sweep: {e}')
                        continue
                
                if len(sweep_all_points) == 0:
                    continue
                
                # 合并该sweep的所有雷达点云
                points_sweep = np.concatenate(sweep_all_points, axis=0)
                
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep, self.close_radius)
                
                # 选择需要的维度
                points_sweep = points_sweep[:, self.use_dim]
                
                # 创建点云对象
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)
        
        # 拼接所有帧的点云
        if len(sweep_points_list) > 1:
            points = points.cat(sweep_points_list)
        
        results['radar_points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'sweeps_num={self.sweeps_num}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim}, '
        repr_str += f'pad_empty_sweeps={self.pad_empty_sweeps}, '
        repr_str += f'remove_close={self.remove_close}, '
        repr_str += f'close_radius={self.close_radius}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str

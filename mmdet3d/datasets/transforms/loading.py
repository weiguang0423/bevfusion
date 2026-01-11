# Copyright (c) OpenMMLab. All rights reserved.
"""
mmdet3d 数据加载模块 (Data Loading Module)

本模块实现了3D目标检测中各种数据的加载功能，包括：
- 多视角图像加载 (LoadMultiViewImageFromFiles)
- 单目3D图像加载 (LoadImageFromFileMono3D)
- 点云数据加载 (LoadPointsFromFile)
- 多帧点云加载 (LoadPointsFromMultiSweeps)
- 3D标注加载 (LoadAnnotations3D)
- 推理器加载器 (Inferencer Loaders)

这些加载器作为数据处理 pipeline 的第一步，负责从文件系统读取原始数据
并转换为模型可以处理的格式。
"""
import copy
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(BaseTransform):
    """从多个文件加载多视角图像
    
    Load multi channel images from a list of separate channel files.

    该类用于加载多视角相机图像，常用于 nuScenes 等数据集。
    支持多帧时序数据加载，可用于时序融合模型。
    
    主要功能：
    - 加载多个相机视角的图像
    - 处理不同尺寸的图像（自动 padding）
    - 支持多帧历史数据加载
    - 提取相机内外参矩阵
    
    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): 是否将图像转换为 float32 类型
            Whether to convert the img to float32.
            Defaults to False.
        color_type (str): 图像颜色类型
            Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend. 
            Defaults to None.
        num_views (int): 每帧的视角数量（如 nuScenes 有6个相机）
            Number of view in a frame. Defaults to 5.
        num_ref_frames (int): 参考帧数量，用于时序融合，-1表示不使用
            Number of frame in loading. Defaults to -1.
        test_mode (bool): 是否为测试模式（影响帧选择策略）
            Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): 是否设置默认缩放因子
            Whether to set default scale. Defaults to True.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 5,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.num_views = num_views
        # num_ref_frames 用于多帧时序加载
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # test_mode=False 时随机选择历史帧，否则选择最近的帧
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def transform(self, results: dict) -> Optional[dict]:
        """加载多视角图像的主函数
        
        Call function to load multi-view image from files.

        处理流程：
        1. 如果启用多帧加载，选择并处理历史帧
        2. 从 results['images'] 中提取文件路径和相机参数
        3. 读取所有图像并处理尺寸差异
        4. 设置输出的元信息
        
        Args:
            results (dict): 包含图像文件名的结果字典
                Result dict containing multi-view image filenames.

        Returns:
            dict: 包含多视角图像数据的结果字典，添加的键值：
                The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): 多视角图像文件名
                    Multi-view image filenames.
                - img (np.ndarray): 多视角图像数组
                    Multi-view image arrays.
                - img_shape (tuple[int]): 图像形状
                    Shape of multi-view image arrays.
                - ori_shape (tuple[int]): 原始图像形状
                    Shape of original image arrays.
                - pad_shape (tuple[int]): padding后的形状
                    Shape of padded image arrays.
                - scale_factor (float): 缩放因子
                    Scale factor.
                - img_norm_cfg (dict): 图像归一化配置
                    Normalization configuration of images.
        """
        # TODO: 考虑将多帧处理部分拆分出去
        # TODO: consider split the multi-sweep part out of this pipeline
        # 处理多帧时序数据的选择逻辑
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # 初始化选择：当前帧索引为0
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            # 计算可用的历史帧数量
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # 没有历史帧，复制当前帧
                # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # 历史帧数量充足
                # NOTE: 假设信息按从最新到最早的顺序保存
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    # 测试模式：选择最近的 N 帧
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 是为了跳过当前帧，选择历史帧
                # NOTE: +1 is for selecting previous frames
                else:
                    # 训练模式：随机选择历史帧
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                # 历史帧数量不足，需要重复采样
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
            # 合并当前帧和选择的历史帧
            choices = np.concatenate([init_choice, choices])
            # 根据选择的帧索引，提取对应的文件名
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            # 同步更新相机参数
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
            # 将 lidar2cam 变换为 [当前LiDAR]到[历史图像] 的变换
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # 只修改历史帧的变换矩阵
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
                        # 计算当前帧到历史帧的相对变换
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        # 更新历史帧的 lidar2cam 矩阵
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # ============ 从 images 字典中提取相机信息 ============
        # Support multi-view images with different shapes
        # TODO: 记录原始形状和 padding 后的形状
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam = [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        # 保存原始相机内参（用于后续数据增强的逆变换）
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # ============ 加载图像数据 ============
        # 图像形状为 (h, w, c, num_views)
        # 不同视角的 h 和 w 可能不同
        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # ============ 处理不同尺寸的图像 ============
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        # 确保所有图像通道数相同
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            # 如果尺寸不同，padding 到最大尺寸
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

        # ============ 设置输出结果 ============
        results['filename'] = filename
        # 展开为列表，参见 formating.py 中的 DefaultFormatBundle
        # 每个图像会单独转置，然后堆叠成数组
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

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """单目3D检测图像加载器
    
    Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    与2D检测相比，单目3D检测需要额外加载相机参数（内参矩阵）。
    支持 KITTI（CAM2）和 nuScenes（CAM_FRONT）数据集。
    
    Args:
        kwargs (dict): 参数与 LoadImageFromFile 相同
            Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def transform(self, results: dict) -> dict:
        """加载图像并获取相机参数
        
        Call functions to load image and get image meta information.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: 包含图像和元信息的字典
                The dict contains loaded image and meta information.
        """
        # TODO: 从 data info 中加载不同相机的图像
        # KITTI 数据集加载 'CAM2' 图像
        # nuScenes 数据集加载 'CAM_FRONT' 图像
        # TODO: load different camera image from data info,
        # for kitti dataset, we load 'CAM2' image.
        # for nuscenes dataset, we load 'CAM_FRONT' image.

        if 'CAM2' in results['images']:
            filename = results['images']['CAM2']['img_path']
            results['cam2img'] = results['images']['CAM2']['cam2img']
        elif len(list(results['images'].keys())) == 1:
            camera_type = list(results['images'].keys())[0]
            filename = results['images'][camera_type]['img_path']
            results['cam2img'] = results['images'][camera_type]['cam2img']
        else:
            raise NotImplementedError(
                'Currently we only support load image from kitti and '
                'nuscenes datasets')

        try:
            img_bytes = get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """从 numpy 数组加载图像
    
    Load an image from ``results['img']``.
    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.
    
    与 LoadImageFromFile 类似，但图像已经作为 np.ndarray 存在于
    results['img'] 中。可用于从摄像头加载图像的场景。
    
    Required Keys:
    - img
    
    Modified Keys:
    - img
    - img_path
    - img_shape
    - ori_shape
    
    Args:
        to_float32 (bool): 是否将图像转换为 float32
            Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """添加图像元信息
        
        Transform function to add image meta information.

        Args:
            results (dict): 包含摄像头读取图像的结果字典
                Result dict with Webcam read image in
                ``results['img']``.
        Returns:
            dict: 包含图像和元信息的字典
                The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweeps(BaseTransform):
    """从多帧扫描加载点云（时序点云融合）
    
    Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    该类通常用于 nuScenes 数据集，利用历史帧的点云数据增强当前帧。
    通过融合多帧点云，可以增加点云密度，提高检测效果。
    
    主要功能：
    - 加载指定数量的历史帧点云
    - 将历史帧点云变换到当前帧坐标系
    - 添加时间戳差作为额外特征
    - 可选择移除原点附近的点（避免自车点）
    
    Args:
        sweeps_num (int): 加载的历史帧数量
            Number of sweeps. Defaults to 10.
        load_dim (int): 加载点云的维度数
            Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): 使用哪些维度
            Which dimension to use. Defaults to [0, 1, 2, 4].
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend. 
            Defaults to None.
        pad_empty_sweeps (bool): 当没有历史帧时是否复制当前帧
            Whether to repeat keyframe when sweeps is empty. 
            Defaults to False.
        remove_close (bool): 是否移除原点附近的点
            Whether to remove close points. Defaults to False.
        test_mode (bool): 测试模式下选择最近的N帧，否则随机选择
            If `test_mode=True`, it will not randomly sample
            sweeps but select the nearest N frames. Defaults to False.
    """

    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 5,
                 use_dim: List[int] = [0, 1, 2, 4],
                 backend_args: Optional[dict] = None,
                 pad_empty_sweeps: bool = False,
                 remove_close: bool = False,
                 test_mode: bool = False) -> None:
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """加载点云数据的私有函数
        
        Private function to load point clouds data.

        支持从远程后端或本地文件加载，支持 .bin 和 .npy 格式。
        
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
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self,
                      points: Union[np.ndarray, BasePoints],
                      radius: float = 1.0) -> Union[np.ndarray, BasePoints]:
        """移除原点附近的点（通常是自车上的点）
        
        Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 点云数据
                Sweep points.
            radius (float): 移除半径，小于此距离的点将被移除
                Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: 移除后的点云
                Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.numpy()
        else:
            raise NotImplementedError
        # 计算 x 和 y 方向上是否在半径内
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        # 保留不在原点附近的点
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """加载多帧点云并融合
        
        Call function to load multi-sweep point clouds from files.

        处理流程：
        1. 获取当前帧点云，设置时间戳为0
        2. 选择历史帧（随机或顺序）
        3. 加载历史帧点云并变换到当前帧坐标系
        4. 添加时间戳差作为特征
        5. 拼接所有帧的点云
        
        Args:
            results (dict): 包含多帧点云文件名的结果字典
                Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: 包含融合后点云的结果字典
                The result dict containing the multi-sweep points data.
                Updated key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): 多帧融合点云
                    Multi-sweep point cloud arrays.
        """
        points = results['points']
        # 当前帧的时间戳设为0
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        
        # 如果没有历史帧信息
        if 'lidar_sweeps' not in results:
            if self.pad_empty_sweeps:
                # 复制当前帧填充
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(self._remove_close(points))
                    else:
                        sweep_points_list.append(points)
        else:
            # 选择历史帧
            if len(results['lidar_sweeps']) <= self.sweeps_num:
                # 历史帧不足，全部使用
                choices = np.arange(len(results['lidar_sweeps']))
            elif self.test_mode:
                # 测试模式：选择最近的帧
                choices = np.arange(self.sweeps_num)
            else:
                # 训练模式：随机选择
                choices = np.random.choice(
                    len(results['lidar_sweeps']),
                    self.sweeps_num,
                    replace=False)
            
            # 加载并处理每一帧历史点云
            for idx in choices:
                sweep = results['lidar_sweeps'][idx]
                points_sweep = self._load_points(
                    sweep['lidar_points']['lidar_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # 注意：pkl 文件中的时间戳已经除以了 1e6
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                sweep_ts = sweep['timestamp']
                # 获取历史帧到当前帧的变换矩阵
                lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                # 将历史帧点云变换到当前帧坐标系
                points_sweep[:, :
                             3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                # 设置时间戳差作为特征
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        # 拼接所有帧的点云
        points = points.cat(sweep_points_list)
        # 只保留需要的维度
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@TRANSFORMS.register_module()
class PointSegClassMapping(BaseTransform):
    """点云语义分割类别映射
    
    Map original semantic class to valid category ids.

    将原始语义类别映射到有效的类别ID。
    有效类别映射为 0~len(valid_cat_ids)-1，
    其他类别映射为 len(valid_cat_ids)。
    
    Required Keys:
    - seg_label_mapping (np.ndarray): 类别映射表
    - pts_semantic_mask (np.ndarray): 原始语义掩码

    Added Keys:
    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """执行类别映射
        
        Call function to map original semantic class to valid category ids.

        Args:
            results (dict): 包含点云语义掩码的结果字典
                Result dict containing point semantic masks.

        Returns:
            dict: 包含映射后类别ID的结果字典
                The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): 映射后的语义掩码
                    Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = label_mapping[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NormalizePointsColor(BaseTransform):
    """点云颜色归一化
    
    Normalize color of points.

    将点云的颜色值归一化到 [0, 1] 范围，并可选择减去均值。
    常用于室内点云数据集（如 ScanNet）。
    
    Args:
        color_mean (list[float]): 点云颜色均值
            Mean color of the point cloud.
    """

    def __init__(self, color_mean: List[float]) -> None:
        self.color_mean = color_mean

    def transform(self, input_dict: dict) -> dict:
        """执行颜色归一化
        
        Call function to normalize color of points.

        Args:
            results (dict): 包含点云数据的结果字典
                Result dict containing point clouds data.

        Returns:
            dict: 包含归一化后点云的结果字典
                The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): 颜色归一化后的点云
                    Points after color normalization.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        input_dict['points'] = points
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromFile(BaseTransform):
    """从文件加载点云数据
    
    Load Points From File.

    这是最基础的点云加载器，从 .bin 或 .npy 文件读取点云数据，
    并转换为 BasePoints 对象。
    
    主要功能：
    - 支持 LIDAR、DEPTH、CAMERA 三种坐标系
    - 支持强度归一化和延展度归一化
    - 支持高度偏移（室内数据集常用）
    - 支持颜色特征
    
    Required Keys:
    - lidar_points (dict)
        - lidar_path (str): 点云文件路径

    Added Keys:
    - points (np.float32): 点云数据

    Args:
        coord_type (str): 点云坐标系类型
            The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': LiDAR 坐标系（车载激光雷达）
                Points in LiDAR coordinates.
            - 'DEPTH': 深度坐标系（室内数据集）
                Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': 相机坐标系
                Points in camera coordinates.
        load_dim (int): 加载点云的维度数
            The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): 使用哪些维度
            Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): 是否使用高度偏移（减去地面高度）
            Whether to use shifted height. Defaults to False.
        use_color (bool): 是否使用颜色特征
            Whether to use color features. Defaults to False.
        norm_intensity (bool): 是否归一化强度值（使用 tanh）
            Whether to normlize the intensity. Defaults to False.
        norm_elongation (bool): 是否归一化延展度（Waymo 数据集常用）
            Whether to normlize the elongation. This is
            usually used in Waymo dataset. Defaults to False.
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend. 
            Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """加载点云数据的私有函数
        
        Private function to load point clouds data.

        支持从远程后端或本地文件加载，支持 .bin 和 .npy 格式。
        
        Args:
            pts_filename (str): 点云文件路径
                Filename of point clouds data.

        Returns:
            np.ndarray: 点云数据数组
                An array containing point clouds data.
        """
        try:
            # 尝试从后端加载
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            # 后端连接失败，从本地文件加载
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """从文件加载点云数据
        
        Method to load points data from file.

        处理流程：
        1. 从文件读取原始点云数据
        2. 重塑为 (N, load_dim) 形状
        3. 选择需要的维度
        4. 可选：强度/延展度归一化
        5. 可选：高度偏移处理
        6. 可选：颜色特征处理
        7. 转换为 BasePoints 对象
        
        Args:
            results (dict): 包含点云数据的结果字典
                Result dict containing point clouds data.

        Returns:
            dict: 包含点云数据的结果字典
                The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): 点云数据对象
                    Point clouds data.
        """
        # 获取点云文件路径
        pts_file_path = results['lidar_points']['lidar_path']
        # 加载原始点云数据
        points = self._load_points(pts_file_path)
        # 重塑为 (N, load_dim) 形状
        points = points.reshape(-1, self.load_dim)
        # 选择需要的维度
        points = points[:, self.use_dim]
        
        # 强度归一化：使用 tanh 函数
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        # 延展度归一化：Waymo 数据集特有
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        # 高度偏移处理：减去地面高度（室内数据集常用）
        if self.shift_height:
            # 使用第 0.99 百分位作为地面高度估计
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            # 将高度作为额外特征插入
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        # 颜色特征处理
        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            # 记录颜色特征的维度索引
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        # 根据坐标系类型获取对应的点云类
        points_class = get_points_type(self.coord_type)
        # 创建点云对象
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """从字典加载点云数据
    
    Load Points From Dict.

    与 LoadPointsFromFile 类似，但点云数据已经作为 numpy 数组
    存在于 results['points'] 中。用于推理或在线数据处理场景。
    """

    def transform(self, results: dict) -> dict:
        """将 numpy 数组转换为对应的点云类对象
        
        Convert the type of points from ndarray to corresponding
        `point_class`.

        Args:
            results (dict): 输入结果字典，'points' 键的值为 numpy 数组
                input result. The value of key `points` is a
                numpy array.

        Returns:
            dict: 处理后的结果字典
                The processed results.
        """
        assert 'points' in results
        points = results['points']

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results


@TRANSFORMS.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """加载3D标注数据
    
    Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    加载3D目标检测和分割所需的各种标注，包括：
    - 3D边界框和类别标签
    - 2D边界框和类别标签
    - 点云实例掩码和语义掩码
    - 属性标签（如 nuScenes 的行人姿态）
    - 2.5D 边界框（深度和2D中心）
    
    Required Keys:
    - ann_info (dict)
        - gt_bboxes_3d: 3D边界框
            (:obj:`LiDARInstance3DBoxes` |
            :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
            3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): 3D标签
            Labels of ground truths. Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D边界框
            2D ground truth bboxes. Only when `with_bbox` is True.
        - gt_labels (np.ndarray): 2D标签
            Labels of ground truths. Only when `with_label` is True.
        - depths (np.ndarray): 深度值
            Only when `with_bbox_depth` is True.
        - centers_2d (np.ndarray): 2D中心点
            Only when `with_bbox_depth` is True.
        - attr_labels (np.ndarray): 属性标签
            Attribute labels of instances. Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): 实例掩码文件路径
        Path of instance mask file. Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): 语义掩码文件路径
        Path of semantic mask file. Only when `with_seg_3d` is True.
    - pts_panoptic_mask_path (str): 全景掩码文件路径
        Path of panoptic mask file. Only when both `with_panoptic_3d` is True.

    Added Keys:
    - gt_bboxes_3d: 3D边界框
    - gt_labels_3d: 3D标签
    - gt_bboxes: 2D边界框
    - gt_labels: 2D标签
    - depths: 深度值
    - centers_2d: 2D中心点
    - attr_labels: 属性标签
    - pts_instance_mask: 点云实例掩码
    - pts_semantic_mask: 点云语义掩码

    Args:
        with_bbox_3d (bool): 是否加载3D边界框
            Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): 是否加载3D标签
            Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): 是否加载属性标签
            Whether to load attribute label. Defaults to False.
        with_mask_3d (bool): 是否加载点云实例掩码
            Whether to load 3D instance masks for points. Defaults to False.
        with_seg_3d (bool): 是否加载点云语义掩码
            Whether to load 3D semantic masks for points. Defaults to False.
        with_bbox (bool): 是否加载2D边界框
            Whether to load 2D boxes. Defaults to False.
        with_label (bool): 是否加载2D标签
            Whether to load 2D labels. Defaults to False.
        with_mask (bool): 是否加载2D实例掩码
            Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): 是否加载2D语义掩码
            Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): 是否加载2.5D边界框
            Whether to load 2.5D boxes. Defaults to False.
        with_panoptic_3d (bool): 是否加载3D全景掩码
            Whether to load 3D panoptic masks for points. Defaults to False.
        poly2mask (bool): 是否将多边形标注转换为位图掩码
            Whether to convert polygon annotations to bitmasks. Defaults to True.
        seg_3d_dtype (str): 3D语义掩码的数据类型
            String of dtype of 3D semantic masks. Defaults to 'np.int64'.
        seg_offset (int): 分割全景标签的偏移量
            The offset to split semantic and instance labels from
            panoptic labels. Defaults to None.
        dataset_type (str): 数据集类型（用于特殊处理）
            Type of dataset used for splitting semantic and
            instance labels. Defaults to None.
        backend_args (dict, optional): 文件后端参数
            Arguments to instantiate the corresponding backend. 
            Defaults to None.
    """

    def __init__(self,
                 with_bbox_3d: bool = True,
                 with_label_3d: bool = True,
                 with_attr_label: bool = False,
                 with_mask_3d: bool = False,
                 with_seg_3d: bool = False,
                 with_bbox: bool = False,
                 with_label: bool = False,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 with_bbox_depth: bool = False,
                 with_panoptic_3d: bool = False,
                 poly2mask: bool = True,
                 seg_3d_dtype: str = 'np.int64',
                 seg_offset: int = None,
                 dataset_type: str = None,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            backend_args=backend_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_panoptic_3d = with_panoptic_3d
        self.seg_3d_dtype = eval(seg_3d_dtype)
        self.seg_offset = seg_offset
        self.dataset_type = dataset_type

    def _load_bboxes_3d(self, results: dict) -> dict:
        """加载3D边界框标注
        
        Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含3D边界框标注的字典
                The dict containing loaded 3D bounding box annotations.
        """

        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        """加载2.5D边界框标注（深度和2D中心）
        
        Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含2.5D边界框标注的字典
                The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        """加载3D标签
        
        Private function to load label annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含标签标注的字典
                The dict containing loaded label annotations.
        """

        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        """加载属性标签（如 nuScenes 的行人姿态）
        
        Private function to load label annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含属性标签的字典
                The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results: dict) -> dict:
        """加载3D实例掩码
        
        Private function to load 3D mask annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含3D实例掩码的字典
                The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['pts_instance_mask_path']

        try:
            mask_bytes = get(
                pts_instance_mask_path, backend_args=self.backend_args)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' 会传递给评估器
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """加载3D语义分割标注
        
        Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含语义分割标注的字典
                The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        try:
            mask_bytes = get(
                pts_semantic_mask_path, backend_args=self.backend_args)
            # 添加 .copy() 修复只读 bug
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        # SemanticKITTI 数据集特殊处理
        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes 从不同文件加载语义和全景标签
        # nuScenes loads semantic and panoptic labels from different files.

        results['pts_semantic_mask'] = pts_semantic_mask

        # 'eval_ann_info' 会传递给评估器
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results

    def _load_panoptic_3d(self, results: dict) -> dict:
        """加载3D全景分割标注
        
        Private function to load 3D panoptic segmentation annotations.

        全景分割同时包含语义和实例信息，需要根据偏移量分离。
        
        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含全景分割标注的字典
                The dict containing the panoptic segmentation annotations.
        """
        pts_panoptic_mask_path = results['pts_panoptic_mask_path']

        try:
            mask_bytes = get(
                pts_panoptic_mask_path, backend_args=self.backend_args)
            # 添加 .copy() 修复只读 bug
            # add .copy() to fix read-only bug
            pts_panoptic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_panoptic_mask_path)
            pts_panoptic_mask = np.fromfile(
                pts_panoptic_mask_path, dtype=np.int64)

        # 根据数据集类型分离语义和实例标签
        if self.dataset_type == 'semantickitti':
            # SemanticKITTI: 语义标签 = 全景标签 % 偏移量
            pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        elif self.dataset_type == 'nuscenes':
            # nuScenes: 语义标签 = 全景标签 // 偏移量
            pts_semantic_mask = pts_semantic_mask // self.seg_offset

        results['pts_semantic_mask'] = pts_semantic_mask

        # 全景标签可以直接作为实例ID使用
        # We can directly take panoptic labels as instance ids.
        pts_instance_mask = pts_panoptic_mask.astype(np.int64)
        results['pts_instance_mask'] = pts_instance_mask

        # 'eval_ann_info' 会传递给评估器
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_bboxes(self, results: dict) -> None:
        """加载2D边界框标注
        
        Private function to load bounding box annotations.

        与父类的唯一区别是移除了 ignore_flag 的处理。
        The only difference is it remove the proceess for `ignore_flag`

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: 包含2D边界框标注的字典
                The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        """加载2D标签
        
        Private function to load label annotations.

        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: 包含标签标注的字典
                The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

    def transform(self, results: dict) -> dict:
        """加载多种类型的标注
        
        Function to load multiple types annotations.

        根据初始化时的配置，加载对应类型的标注数据。
        
        Args:
            results (dict): 来自数据集的结果字典
                Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: 包含3D边界框、标签、掩码和语义分割标注的字典
                The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        # 先调用父类加载2D标注
        results = super().transform(results)
        # 根据配置加载各种3D标注
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_panoptic_3d={self.with_panoptic_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        repr_str += f'{indent_str}seg_offset={self.seg_offset})'

        return repr_str


@TRANSFORMS.register_module()
class LidarDet3DInferencerLoader(BaseTransform):
    """LiDAR 3D检测推理加载器
    
    Load point cloud in the Inferencer's pipeline.

    用于推理 pipeline 中加载点云数据，支持从文件路径或 numpy 数组加载。
    
    Added keys:
      - points: 点云数据
      - timestamp: 时间戳
      - axis_align_matrix: 轴对齐矩阵（ScanNet 需要）
      - box_type_3d: 3D边界框类型
      - box_mode_3d: 3D边界框模式
    """

    def __init__(self, coord_type='LIDAR', **kwargs) -> None:
        super().__init__()
        # 从文件加载的变换
        self.from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', coord_type=coord_type, **kwargs))
        # 从数组加载的变换
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', coord_type=coord_type, **kwargs))
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """加载点云数据
        
        Transform function to add image meta information.
        
        Args:
            single_input (dict): 单个输入，必须包含 'points' 键
                Single input.

        Returns:
            dict: 包含点云和元信息的字典
                The dict contains loaded image and meta information.
        """
        assert 'points' in single_input, "key 'points' must be in input dict"
        # 根据输入类型选择加载方式
        if isinstance(single_input['points'], str):
            # 从文件路径加载
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # ScanNet demo 需要轴对齐矩阵
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            # 从 numpy 数组加载
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # ScanNet demo 需要轴对齐矩阵
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class MonoDet3DInferencerLoader(BaseTransform):
    """单目3D检测推理加载器
    
    Load an image from ``results['images']['CAMX']['img']``. Similar with
    :obj:`LoadImageFromFileMono3D`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['images']['CAMX']['img']``.

    用于单目3D检测推理 pipeline 中加载图像数据。
    
    Added keys:
      - img: 图像数据
      - box_type_3d: 3D边界框类型
      - box_mode_3d: 3D边界框模式
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFileMono3D', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: dict) -> dict:
        """加载图像数据
        
        Transform function to add image meta information.

        Args:
            single_input (dict): 包含图像的结果字典
                Result dict with Webcam read image in
                ``results['images']['CAMX']['img']``.
        Returns:
            dict: 包含图像和元信息的字典
                The dict contains loaded image and meta information.
        """
        box_type_3d, box_mode_3d = get_box_type('camera')

        # 根据输入类型选择加载方式
        if isinstance(single_input['img'], str):
            # 从文件路径加载
            inputs = dict(
                images=dict(
                    CAM_FRONT=dict(
                        img_path=single_input['img'],
                        cam2img=single_input['cam2img'])),
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            # 从 numpy 数组加载
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class MultiModalityDet3DInferencerLoader(BaseTransform):
    """多模态3D检测推理加载器
    
    Load point cloud and image in the Inferencer's pipeline.

    用于多模态（点云+图像）3D检测推理 pipeline 中同时加载点云和图像数据。
    
    Added keys:
      - points: 点云数据
      - img: 图像数据
      - cam2img: 相机内参矩阵
      - lidar2cam: LiDAR到相机变换矩阵
      - lidar2img: LiDAR到图像投影矩阵
      - timestamp: 时间戳
      - axis_align_matrix: 轴对齐矩阵
      - box_type_3d: 3D边界框类型
      - box_mode_3d: 3D边界框模式
    """

    def __init__(self, load_point_args: dict, load_img_args: dict) -> None:
        super().__init__()
        # 点云加载器
        self.points_from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', **load_point_args))
        self.points_from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', **load_point_args))
        coord_type = load_point_args['coord_type']
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

        # 图像加载器
        self.imgs_from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **load_img_args))
        self.imgs_from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **load_img_args))

    def transform(self, single_input: dict) -> dict:
        """加载点云和图像数据
        
        Transform function to add image meta information.
        
        Args:
            single_input (dict): 单个输入，必须包含 'points' 和 'img' 键
                Single input.

        Returns:
            dict: 包含点云、图像和元信息的字典
                The dict contains loaded image, point cloud and meta
                information.
        """
        assert 'points' in single_input and 'img' in single_input, \
            "key 'points', 'img' and must be in input dict," \
            f'but got {single_input}'
        
        # ============ 加载点云数据 ============
        if isinstance(single_input['points'], str):
            # 从文件路径加载点云
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # ScanNet demo 需要轴对齐矩阵
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            # 从 numpy 数组加载点云
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # ScanNet demo 需要轴对齐矩阵
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            points_inputs = self.points_from_ndarray(inputs)
        else:
            points_inputs = self.points_from_file(inputs)

        multi_modality_inputs = points_inputs

        # ============ 加载图像数据 ============
        box_type_3d, box_mode_3d = get_box_type('lidar')

        if isinstance(single_input['img'], str):
            # 从文件路径加载图像
            inputs = dict(
                img_path=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            # 从 numpy 数组加载图像
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if isinstance(single_input['img'], np.ndarray):
            imgs_inputs = self.imgs_from_ndarray(inputs)
        else:
            imgs_inputs = self.imgs_from_file(inputs)

        # 合并点云和图像的输入
        multi_modality_inputs.update(imgs_inputs)

        return multi_modality_inputs

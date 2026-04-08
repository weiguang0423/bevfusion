# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Set, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class CBGSDataset:
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`BaseDataset` or dict): The dataset to be class sampled.
        lazy_init (bool): Whether to load annotation during instantiation.
            Defaults to False.
        weather_resampling_ratios (dict, optional): Extra weather-based
            resampling multipliers applied on top of CBGS. Defaults to
            ``dict(day=1, night=2, rain=3)`` to preserve the current
            repository behavior.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 lazy_init: bool = False,
                 weather_resampling_ratios: Union[dict, None] = None) -> None:
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo
        default_weather_ratios = dict(day=1, night=2, rain=3)
        if weather_resampling_ratios is not None:
            default_weather_ratios.update(weather_resampling_ratios)
        self.weather_resampling_ratios = default_weather_ratios

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get sample_indices
        self.sample_indices = self._get_sample_indices(self.dataset)

        self._fully_initialized = True

    def _get_sample_indices(self, dataset: BaseDataset) -> List[int]:
        """Load sample indices according to ann_file.

        Args:
            dataset (:obj:`BaseDataset`): The dataset.

        Returns:
            List[dict]: List of indices after class sampling.
        """
        classes = self.metainfo['classes']
        cat2id = {name: i for i, name in enumerate(classes)}
        class_sample_idxs = {cat_id: [] for cat_id in cat2id.values()}
        for idx in range(len(dataset)):
            sample_cat_ids = dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                if cat_id != -1:
                    # Filter categories that do not need to be cared.
                    # -1 indicates dontcare in MMDet3D.
                    class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(classes)
        ratios = [frac / v for v in class_distribution.values()]

        # Build token to weather description map
        token2desc = {}
        try:
            import json
            import os.path as osp
            data_root = 'data/nuscenes'
            sample_json_path = osp.join(data_root, 'v1.0-trainval', 'sample.json')
            scene_json_path = osp.join(data_root, 'v1.0-trainval', 'scene.json')
            if osp.exists(sample_json_path) and osp.exists(scene_json_path):
                with open(sample_json_path, 'r') as f:
                    samples = json.load(f)
                with open(scene_json_path, 'r') as f:
                    scenes = json.load(f)
                scene_token2desc = {scene['token']: scene['description'].lower() for scene in scenes}
                for sample in samples:
                    token2desc[sample['token']] = scene_token2desc.get(sample['scene_token'], '')
        except Exception as e:
            print(f"Warning: Failed to load weather info. Reason: {e}")

        # Apply CBGS ratios first
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            base_sample_count = int(len(cls_inds) * ratio)
            sampled_idxs = np.random.choice(cls_inds, base_sample_count).tolist()
            
            # Apply weather oversampling on top of the sampled indices
            for idx in sampled_idxs:
                info = dataset.get_data_info(idx)
                
                token = info.get('token', '')
                scene_desc = token2desc.get(token, '')
                
                if 'rain' in scene_desc:
                    multiplier = self.weather_resampling_ratios['rain']
                elif 'night' in scene_desc:
                    multiplier = self.weather_resampling_ratios['night']
                else:
                    multiplier = self.weather_resampling_ratios['day']
                    
                sample_indices.extend([idx] * multiplier)
                
        return sample_indices

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``CBGSDataset``.

        Returns:
            int: Local index of data.
        """
        return self.sample_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> Set[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            Set[int]: All categories in the sample of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``CBGSDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx: int) -> dict:
        """Get item from infos according to the given index.

        Args:
            idx (int): The index of self.sample_indices.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self) -> int:
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``CBGSDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`CBGSDataset` does not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `CBGSDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> BaseDataset:
        """Not supported in ``CBGSDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`CBGSDataset` does not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `CBGSDataset`.')

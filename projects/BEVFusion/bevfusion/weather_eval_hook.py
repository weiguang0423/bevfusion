# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import os.path as osp
from typing import Dict, List, Tuple

from mmengine.dist import barrier, get_dist_info
from mmengine.fileio import dump, load
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FreezeModulesHook(Hook):
    """Hard-freeze selected modules for stage1 training.

    Compared with ``lr_mult=0``, this hook disables gradients at the
    parameter level and forces frozen modules to stay in eval mode,
    which avoids storing unnecessary backward activations.
    """

    priority = 'VERY_HIGH'

    def __init__(self, module_names, set_eval=True):
        self.module_names = list(module_names)
        self.set_eval = set_eval
        self._logged = False

    def _unwrap_model(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def _apply_freeze(self, runner):
        model = self._unwrap_model(runner)
        frozen_modules = []
        frozen_params = 0

        for module_name in self.module_names:
            module = getattr(model, module_name, None)
            if module is None:
                runner.logger.warning('FreezeModulesHook: module %s not found', module_name)
                continue

            if self.set_eval:
                module.eval()

            for param in module.parameters():
                if param.requires_grad:
                    frozen_params += param.numel()
                param.requires_grad_(False)

            frozen_modules.append(module_name)

        if not self._logged:
            runner.logger.info(
                'FreezeModulesHook: hard-froze modules=%s, params=%d',
                frozen_modules, frozen_params)
            self._logged = True

    def before_train(self, runner):
        self._apply_freeze(runner)

    def before_train_epoch(self, runner):
        self._apply_freeze(runner)


@HOOKS.register_module()
class EvidenceEpochHook(Hook):
    """在每个 epoch 开始时更新模型的 _evi_epoch 计数器。
    
    用于证据损失中 KL 散度的 annealing 调度。
    """
    priority = 'NORMAL'

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'sync_resume_state'):
            synced = model.sync_resume_state(runner.iter)
            if len(synced) > 0:
                runner.logger.info(
                    'EvidenceEpochHook: synced resume state from iter=%d -> %s',
                    runner.iter, synced)

    def before_train_epoch(self, runner):
        model = runner.model
        # 处理 DDP 包装
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, '_evi_epoch'):
            model._evi_epoch = runner.epoch
        fusion_layer = getattr(model, 'fusion_layer', None)
        if fusion_layer is not None and hasattr(fusion_layer, 'set_epoch'):
            fusion_layer.set_epoch(runner.epoch)


@HOOKS.register_module()
class WeatherEvalHook(Hook):
    """Run weather-grouped evaluation after each validation epoch.

    This hook builds NuScenes subset ann files for rain/night/day and runs
    subgroup evaluation in the current distributed training process right
    after the full validation loop. Metrics are written to
    ``work_dir/weather_eval/epoch_xxx.json``.
    """

    priority = 'LOWEST'

    def __init__(self,
                 enabled: bool = True,
                 interval: int = 1,
                 groups: Dict[str, List[str]] = None,
                 adverse_version: str = 'v1.0-trainval',
                 output_dir: str = 'weather_eval',
                 test_num_workers: int = 8,
                 test_persistent_workers: bool = True,
                 test_pin_memory: bool = True,
                 test_prefetch_factor: int = 2) -> None:
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.groups = groups or {
            'rain': ['rain'],
            'night': ['night'],
            'day': [],
        }
        self.adverse_version = adverse_version
        self.output_dir = output_dir
        self.test_num_workers = max(0, int(test_num_workers))
        self.test_persistent_workers = bool(test_persistent_workers)
        self.test_pin_memory = bool(test_pin_memory)
        self.test_prefetch_factor = max(1, int(test_prefetch_factor))

        self._subset_ann_paths: Dict[str, str] = {}

    def before_train(self, runner) -> None:
        if not self.enabled:
            return

        try:
            self._subset_ann_paths = self._build_weather_subsets(runner)
            if len(self._subset_ann_paths) == 0:
                runner.logger.warning('WeatherEvalHook: no subset ann files generated, disable weather eval.')
                self.enabled = False
            else:
                runner.logger.info(
                    'WeatherEvalHook: generated subset ann files: %s',
                    self._subset_ann_paths)
        except Exception as e:
            runner.logger.warning('WeatherEvalHook init failed: %s', str(e))
            self.enabled = False

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self.enabled:
            return

        # MMEngine 的 EpochBasedTrainLoop 在 after_train_epoch 保存检查点后
        # 递增 _epoch，再执行 val loop。所以此刻 runner.epoch 已经是
        # 递增后的值，恰好等于刚保存的检查点编号。
        current_epoch = runner.epoch
        rank, world_size = get_dist_info()

        if current_epoch % self.interval != 0:
            return

        ckpt_path = osp.join(runner.work_dir, f'epoch_{current_epoch}.pth')
        if not osp.exists(ckpt_path) and rank == 0:
            runner.logger.warning('WeatherEvalHook: checkpoint not found: %s', ckpt_path)

        weather_dir = osp.join(runner.work_dir, self.output_dir)
        os.makedirs(weather_dir, exist_ok=True)

        full_metrics = self._sanitize_metrics(metrics)
        full_nds = self._extract_metric(full_metrics, 'NDS')
        full_m_ap = self._extract_metric(full_metrics, 'mAP')
        if rank == 0 and full_metrics:
            runner.logger.info(
                'FullEval epoch=%d NDS=%s mAP=%s',
                current_epoch,
                'NA' if full_nds is None else f'{full_nds:.4f}',
                'NA' if full_m_ap is None else f'{full_m_ap:.4f}')

        epoch_result = {
            'epoch': int(current_epoch),
            'checkpoint': ckpt_path,
            'full_metrics': full_metrics,
            'groups': {},
        }

        try:
            for group_name, ann_file in self._subset_ann_paths.items():
                try:
                    group_metrics = self._run_group_test(
                        runner=runner,
                        epoch=current_epoch,
                        ann_file=ann_file,
                        group_name=group_name)
                    epoch_result['groups'][group_name] = group_metrics

                    nds = group_metrics.get('NDS', None)
                    m_ap = group_metrics.get('mAP', None)
                    if rank == 0:
                        runner.logger.info(
                            'WeatherEval epoch=%d group=%s NDS=%s mAP=%s',
                            current_epoch,
                            group_name,
                            'NA' if nds is None else f'{nds:.4f}',
                            'NA' if m_ap is None else f'{m_ap:.4f}')
                except Exception as e:
                    if rank == 0:
                        runner.logger.warning(
                            'WeatherEvalHook group=%s failed at epoch=%d: %s',
                            group_name, current_epoch, str(e))

            if rank == 0:
                out_path = osp.join(weather_dir, f'epoch_{current_epoch:03d}.json')
                dump(epoch_result, out_path)
        finally:
            if world_size > 1:
                barrier()

    def _run_group_test(self, runner, epoch: int, ann_file: str,
                        group_name: str) -> Dict[str, float]:
        ann_file = osp.abspath(ann_file)
        work_dir = osp.join(runner.work_dir, self.output_dir, f'epoch_{epoch:03d}_{group_name}')
        os.makedirs(work_dir, exist_ok=True)

        dataloader_cfg = copy.deepcopy(runner.cfg.test_dataloader)
        evaluator_cfg = copy.deepcopy(runner.cfg.test_evaluator)
        test_loop_cfg = copy.deepcopy(getattr(runner.cfg, 'test_cfg', dict()))

        self._set_dataset_ann_file(dataloader_cfg['dataset'], ann_file)
        self._set_evaluator_ann_file(evaluator_cfg, ann_file)
        dataloader_cfg['num_workers'] = self.test_num_workers
        dataloader_cfg['persistent_workers'] = (
            self.test_persistent_workers if self.test_num_workers > 0 else False)
        dataloader_cfg['pin_memory'] = self.test_pin_memory
        if self.test_num_workers > 0:
            dataloader_cfg['prefetch_factor'] = self.test_prefetch_factor
        else:
            dataloader_cfg.pop('prefetch_factor', None)

        dataloader = runner.build_dataloader(dataloader_cfg)
        evaluator = runner.build_evaluator(evaluator_cfg)

        prev_test_dataloader = runner._test_dataloader
        prev_test_evaluator = runner._test_evaluator
        try:
            runner._test_dataloader = dataloader
            runner._test_evaluator = evaluator
            metrics = self._sanitize_metrics(runner.build_test_loop(test_loop_cfg).run())
        finally:
            runner._test_dataloader = prev_test_dataloader
            runner._test_evaluator = prev_test_evaluator

        rank, _ = get_dist_info()
        if rank == 0:
            dump(metrics, osp.join(work_dir, 'metrics.json'))
        return metrics

    def _sanitize_metrics(self, metrics) -> Dict[str, float]:
        if metrics is None:
            return {}
        if not isinstance(metrics, dict):
            return {'raw': str(metrics)}

        sanitized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                sanitized[key] = value
                continue
            if hasattr(value, 'item'):
                try:
                    sanitized[key] = value.item()
                    continue
                except Exception:
                    pass
            sanitized[key] = str(value)
        return sanitized

    def _extract_metric(self, metrics: Dict[str, float], metric_name: str):
        for key in [
                f'NuScenes metric/pred_instances_3d_NuScenes/{metric_name}',
                metric_name]:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _set_dataset_ann_file(self, dataset_cfg: dict, ann_file: str) -> None:
        target_cfg = dataset_cfg
        while isinstance(target_cfg, dict) and 'dataset' in target_cfg:
            target_cfg = target_cfg['dataset']
        if not isinstance(target_cfg, dict):
            raise RuntimeError('WeatherEvalHook: invalid test dataset config for ann_file override.')
        target_cfg['ann_file'] = ann_file

    def _set_evaluator_ann_file(self, evaluator_cfg, ann_file: str) -> None:
        if isinstance(evaluator_cfg, list):
            for metric_cfg in evaluator_cfg:
                if isinstance(metric_cfg, dict):
                    metric_cfg['ann_file'] = ann_file
            return
        if isinstance(evaluator_cfg, dict):
            evaluator_cfg['ann_file'] = ann_file
            return
        raise RuntimeError('WeatherEvalHook: invalid evaluator config for ann_file override.')

    def _build_weather_subsets(self, runner) -> Dict[str, str]:
        cfg = runner.cfg
        data_root = cfg.get('data_root', 'data/nuscenes/')
        val_dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
        ann_file = val_dataset_cfg.get('ann_file', 'nuscenes_infos_val.pkl')

        if not osp.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)

        ann_data = load(ann_file)
        use_data_list = isinstance(ann_data, dict) and 'data_list' in ann_data
        if use_data_list:
            records = ann_data.get('data_list', [])
        else:
            records = ann_data.get('infos', [])

        sample_desc = self._build_sample_desc_map(data_root)
        if len(sample_desc) == 0:
            raise RuntimeError('NuScenes scene/sample metadata not found for weather split.')

        splits: Dict[str, List[dict]] = {k: [] for k in self.groups.keys()}

        for info in records:
            token = info.get('token', '')
            desc = sample_desc.get(token, '')
            for group_name, kws in self.groups.items():
                if group_name == 'day':
                    # day = not matched by adverse keywords from other groups.
                    adverse_match = False
                    for k2, kws2 in self.groups.items():
                        if k2 == 'day':
                            continue
                        if any(kw.lower() in desc for kw in kws2):
                            adverse_match = True
                            break
                    if not adverse_match:
                        splits[group_name].append(info)
                else:
                    if any(kw.lower() in desc for kw in kws):
                        splits[group_name].append(info)

        out_dir = osp.join(runner.work_dir, self.output_dir, 'ann_splits')
        os.makedirs(out_dir, exist_ok=True)

        out_paths = {}
        for group_name, group_infos in splits.items():
            if len(group_infos) == 0:
                continue
            # Keep subset file lightweight and avoid expensive deep copy.
            out_data = {}
            if isinstance(ann_data, dict) and 'metainfo' in ann_data:
                out_data['metainfo'] = ann_data['metainfo']
            if use_data_list:
                out_data['data_list'] = group_infos
            else:
                out_data['infos'] = group_infos
                if isinstance(ann_data, dict) and 'metadata' in ann_data:
                    out_data['metadata'] = ann_data['metadata']
            out_path = osp.join(out_dir, f'nuscenes_infos_val_{group_name}.pkl')
            dump(out_data, out_path)
            out_paths[group_name] = out_path
        return out_paths

    def _build_sample_desc_map(self, data_root: str) -> Dict[str, str]:
        version_dir = osp.join(data_root, self.adverse_version)
        sample_path = osp.join(version_dir, 'sample.json')
        scene_path = osp.join(version_dir, 'scene.json')
        if not (osp.exists(sample_path) and osp.exists(scene_path)):
            return {}

        with open(sample_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        with open(scene_path, 'r', encoding='utf-8') as f:
            scenes = json.load(f)

        scene_desc = {}
        for scene in scenes:
            token = scene.get('token', '')
            desc = scene.get('description', '')
            name = scene.get('name', '')
            scene_desc[token] = f'{desc} {name}'.lower()

        sample_desc = {}
        for sample in samples:
            sample_token = sample.get('token', '')
            scene_token = sample.get('scene_token', '')
            sample_desc[sample_token] = scene_desc.get(scene_token, '')

        return sample_desc

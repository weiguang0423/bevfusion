import argparse
import gc
import math
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS, VISUALIZERS


WEATHER_TO_SPLIT = {
    'day': 'nuscenes_infos_val_day.pkl',
    'night': 'nuscenes_infos_val_night.pkl',
    'rainy': 'nuscenes_infos_val_rain.pkl',
}

IMAGE_SUFFIX = '.jpg'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate paired weather visual comparisons for two checkpoints.')
    parser.add_argument(
        '--split-dir',
        default='work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1/weather_eval/ann_splits',
        help='Directory containing weather split PKL files.')
    parser.add_argument(
        '--official-config',
        default='projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
        help='Config for the official lidar-camera checkpoint.')
    parser.add_argument(
        '--official-checkpoint',
        default='checkpoints/bevfusion_lidar_cam_converted.pth',
        help='Official checkpoint path.')
    parser.add_argument(
        '--best-config',
        default='work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1/bevfusion_lidar-cam-radar_geometry_enhanced_stage2_e12.py',
        help='Config matching the best radar checkpoint.')
    parser.add_argument(
        '--best-checkpoint',
        default='work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_10.pth',
        help='Best checkpoint path.')
    parser.add_argument(
        '--output-root',
        default='outputs',
        help='Root directory for all generated outputs.')
    parser.add_argument(
        '--output-name',
        default='',
        help='Optional fixed output folder name under output-root.')
    parser.add_argument(
        '--weathers',
        nargs='+',
        default=['day', 'rainy', 'night'],
        help='Weather groups to sample. Supported: day rainy night.')
    parser.add_argument(
        '--num-per-weather',
        type=int,
        default=10,
        help='Number of random samples per weather.')
    parser.add_argument(
        '--seed',
        type=int,
        default=20260408,
        help='Random seed for reproducible sampling.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.2,
        help='Prediction score threshold used for drawing boxes.')
    parser.add_argument(
        '--comparison-layout',
        choices=['horizontal', 'vertical'],
        default='horizontal',
        help='Comparison layout for official vs best visualizations.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Inference device.')
    parser.add_argument(
        '--keep-single-views',
        action='store_true',
        help='Keep intermediate official/best render directories.')
    return parser.parse_args()


def canonical_weather_name(weather: str) -> str:
    weather = weather.lower().strip()
    if weather == 'rain':
        return 'rainy'
    if weather not in WEATHER_TO_SPLIT:
        raise ValueError(f'Unsupported weather group: {weather}')
    return weather


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    if args.output_name:
        output_dir = output_root / args.output_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_root / f'official_vs_best_weather_compare_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_split(split_path: Path) -> Dict:
    split_obj = mmengine.load(str(split_path))
    if not isinstance(split_obj, dict) or 'data_list' not in split_obj:
        raise ValueError(f'Unexpected split format in {split_path}')
    return split_obj


def make_sample_basename(order: int, sample_info: Dict, fallback_index: int) -> str:
    token = str(sample_info.get('token', ''))
    sample_idx = sample_info.get('sample_idx', fallback_index)
    if isinstance(sample_idx, int):
        sample_idx_text = f'{sample_idx:06d}'
    else:
        sample_idx_text = str(sample_idx)
    token_text = token[:8] if token else f'idx{fallback_index:04d}'
    return f'{order:02d}_sample{sample_idx_text}_{token_text}'


def select_samples(split_dir: Path,
                   weathers: List[str],
                   num_per_weather: int,
                   seed: int) -> Dict[str, Dict]:
    selection: Dict[str, Dict] = {}
    for weather in weathers:
        split_path = split_dir / WEATHER_TO_SPLIT[weather]
        split_obj = load_split(split_path)
        data_list = split_obj['data_list']
        if not data_list:
            raise ValueError(f'No samples found in {split_path}')

        rng = random.Random(f'{seed}:{weather}')
        sample_count = min(num_per_weather, len(data_list))
        chosen_indices = sorted(rng.sample(range(len(data_list)), sample_count))

        samples = []
        for order, index in enumerate(chosen_indices, start=1):
            info = data_list[index]
            basename = make_sample_basename(order, info, index)
            first_image = next(iter(info.get('images', {}).values()), {})
            samples.append({
                'order': order,
                'index': index,
                'basename': basename,
                'token': str(info.get('token', '')),
                'sample_idx': info.get('sample_idx', index),
                'lidar_path': info.get('lidar_points', {}).get('lidar_path', ''),
                'first_image_path': first_image.get('img_path', ''),
            })

        selection[weather] = {
            'split_path': str(split_path.resolve()),
            'available_samples': len(data_list),
            'samples': samples,
        }
    return selection


def build_dataset(model, ann_file: str):
    dataset_cfg = deepcopy(model.cfg.test_dataloader.dataset)
    dataset_cfg.ann_file = str(Path(ann_file).resolve())
    dataset_cfg.test_mode = True
    dataset_cfg.lazy_init = False
    return DATASETS.build(dataset_cfg)


def build_visualizer(model):
    vis_cfg = deepcopy(getattr(model.cfg, 'visualizer', None))
    if vis_cfg is None:
        vis_cfg = dict(type='Det3DLocalVisualizer', name='visualizer')
    visualizer = VISUALIZERS.build(vis_cfg)
    visualizer.dataset_meta = model.dataset_meta
    return visualizer


def load_rgb_images(img_paths):
    if isinstance(img_paths, str):
        img_paths = [img_paths]

    images = []
    for img_path in img_paths:
        image = mmcv.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')
        images.append(mmcv.imconvert(image, 'bgr', 'rgb'))
    return images


def write_image(image: np.ndarray, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_file.suffix.lower()
    params = None
    if suffix in {'.jpg', '.jpeg'}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    mmcv.imwrite(image, str(out_file), params=params)


def compose_multiview_image(images: List[np.ndarray], columns: int = 3) -> np.ndarray:
    if not images:
        raise ValueError('No images to compose.')

    height, width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        canvas[row * height:(row + 1) * height,
               col * width:(col + 1) * width] = image
    return canvas


def count_visible_predictions(result, score_thr: float) -> int:
    pred_instances_3d = getattr(result, 'pred_instances_3d', None)
    if pred_instances_3d is None or len(pred_instances_3d) == 0:
        return 0
    return int((pred_instances_3d.scores_3d > score_thr).sum().item())


def render_single_sample(dataset,
                         model,
                         visualizer,
                         index: int,
                         out_file: Path,
                         score_thr: float) -> None:
    data = dataset[index]
    batch = pseudo_collate([data])
    with torch.no_grad():
        result = model.test_step(batch)[0]

    img_paths = getattr(result, 'img_path', None)
    if img_paths is None:
        raise ValueError('Result does not contain img_path metainfo.')

    images = load_rgb_images(img_paths)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if count_visible_predictions(result, score_thr) == 0:
        fallback = compose_multiview_image(images)
        write_image(fallback[..., ::-1], out_file)
        return

    visualizer.add_datasample(
        out_file.stem,
        dict(img=images),
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        show=False,
        out_file=str(out_file),
        pred_score_thr=score_thr,
        vis_task='mono_det')

    if not out_file.is_file():
        fallback = compose_multiview_image(images)
        write_image(fallback[..., ::-1], out_file)


def render_model_outputs(model_name: str,
                         config_path: Path,
                         checkpoint_path: Path,
                         selection: Dict[str, Dict],
                         output_dir: Path,
                         score_thr: float,
                         device: str) -> None:
    print(f'Loading {model_name} model...')
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    visualizer = build_visualizer(model)

    try:
        for weather, weather_meta in selection.items():
            print(f'[{model_name}] rendering {weather}...')
            dataset = build_dataset(model, weather_meta['split_path'])
            model_output_dir = output_dir / model_name / weather
            for sample_meta in weather_meta['samples']:
                out_file = model_output_dir / f"{sample_meta['basename']}{IMAGE_SUFFIX}"
                render_single_sample(
                    dataset=dataset,
                    model=model,
                    visualizer=visualizer,
                    index=sample_meta['index'],
                    out_file=out_file,
                    score_thr=score_thr)
                sample_meta[f'{model_name}_path'] = str(out_file.resolve())
    finally:
        del visualizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = target_height / image.shape[0]
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image
    scale = target_width / image.shape[1]
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_comparison_image(left: np.ndarray,
                          right: np.ndarray,
                          weather: str,
                          basename: str,
                          sample_token: str,
                          comparison_layout: str) -> np.ndarray:
    gap = 16

    if comparison_layout == 'vertical':
        common_width = min(left.shape[1], right.shape[1])
        top = resize_to_width(left, common_width)
        bottom = resize_to_width(right, common_width)

        banner = np.full((100, common_width, 3), 248, dtype=np.uint8)
        cv2.putText(
            banner,
            'Top: Official lidar-cam checkpoint',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA)
        cv2.putText(
            banner,
            'Bottom: Best radar checkpoint',
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA)
        cv2.putText(
            banner,
            f'Weather: {weather} | Sample: {basename} | Token: {sample_token[:12]}',
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (70, 70, 70),
            1,
            cv2.LINE_AA)

        column = np.full((top.shape[0] + gap + bottom.shape[0], common_width, 3), 255,
                         dtype=np.uint8)
        column[:top.shape[0], :top.shape[1]] = top
        column[top.shape[0] + gap:top.shape[0] + gap + bottom.shape[0], :bottom.shape[1]] = bottom
        return np.concatenate([banner, column], axis=0)

    common_height = min(left.shape[0], right.shape[0])
    left = resize_to_height(left, common_height)
    right = resize_to_height(right, common_height)

    canvas_width = left.shape[1] + gap + right.shape[1]
    banner = np.full((72, canvas_width, 3), 248, dtype=np.uint8)
    cv2.putText(
        banner,
        'Left: Official lidar-cam checkpoint',
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA)
    cv2.putText(
        banner,
        'Right: Best radar checkpoint',
        (left.shape[1] + gap + 20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA)
    cv2.putText(
        banner,
        f'Weather: {weather} | Sample: {basename} | Token: {sample_token[:12]}',
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 70, 70),
        1,
        cv2.LINE_AA)

    row = np.full((common_height, canvas_width, 3), 255, dtype=np.uint8)
    row[:, :left.shape[1]] = left
    row[:, left.shape[1] + gap:left.shape[1] + gap + right.shape[1]] = right
    return np.concatenate([banner, row], axis=0)


def combine_outputs(selection: Dict[str, Dict],
                    output_dir: Path,
                    comparison_layout: str) -> None:
    comparison_root = output_dir / 'comparison'
    for weather, weather_meta in selection.items():
        comparison_dir = comparison_root / weather
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for sample_meta in weather_meta['samples']:
            official_path = Path(sample_meta['official_path'])
            best_path = Path(sample_meta['best_path'])
            if not official_path.is_file():
                raise FileNotFoundError(f'Missing official render: {official_path}')
            if not best_path.is_file():
                raise FileNotFoundError(f'Missing best render: {best_path}')

            official_image = mmcv.imread(str(official_path))
            best_image = mmcv.imread(str(best_path))
            comparison = make_comparison_image(
                official_image,
                best_image,
                weather=weather,
                basename=sample_meta['basename'],
                sample_token=sample_meta['token'],
                comparison_layout=comparison_layout)

            comparison_path = comparison_dir / f"{sample_meta['basename']}{IMAGE_SUFFIX}"
            write_image(comparison, comparison_path)
            sample_meta['comparison_path'] = str(comparison_path.resolve())


def build_manifest_selection(selection: Dict[str, Dict],
                             keep_single_views: bool) -> Dict[str, Dict]:
    manifest_selection = deepcopy(selection)
    if keep_single_views:
        return manifest_selection

    for weather_meta in manifest_selection.values():
        for sample_meta in weather_meta['samples']:
            sample_meta.pop('official_path', None)
            sample_meta.pop('best_path', None)
    return manifest_selection


def save_manifest(args: argparse.Namespace,
                  output_dir: Path,
                  selection: Dict[str, Dict]) -> None:
    manifest_selection = build_manifest_selection(
        selection, keep_single_views=args.keep_single_views)
    manifest = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'seed': args.seed,
        'score_thr': args.score_thr,
        'num_per_weather': args.num_per_weather,
        'comparison_layout': args.comparison_layout,
        'device': args.device,
        'split_dir': str(Path(args.split_dir).resolve()),
        'official': {
            'config': str(Path(args.official_config).resolve()),
            'checkpoint': str(Path(args.official_checkpoint).resolve()),
        },
        'best': {
            'config': str(Path(args.best_config).resolve()),
            'checkpoint': str(Path(args.best_checkpoint).resolve()),
        },
        'selection': manifest_selection,
    }
    manifest_path = output_dir / 'selection_manifest.json'
    mmengine.dump(manifest, str(manifest_path), indent=2)
    print(f'Manifest saved to: {manifest_path}')


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Split directory not found: {split_dir}')

    weathers = [canonical_weather_name(weather) for weather in args.weathers]
    output_dir = resolve_output_dir(args)
    selection = select_samples(
        split_dir=split_dir,
        weathers=weathers,
        num_per_weather=args.num_per_weather,
        seed=args.seed)

    render_model_outputs(
        model_name='official',
        config_path=Path(args.official_config).resolve(),
        checkpoint_path=Path(args.official_checkpoint).resolve(),
        selection=selection,
        output_dir=output_dir,
        score_thr=args.score_thr,
        device=args.device)
    render_model_outputs(
        model_name='best',
        config_path=Path(args.best_config).resolve(),
        checkpoint_path=Path(args.best_checkpoint).resolve(),
        selection=selection,
        output_dir=output_dir,
        score_thr=args.score_thr,
        device=args.device)

    combine_outputs(selection, output_dir, args.comparison_layout)
    save_manifest(args, output_dir, selection)
    if not args.keep_single_views:
        shutil.rmtree(output_dir / 'official', ignore_errors=True)
        shutil.rmtree(output_dir / 'best', ignore_errors=True)
    print(f'All outputs saved under: {output_dir}')


if __name__ == '__main__':
    main()
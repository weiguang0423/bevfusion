import json
import os
import os.path as osp

from mmengine.fileio import dump, load

ROOT = '/root/mmdetection3d-main'
DATA_ROOT = osp.join(ROOT, 'data/nuscenes')
ANN_FILE = osp.join(DATA_ROOT, 'nuscenes_infos_val.pkl')
OUT_DIR = osp.join(
    ROOT,
    'work_dirs/bevfusion_radar_weather_stage1_e6_depthsup_epoch7init_cameraaware/weather_eval/ann_splits',
)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(osp.join(DATA_ROOT, 'v1.0-trainval/sample.json'), 'r', encoding='utf-8') as f:
        samples = json.load(f)
    with open(osp.join(DATA_ROOT, 'v1.0-trainval/scene.json'), 'r', encoding='utf-8') as f:
        scenes = json.load(f)

    scene_desc = {
        scene.get('token', ''): f"{scene.get('description', '')} {scene.get('name', '')}".lower()
        for scene in scenes
    }
    sample_desc = {
        sample.get('token', ''): scene_desc.get(sample.get('scene_token', ''), '')
        for sample in samples
    }

    ann_data = load(ANN_FILE)
    use_data_list = isinstance(ann_data, dict) and 'data_list' in ann_data
    records = ann_data.get('data_list', []) if use_data_list else ann_data.get('infos', [])

    splits = {'rain': [], 'night': [], 'day': []}
    for info in records:
        token = info.get('token', '')
        desc = sample_desc.get(token, '')
        is_rain = 'rain' in desc
        is_night = 'night' in desc
        if is_rain:
            splits['rain'].append(info)
        if is_night:
            splits['night'].append(info)
        if not (is_rain or is_night):
            splits['day'].append(info)

    for group_name, group_infos in splits.items():
        out_data = {}
        if isinstance(ann_data, dict) and 'metainfo' in ann_data:
            out_data['metainfo'] = ann_data['metainfo']
        if use_data_list:
            out_data['data_list'] = group_infos
        else:
            out_data['infos'] = group_infos
            if isinstance(ann_data, dict) and 'metadata' in ann_data:
                out_data['metadata'] = ann_data['metadata']
        out_path = osp.join(OUT_DIR, f'nuscenes_infos_val_{group_name}.pkl')
        dump(out_data, out_path)
        print(f'{group_name}\t{len(group_infos)}\t{out_path}')


if __name__ == '__main__':
    main()

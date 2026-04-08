#!/usr/bin/env python3
"""
将2模态(cam+lidar) BEVFusion 预训练权重转换为3模态(cam+lidar+radar)残差雷达融合格式。

权重映射关系：
  2模态 ConvFuser (nn.Sequential):
    fusion_layer.0.weight  [256, 336, 3, 3]  →  SEConvFuser.primary_conv.weight
    fusion_layer.1.*                          →  SEConvFuser.bn.*

  3模态 SEConvFuser (residual_radar=True) 的新增参数:
    fusion_layer.radar_proj.*    → 随机初始化 (kaiming)
    fusion_layer.radar_gate      → 初始化为 0.0
    fusion_layer.se_modules.*    → 随机初始化
    fusion_layer.evidence_heads.* → 随机初始化
    fusion_layer.conv.*          → 随机初始化 (兼容层，不使用)

用法:
    python tools/convert_2modal_to_3modal.py \
        --src checkpoints/bevfusion_lidar_cam_converted.pth \
        --dst checkpoints/bevfusion_lidar_cam_radar_stage1_init.pth
"""
import argparse
import torch


def convert_sparse_middle_encoder_weight(key: str, value: torch.Tensor) -> torch.Tensor:
    """Convert sparse conv weights to the layout expected by the stage1 config.

    Depth-supervision checkpoints in this repo store pts_middle_encoder 5D weights as
    [out_channels, kx, ky, kz, in_channels], while the loadable stage1 init uses
    [kx, ky, kz, in_channels, out_channels].
    """
    if key.startswith('pts_middle_encoder.') and key.endswith('.weight') and value.ndim == 5:
        converted = value.permute(1, 2, 3, 4, 0).contiguous()
        print(
            f'  {key} {list(value.shape)} → sparse-layout {list(converted.shape)}')
        return converted
    return value


def convert(src_path: str, dst_path: str, keep_camera_aware: bool = False):
    print(f'Loading source checkpoint: {src_path}')
    ckpt = torch.load(src_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    new_state_dict = {}
    mapped_keys = set()
    dropped_keys = []

    for key, value in state_dict.items():
        if key.startswith('view_transform.camera_aware_depthnet.') and not keep_camera_aware:
            # DepthLSSTransform-based stage1 configs do not instantiate camera_aware_depthnet.
            dropped_keys.append(key)
            continue

        value = convert_sparse_middle_encoder_weight(key, value)

        if key == 'fusion_layer.0.weight':
            # ConvFuser.0 (Conv2d) → SEConvFuser.primary_conv
            new_state_dict['fusion_layer.primary_conv.weight'] = value
            mapped_keys.add(key)
            print(f'  {key} {list(value.shape)} → fusion_layer.primary_conv.weight')
        elif key.startswith('fusion_layer.1.'):
            # ConvFuser.1 (BatchNorm2d) → SEConvFuser.bn
            suffix = key[len('fusion_layer.1.'):]
            new_key = f'fusion_layer.bn.{suffix}'
            new_state_dict[new_key] = value
            mapped_keys.add(key)
            print(f'  {key} → {new_key}')
        else:
            # 其他所有权重原样保留
            new_state_dict[key] = value

    unmapped = [k for k in state_dict if k.startswith('fusion_layer.') and k not in mapped_keys]
    if unmapped:
        print(f'\n  Warning: unmapped fusion keys: {unmapped}')

    if dropped_keys:
        print(f'\n  Dropped {len(dropped_keys)} camera-aware-only keys')
    elif keep_camera_aware:
        print('\n  Kept camera-aware depthnet keys for camera-aware stage1 config')

    # 保存
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = new_state_dict
    else:
        ckpt = new_state_dict

    # 清除optimizer/scheduler state避免resume冲突
    for remove_key in ['optimizer', 'param_schedulers', 'message_hub']:
        if isinstance(ckpt, dict) and remove_key in ckpt:
            del ckpt[remove_key]
            print(f'  Removed {remove_key} from checkpoint')

    torch.save(ckpt, dst_path)
    print(f'\nSaved converted checkpoint to: {dst_path}')
    print(f'  Mapped {len(mapped_keys)} fusion_layer keys')
    print(f'  Total keys: {len(new_state_dict)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert 2-modal to 3-modal checkpoint')
    parser.add_argument('--src', required=True, help='Source 2-modal checkpoint path')
    parser.add_argument('--dst', required=True, help='Destination 3-modal checkpoint path')
    parser.add_argument(
        '--keep-camera-aware',
        action='store_true',
        help='Preserve view_transform.camera_aware_depthnet.* weights for CameraAwareDepthLSSTransform-based stage1 configs')
    args = parser.parse_args()
    convert(args.src, args.dst, keep_camera_aware=args.keep_camera_aware)

"""
BEVFusion 模块初始化

本模块支持在没有CUDA编译的环境下进行测试。
CUDA相关模块（如BEVFusion、BEVFusionSparseEncoder等）会延迟导入，
只有在实际使用时才会触发CUDA扩展的加载。

非CUDA依赖的模块（如数据加载、预处理变换）可以直接导入使用。
"""
import warnings

# ============ 非CUDA依赖模块（可直接导入）============
# 数据加载模块
from .weather_eval_hook import WeatherEvalHook, EvidenceEpochHook, FreezeModulesHook
from .loading import (BEVLoadMultiViewImageFromFiles, 
                      LoadRadarPointsFromFile,
                      LoadRadarPointsFromMultiSweeps,
                      LoadDepthFromPoints)

# 数据预处理变换模块
from .transforms_3d import (BEVFusionGlobalRotScaleTrans,
                            BEVFusionRandomFlip3D, GridMask, ImageAug3D,
                            RadarPointsRangeFilter, RadarGeometryEnhancer)

# 雷达速度编码器（非CUDA依赖）
from .radar_velocity_encoder import (GeometryAwareVelocityEncoder, 
                                     VelocityRefinementModule,
                                     RadarVelocityBEVEncoder, 
                                     VelocityAwareFuser)

# ============ CUDA依赖模块（直接导入以确保注册）============
# MMEngine 的注册机制需要在模块导入时执行 @MODELS.register_module()
# 因此这些模块必须直接导入，不能使用延迟导入
try:
    from .bevfusion import BEVFusion, BEVFusionWithRadar
    from .bevfusion_necks import GeneralizedLSSFPN
    from .depth_lss import (DepthLSSTransform, LSSTransform,
                           DepthSupervisionLoss, CameraAwareDepthNet,
                           CameraAwareDepthLSSTransform)
    from .sparse_encoder import BEVFusionSparseEncoder
    from .transformer import TransformerDecoderLayer
    from .transfusion_head import ConvFuser, SEConvFuser, TransFusionHead
    from .utils import (BBoxBEVL1Cost, HeuristicAssigner3D, 
                       HungarianAssigner3D, IoU3DCost)
except ImportError as e:
    warnings.warn(
        f'Failed to load CUDA-dependent modules: {e}. '
        'These modules require CUDA compilation. '
        'Non-CUDA modules (loading, transforms) are still available.'
    )


__all__ = [
    # 非CUDA依赖模块
    'WeatherEvalHook', 'EvidenceEpochHook', 'FreezeModulesHook', 'BEVLoadMultiViewImageFromFiles', 'LoadRadarPointsFromFile',
    'LoadRadarPointsFromMultiSweeps', 'LoadDepthFromPoints',
    'BEVFusionGlobalRotScaleTrans',
    'BEVFusionRandomFlip3D', 'GridMask', 'ImageAug3D', 'RadarPointsRangeFilter',
    'RadarGeometryEnhancer',
    # 雷达速度编码器
    'GeometryAwareVelocityEncoder', 'VelocityRefinementModule',
    'RadarVelocityBEVEncoder', 'VelocityAwareFuser',  # 向后兼容别名
    # CUDA依赖模块
    'BEVFusion', 'BEVFusionWithRadar', 'TransFusionHead', 
    'ConvFuser', 'SEConvFuser',  # 融合器
    'GeneralizedLSSFPN', 'HungarianAssigner3D', 'BBoxBEVL1Cost', 'IoU3DCost', 
    'HeuristicAssigner3D', 'DepthLSSTransform', 'LSSTransform',
    'DepthSupervisionLoss', 'CameraAwareDepthNet', 'CameraAwareDepthLSSTransform',
    'BEVFusionSparseEncoder', 'TransformerDecoderLayer'
]

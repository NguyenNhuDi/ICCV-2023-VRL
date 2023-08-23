import importlib

CONCAT = 'concat'
ADD = 'add'
TWO_D = '2d'
THREE_D = '3d'
INSTANCE = 'instance'
BATCH = "batch"

import sys
sys.path.append('/home/andrew.heschl/Documents/ICCV-2023-VRL')

def my_import(class_name: str, dropout_package: str = 'torch.nn'):
    """
    Returns a class based on a string name.
    """
    if class_name == 'UpsamplingConv':
        from src.json_models.modules import UpsamplingConv
        return UpsamplingConv
    elif class_name == 'ConvPixelShuffle':
        from src.json_models.modules import ConvPixelShuffle
        return ConvPixelShuffle
    elif class_name == 'SelfAttention':
        from src.json_models.modules import SelfAttention
        return SelfAttention
    elif class_name == 'SpatialAttentionModule':
        from src.json_models.modules import SpatialAttentionModule
        return SpatialAttentionModule
    elif class_name == 'DepthWiseSeparableConv':
        from src.json_models.modules import DepthWiseSeparableConv
        return DepthWiseSeparableConv
    elif class_name == "MyConv2d":
        from src.json_models.modules import MyConv2d
        return MyConv2d
    elif class_name == "Residual":
        from src.json_models.modules import Residual
        return Residual
    elif class_name == "Linker":
        from src.json_models.modules import Linker
        return Linker
    elif class_name == "XModule":
        from src.json_models.modules import XModule
        return XModule
    elif class_name == "XModule_Norm":
        from src.json_models.modules import XModule_Norm
        return XModule_Norm
    elif class_name == "AttentionX":
        from src.json_models.modules import AttentionX
        return AttentionX
    elif class_name == "XModuleInverse":
        from src.json_models.modules import XModuleInverse
        return XModuleInverse
    elif class_name == "CBAM":
        from src.json_models.modules import CBAM
        return CBAM
    elif class_name == "CBAMResidual":
        from src.json_models.modules import CBAMResidual
        return CBAMResidual
    elif class_name == "XModuleNoPW":
        from src.json_models.modules import XModuleNoPW
        return XModuleNoPW
    elif class_name == "CAM":
        from src.json_models.modules import CAM
        return CAM
    elif class_name == "XModuleReluBetween":
        from src.json_models.modules import XModuleReluBetween
        return XModuleReluBetween
    elif class_name == "XModuleNoPWReluBetween":
        from src.json_models.modules import XModuleNoPWReluBetween
        return XModuleNoPWReluBetween
    elif class_name == "LearnableCAM":
        from src.json_models.modules import LearnableCAM
        return LearnableCAM
    elif class_name == "EfficientCC_Wrapper":
        from src.json_models.modules import EfficientCC_Wrapper
        return EfficientCC_Wrapper
    elif class_name == "CC_module":
        from src.json_models.modules import CC_module
        return CC_module
    elif class_name == "DecoderBlock":
        from src.json_models.modules import DecoderBlock
        return DecoderBlock
    elif class_name == "InstanceNorm":
        from src.json_models.modules import InstanceNorm
        return InstanceNorm
    elif class_name == "BatchNorm":
        from src.json_models.modules import BatchNorm
        return BatchNorm
    elif class_name == "ConvTranspose":
        from src.json_models.modules import ConvTranspose
        return ConvTranspose
    elif class_name == "Conv":
        from src.json_models.modules import Conv
        return Conv
    elif class_name == "SpatialGatedConv2d":
        from src.json_models.modules import SpatialGatedConv2d
        return SpatialGatedConv2d
    elif class_name == "AveragePool":
        from src.json_models.modules import AveragePool
        return AveragePool
    elif class_name == "CC_Wrapper3d":
        from src.json_models.modules import CC_Wrapper3d
        return CC_Wrapper3d
    elif class_name == "CC_Linker":
        from src.json_models.modules import CC_Linker
        return CC_Linker
    elif class_name == "MCDropout":
        from src.json_models.modules import MCDropout
        return MCDropout
    elif class_name == "ReverseLinearBottleneck":
        from src.json_models.modules import ReverseLinearBottleneck
        return ReverseLinearBottleneck
    elif class_name == "EfficientNetWrapper":
        from src.json_models.efficient_net.efficient_net import EfficientNetWrapper
        return EfficientNetWrapper
    elif class_name == "EfficientNetv2":
        from src.json_models.efficientnetv2.efficientnetv2 import EfficientNetv2
        return EfficientNetv2
    else:
        module = importlib.import_module(dropout_package)
        class_ = getattr(module, class_name)
        return class_

import importlib, sys
sys.path.append('/home/andrewheschl/Documents/3DSegmentation')

def my_import(class_name:str, dropout_package:str = 'torch.nn'):
    if class_name == 'BaseModel':
        from src.models.base_model import BaseModel
        return BaseModel
    elif class_name == 'UpsamplingConv':
        from src.models.utils import UpsamplingConv
        return UpsamplingConv
    elif class_name == 'ConvPixelShuffle':
        from src.models.utils import ConvPixelShuffle
        return ConvPixelShuffle
    elif class_name == 'SelfAttention':
        from src.models.utils import SelfAttention
        return SelfAttention
    elif class_name == 'SpatialAttentionModule':
        from src.models.utils import SpatialAttentionModule
        return SpatialAttentionModule
    elif class_name == 'DepthWiseSeparableConv':
        from src.models.utils import DepthWiseSeparableConv
        return DepthWiseSeparableConv
    elif class_name == "MyConvTranspose2d":
        from src.models.utils import MyConvTranspose2d
        return MyConvTranspose2d
    elif class_name == "MyConv2d":
        from src.models.utils import MyConv2d
        return MyConv2d
    elif class_name == "Residual":
        from src.models.utils import Residual
        return Residual
    elif class_name == "Linker":
        from src.models.utils import Linker
        return Linker
    elif class_name == "XModule":
        from src.models.utils import XModule
        return XModule
    elif class_name == "XModule_Norm":
        from src.models.utils import XModule_Norm
        return XModule_Norm
    elif class_name == "AttentionX":
        from src.models.utils import AttentionX
        return AttentionX
    elif class_name == "XModuleInverse":
        from src.models.utils import XModuleInverse
        return XModuleInverse
    elif class_name == "CBAM":
        from src.models.utils import CBAM
        return CBAM
    elif class_name == "CBAMResidual":
        from src.models.utils import CBAMResidual
        return CBAMResidual
    elif class_name == "XModuleNoPW":
        from src.models.utils import XModuleNoPW
        return XModuleNoPW
    elif class_name == "CAM":
        from src.models.utils import CAM
        return CAM
    elif class_name == "XModuleReluBetween":
        from src.models.utils import XModuleReluBetween
        return XModuleReluBetween
    elif class_name == "XModuleNoPWReluBetween":
        from src.models.utils import XModuleNoPWReluBetween
        return XModuleNoPWReluBetween
    elif class_name == "LearnableCAM":
        from src.models.utils import LearnableCAM
        return LearnableCAM
    elif class_name == "EfficientCC_Wrapper":
        from src.models.utils import EfficientCC_Wrapper
        return EfficientCC_Wrapper
    elif class_name == "CC_module":
        from src.models.utils import CC_module
        return CC_module
    elif class_name == "DecoderBlock":
        from src.models.utils import DecoderBlock
        return DecoderBlock
    elif class_name == "InstanceNorm":
        from src.models.utils import InstanceNorm
        return InstanceNorm
    elif class_name == "ConvTranspose":
        from src.models.utils import ConvTranspose
        return ConvTranspose
    elif class_name == "Conv":
        from src.models.utils import Conv
        return Conv
        
    else:
        module = importlib.import_module(dropout_package)
        class_ = getattr(module, class_name)
        return class_
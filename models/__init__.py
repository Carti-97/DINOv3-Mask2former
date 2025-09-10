"""
Models module for DINOv3-Mask2Former implementation
"""

from .mask2former_dinov3_vitsmallplus import (
    create_mask2former_dinov3_model as create_small_model,
    get_model_info as get_small_model_info,
    Adapter as SmallAdapter,
    DinoV3WithAdapterBackbone as SmallDinoV3Backbone
)

from .mask2former_dinov3_vitlarge import (
    create_mask2former_dinov3_model as create_large_model,
    get_model_info as get_large_model_info,
    Adapter as LargeAdapter,
    DinoV3WithAdapterBackbone as LargeDinoV3Backbone
)

__all__ = [
    "create_small_model",
    "get_small_model_info", 
    "SmallAdapter",
    "SmallDinoV3Backbone",
    "create_large_model",
    "get_large_model_info",
    "LargeAdapter", 
    "LargeDinoV3Backbone"
]

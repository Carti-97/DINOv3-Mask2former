"""
Models module for the DINOv3-Mask2Former implementation.

Variant files (mask2former_dinov3_*.py) are normally loaded dynamically by
file path via the --model argument; this package interface is provided for
programmatic use.
"""

from .dinov3_mask2former_base import (
    Adapter,
    DinoV3WithAdapterBackbone,
    PyramidAdapter,
    build_mask2former_dinov3_model,
    get_model_info,
)
from .mask2former_dinov3_vitsmallplus import (
    create_mask2former_dinov3_model as create_small_model,
)
from .mask2former_dinov3_vitlarge import (
    create_mask2former_dinov3_model as create_large_model,
)

__all__ = [
    "Adapter",
    "PyramidAdapter",
    "DinoV3WithAdapterBackbone",
    "build_mask2former_dinov3_model",
    "get_model_info",
    "create_small_model",
    "create_large_model",
]

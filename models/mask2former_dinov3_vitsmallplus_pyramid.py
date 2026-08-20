"""
DINOv3 ViT-Small+ backbone + Mask2Former (Swin-Small head configuration),
with a ViTDet-style simple feature pyramid.

Unlike the plain variant (all feature maps at stride 16), the tapped features
are resampled to strides 4/8/16/32, matching the multi-scale pyramid the
Mask2Former pixel decoder was designed for. Checkpoints are NOT compatible
with the plain variant.
"""

import os
import sys
from typing import Dict

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from dinov3_mask2former_base import (  # noqa: E402
    DinoV3WithAdapterBackbone,
    PyramidAdapter,
    build_mask2former_dinov3_model,
    get_model_info,
)

# Read by the training script to load the matching image processor.
MASK2FORMER_MODEL_NAME = "facebook/mask2former-swin-small-coco-instance"
DINOV3_MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
EXPECTED_CHANNELS = [96, 192, 384, 768]
LAYERS_TO_EXTRACT = [2, 5, 8, 11]

__all__ = [
    "DinoV3WithAdapterBackbone",
    "PyramidAdapter",
    "MASK2FORMER_MODEL_NAME",
    "create_mask2former_dinov3_model",
    "get_model_info",
]


def create_mask2former_dinov3_model(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    freeze_backbone: bool = True,
    hub_token: str = None,
):
    return build_mask2former_dinov3_model(
        label2id=label2id,
        id2label=id2label,
        mask2former_model_name=MASK2FORMER_MODEL_NAME,
        dinov3_model_name=DINOV3_MODEL_NAME,
        expected_channels=EXPECTED_CHANNELS,
        layers_to_extract=LAYERS_TO_EXTRACT,
        freeze_backbone=freeze_backbone,
        hub_token=hub_token,
        use_feature_pyramid=True,
    )

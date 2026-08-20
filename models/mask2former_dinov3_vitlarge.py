"""
DINOv3 ViT-Large backbone + Mask2Former (Swin-Large head configuration).

This file is loaded dynamically by file path from the training/inference
scripts. The shared implementation lives in dinov3_mask2former_base.py.
"""

import os
import sys
from typing import Dict

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from dinov3_mask2former_base import (  # noqa: E402
    Adapter,
    DinoV3WithAdapterBackbone,
    build_mask2former_dinov3_model,
    get_model_info,
)

# Read by the training script to load the matching image processor.
MASK2FORMER_MODEL_NAME = "facebook/mask2former-swin-large-coco-instance"
DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EXPECTED_CHANNELS = [192, 384, 768, 1536]
LAYERS_TO_EXTRACT = [5, 11, 17, 23]

__all__ = [
    "Adapter",
    "DinoV3WithAdapterBackbone",
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
        use_feature_pyramid=False,
    )

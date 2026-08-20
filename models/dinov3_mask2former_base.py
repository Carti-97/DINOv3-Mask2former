"""
Shared implementation for DINOv3-backed Mask2Former models.

Model variant files (e.g. mask2former_dinov3_vitsmallplus.py) are thin wrappers
around `build_mask2former_dinov3_model` and only define the variant-specific
constants. They are loaded dynamically by file path from the training and
inference scripts, so this module is imported as a top-level module (the
variant files put this directory on sys.path).
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForUniversalSegmentation
from transformers.modeling_outputs import BackboneOutput

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    """
    Adapter module to convert DINOv3 features to expected channels for the
    Mask2Former head. All output feature maps keep the ViT patch resolution
    (stride = patch_size for every stage).
    """

    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projections[i](feat) for i, feat in enumerate(features)]


class PyramidAdapter(nn.Module):
    """
    ViTDet-style simple feature pyramid adapter.

    Plain ViT backbones produce single-scale (stride = patch_size) features,
    while the Mask2Former pixel decoder was designed for a real multi-scale
    pyramid (strides 4/8/16/32 with the Swin backbone). This adapter resamples
    the four tapped feature maps to strides 4/8/16/32 before projecting them
    to the channel counts the head expects.
    """

    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError(
                f"PyramidAdapter expects exactly 4 output stages, got {len(out_channels)}"
            )
        if (in_channels // 2) % 32 != 0:
            raise ValueError(
                f"PyramidAdapter requires in_channels/2 divisible by 32 for GroupNorm, got in_channels={in_channels}"
            )
        # stage 0: x4 upsample (stride 16 -> 4), stage 1: x2 (16 -> 8),
        # stage 2: identity (16), stage 3: x0.5 downsample (16 -> 32)
        self.resamplers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    nn.GroupNorm(32, in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
                ),
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        resampled_dims = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(dim, out_ch, kernel_size=1)
                for dim, out_ch in zip(resampled_dims, out_channels)
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            proj(resample(feat))
            for feat, resample, proj in zip(features, self.resamplers, self.projections)
        ]


class DinoV3WithAdapterBackbone(nn.Module):
    """
    Custom backbone that combines DINOv3 with adapter layers for Mask2Former
    compatibility.
    """

    def __init__(
        self,
        model_name: str,
        out_channels: List[int],
        layers_to_extract: List[int],
        use_feature_pyramid: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        patch_size = self.model.config.patch_size

        if use_feature_pyramid:
            self.adapter = PyramidAdapter(hidden_size, out_channels)
            strides = [patch_size // 4, patch_size // 2, patch_size, patch_size * 2]
        else:
            self.adapter = Adapter(hidden_size, out_channels)
            # Plain ViT features: every stage has the same (patch) stride.
            strides = [patch_size] * len(out_channels)

        self.channels = list(out_channels)
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = dict(zip(self.out_features, out_channels))
        self._out_feature_strides = dict(zip(self.out_features, strides))

        # Transformer blocks to tap for multi-stage features.
        num_layers = self.model.config.num_hidden_layers
        for layer_idx in layers_to_extract:
            if not 0 <= layer_idx < num_layers:
                raise ValueError(
                    f"layers_to_extract contains index {layer_idx}, but {model_name} has {num_layers} layers"
                )
        self.layers_to_extract = list(layers_to_extract)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size

        # DINOv3 prepends a CLS token and optional register tokens before patch tokens.
        prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)

        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]
            patch_tokens = layer_output[:, prefix_tokens:, :]
            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(
                batch_size, self.model.config.hidden_size, patch_height, patch_width
            )
            extracted_features.append(feature_map)

        adapted_features = self.adapter(extracted_features)

        return BackboneOutput(
            feature_maps=tuple(adapted_features),
            hidden_states=None,
            attentions=None,
        )


def build_mask2former_dinov3_model(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    mask2former_model_name: str,
    dinov3_model_name: str,
    expected_channels: List[int],
    layers_to_extract: List[int],
    freeze_backbone: bool = True,
    hub_token: str = None,
    use_feature_pyramid: bool = False,
) -> AutoModelForUniversalSegmentation:
    """
    Create a complete DINOv3-Mask2Former model with custom backbone replacement.

    Args:
        label2id: Dictionary mapping label names to IDs
        id2label: Dictionary mapping IDs to label names
        mask2former_model_name: HuggingFace model name of the Mask2Former base
        dinov3_model_name: HuggingFace model name of the DINOv3 backbone
        expected_channels: Output channels per stage expected by the head
        layers_to_extract: DINOv3 transformer block indices to tap
        freeze_backbone: Whether to freeze DINOv3 backbone weights
        hub_token: HuggingFace Hub token if needed
        use_feature_pyramid: Resample tapped features to strides 4/8/16/32
            (ViTDet-style) instead of keeping them all at the patch stride

    Returns:
        Complete DINOv3-Mask2Former model ready for training/inference
    """
    logger.info("Creating DINOv3-Mask2Former model...")
    logger.info(f"  - Mask2Former base: {mask2former_model_name}")
    logger.info(f"  - DINOv3 backbone: {dinov3_model_name}")
    logger.info(f"  - Expected channels: {expected_channels}")
    logger.info(f"  - Layers to extract: {layers_to_extract}")
    logger.info(f"  - Feature pyramid: {use_feature_pyramid}")
    logger.info(f"  - Freeze backbone: {freeze_backbone}")

    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )

    custom_backbone = DinoV3WithAdapterBackbone(
        dinov3_model_name,
        expected_channels,
        layers_to_extract,
        use_feature_pyramid=use_feature_pyramid,
    )

    # Replace the encoder actually used by Mask2Former during training/inference.
    model.model.pixel_level_module.encoder = custom_backbone

    if freeze_backbone:
        for param in model.model.pixel_level_module.encoder.model.parameters():
            param.requires_grad = False
        logger.info("DINOv3 backbone weights frozen.")
    else:
        logger.info("DINOv3 backbone weights remain trainable.")

    logger.info("Successfully created DINOv3-Mask2Former model.")
    return model


def get_model_info(model: AutoModelForUniversalSegmentation) -> Dict:
    """Get parameter counts and backbone info for a DINOv3-Mask2Former model."""
    backbone = model.model.pixel_level_module.encoder

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in backbone.model.parameters())
    frozen_params = sum(p.numel() for p in backbone.model.parameters() if not p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "frozen_parameters": frozen_params,
        "backbone_model": getattr(backbone.model.config, "name_or_path", "DINOv3"),
        "output_channels": list(backbone._out_feature_channels.values()),
        "output_strides": list(backbone._out_feature_strides.values()),
    }

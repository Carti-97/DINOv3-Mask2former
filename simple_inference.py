#!/usr/bin/env python
"""
Simple DINOv3 Mask2Former Inference Script
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Color palette (per class id, cycled)
COLORS = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [255, 128, 0],  # Orange
    [128, 0, 255],  # Purple
]


def _load_font(size=20):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def load_model(model_path):
    """Load model with correct DINOv3 backbone reconstruction."""
    print(f"Loading model: {model_path}")

    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)

    # Check if DINOv3 backbone config exists
    dinov3_config_path = os.path.join(model_path, "dinov3_backbone_config.json")
    if os.path.exists(dinov3_config_path):
        print("Found dinov3_backbone_config.json - reconstructing DINOv3 backbone")
        with open(dinov3_config_path, "r") as f:
            dinov3_config = json.load(f)

        # Load model creation function from the model file
        model_file = dinov3_config["model_file"]
        if not os.path.isabs(model_file):
            # Resolve relative path from project root
            project_root = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.join(project_root, model_file)
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model definition file referenced by {dinov3_config_path} not found: {model_file}. "
                "The checkpoint cannot be reconstructed without it."
            )

        module_name = os.path.splitext(os.path.basename(model_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Read label mappings from saved config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)

        label2id = model_config.get("label2id")
        id2label = model_config.get("id2label")
        if not label2id or not id2label:
            raise ValueError(
                f"{config_path} does not contain label2id/id2label mappings - "
                "cannot rebuild the classification head."
            )

        # Recreate model with correct DINOv3 backbone
        model = module.create_mask2former_dinov3_model(
            label2id=label2id,
            id2label=id2label,
            freeze_backbone=False,
        )

        # Load saved weights (fail fast if missing or mismatched)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            weights_path = safetensors_path
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            weights_path = bin_path
        else:
            raise FileNotFoundError(
                f"No weight file found in {model_path} "
                "(looked for model.safetensors and pytorch_model.bin)."
            )

        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                f"Weight mismatch when loading {weights_path}:\n"
                f"  missing keys: {load_result.missing_keys}\n"
                f"  unexpected keys: {load_result.unexpected_keys}\n"
                "The checkpoint does not match the model definition in "
                f"{model_file}. Refusing to run with partially loaded weights."
            )
        print(f"Loaded weights from {os.path.basename(weights_path)}")
    else:
        print("No dinov3_backbone_config.json found - using default HuggingFace loading")
        model = AutoModelForUniversalSegmentation.from_pretrained(model_path)

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    # Class names come from the trained model's config (not hardcoded)
    id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}

    return model, image_processor, id2label


def inference_and_visualize(model, image_processor, id2label, image_path, save_path=None, threshold=0.5):
    """Run inference on one image and save a visualization."""
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    font = _load_font(20)

    # Preprocessing
    inputs = image_processor(images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processing
    target_sizes = [(image.size[1], image.size[0])]  # (height, width)
    results = image_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
        return_binary_maps=False,
    )
    result = results[0]

    if not save_path:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"{base_name}_result.jpg"

    if result["segments_info"]:
        print(f"Detected objects: {len(result['segments_info'])}")

        img_array = np.array(image)
        segmentation = result["segmentation"].cpu().numpy()
        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)

        # Mask overlay
        overlay = img_array.copy()
        for segment_info in result["segments_info"]:
            class_id = int(segment_info.get("label_id", segment_info.get("label", 0)))
            segment_id = segment_info["id"]
            mask = segmentation == segment_id
            print(
                f"Segment ID {segment_id}, Class {class_id}: "
                f"Score {segment_info['score']:.3f}, Mask pixels {int(mask.sum())}"
            )

            color = COLORS[class_id % len(COLORS)]
            if mask.any():
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask] = color
                overlay[mask] = (overlay[mask] * 0.6 + colored_mask[mask] * 0.4).astype(np.uint8)

        result_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(result_image)

        # Text overlay
        y_offset = 10
        for segment_info in result["segments_info"]:
            class_id = int(segment_info.get("label_id", segment_info.get("label", 0)))
            class_name = id2label.get(class_id, f"class_{class_id}")
            confidence = segment_info["score"]
            color = COLORS[class_id % len(COLORS)]

            text = f"{class_name}: {confidence:.3f}"
            bbox = draw.textbbox((10, y_offset), text, font=font)
            padded_bbox = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)
            draw.rectangle(padded_bbox, fill=(0, 0, 0))
            draw.text((10, y_offset), text, fill=tuple(color), font=font)
            y_offset += 30

        result_image.save(save_path)
        print(f"Result saved: {save_path}")
        return result_image

    print("No objects detected.")
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    text = "No objects detected"
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    result_image.save(save_path)
    print(f"Result saved: {save_path}")
    return result_image


def process_directory(model, image_processor, id2label, input_dir, output_dir, threshold=0.5, recursive=False):
    """Batch process all images in a directory."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    image_files = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        for path in glob.glob(os.path.join(input_dir, "*")):
            if os.path.isfile(path) and path.lower().endswith(image_extensions):
                image_files.append(path)

    image_files.sort()
    print(f"Images to process: {len(image_files)} (Recursive search: {'On' if recursive else 'Off'})")

    failures = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

        try:
            inference_and_visualize(
                model, image_processor, id2label, image_path, output_path, threshold
            )
            print(f"Completed: {output_path}")
        except Exception:
            print(f"ERROR while processing {image_path}:")
            traceback.print_exc()
            failures.append(image_path)

    print(f"\nDone. {len(image_files) - len(failures)}/{len(image_files)} images processed, results in {output_dir}")
    if failures:
        print("Failed images:")
        for path in failures:
            print(f"  - {path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Simple Mask2Former Inference")
    parser.add_argument("--model_path", "-m", required=True, help="Path to the trained model directory")
    parser.add_argument("--image_path", "-i", help="Path to a single image (single image mode)")
    parser.add_argument("--input_dir", "-d", help="Input directory (batch mode)")
    parser.add_argument("--output_dir", "-od", default="results", help="Output directory for batch mode (default: results)")
    parser.add_argument("--output", "-o", help="Output path for the single image result")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories recursively (batch mode)")

    args = parser.parse_args()

    if bool(args.image_path) == bool(args.input_dir):
        parser.error("Provide exactly one of --image_path (single mode) or --input_dir (batch mode).")

    model, image_processor, id2label = load_model(args.model_path)

    if args.image_path:
        print("Single image mode")
        inference_and_visualize(
            model,
            image_processor,
            id2label,
            args.image_path,
            args.output,
            args.threshold,
        )
        print("Completed!")
    else:
        print("Batch processing mode")
        process_directory(
            model,
            image_processor,
            id2label,
            args.input_dir,
            args.output_dir,
            args.threshold,
            args.recursive,
        )


if __name__ == "__main__":
    main()

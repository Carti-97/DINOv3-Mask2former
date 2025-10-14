#!/usr/bin/env python
"""
Simple DINOv3 Mask2Former Inference Script
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
)
import glob

# Class ID to name mapping
CLASS_NAMES = {
    0: "bead 1",
    1: "bead 2",
    2: "bead 3",
    3: "bead 4",
    4: "bead 5",
    5: "bead 6",
    6: "bead 7",
    7: "bead 8"
}

def load_model(model_path):
    """Load model"""
    print(f"Loading model: {model_path}")
    
    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    
    return model, image_processor

def inference_and_visualize(model, image_processor, image_path, save_path=None, threshold=0.5):
    """Inference and visualization (modified version)"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Font setup (for text display)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        # Use default font
        font = ImageFont.load_default()

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

    # Visualize results
    if result["segments_info"]:
        print(f"Detected objects: {len(result['segments_info'])}")

        # Convert original image to numpy array
        img_array = np.array(image)
        segmentation = result["segmentation"].cpu().numpy()

        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)
            
        # Debug: print segmentation information
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation unique values: {np.unique(segmentation)}")
        
        # Print each segment information
        for i, seg_info in enumerate(result["segments_info"]):
            print(f"Segment {i}: ID={seg_info['id']}, Label={seg_info.get('label_id', seg_info.get('label', 0))}, Score={seg_info['score']:.3f}")

        # Color generation (different colors for each class)
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]

        # Mask overlay
        overlay = img_array.copy()
        
        # Prepare PIL Image for text overlay
        result_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(result_image)

        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            segment_id = segment_info["id"]
            
            # Create mask (original logic)
            mask = (segmentation == segment_id)
            
            # Mask debugging information
            mask_pixels = mask.sum()
            total_pixels = mask.shape[0] * mask.shape[1]
            
            print(f"Segment ID {segment_id}, Class {class_id}: Mask pixels {mask_pixels}/{total_pixels}")
            
            # Mask verification and correction
            mask_ratio = mask_pixels / total_pixels

            # --- Key modification: select color based on class_id ---
            color = colors[class_id % len(colors)]

            # Apply semi-transparent mask
            if mask.sum() > 0:
                # Create a new image filled with color in the mask area
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask] = color
                
                # Blend original image with mask
                overlay[mask] = (overlay[mask] * 0.6 + colored_mask[mask] * 0.4).astype(np.uint8)

        # Final conversion to PIL Image
        result_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(result_image)

        y_offset = 10
        # Add text overlay
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            confidence = segment_info['score']
            
            # --- Key modification: select color based on class_id ---
            color = colors[class_id % len(colors)]

            text = f"{class_name}: {confidence:.3f}"
            
            # Text background rectangle
            bbox = draw.textbbox((10, y_offset), text, font=font)
            # Add padding
            padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
            draw.rectangle(padded_bbox, fill=(0, 0, 0, 128))

            # Draw text
            draw.text((10, y_offset), text, fill=tuple(color), font=font)
            y_offset += 30

        # Save
        save_filename = save_path
        if not save_filename:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_filename = f"{base_name}_result.jpg"

        result_image.save(save_filename)
        print(f"Result saved: {save_filename}")

        return result_image

    else:
        print("No objects detected.")
        # (same as original code)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        
        text = "No objects detected"
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        if save_path:
            image.save(save_path)
        return image

def process_directory(model, image_processor, input_dir, output_dir, threshold=0.5, recursive=False):
    """Batch process all images in directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Find all image files
    image_files = []
    
    if recursive:
        # Recursively search subdirectories
        for root, dirs, files in os.walk(input_dir):
            for ext in image_extensions:
                pattern = ext.lower()
                for file in files:
                    if file.lower().endswith(pattern[1:]):  # *.jpg -> .jpg
                        image_files.append(os.path.join(root, file))
    else:
        # Search current directory only
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    image_files.sort()
    
    print(f"Images to process: {len(image_files)} (Recursive search: {'On' if recursive else 'Off'})")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        
        try:
            # Inference and visualization
            result_image = inference_and_visualize(
                model, image_processor, image_path, output_path, threshold
            )
            print(f"✅ Completed: {output_path}")
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            continue
    
    print(f"\n🎉 All processing completed! Results saved in {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Simple Mask2Former Inference")
    parser.add_argument("--model_path", "-m", help="Path to model", 
                        default="/home/chbae/ImageModel_Research/DINOv3-Mask2former/output/dinov3-smallplus-mask2former-1e4/step_1400_model")
    parser.add_argument("--image_path", "-i", help="Path to single image")
    parser.add_argument("--input_dir", "-d", help="Input directory (for batch processing)", 
                        default="/home/chbae/ImageModel_Research/dataset/ON_diverse_1024/valid")
    parser.add_argument("--output_dir", "-od", help="Output directory (for batch processing)", 
                        default="/home/chbae/ImageModel_Research/results_temp")
    parser.add_argument("--output", "-o", help="Output path for single image result")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch processing mode")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Load model
    model, image_processor = load_model(args.model_path)
    
    if args.batch or args.input_dir:
        # Batch processing mode
        print("🚀 Batch processing mode")
        process_directory(
            model, 
            image_processor, 
            args.input_dir, 
            args.output_dir, 
            args.threshold,
            args.recursive
        )
    else:
        # Single image processing mode
        if not args.image_path:
            print("❌ Error: --image_path is required in single image mode.")
            return
        
        print("🎯 Single image processing mode")
        result_image = inference_and_visualize(
            model, 
            image_processor, 
            args.image_path, 
            args.output, 
            args.threshold
        )
        print("✅ Completed!")

if __name__ == "__main__":
    main()

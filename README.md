# Mask2Former with DINOv3 Backbones

> Last Updated: 2025.10.14

## Recent Updates

### 2025.10.14
- **Inference Script Added**: New `simple_inference.py` for easy model inference
  - Single image and batch processing modes
  - Recursive directory processing support
  - Visual output with colored segmentation masks
  - Adjustable confidence threshold

### 2025.09.30
- **Enhanced Checkpointing System**: Improved `checkpointing_steps` functionality
  - Automatic model and checkpoint saving at specified intervals
  - Integrated validation process at each checkpoint
  - Comprehensive metrics logging and storage
  - Better training resumption and model selection capabilities

---

This project replaces the Swin-Small and Swin-Large backbones in Mask2Former with the ViT Small Plus and ViT Large models from DINOv3, respectively. This modification aims to leverage the powerful, self-supervised learned features of DINOv3 for instance segmentation tasks.

The training methodology is based on the instance segmentation examples provided in the Hugging Face Transformers library. The implementation can be found at: [https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).

## License

This project incorporates code and models from different sources, each with its own license. Please review the following details carefully.

*   **Project Codebase (Apache 2.0)**
    *   The main codebase of this project is licensed under the Apache 2.0 License. You can find the full license text in the [`LICENSE`](./LICENSE) file.
    *   This code is a derivative of examples from the Hugging Face Transformers library and includes necessary attributions in the [`NOTICE`](./NOTICE) file.

*   **DINOv3 Backbone**
    *   The DINOv3 models used as a backbone in this project are subject to a separate license. 
    *   The terms can be found in the [`DINOV3_LICENSE.md`](./DINOV3_LICENSE.md) file. For complete details, please refer to the official DINOv3 repository and its licensing terms.

## Dataset

This model is designed to be trained on datasets with the COCO format.
### Image Size Requirements
Important: The Vision Transformer (ViT) architecture, which forms the backbone of this model, processes images by dividing them into fixed-size patches.[1] The patch size for the DINOv3 ViT models is 16x16 pixels.[2] Consequently, the height and width of the input images must be a multiple of 16. If the image dimensions do not meet this requirement, the training process will fail.

## Usage

This project supports dynamic model loading through configuration files. You can choose between DINOv3-Small+ and DINOv3-Large models by specifying different model files in the configuration.

### Method 1: Using JSON Configuration File (Recommended)

Train with DINOv3-Small+ backbone:
```bash
accelerate launch mask2former_dinov3_no_trainer_coco.py --config mask2former-dinov3_smallplus_1024_train_args.json
```

Train with DINOv3-Large backbone:
```bash
accelerate launch mask2former_dinov3_no_trainer_coco.py --config mask2former-dinov3_large_1024_train_args.json
```

### Method 2: Using Command Line Arguments

You can also specify parameters directly via command line:
```bash
accelerate launch mask2former_dinov3_no_trainer_coco.py \
    --model models/mask2former_dinov3_vitsmallplus.py \
    --dataset_name /path/to/your/coco/dataset \
    --output_dir ./output/dinov3-smallplus-experiment \
    --image_height 1024 \
    --image_width 1024 \
    --num_train_epochs 50 \
    --learning_rate 1e-6
```

For DINOv3-Large:
```bash
accelerate launch mask2former_dinov3_no_trainer_coco.py \
    --model models/mask2former_dinov3_vitlarge.py \
    --dataset_name /path/to/your/coco/dataset \
    --output_dir ./output/dinov3-large-experiment \
    --image_height 1024 \
    --image_width 1024 \
    --num_train_epochs 50 \
    --learning_rate 5e-5
```

### Method 3: Hybrid Approach

You can use a configuration file as a base and override specific parameters:
```bash
accelerate launch mask2former_dinov3_no_trainer_coco.py \
    --config mask2former-dinov3_smallplus_1024_train_args.json \
    --learning_rate 2e-6 \
    --output_dir ./custom_output_dir
```

### Configuration Files

The project includes pre-configured JSON files:
- `mask2former-dinov3_smallplus_1024_train_args.json`: Configuration for DINOv3-Small+ model
- `mask2former-dinov3_large_1024_train_args.json`: Configuration for DINOv3-Large model

You will need to adapt the `dataset_name` parameter in these files to point to your specific COCO dataset directory.

## Inference

The project now includes `simple_inference.py`, a user-friendly script for running inference with trained Mask2Former models.

### Features
- **Single Image Mode**: Process individual images
- **Batch Mode**: Process entire directories of images
- **Recursive Processing**: Option to process subdirectories
- **Visual Results**: Automatically generates annotated images with colored masks and confidence scores
- **Flexible Thresholding**: Adjustable detection confidence threshold

### Usage Examples

#### Process a Single Image
```bash
python simple_inference.py \
    --model_path ./output/dinov3-smallplus-mask2former-1e4/step_1400_model \
    --image_path /path/to/image.jpg \
    --output result.jpg \
    --threshold 0.5
```

#### Batch Process a Directory
```bash
python simple_inference.py \
    --model_path ./output/dinov3-smallplus-mask2former-1e4/step_1400_model \
    --input_dir /path/to/images/ \
    --output_dir /path/to/results/ \
    --threshold 0.5 \
    --batch
```

#### Recursive Directory Processing
```bash
python simple_inference.py \
    --model_path ./output/dinov3-smallplus-mask2former-1e4/step_1400_model \
    --input_dir /path/to/images/ \
    --output_dir /path/to/results/ \
    --recursive \
    --batch
```

### Parameters
- `--model_path, -m`: Path to the trained model directory
- `--image_path, -i`: Path to a single image (for single image mode)
- `--input_dir, -d`: Input directory containing images (for batch mode)
- `--output_dir, -od`: Output directory for results (for batch mode)
- `--output, -o`: Output path for single image result
- `--threshold, -t`: Detection confidence threshold (default: 0.5)
- `--batch, -b`: Enable batch processing mode
- `--recursive, -r`: Process subdirectories recursively

## Features

### Checkpointing System

The `checkpointing_steps` parameter supports two modes:

#### Step-based Checkpointing
```json
{
  "checkpointing_steps": "200"  // Save every 200 steps
}
```

- Saves checkpoint every N steps
- Runs validation at each checkpoint
- Output: `step_{N}_state/` (for resuming), `step_{N}_model/` (for inference)

#### Epoch-based Checkpointing
```json
{
  "checkpointing_steps": "epoch"  // Save every epoch
}
```

- Saves checkpoint at the end of each epoch
- Uses existing end-of-epoch validation
- Output: `checkpoint_epoch_{N}/` (for resuming), `epoch_{N}/` (for inference), `best_model/`

#### Training Resumption

```bash
# Resume from step checkpoint
accelerate launch mask2former_dinov3_no_trainer_coco.py \
    --config your_config.json \
    --resume_from_checkpoint output_dir/step_400_state

# Resume from epoch checkpoint
accelerate launch mask2former_dinov3_no_trainer_coco.py \
    --config your_config.json \
    --resume_from_checkpoint output_dir/checkpoint_epoch_5
```

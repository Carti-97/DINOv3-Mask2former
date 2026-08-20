# Mask2Former with DINOv3 Backbones

> Last Updated: 2026.08.20

## Recent Updates

### 2026.08.20
- **Bug fixes & fail-fast overhaul**
  - `--resume_from_checkpoint` now correctly parses the checkpoint folder names this script creates (`step_{N}_state`, `checkpoint_epoch_{N}`); previously resuming crashed with a `ValueError`.
  - Single-image inference mode is reachable again (batch mode was always triggered by a default `--input_dir`).
  - The Mask2Former base model name is now read from a `MASK2FORMER_MODEL_NAME` constant in the model file instead of parsing Python source code; a missing constant is a hard error (no silent fallback to swin-small).
  - Inference weight loading fails fast on missing weight files or mismatched keys (no more silent `strict=False`).
  - Unknown keys in a `--config` JSON file now raise an error instead of being silently ignored.
  - COCO annotation cache is invalidated when the annotation file changes, is built only once in distributed runs, and RLE segmentations are decoded correctly in cached runs.
  - End-of-epoch validation now runs with the model in `eval()` mode (dropout was previously active during evaluation).
  - Compatible with transformers v5 (removed the deleted `send_example_telemetry` API).
- **New training options**: `--freeze_backbone/--no-freeze_backbone`, `--use_augmentation`, `--max_grad_norm`, `--logging_steps` (periodic loss/LR logging).
- **Final evaluation uses the `test` split** when `dataset/<name>/test/_annotations.coco.json` exists (falls back to reporting the last validation metrics, clearly logged).
- **Feature-pyramid model variants** (`models/*_pyramid.py`): ViTDet-style simple feature pyramid producing true multi-scale features (strides 4/8/16/32) instead of four stride-16 maps. Not weight-compatible with the plain variants. In an A/B test on a private industrial dataset the pyramid variant improved valid mAP from 0.9431 to 0.9469 with 11 of 14 classes improving — see [docs/pyramid-adapter-benchmark](./docs/pyramid-adapter-benchmark/README.md).
- **Model code deduplicated**: shared implementation in `models/dinov3_mask2former_base.py`; variant files only define constants.
- **Class names at inference come from the checkpoint's `id2label`** instead of a hardcoded list.
- `requirements.txt` cleaned up (removed the bogus `huggingface` package, added missing dependencies).

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

You will need to adapt the `dataset_name` parameter in these files to point to your specific COCO dataset directory. Unknown keys in a config file raise an error, so typos are caught immediately.

### Model Variants

| Model file | Backbone | Head | Feature scales |
|---|---|---|---|
| `models/mask2former_dinov3_vitsmallplus.py` | DINOv3 ViT-S+/16 | Swin-Small config | 4 maps, all stride 16 |
| `models/mask2former_dinov3_vitlarge.py` | DINOv3 ViT-L/16 | Swin-Large config | 4 maps, all stride 16 |
| `models/mask2former_dinov3_vitsmallplus_pyramid.py` | DINOv3 ViT-S+/16 | Swin-Small config | strides 4/8/16/32 (ViTDet-style pyramid) |
| `models/mask2former_dinov3_vitlarge_pyramid.py` | DINOv3 ViT-L/16 | Swin-Large config | strides 4/8/16/32 (ViTDet-style pyramid) |

The `_pyramid` variants resample the tapped ViT features into a real multi-scale pyramid, matching what the Mask2Former pixel decoder was designed for. Checkpoints are **not** weight-compatible between plain and pyramid variants.

### Additional Training Options

- `--freeze_backbone` / `--no-freeze_backbone`: freeze or finetune the DINOv3 backbone (default: frozen). In a JSON config use `"freeze_backbone": false`.
- `--use_augmentation`: enable training-time augmentations (horizontal flip + brightness/contrast). Do **not** enable flips when class labels depend on object position.
- `--max_grad_norm`: gradient clipping max norm (disabled by default; the Mask2Former paper uses 0.01).
- `--logging_steps`: log training loss and learning rate every N optimization steps (default: 50).

Mixed precision is handled by Accelerate, e.g.:
```bash
accelerate launch --mixed_precision bf16 mask2former_dinov3_no_trainer_coco.py --config ...
```

### Final Evaluation

If `<dataset>/test/_annotations.coco.json` exists, the final evaluation after training runs on the test split and results are stored with a `test_` prefix in `all_results.json`. Otherwise the last validation metrics are reported with a `valid_` prefix.

## Inference

The project now includes `simple_inference.py`, a user-friendly script for running inference with trained Mask2Former models.

### Features
- **Single Image Mode**: Process individual images
- **Batch Mode**: Process entire directories of images
- **Recursive Processing**: Option to process subdirectories
- **Visual Results**: Automatically generates annotated images with colored masks and confidence scores
- **Flexible Thresholding**: Adjustable detection confidence threshold

Class names shown on the output images are read from the checkpoint's `id2label` mapping.

### Usage Examples

Provide exactly one of `--image_path` (single mode) or `--input_dir` (batch mode).

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
    --threshold 0.5
```

#### Recursive Directory Processing
```bash
python simple_inference.py \
    --model_path ./output/dinov3-smallplus-mask2former-1e4/step_1400_model \
    --input_dir /path/to/images/ \
    --output_dir /path/to/results/ \
    --recursive
```

### Parameters
- `--model_path, -m`: Path to the trained model directory (required)
- `--image_path, -i`: Path to a single image (single image mode)
- `--input_dir, -d`: Input directory containing images (batch mode)
- `--output_dir, -od`: Output directory for batch results (default: `results`)
- `--output, -o`: Output path for single image result
- `--threshold, -t`: Detection confidence threshold (default: 0.5)
- `--recursive, -r`: Process subdirectories recursively (batch mode)

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

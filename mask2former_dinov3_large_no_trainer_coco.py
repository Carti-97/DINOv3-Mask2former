#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 [Chan Ho Bae / GitHub @Carti-97]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a modification of the Hugging Face Transformers instance segmentation example.
It has been adapted to replace the original backbone with DINOv3 models and support the COCO dataset format.
"""

"""Finetuning 🤗 Transformers model for instance segmentation with Accelerate 🚀."""

import argparse
import json
import logging
import math
import os
import sys
import pickle
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import datasets
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    SchedulerType,
    get_scheduler,
)
from transformers.image_processing_utils import BatchFeature
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from typing import List, Dict
from transformers import AutoModel

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.56.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/instance-segmentation/requirements.txt")


# ===================== COCO Dataset Class =====================
class COCOInstanceDataset(Dataset):
    """COCO dataset optimized for instance segmentation"""
    
    def __init__(self, data_dir, split, image_processor, transform=None, use_cache=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_processor = image_processor
        self.transform = transform
        
        ann_file = self.data_dir / split / "_annotations.coco.json"
        cache_file = self.data_dir / split / "_cache.pkl"
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        # Cache handling
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached annotations from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.image_ids = cache_data['image_ids']
                self.annotations = cache_data['annotations']
                self.categories = cache_data['categories']
                self.coco = None  # Don't need COCO object when using cache
        else:
            logger.info(f"Building annotation cache for {split}")
            self.coco = COCO(ann_file)
            self.image_ids = list(self.coco.imgs.keys())
            self.categories = {cat['id']: cat['name'] 
                              for cat in self.coco.loadCats(self.coco.getCatIds())}
            
            # Pre-process annotations
            self.annotations = {}
            for idx, img_id in enumerate(tqdm(self.image_ids, desc="Caching annotations")):
                img_info = self.coco.loadImgs(img_id)[0]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                self.annotations[img_id] = {
                    'file_name': img_info['file_name'],
                    'height': img_info['height'],
                    'width': img_info['width'],
                    'anns': anns  # Keep annotations, generate masks on demand
                }
            
            # Save cache
            if use_cache:
                cache_data = {
                    'image_ids': self.image_ids,
                    'annotations': self.annotations,
                    'categories': self.categories
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cache saved to {cache_file}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_data = self.annotations[img_id]
        
        # Load image
        img_path = self.data_dir / self.split / ann_data['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Generate masks on-demand
        h, w = ann_data['height'], ann_data['width']
        instance_mask = np.zeros((h, w), dtype=np.int32)
        instance_to_semantic = {}
        
        for i, ann in enumerate(ann_data['anns'], 1):
            if 'segmentation' in ann:
                if self.coco:
                    mask = self.coco.annToMask(ann)
                else:
                    mask = self._poly_to_mask(ann['segmentation'], h, w)
                instance_mask[mask > 0] = i
                instance_to_semantic[i] = ann['category_id']
        
        # Apply transforms
        if self.transform:
            image_np = np.array(image)
            output = self.transform(image=image_np, mask=instance_mask)
            image = output["image"]
            instance_mask = output["mask"]
            
            # Remap instance IDs after augmentation
            unique_ids = np.unique(instance_mask)
            instance_to_semantic = {
                int(inst_id): instance_to_semantic.get(inst_id, 0)
                for inst_id in unique_ids if inst_id > 0
            }
        else:
            image = np.array(image)
        
        # Apply image processor
        inputs = self.image_processor(
            images=[image],
            segmentation_maps=[instance_mask],
            instance_id_to_semantic_id=instance_to_semantic,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": inputs.pixel_values[0],
            "mask_labels": inputs.mask_labels[0],
            "class_labels": inputs.class_labels[0],
            "original_size": (h, w),  # 원본 크기 추가
        }
    
    @staticmethod
    def _poly_to_mask(polygons, h, w):
        """Convert polygon to mask without COCO"""
        import cv2
        mask = np.zeros((h, w), dtype=np.uint8)
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model for instance segmentation task")

    # JSON config file option (at the beginning)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file containing training arguments"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to a pretrained model or model identifier from huggingface.co/models.",
        default="facebook/mask2former-swin-tiny-coco-instance",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset on the hub or path to local COCO dataset.",
        default="qubvel-hf/ade20k-mini",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=384,
        help="The height of the images to feed the model.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=384,
        help="The width of the images to feed the model.",
    )
    parser.add_argument(
        "--do_reduce_labels",
        action="store_true",
        help="Whether to reduce the number of labels by removing the background class.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path to a folder in which the model and dataset will be cached.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers to use for the dataloaders.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    # JSON config file handling
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update args with values from JSON
        for key, value in config.items():
            if hasattr(args, key):
                # Convert string to SchedulerType for lr_scheduler_type
                if key == 'lr_scheduler_type' and isinstance(value, str):
                    value = SchedulerType(value)
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown argument in config: {key}")

    # Sanity checks
    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        image = np.array(pil_image)
        semantic_and_instance_masks = np.array(pil_annotation)[..., :2]

        # Apply augmentations
        output = transform(image=image, mask=semantic_and_instance_masks)

        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

        # Create mapping from instance id to semantic id
        unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
        }

        # Apply the image processor transformations: resizing, rescaling, normalization
        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[aug_instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )

        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    
    # ================== FIX START ==================
    if "original_size" in examples[0]:
        batch["original_sizes"] = [example["original_size"] for example in examples]
    # ================== FIX END ==================
    
    return batch


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


def evaluation_loop(model, image_processor, accelerator: Accelerator, dataloader, id2label):
    # Each GPU maintains its own metric instance
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True).to(accelerator.device)

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not accelerator.is_local_main_process):
        original_sizes = inputs.pop("original_sizes", None)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get target sizes for current batch
        if original_sizes is not None:
            current_target_sizes = original_sizes
        else:
            current_target_sizes = [masks.shape[-2:] for masks in inputs["mask_labels"]]
        
        # Process predictions for current batch (local GPU only)
        post_processed_output = image_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            target_sizes=current_target_sizes,
            return_binary_maps=True,
        )
        
        # Prepare predictions and targets for current batch
        post_processed_predictions = []
        post_processed_targets = []
        
        for idx, (image_predictions, target_size) in enumerate(zip(post_processed_output, current_target_sizes)):
            # Prediction
            if image_predictions["segments_info"]:
                pred = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]], 
                                          device=accelerator.device),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]], 
                                          device=accelerator.device),
                }
            else:
                pred = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool, device=accelerator.device),
                    "labels": torch.tensor([], device=accelerator.device),
                    "scores": torch.tensor([], device=accelerator.device),
                }
            post_processed_predictions.append(pred)
            
            # Target
            target = {
                "masks": inputs["mask_labels"][idx].to(dtype=torch.bool),
                "labels": inputs["class_labels"][idx],
            }
            post_processed_targets.append(target)
        
        # Update metric locally (no gathering)
        metric.update(post_processed_predictions, post_processed_targets)

    results = metric.compute()
    return results


def setup_logging(accelerator: Accelerator) -> None:
    """Setup logging according to `training_args`."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        logger.setLevel(logging.INFO)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def handle_repository_creation(accelerator: Accelerator, args: argparse.Namespace):
    """Create a repository for the model and dataset if `args.push_to_hub` is set."""

    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    return repo_id


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_instance_segmentation_no_trainer", args)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    setup_logging(accelerator)

    # If passed along, set the training seed now.
    # We set device_specific to True as we want different data augmentation per device.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Create repository if push ot hub is specified
    repo_id = handle_repository_creation(accelerator, args)

    if args.push_to_hub:
        api = HfApi()

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits - MODIFIED FOR COCO SUPPORT
    # ------------------------------------------------------------------------------------------------
    
    # Check if dataset_name is a local directory (COCO dataset)
    if Path(args.dataset_name).is_dir():
        logger.info(f"Loading local COCO dataset from {args.dataset_name}")
        
        # Initialize image processor first
        image_processor = AutoImageProcessor.from_pretrained(
            args.model_name_or_path,
            do_resize=True,
            size={"height": args.image_height, "width": args.image_width},
            do_reduce_labels=args.do_reduce_labels,
            reduce_labels=args.do_reduce_labels,
            token=args.hub_token,
        )
        
        # Define augmentations
        train_transform = A.Compose([A.NoOp()])
        
        val_transform = A.Compose([A.NoOp()])
        
        # Create datasets
        train_dataset = COCOInstanceDataset(
            args.dataset_name,
            "train",
            image_processor,
            transform=train_transform,
            use_cache=True
        )
        
        # Try to load validation set, if not available use subset of training
        try:
            val_dataset = COCOInstanceDataset(
                args.dataset_name,
                "valid",
                image_processor,
                transform=val_transform,
                use_cache=True
            )
        except FileNotFoundError:
            logger.warning("Validation dataset not found. Using 10% of training set.")
            from torch.utils.data import random_split
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Setup label mappings
        if hasattr(train_dataset, 'categories'):
            label2id = {name: cat_id for cat_id, name in train_dataset.categories.items()}
        else:
            # If using random_split, access the underlying dataset
            label2id = {name: cat_id for cat_id, name in train_dataset.dataset.categories.items()}
        
        if args.do_reduce_labels:
            label2id = {name: idx - 1 for name, idx in label2id.items() if idx != 0}
        
        id2label = {v: k for k, v in label2id.items()}
        
        # Create model with the label mappings
        # ================== START: MODIFICATION FOR DINOv3 BACKBONE ==================
        
        # 1. 여기에 DINOv3 커스텀 백본과 어댑터 클래스를 붙여넣습니다.
        
        
        class Adapter(nn.Module):
            def __init__(self, in_channels: int, out_channels: List[int]):
                super().__init__()
                self.projections = nn.ModuleList([
                    nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels
                ])
            def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
                return [self.projections[i](feat) for i, feat in enumerate(features)]

        class DinoV3WithAdapterBackbone(nn.Module):
            def __init__(self, model_name: str, out_channels: List[int]):
                super().__init__()
                self.model = AutoModel.from_pretrained(model_name)
                self.adapter = Adapter(self.model.config.hidden_size, out_channels)
                self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
                self._out_feature_channels = {name: ch for name, ch in zip(self.out_features, out_channels)}
                self._out_feature_strides = {"stage_0": 8, "stage_1": 16, "stage_2": 32, "stage_3": 32}
                self.layers_to_extract = [5, 11, 17, 23]
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                batch_size, _, height, width = x.shape
                patch_size = self.model.config.patch_size
                patch_height, patch_width = height // patch_size, width // patch_size
                extracted_features = []
                for layer_idx in self.layers_to_extract:
                    layer_output = hidden_states[layer_idx + 1]
                    feature_map = layer_output[:, 1:, :].permute(0, 2, 1).reshape(
                        batch_size, self.model.config.hidden_size, patch_height, patch_width)
                    extracted_features.append(feature_map)
                adapted_features = self.adapter(extracted_features)
                return {name: feat for name, feat in zip(self.out_features, adapted_features)}

        # 2. DINOv3 모델 정보와 Mask2Former 헤드가 기대하는 채널 정의
        dinov3_backbone_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        expected_channels = [192, 384, 768, 1536] # Swin-Large 헤드 기준

        # 3. 커스텀 백본 생성
        custom_backbone = DinoV3WithAdapterBackbone(dinov3_backbone_name, expected_channels)
        
        # 4. Mask2Former 모델 로드 (헤드 부분만 사용)
        model = AutoModelForUniversalSegmentation.from_pretrained(
            args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            token=args.hub_token,
        )
        
        # 5. 백본 교체 및 동결(Freeze)
        model.model.backbone = custom_backbone
        for param in model.model.backbone.model.parameters():
            param.requires_grad = False
            
        logger.info("Successfully replaced and froze the DINOv3 backbone.")
        # =================== END: MODIFICATION FOR DINOv3 BACKBONE ===================
        
    else:
        # Original HuggingFace dataset loading code
        logger.info(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
        
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code)
        
        label2id = dataset["train"][0]["semantic_class_to_id"]
        if args.do_reduce_labels:
            label2id = {name: idx for name, idx in label2id.items() if idx != 0}
            label2id = {name: idx - 1 for name, idx in label2id.items()}
        
        id2label = {v: k for k, v in label2id.items()}
        
        # Load pretrained model and image processor
        model = AutoModelForUniversalSegmentation.from_pretrained(
            args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            token=args.hub_token,
        )
        
        image_processor = AutoImageProcessor.from_pretrained(
            args.model_name_or_path,
            do_resize=True,
            size={"height": args.image_height, "width": args.image_width},
            do_reduce_labels=args.do_reduce_labels,
            reduce_labels=args.do_reduce_labels,
            token=args.hub_token,
        )
        
        # Define image augmentations
        train_augment_and_transform = A.Compose([A.NoOp()])
        validation_transform = A.Compose([A.NoOp()])
        
        # Transform functions for batch
        train_transform_batch = partial(
            augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
        )
        validation_transform_batch = partial(
            augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
        )
        
        with accelerator.main_process_first():
            dataset["train"] = dataset["train"].with_transform(train_transform_batch)
            dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
        
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

    # ------------------------------------------------------------------------------------------------
    # Create dataloaders
    # ------------------------------------------------------------------------------------------------
    
    dataloader_common_args = {
        "num_workers": args.dataloader_num_workers,
        "persistent_workers": args.dataloader_num_workers > 0,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.per_device_train_batch_size, 
        **dataloader_common_args
    )
    
    valid_dataloader = DataLoader(
        val_dataset, 
        shuffle=False, 
        batch_size=args.per_device_eval_batch_size, 
        **dataloader_common_args
    )

    # ------------------------------------------------------------------------------------------------
    # Define optimizer, scheduler and prepare everything with the accelerator
    # ------------------------------------------------------------------------------------------------

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ------------------------------------------------------------------------------------------------
    # Run training with evaluation on each epoch
    # ------------------------------------------------------------------------------------------------

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    
    # Best epoch tracking
    best_epoch = -1
    best_metric = -1.0  # mAP 기준으로 추적
    best_metrics = {}

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # ================== FIX START ==================
                # 모델이 예상하지 않는 'original_sizes' 인자를 제거합니다.
                # 이 정보는 평가 시에만 필요합니다.
                if "original_sizes" in batch:
                    batch.pop("original_sizes")
                # ================== FIX END ====================
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            api.upload_folder(
                                repo_id=repo_id,
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=args.output_dir,
                                repo_type="model",
                                token=args.hub_token,
                            )

            if completed_steps >= args.max_train_steps:
                break

        logger.info("***** Running evaluation *****")
        metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label)

        logger.info(f"epoch {epoch}: {metrics}")

        # Best epoch 확인 및 업데이트
        current_metric = 0.0
        if 'map' in metrics:
            current_metric = metrics['map'].item() if isinstance(metrics['map'], torch.Tensor) else metrics['map']
        elif 'map_50' in metrics:
            current_metric = metrics['map_50'].item() if isinstance(metrics['map_50'], torch.Tensor) else metrics['map_50']
        elif 'map_75' in metrics:
            current_metric = metrics['map_75'].item() if isinstance(metrics['map_75'], torch.Tensor) else metrics['map_75']
        
        is_best_epoch = current_metric > best_metric
        if is_best_epoch:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = dict(metrics)
            logger.info(f"🏆 새로운 BEST EPOCH: {epoch}, mAP: {current_metric:.4f}")

        # 각 epoch마다 모델 저장 (로컬)
        if args.output_dir is not None:
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                epoch_output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            
            if accelerator.is_main_process:
                image_processor.save_pretrained(epoch_output_dir)
                
                # epoch 메트릭 저장
                epoch_metrics = {f"epoch_{epoch}_{k}": v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
                with open(os.path.join(epoch_output_dir, f"epoch_{epoch}_metrics.json"), "w") as f:
                    json.dump(epoch_metrics, f, indent=2)
                
                logger.info(f"모델과 메트릭이 {epoch_output_dir}에 저장되었습니다.")

            # Best epoch 모델 저장
            if is_best_epoch:
                best_output_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_output_dir, exist_ok=True)
                
                unwrapped_model.save_pretrained(
                    best_output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                
                if accelerator.is_main_process:
                    image_processor.save_pretrained(best_output_dir)
                    
                    # Best 메트릭과 epoch 정보 저장
                    best_info = {
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "metric_name": "map" if "map" in metrics else "map_50" if "map_50" in metrics else "map_75",
                        "all_metrics": {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in best_metrics.items()}
                    }
                    
                    with open(os.path.join(best_output_dir, "best_model_info.json"), "w") as f:
                        json.dump(best_info, f, indent=2)
                    
                    logger.info(f"🏆 BEST 모델이 {best_output_dir}에 저장되었습니다!")

        # Hub에 푸시하는 경우 (선택사항)
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        # Accelerator state 저장 (체크포인트)
        if args.checkpointing_steps == "epoch":
            checkpoint_dir = f"checkpoint_epoch_{epoch}"
            if args.output_dir is not None:
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_dir)
            accelerator.save_state(checkpoint_dir)

    # ------------------------------------------------------------------------------------------------
    # Run evaluation on test dataset and save the model
    # ------------------------------------------------------------------------------------------------

    logger.info("***** Running evaluation on test dataset *****")
    metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label)
    
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            # 텐서가 단일 값(scalar)을 가지면 .item()으로, 여러 값을 가지면 .tolist()로 변환
            processed_metrics[f"test_{key}"] = value.tolist() if value.numel() > 1 else value.item()
        else:
            processed_metrics[f"test_{key}"] = value

    logger.info(f"Test metrics: {processed_metrics}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            # 최종 결과에 best epoch 정보 포함
            final_results = {
                **processed_metrics,
                "training_summary": {
                    "total_epochs": args.num_train_epochs,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_metric_name": "map" if "map" in best_metrics else "map_50" if "map_50" in best_metrics else "map_75",
                    "best_epoch_metrics": {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in best_metrics.items()} if best_metrics else {}
                }
            }
            
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            # Best epoch 요약 로그
            if best_epoch >= 0:
                logger.info(f"🏆 훈련 완료! BEST EPOCH: {best_epoch}, 최고 성능: {best_metric:.4f}")
                logger.info(f"BEST 모델 위치: {os.path.join(args.output_dir, 'best_model')}")
            else:
                logger.info("훈련 완료! (Best epoch 정보 없음)")

            image_processor.save_pretrained(args.output_dir)

            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                    ignore_patterns=["epoch_*"],
                )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
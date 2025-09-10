# Mask2Former with DINOv3 Backbones

This project replaces the Swin-Small and Swin-Large backbones in Mask2Former with the ViT Small Plus and ViT Large models from DINOv3, respectively. This modification aims to leverage the powerful, self-supervised learned features of DINOv3 for instance segmentation tasks.

The training methodology is based on the instance segmentation examples provided in the Hugging Face Transformers library. The implementation can be found at: [https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).

## License

This project adheres to the DINOv3 license. DINOv3 is released under a commercial license, and its training code and pre-trained backbones are open-sourced to foster innovation in the computer vision community. For detailed terms, please refer to the official DINOv3 license agreement.

## Dataset

This model is designed to be trained on datasets with the COCO format.

## Usage

To train the model, you can use the following command as an example:

```bash
accelerate launch mask2former_dinov3_smallplus_no_trainer_coco.py --config mask2former-dinov3_smallplus_1024_train_args.json
```

In this example, `mask2former_dinov3_smallplus_no_trainer_coco.py` is the training script, and `mask2former-dinov3_smallplus_1024_train_args.json` is the configuration file containing the training arguments. You will need to adapt these files to your specific dataset and training setup.
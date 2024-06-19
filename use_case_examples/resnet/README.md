# ResNet18 with Fully Homomorphic Encryption

## Overview

This project executes the ResNet18 image classification model using Fully Homomorphic Encryption (FHE) with Concrete ML. The model is adapted for FHE compatibility and tested on a small subset of tiny-imagenet (up-sampled) images.

## ResNet18

The ResNet18 model is adapted from torchvision the original https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py where the adaptive average pooling layer `AdaptiveAvgPool2d` (not yet supported by Concrete ML) is replaced with a standard `AvgPool2d` layer as follows:

```diff
-        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
+        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
```

The rest is left unchanged.

## Evaluation dataset

The model is evaluated on images from [the Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet).

The `TinyImageNetProcessor` class in `utils_resnet.py` preprocesses the Tiny ImageNet data-set and aligns it with ImageNet labels for model evaluation.

## Usage

1. Install a virtual Python environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install Concrete ML:

```bash
pip install concrete-ml
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the script:

```bash
python resnet_fhe.py [--run_fhe] [--export_statistics]
```

Example of output when running the script:

```bash
python resnet_fhe.py --run_fhe
```

```
Accuracy of the ResNet18 model on the images: 56.00%
Top-5 Accuracy of the ResNet18 model on the images: 82.00%
Compiling the model...
Model compiled successfully.
Quantized Model Accuracy of the FHEResNet18 on the images: 54.00%
Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: 77.00%
FHE Simulation Accuracy of the FHEResNet18 on the images: 53.00%
FHE Simulation Top-5 Accuracy of the FHEResNet18 on the images: 75.00%
Time taken for one FHE execution: 5482.5433 seconds
```

## Timings and Accuracy in FHE

CPU machine: 196 cores CPU machine (hp7c from AWS)
GPU machine: TBD

Summary of the accuracy evaluation on tinyImageNet (100 images):

| w&a bits | p_error | Accuracy | Top-5 Accuracy | Runtime (hours) | Device |
| -------- | ------- | -------- | -------------- | --------------- | ------ |
| 6/6      | 0.05    | 50%      | 75%            | **1.52**        | CPU    |
| 6/6      | 0.05    | 50%      | 75%            | TBD             | GPU    |
| 6/7      | 0.05    | 53%      | 76%            | 2.2             | CPU    |
| 6/7      | 0.005   | 57%      | 74%            | 5.2             | CPU    |

Note that the original model in fp32 achieved 56% accuracy and 82% top-5 accuracy.

Recommended configuration: 6/6 with p_error = 0.05

6/6 `n_bits` configuration: {"model_inputs": 8, "op_inputs": 6, "op_weights": 6, "model_outputs": 8}

6/7 `n_bits` configuration: {"model_inputs": 8, "op_inputs": 7, "op_weights": 6, "model_outputs": 8}

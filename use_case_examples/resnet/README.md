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

The `TinyImageNetProcessor` class in `utils_resnet.py` preprocesses the Tiny ImageNet dataset and aligns it with ImageNet labels for model evaluation.

## Usage

1. Install a virtual python environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```
2. Install concrete-ml:

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

## Timing

| Device | Time (seconds) |
|--------|----------------|
| CPU    | 5482.5433      |
| GPU    | TBD            |

# ResNet18 with Fully Homomorphic Encryption

## Overview

This project executes the ResNet18 image classification model using Fully Homomorphic Encryption (FHE) with Concrete ML. The model is adapted for FHE compatibility and tested on a small subset of imagenet images.

## ResNet18

The ResNet18 model is adapted from torchvision the original https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py where the adaptive average pooling layer `AdaptiveAvgPool2d` (not yet supported by Concrete ML) is replaced with a standard `AvgPool2d` layer as follows:

```diff
-        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
+        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
```

The rest is left unchanged.

## Evaluation data-set

The model is evaluated on images from the [ImageNet-1k data-set](https://huggingface.co/datasets/timm/imagenet-1k-wds).

The `ImageNetProcessor` class in `utils_resnet.py` preprocesses the ImageNet validation set for model evaluation. It uses a subset of the validation data to ensure efficient processing and evaluation.

## Usage

1. Install a virtual Python environment and activate it:

<!--pytest-codeblocks:skip-->

```bash
python -m venv venv
source venv/bin/activate
```

2. Install Concrete ML:

<!--pytest-codeblocks:skip-->

```bash
pip install concrete-ml
```

3. Install the dependencies:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt
```

4. Run the script:

<!--pytest-codeblocks:skip-->

```bash
python run_resnet18_fhe.py [--run_fhe] [--export_statistics] [--use_gpu] [--run_experiment] [--dataset_cache_dir <path>] [--num_images <number>]
```

The script `run_resnet18_fhe.py` accepts several command-line arguments to control its behavior:

- `--run_fhe`: runs the actual FHE execution of the ResNet18 model. If not set, the script will run the model without FHE.

- `--export_statistics`: exports the circuit statistics after running the model. This can be useful for analyzing the performance and characteristics of the FHE execution.

- `--use_gpu`: utilizes the available GPU for FHE runtime, potentially speeding up the execution. If not set, the script will run on the CPU.

- `--run_experiment`: runs experiments with different `n_bits` and `rounding_threshold_bits` configurations. This can help in finding the optimal settings for the model.

- `--dataset_cache_dir <path>`: specifies the path to the directory where the data-set is cached. If not provided, the data-set will be downloaded and cached in the default location.

- `--num_images <number>`: specifies the number of images to process in the FHE execution. The default value is 1. Increasing this number will process more images but may take longer to execute.

Example of output when running the script:

<!--pytest-codeblocks:skip-->

```bash
python resnet_fhe.py --run_fhe
```

```
Accuracy of the ResNet18 model on the images: 67.00%
Top-5 Accuracy of the ResNet18 model on the images: 87.00%
Compiling the model with compile_torch_model...
Model compiled successfully.
Quantized Model Accuracy of the FHEResNet18 on the images: 67.00%
Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: 87.00%
FHE simulation Accuracy of the FHEResNet18 on the images: 66.00%
FHE simulation Top-5 Accuracy of the FHEResNet18 on the images: 87.00%
Processing 1 image(s)...

Image 1:
  Running FHE execution...
  FHE execution completed in 811.8710 seconds
  FHE top 5 labels: 636, 588, 502, 774, 459
  Running simulation...
  Simulation top 5 labels: 636, 588, 502, 774, 459
```

## Timings and Accuracy in FHE

CPU machine: 196 cores CPU machine (hp7c from AWS)
GPU machine: 8xH100 GPU machine

Summary of the accuracy evaluation on ImageNet (100 images):

| w&a bits | p_error | Accuracy | Top-5 Accuracy | Runtime        | Device |
| -------- | ------- | -------- | -------------- | -------------- | ------ |
| fp32     | -       | 67%      | 87%            | -              | -      |
| 6/6      | 0.05    | 55%      | 78%            | 56 min         | GPU    |
| 6/6      | 0.05    | 55%      | 78%            | 1 h 31 min     | CPU    |
| 7/7      | 0.05    | **66%**  | **87%**        | **2 h 12 min** | CPU    |

6/6 `n_bits` configuration: {"model_inputs": 8, "op_inputs": 6, "op_weights": 6, "model_outputs": 9}

7/7 `n_bits` configuration: {"model_inputs": 8, "op_inputs": 7, "op_weights": 7, "model_outputs": 9}

For each setting, we use a the following config for the `rounding_threshold_bits`: `{"n_bits": 7, "method": "APPROXIMATE"}`.

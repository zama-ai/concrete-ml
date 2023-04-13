# Description

In this directory we show how to solve a classification task on CIFAR-10 and CIFAR-100, by converting a pre-trained CNN to its fully homomorphic encryption (FHE) equivalent using Quantization Aware Training (QAT) and Concrete ML. We evaluate it using the FHE simulation
mode provided by the Concrete stack.

To do so, we divided this use case in 4 notebooks :

1. [FromImageNetToCifar.ipynb](FromImageNetToCifar.ipynb) : in this notebook, we used a VGG11 network from [torch.hub](https://pytorch.org/hub/pytorch_vision_vgg/), on which we applied some changes on the original architecture in order to speed up the FHE execution and therefore make it more user-friendly :

   - replacing the `MaxPool2d` by the `AvgPool2d`, because in the current version of Concrete ML `MaxPool2d` isn't available yet
   - changing the kernel and stride size in some pooling layers, because we have chosen to keep the initial input size of $3*32*32$ instead of the recommended input size of $3*224*224$
   - changing the kernel size from $7$ to $1$ in the `AdaptiveAvgPool2d` to further reduce the image's size
   - removing the classification part of the original VGG11 architecture to speed up computation

   This notebook may be skipped if the user already has a pre-trained floating point CIFAR-10 / CIFAR-100 model.

1. [CifarQuantizationAwareTraining.ipynb](CifarQuantizationAwareTraining.ipynb): explains the **(FHE) constraints** and how to **quantize the pre-trained** neural network to make it work in FHE.

1. [CifarInFhe.ipynb](CifarInFhe.ipynb): computes the accuracy of the quantized models using FHE simulation

1. [CifarInFheWithSmallerAccumulators.ipynb](./CifarInFheWithSmallerAccumulators.ipynb): shows how to use the rounded PBS operation
   to lower the accumulator size, thus decreasing the inference time in FHE

## Installation

To use this code, you need to have Python 3.8 and install the following dependencies:

```
pip install -r requirements.txt
```

## Results

<!-- Add FHE inference accuracy -->

<!-- FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2420 -->

CIFAR-100:

| Runtime            | Top1 Accuracy |
| ------------------ | ------------- |
| VGG PyTorch        | 70.04%        |
| VGG Brevitas       | 68.40%        |
| VGG FHE simulation | 68.28%        |

CIFAR-10:

| Runtime            | Top1 Accuracy |
| ------------------ | ------------- |
| VGG PyTorch        | 90.11%        |
| VGG Brevitas       | 90.40%        |
| VGG FHE simulation | 90.28%        |

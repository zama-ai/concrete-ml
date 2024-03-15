# CIFAR-10 and CIFAR-100 Classification with FHE

This repository provides resources and documentation on different use-cases for classifying CIFAR-10 and CIFAR-100 images using Fully Homomorphic Encryption (FHE). Each use-case demonstrates different techniques and adaptations to work within the constraints of FHE.Notably, a fine-tuning from a public pre-trained model, a training from scratch using quantization aware training (QAT) and finally a hybrid approach where only a subset of the model is done in FHE.

## Table of Contents

1. [Use-Cases](#use-cases)
   - [Fine-Tuning VGG11 CIFAR-10/100](#fine-tuning-cifar)
   - [Training Ternary VGG9 on CIFAR10](#training-ternary-vgg-on-cifar10)
1. [Installation](#installation)
1. [Further Reading & Resources](#further-reading)

## Use cases

### Fine-Tuning CIFAR

- **Description**: This use-case explores how to convert a pre-trained CNN (on imagenet) to its FHE equivalent using Quantization Aware Training (QAT) and Concrete ML. The conversion process involves adapting a VGG11 network and quantizing the network for FHE.

Notebooks:

1. [Adapting VGG11 for CIFAR datasets](cifar_brevitas_finetuning/FromImageNetToCifar.ipynb).
1. [Quantizing the pre-trained network](cifar_brevitas_finetuning/CifarQuantizationAwareTraining.ipynb).
1. [Computing the accuracy of the quantized models with FHE simulation](cifar_brevitas_finetuning/CifarInFhe.ipynb).
1. [Enhancing inference time in FHE using smaller accumulators](cifar_brevitas_finetuning/CifarInFheWithSmallerAccumulators.ipynb).

[Results & Metrics](./cifar_brevitas_finetuning/README.md#results)

### Training Ternary VGG on CIFAR10

- **Description**: Train a VGG-like neural network from scratch using Brevitas on CIFAR-10 and run it in FHE. This use-case modifies the original VGG model for compatibility with Concrete ML, and explores the performance gains of rounding operations in FHE.
- **Training & Inference**: Scripts provided to train the network and evaluate its performance. It also includes simulations in Concrete ML and insights into the performance enhancement using rounding.

[Results & Metrics](./cifar_brevitas_training/README.md#accuracy-and-performance)

## Installation

All use-cases can be quickly set up with:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt
```

## Further Reading

- [Concrete ML Documentation](https://docs.zama.ai/concrete-ml/)
- [Brevitas Github Repository](https://github.com/Xilinx/brevitas)
- [Fully Homomorphic Encryption Basics](https://www.zama.ai/post/tfhe-deep-dive-part-1)
- [CIFAR Datasets Overview](https://www.cs.toronto.edu/~kriz/cifar.html)

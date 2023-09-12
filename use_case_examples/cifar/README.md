# CIFAR-10 and CIFAR-100 Classification with FHE

This repository provides resources and documentation on different use-cases for classifying CIFAR-10 and CIFAR-100 images using Fully Homomorphic Encryption (FHE). Each use-case demonstrates different techniques and adaptations to work within the constraints of FHE.

## Table of Contents
1. [Use-Cases](#use-cases)
   - [Fine-Tuning CIFAR-10/100](#fine-tuning-cifar-10100)
   - [Training Ternary VGG on CIFAR10](#training-ternary-vgg-on-cifar10)
   - [CIFAR-10 with a Split Clear/FHE Model](#cifar-10-with-a-split-clearfhe-model)
2. [Installation](#installation)
3. [Further Reading & Resources](#further-reading--resources)

## Use cases

### Fine-Tuning CIFAR-10/100

- **Description**: This use-case explores how to convert a pre-trained CNN (on imagenet) to its FHE equivalent using QAT and Concrete ML. The conversion process involves adapting a VGG11 network and quantizing the network for FHE.

- Notebooks:
1. [Adapting VGG11 for CIFAR datasets](cifar_brevitas_finetuning/FromImageNetToCifar.ipynb).
2. [Quantizing the pre-trained network](cifar_brevitas_finetuning/CifarQuantizationAwareTraining.ipynb).
3. [Computing the accuracy of the quantized models with FHE simulation](cifar_brevitas_finetuning/CifarInFhe.ipynb).
4. [Enhancing inference time in FHE using smaller accumulators](cifar_brevitas_finetuning/CifarInFheWithSmallerAccumulators.ipynb).

[Results & Metrics](cifar_brevitas_finetuning/README.md#results)

### Training Ternary VGG on CIFAR10

- **Description**: Train a VGG-like neural network from scratch using Brevitas on CIFAR-10 and run it in FHE. This use-case modifies the original VGG model for compatibility with Concrete ML, and explores the performance gains of rounding operations in FHE.
- **Training & Inference**: Scripts provided to train the network and evaluate its performance. Also includes simulations in Concrete ML and insights into the performance enhancement using rounding.

[Results & Metrics](cifar_brevitas_training/README.md#Accuracy_/and_/performance)

### CIFAR-10 with a Split Clear/FHE Model

- **Description**: An approach that splits the model into two parts: one running in clear and the other in FHE. By doing this, higher precision can be achieved in the input layer while still benefiting from FHE in subsequent layers.
- **Model Design**: Targets 8-bit accumulators for faster FHE inference. Pruning and 2-bit weights are used.
- **Implementation**: Detailed steps on how to run the model in FHE and the trade-offs involved in choosing the appropriate p_error value.

[Results & Metrics](cifar_brevitas_with_model_splitting/README.md#results)

## Installation

All use-cases can be quickly set up with:

```bash
pip install -r requirements.txt
```

## Further Reading & Resources

- [Concrete ML Documentation](https://docs.zama.ai/concrete-ml/)
- [Brevitas Github Repository](https://github.com/Xilinx/brevitas)
- [Fully Homomorphic Encryption Basics](https://www.zama.ai/post/tfhe-deep-dive-part-1)
- [CIFAR Datasets Overview](https://www.cs.toronto.edu/~kriz/cifar.html)

# Optimizing inference

This document introduces several approaches to reduce the overall latency of a neural network.

## Introduction

Neural networks are challenging for encrypted inference. Each neuron in a network has to apply an activation function that requires a [Programmable Bootstrapping(PBS)](../getting-started/concepts.md#cryptography-concepts) operation. The latency of a single PBS depends on the bit-width of its input.

## Circuit bit-width optimization

[Quantization Aware Training](../explanations/quantization.md) and [pruning](../explanations/pruning.md) introduce specific hyper-parameters that influence the accumulator sizes. You can chose quantization and pruning configurations to reduce the accumulator size. To obtain a trade-off between latency and accuracy, you can manually set these hyper-parameters as described in the [deep learning design guide](torch_support.md#configuring-quantization-parameters).

## Structured pruning

While using unstructured pruning ensures the accumulator bit-width stays low, [structured pruning](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html) can eliminate entire neurons from the network as many neural networks are over-parametrized for easier training. You can apply structured pruning to a trained network as a fine-tuning step. [This example](../advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb) demonstrates how to apply structured pruning to built-in neural networks using the [prune](../references/api/concrete.ml.sklearn.base.md#method-prune) helper function. To apply structured pruning to custom models, it is recommended to use the [torch-pruning](https://github.com/VainF/Torch-Pruning) package.

## Rounded activations and quantizers

Reducing the bit-width of inputs to the Table Lookup (TLU) operations significantly improves latency. Post-training, you can leverage properties of the fused activation and quantization functions in the TLUs to further reduce the accumulator size. This is achieved through the _rounded PBS_ feature as described in the [rounded activations and quantizers reference](../explanations/advanced_features.md#rounded-activations-and-quantizers). Adjusting the rounding amount relative to the initial accumulator size can improve latency while maintaining accuracy.

## TLU error tolerance adjustment

Finally, the TFHE scheme introduces a TLU error tolerance parameter that has an impact on crypto-system parameters that influence latency. A higher tolerance of TLU off-by-one errors results in faster computations but may reduce accuracy. You can think of the error of obtaining $$T[x]$$ as a Gaussian distribution centered on $$x$$: $$TLU[x]$$ is obtained with probability of `1 - p_error`, while $$T[x-1]$$, $$T[x+1]$$ are obtained with much lower probability, etc. In Deep NNs, these type of errors can be tolerated up to some point. See the [`p_error` documentation for details](../explanations/advanced_features.md#approximate-computations) and more specifically [the API for finding the best `p_error`](../explanations/advanced_features.md#searching-for-the-best-error-probability).

# Optimizing inference

Neural networks pose unique challenges with regards to encrypted inference. Each neuron in a network applies an activation function that requires a PBS operation. The latency of a single PBS depends on the bit-width of the input of the PBS.

Several approaches can be used to reduce the overall latency of a neural network.

## Circuit bit-width optimization

[Quantization Aware Training](../explanations/quantization.md) and [pruning](../explanations/pruning.md) introduce specific hyper-parameters that influence the accumulator sizes. It is possible to chose quantization and pruning configurations that reduce the accumulator size. A trade-off between latency and accuracy can be obtained by varying these hyper-parameters as described in the [deep learning design guide](torch_support.md#configuring-quantization-parameters).

## Structured pruning

While un-structured pruning is used to ensure the accumulator bit-width stays low, [structured pruning](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html) can eliminate entire neurons from the network. Many neural networks are over-parametrized (since this enables easier training) and some neurons can be removed. Structured pruning, applied to a trained network as a fine-tuning step, can be applied to built-in neural networks using the [prune](../references/api/concrete.ml.sklearn.base.md#method-prune) helper function as shown in [this example](../advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb). To apply structured pruning to custom models, it is recommended to use the [torch-pruning](https://github.com/VainF/Torch-Pruning) package.

## Rounded activations and quantizers

Reducing the bit-width of the inputs to the Table Lookup (TLU) operations is a major source of improvements in the latency. Post-training, it is possible to leverage some properties of the fused activation and quantization functions expressed in the TLUs to further reduce the accumulator. This is achieved through the _rounded PBS_ feature as described in the [rounded activations and quantizers reference](../explanations/advanced_features.md#rounded-activations-and-quantizers). Adjusting the rounding amount, relative to the initial accumulator size, can bring large improvements in latency while maintaining accuracy.

## TLU error tolerance adjustment

Finally, the TFHE scheme exposes a TLU error tolerance parameter that has an impact on crypto-system parameters that influence latency. A higher tolerance of TLU off-by-one errors results in faster computations but may reduce accuracy. One can think of the error of obtaining $$T[x]$$ as a Gaussian distribution centered on $$x$$: $$TLU[x]$$ is obtained with probability of `1 - p_error`, while $$T[x-1]$$, $$T[x+1]$$ are obtained with much lower probability, etc. In Deep NNs, these type of errors can be tolerated up to some point. See the [`p_error` documentation for details](../explanations/advanced_features.md#approximate-computations) and more specifically the usage example of [the API for finding the best `p_error`](../explanations/advanced_features.md#searching-for-the-best-error-probability).

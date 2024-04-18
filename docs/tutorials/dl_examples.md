# Deep learning examples

These examples illustrate the basic usage of Concrete ML to build various types of neural networks. They use simple data-sets, focusing on the syntax and usage of Concrete ML. For examples showing how to train high-accuracy models on more complex data-sets, see the [Demos and Tutorials](showcase.md) section.

## FHE constraints considerations

The examples listed here make use of [simulation](../explanations/compilation.md#fhe-simulation) to perform evaluation over large test sets. Since FHE execution can be slow, only a few FHE executions can be performed. The [correctness guarantees](../getting-started/concepts.md#cryptography-concepts) of Concrete ML ensure that accuracy measured with simulation is the same as that which will be obtained during FHE execution.

Some examples constrain accumulators to 7-8 bits, which can be sufficient for simple data-sets. Up to 16-bit accumulators can be used, but this introduces a slowdown of 4-5x compared to 8-bit accumulators.

## List of Examples

### 1. Step-by-step guide to building a custom NN

- [Quantization aware training example](../advanced_examples/QuantizationAwareTraining.ipynb)

This shows how to use Quantization Aware Training and pruning when starting out from a classical PyTorch network. This example uses a simple data-set and a small NN, which achieves good accuracy with low accumulator size.

### 2. Custom convolutional NN on the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) data-set

- [Convolutional Neural Network](../advanced_examples/ConvolutionalNeuralNetwork.ipynb)

Following the [Step-by-step guide](../deep-learning/fhe_friendly_models.md), this notebook implements a Quantization Aware Training convolutional neural network on the MNIST data-set. It uses 3-bit weights and activations, giving a 7-bit accumulator.

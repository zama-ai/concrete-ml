# Deep Learning Examples

These examples illustrate the basic usage of Concrete-ML to build various types of neural network. They use simple data-sets, focusing on the syntax and usage of Concrete-ML. For examples showing how to train high-accuracy models on more complex data-sets, see the [Demos and Tutorials](../getting-started/showcase.md) section.

## FHE constraints considerations

The examples listed here make use of simulation (using the [Virtual Library](../advanced-topics/compilation.md#simulation-with-the-virtual-library)) to perform evaluation over large test sets. Since FHE execution can be slow, only for one or few examples actual FHE execution is performed. The [correctness guarantees](../getting-started/concepts.md#cryptography-concepts) of Concrete-ML ensure that accuracy measured with simulation is the same as that that will be obtained during FHE execution.

Some examples constrain accumulators to 7-8 bits, which can be sufficient for simple data-sets. Up to 16-bit accumulators can be used, but this introduces a slow-down of 4-5x compared to 8b accumulators.

## List of Examples

### 1. Step-by-step guide to building a custom NN

[<img src="../.gitbook/assets/jupyter_logo.png" width="20px">  Quantization aware training example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/QuantizationAwareTraining.ipynb)

Shows how to use Quantization Aware Training and pruning when starting out from a classical Pytorch network. This example uses a simple data-set and a small NN which achieves good accuracy with low accumulator size.

### 2. Custom convolutional NN on the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) data-set

[<img src="../.gitbook/assets/jupyter_logo.png" width="20px">   Convolutional Neural Network](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb)

Following the [Step-by-Step guide](./fhe_friendly_models.md), this notebook implements a Quantization Aware Training convolutional neural network on the MNIST data-set. It uses 3-bit weight and activations, giving a 7-bit accumulator.

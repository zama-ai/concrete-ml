# Using Skorch

Concrete-ML uses [Skorch](https://skorch.readthedocs.io/en/stable/) to implement multi-layer, fully-connected torch neural networks in a way that is compatible with the Scikit-learn API.

This wrapper implements Torch training boilerplate code, alleviating the work that needs to be done by the user. It is possible to add hooks during the training phase, for example once an epoch is finished.

Skorch allows the user to easily create a classifier or regressor around a neural network (NN), implemented in Torch as a `nn.Module`, which is used by Concrete-ML to provide a fully-connected multi-layer NN with a configurable number of layers and optional pruning (see [pruning](pruning.md) and the [neural network documentation](../built-in-models/neural-networks.md) for more information).

Under the hood, Concrete-ML uses a Skorch wrapper around a single torch module, `SparseQuantNeuralNetImpl`. More information can be found [in the API guide](../_apidoc/concrete.ml.sklearn.html#concrete.ml.sklearn.qnn.SparseQuantNeuralNetImpl).

```
class SparseQuantNeuralNetImpl(nn.Module):
    """Sparse Quantized Neural Network classifier.
```

## Parameter choice

A linear or convolutional layer of an NN will compute a linear combination of weights and inputs (also called a _'multi-sum'_). For example, a linear layer will compute:

$$\mathsf{output}^k = \sum_i^Nw_{i}^kx_i$$

where $$k$$ is the k-th neuron in the layer. In this case, the sum is taken on a single dimension. A convolutional layer will compute:

$$\mathsf{output}_{xy}^{k} = \sum_c^{N}\sum_j^{K_h}\sum_i^{K_w}w_{cji}^kx_{c,y+j,x+i}^k$$

where $$k$$ is the k-th filter of the convolutional layer and $$N$$, $$K_h$$ and $$K_w$$ are the number of input channels, the kernel height and the kernel width, respectively.

Following the formulas for the resulting bit width of quantized linear combinations described [here](../getting-started/concrete_numpy.md#limitations-for-fhe-friendly-models),  it can be seen that the maximum dimensionality of the input and weights that can make the result exceed 8 bits:

$$\Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right)$$

Here, $$n_{\mathsf{max}} = 8$$ is the maximum precision allowed.

For example, if $$n_{\mathsf{weights}} = 2$$ and $$n_{\mathsf{inputs}} = 2$$ with $$n_{\mathsf{max}} = 8$$, the worst case is where all inputs and weights are equal to their maximal value $$2^2-1=3$$. In this case, there can be at most $$\Omega = 28$$ elements in the multi-sums.

In practice, the distribution of the weights of a neural network is Gaussian, with many weights either 0 or having a small value. This enables exceeding the worst-case number of active neurons without having to risk overflowing the bitwidth. The parameter `n_hidden_neurons_multiplier` is multiplied with $$\Omega$$ to determine the total number of non-zero weights that should be kept in a neuron.

The pruning mechanism is already implemented in `SparseQuantNeuralNetImpl`, and the user only needs to determine the parameters listed above. They can be chosen in a way that is convenient, e.g. maximizing accuracy.

Increasing `n_hidden_neurons_multiplier` can lead to improved performance, as long as the compiled NN does not exceed 8 bits of accumulator bitwidth.

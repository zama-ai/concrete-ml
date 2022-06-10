# Skorch Usage

We use [skorch](https://skorch.readthedocs.io/en/stable/) to implement multi-layer, fully-connected
torch neural networks in **Concrete-ML** in a way that is compatible with the scikit-learn API.

This wrapper implements torch training boilerplate code, alleviating the work that needs to be done
by the user. It is possible to add hooks during the training phase, for example once an epoch
is finished.

Skorch allows the user to easily create a classifier or regressor around a neural network (NN), implemented
in Torch as a `nn.Module`. We provide a simple, fully-connected, multi-layer NN with a configurable
number of layers and optional pruning (see [pruning](pruning.md)). To see how to use these types of networks,
through the `NeuralNetClassifier` and `NeuralNetRegressor` classes, please see the [neural network documentation](quantized_neural_networks.md).

Under the hood, these two classes are simply skorch wrapper around a single torch module, `SparseQuantNeuralNetImpl`.
Please see the documentation on this class [in the API guide](_apidoc/concrete.ml.sklearn.html#concrete.ml.sklearn.qnn.SparseQuantNeuralNetImpl).

```
class SparseQuantNeuralNetImpl(nn.Module):
    """Sparse Quantized Neural Network classifier.
```

## Parameter choice

A linear or convolutional layer of an NN will compute a linear combination of weights and inputs (we also call this a  _'multi-sum'_). For example, a linear layer will compute:

$$\mathsf{output}^k = \sum_i^Nw_{i}^kx_i$$

where $k$ is the k-th neuron in the layer. In this case, the sum is taken on a single dimension. A convolutional layer will compute:

$$\mathsf{output}_{xy}^{k} = \sum_c^{N}\sum_j^{K_h}\sum_i^{K_w}w_{cji}^kx_{c,y+j,x+i}^k$$

where $k$ is the k-th filter of the convolutional layer and $N$, $K_h$, $K_w$ are the number of input channels, the kernel height and the kernel width, respectively.

Following the formulas for the resulting bit width of quantized linear combinations described [here](fhe_constraints.md), notably the maximum dimensionality of the input and weights that can make the result exceed 8 bits:

$$ \Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right) $$

where $ n_{\mathsf{max}} = 8 $ is the maximum precision allowed.

For example, we set $ n_{\mathsf{weights}} = 2$ and $ n_{\mathsf{inputs}} = 2$ with $ n_{\mathsf{max}} = 8$. The worst case is a scenario where all inputs and weights are equal to their maximal value $2^2-1=3$. The formula above tells us that, in this case, we can afford at most $ \Omega = 28 $ elements in the multi-sums detailed above.

In a practical setting, the distribution of the weights of a neural network is Gaussian. Thus, there will be weights that are equal to 0 and many weights will have small values. In a typical scenario, we can exceed the worst-case number of active neurons. The parameter `n_hidden_neurons_multiplier` is a factor that is multiplied with $\Omega$ to determine the total number of non-zero weights that should be kept in a neuron.

The pruning mechanism is already implemented in `SparseQuantNeuralNetImpl`, and the user only needs to determine the parameters listed above. They can choose them in a way that is convenient, e.g. maximizing accuracy.

We could then increase `n_hidden_neurons_multiplier` to improve performance, taking care to verify that the compiled NN does not exceed 8 bits of accumulator bit width.

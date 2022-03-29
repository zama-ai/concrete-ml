# Skorch Usage

We use [Skorch](https://skorch.readthedocs.io/en/stable/) to implement multi-layer fully connected
torch neural networks in **Concrete ML** in a way that is compatible with the scikit-learn API.

This wrapper implements torch training boilerplate code, alleviating the work that needs to be done
by the user. It is possible to add hooks during the training phase, for example once an epoch
is finished.

skorch allows to easily create a classifier or regressor around a Neural Network (NN) implemented
in torch as a `nn.Module`. We provide a simple fully-connected multi-layer NN, with configurable
number of layers and optional pruning (see [pruning](../../user/explanation/pruning.md)).

The `SparseQuantNeuralNetImpl` class implements this neural network. Please see the documentation on this class [in the API guide](../../_apidoc/concrete.ml.sklearn.html#concrete.ml.sklearn.qnn.SparseQuantNeuralNetImpl).

```
class SparseQuantNeuralNetImpl(nn.Module):
    """Sparse Quantized Neural Network classifier.
```

The constructor of this class takes some parameters that influence the FHE-compatibility:

- `n_w_bits` (default 3): number of bits for _weights_
- `n_a_bits` (default 3): number of bits for _activations_ and _inputs_
- `n_accum_bits` (default 7): maximum accumulator bitwidth to impose through pruning
- `n_hidden_neurons_multiplier` (default 4): explained below

A linear or convolutional layer of a NN will compute a linear combination of weights and inputs (we also call this a  _'multi-sum'_). For example a linear layer will compute:

$output^k = \sum_i^Nw_{i}^kx_i$

where $k$ is the k-th neuron in the layer. In this case the sum is taken on a single dimension. A convolutional layer will compute

$output_{xy}^{k} = \sum_c^{N}\sum_j^{K_h}\sum_i^{K_w}w_{cji}^kx_{c,y+j,x+i}^k$

where $k$ is the k-th filter of the convolutional layer, $N$, $K_h$, $K_w$ are the number of input channels, the kernel height and width respectively.

Following the formulas for the resulting bitwidth of quantized linear combinations described [here](../../user/howto/reduce_needed_precision.md), notably the maximum dimensionality of the input and weights that can make the result exceed 7 bits:

$$ \Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right) $$

where $ n_{\mathsf{max}} = 7 $ is the maximum precision allowed.

For example, we set $ n_{\mathsf{weights}} = 2$ and $ n_{\mathsf{inputs}} = 2$ with $ n_{\mathsf{max}} = 7$. The worst case is scenario when all inputs and weights are equal to their maximal value $2^2-1=3$. The formula above tells us that in this case we can afford at most $ \Omega = 14 $ elements in the multi-sums detailed above.

In a practical setting, the distribution of the weights of a neural network is gaussian. Thus, there will be weights that are equal to 0 and many weights will have small values. In a typical scenario, we can exceed the worst-case number of active neurons. The parameter `n_hidden_neurons_multiplier` is a factor that is multiplied with $\Omega$ to determine the total number of non-zero weights that should be kept in a neuron.

The pruning mechanism is already implemented in `SparseQuantNeuralNetImpl` and the user only needs to determine the parameters listed above, and they can choose them in a way that is convenient, e.g. maximizing accuracy.

The skorch wrapper requires that all the parameters that will be passed to the wrapped `nn.Module` be prefixed with `module__`. For example, the code create an FHE compatible **Concrete ML** fully connected NN classifier, for a dataset with 10 input dimensions and two classes, will thus be:

<!--pytest-codeblocks:skip-->

```python
n_inputs = 10
n_outputs = 2
params = {
    "module__n_layers": 2,
    "module__n_w_bits": 2,
    "module__n_a_bits": 2,
    "module__n_accum_bits": 7,
    "module__n_hidden_neurons_multiplier": 1,
    "module__n_outputs": n_outputs,
    "module__input_dim": n_inputs,
    "module__activation_function": nn.ReLU,
    "max_epochs": 10,
}

concrete_classifier = NeuralNetClassifier(**params)
```

We could then increase `n_hidden_neurons_multiplier` to improve performance, taking care to verify that the compiled NN does not exceed 7 bits of accumulator bitwidth.

A similar example is given in the [classifier comparison notebook](../../user/advanced_examples/ClassifierComparison.ipynb).

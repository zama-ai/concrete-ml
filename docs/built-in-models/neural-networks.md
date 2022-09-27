# Neural Networks

Concrete-ML provides simple neural networks models with a Scikit-learn interface through the `NeuralNetClassifier` and `NeuralNetRegressor` classes.

|                                            Concrete-ML                                             |
| :------------------------------------------------------------------------------------------------: |
| [NeuralNetClassifier](../developer-guide/api/concrete.ml.sklearn.qnn.md#class-neuralnetclassifier) |
|  [NeuralNetRegressor](../developer-guide/api/concrete.ml.sklearn.qnn.md#class-neuralnetregressor)  |

The neural network models are built with [Skorch](https://skorch.readthedocs.io/en/stable/index.html), which provides a scikit-learn like interface to Torch models (more [here](../developer-guide/external_libraries.md#skorch)).

These models use a stack of linear layers and the activation function and the number of neurons in each layer is configurable. This approach is similar to what is available in Scikit-learn using the `MLPClassifier`/`MLPRegressor` classes. The built-in fully connected neural network (FCNN) models train easily with a single call to `.fit()`, which will automatically quantize the weights and activations. These models use Quantization Aware Training, allowing good performance for low precision (down to 2-3 bit) weights and activations.

While `NeuralNetClassifier` and `NeuralNetClassifier` provide scikit-learn like models, their architecture is somewhat restricted in order to make training easy and robust. If you need more advanced models you can convert custom neural networks, as described in the [FHE-friendly models documentation](../deep-learning/fhe_friendly_models.md).

{% hint style="warning" %}
Good quantization parameter values are critical to make models [respect FHE constraints](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints). Weights and activations should be quantized to low
precision (e.g. 2-4 bits). Furthermore, in case of overflow, the sparsity of the network can be tuned
[as described below](#overflow-errors).
{% endhint %}

## Example usage

To create an instance of a Fully Connected Neural Network you need to instantiate one of the `NeuralNetClassifier` and `NeuralNetRegressor` classes and configure a number of parameters that are passed to their constructor. Note that some parameters need to be prefixed by `module__`, while others don't. Basically, the parameters that are related to the model, i.e. the underlying `nn.Module`, must have the prefix. The parameters that are related to training options do not require the prefix.

```python
from concrete.ml.sklearn import NeuralNetClassifier
import torch.nn as nn

n_inputs = 10
n_outputs = 2
params = {
    "module__n_layers": 2,
    "module__n_w_bits": 2,
    "module__n_a_bits": 2,
    "module__n_accum_bits": 8,
    "module__n_hidden_neurons_multiplier": 1,
    "module__n_outputs": n_outputs,
    "module__input_dim": n_inputs,
    "module__activation_function": nn.ReLU,
    "max_epochs": 10,
}

concrete_classifier = NeuralNetClassifier(**params)
```

The [Classifier Comparison notebook](ml_examples.md) shows the behavior of built-in neural networks on several synthetic datasets.

![Comparison neural networks](../figures/neural_nets_builtin.png)

The figure above shows, on the right, the Concrete-ML neural network, trained with Quantization Aware Training, in a FHE compatible configuration. The figure compares this network to the floating point equivalent trained with scikit-learn.

### Architecture parameters

- `module__n_layers`: number of layers in the FCNN, must be at least 1. Note that this is the total
  number of layers. For a single hidden layer NN model, set `module__n_layers=2`
- `module__n_outputs`: number of outputs (classes or targets)
- `module__input_dim`: dimensionality of the input
- `module__activation_function`: can be one of the Torch activations (e.g. nn.ReLU, see the full list [here](../deep-learning/torch_support.md#activations))

### Quantization parameters

- `n_w_bits` (default 3): number of bits for weights
- `n_a_bits` (default 3): number of bits for activations and inputs
- `n_accum_bits` (default 8): maximum accumulator bit-width that is desired. The implementation will attempt to keep accumulators under this bit-width through [pruning](../advanced-topics/pruning.md), i.e. setting some weights to zero

### Training parameters (from Skorch)

- `max_epochs`: The number of epochs to train the network (default 10),
- `verbose`: Whether to log loss/metrics during training (default: False)
- `lr`: Learning rate (default 0.001)
- Other parameters from skorch are in the [Skorch documentation](https://skorch.readthedocs.io/en/stable/classifier.html)

### Advanced parameters

- `module__n_hidden_neurons_multiplier`: The number of hidden neurons will be automatically set proportional to the dimensionality of the input (i.e. the vlaue for `module__input_dim`). This parameter controls the proportionality factor, and is by default set to 4. This value gives good accuracy while avoiding accumulator overflow. See the [pruning](../advanced-topics/pruning.md)
  and [quantization](../advanced-topics/quantization.md) sections for more info.

### Network input/output

When you have training data in the form of a NumPy array, and targets in a NumPy 1d array, you can set:

<!--pytest-codeblocks:skip-->

```python
    classes = np.unique(y_all)
    params["module__input_dim"] = x_train.shape[1]
    params["module__n_outputs"] = len(classes)
```

### Class weights

You can give weights to each class, to use in training. Note that this must be supported by the underlying PyTorch loss function.

<!--pytest-codeblocks:skip-->

```python
    from sklearn.utils.class_weight import compute_class_weight
    params["criterion__weight"] = compute_class_weight("balanced", classes=classes, y=y_train)
```

### Overflow errors

The `n_hidden_neurons_multiplier` parameter influences training accuracy as it controls the number of non-zero neurons that are allowed in each layer. Increasing `n_hidden_neurons_multiplier` improves accuracy, but should take into account precision limitations to avoid overflow in the accumulator. The default value is a good compromise that avoids overflow, in most cases, but you may want to change the value of this parameter to reduce the breadth of the network if you have overflow errors. A value of 1 should be completely safe with respect to overflow.

# Neural networks

**Concrete-ML** provides simple Neural Networks models with a scikit-learn interface through the
`NeuralNetClassifier` and `NeuralNetRegressor` clases.  The architecture
of the models is restricted to use only linear layers. We can call these networks Fully Connected
Neural Networks (FCNN). The number of layers, the activation function
and the number of neurons in each layer is configurable. This approach is similar to what
is available in scikit-learn using the `MLPClassifier`/`MLPRegressor` classes.

The builtin FCNN models train easily with a single call to `.fit()`, but, during this process,
also quantize the weights and activations.

The FCNN models are built upon [skorch](https://skorch.readthedocs.io/en/stable/index.html), which provides a scikit-learn like interface to torch models, as detailed
[in the development guide](skorch_usage.md).

While `NeuralNetClassifier` and `NeuralNetClassifier` provide scikit-learn like models,
their architecture is somewhat restricted in order to make training easy and robust. If you
need more advanced models you can convert custom neural networks, as described in the [custom models
documentation](custom_models.md).

## Usage

To create an instance of a Fully Connected Neural Network you need to instantiate one of the
`NeuralNetClassifier` and `NeuralNetRegressor` classes and configure a number of
parameters that are passed to their constructor. Note that some parameters need to be prefixed by
`module__`, while others don't. Basically, the parameters that are related to the model, i.e.
the underlying `nn.Module`, must have the prefix. The parameters that are related to training options
do not require the prefix.

```
from concrete.ml.sklearn import NeuralNetClassifier

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

Architecture parameters:

- `module__n_layers`: number of layers in the FCNN, must be at least 1
- `module__n_outputs`: number of outputs (classes or targets)
- `module__input_dim`: dimensionality of the input
- `module__activation_function`: can be one of the torch activations (e.g. nn.ReLU). See the
  [full list here](torch_support.md)

Quantization parameters:

- `n_w_bits` (default 3): number of bits for _weights_
- `n_a_bits` (default 3): number of bits for _activations_ and _inputs_
- `n_accum_bits` (default 8): maximum accumulator bit width that is desired. The implementation
  will attempt to keep accumulators under this bitwidth through pruning, i.e. setting some weights to
  zero

Training parameters (from skorch):

- `max_epochs`: The number of epochs to train the network (default 10),
- `verbose`: Whether to log loss/metrics during training (default: False)
- `lr`: Learning rate (default 0.001)
- Other parameters from skorch: [see skorch documentation](https://skorch.readthedocs.io/en/stable/classifier.html)

Advanced parameters:

- `module__n_hidden_neurons_multiplier`: The number of hidden neurons will be automatically set
  proportional to the dimensionality of the input (i.e. the vlaue for `module__input_dim`). This parameter
  controls the proportionality factor, and is by default set to 4. This value gives good accuracy
  while avoiding accumulator overflow.

## Tips

### Network input/output

When when you have training data in the form of a numpy array, and targets in a numpy 1d array, you
can set:

```
        classes = np.unique(y_all)
        config["module__input_dim"] = x_train.shape[1]
        config["module__n_outputs"] = len(classes)
```

### Class weights

You can give weights to each class, to use in training. Note that this must be supported
by the underlying torch loss function.

```
    from sklearn.utils.class_weight import compute_class_weight
    config["criterion__weight"] = compute_class_weight("balanced", classes=classes, y=y_train)
```

### Overflow errors

The `n_hidden_neurons_multiplier` parameter influences training accuracy as it controls the number
of non-zero neurons that are allowed in each layer. We could then increase `n_hidden_neurons_multiplier` to improve accuracy, taking care to verify that the compiled NN does not exceed 8 bits of accumulator bit width. The default value is a good compromise that avoids overflow, in most cases, but you may want to change the value of this parameter to reduce the breadth of the network if you have
overflow errors. A value of 1 should be completely safe with respect to overflow.

### Examples

A similar example is given in the [classifier comparison notebook](advanced_examples.md).

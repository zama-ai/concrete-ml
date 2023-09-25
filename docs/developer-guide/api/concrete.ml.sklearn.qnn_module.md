<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.qnn_module`

Sparse Quantized Neural Network torch module.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseQuantNeuralNetwork`

Sparse Quantized Neural Network.

This class implements an MLP that is compatible with FHE constraints. The weights and activations are quantized to low bit-width and pruning is used to ensure accumulators do not surpass an user-provided accumulator bit-width. The number of classes and number of layers are specified by the user, as well as the breadth of the network

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_dim: int,
    n_layers: int,
    n_outputs: int,
    n_hidden_neurons_multiplier: int = 4,
    n_w_bits: int = 4,
    n_a_bits: int = 4,
    n_accum_bits: int = 32,
    n_prune_neurons_percentage: float = 0.0,
    activation_function: Type = <class 'torch.nn.modules.activation.ReLU'>,
    quant_narrow: bool = False,
    quant_signed: bool = True,
    power_of_two_scaling: bool = True
)
```

Sparse Quantized Neural Network constructor.

**Args:**

- <b>`input_dim`</b> (int):  Number of dimensions of the input data.
- <b>`n_layers`</b> (int):  Number of linear layers for this network.
- <b>`n_outputs`</b> (int):  Number of output classes or regression targets.
- <b>`n_w_bits`</b> (int):  Number of weight bits.
- <b>`n_a_bits`</b> (int):  Number of activation and input bits.
- <b>`n_accum_bits`</b> (int):  Maximal allowed bit-width of intermediate accumulators.
- <b>`n_hidden_neurons_multiplier`</b> (int):  The number of neurons on the hidden will be the  number of dimensions of the input multiplied by `n_hidden_neurons_multiplier`. Note  that pruning is used to adjust the accumulator size to attempt to keep the maximum  accumulator bit-width to `n_accum_bits`, meaning that not all hidden layer neurons  will be active. The default value for `n_hidden_neurons_multiplier` is chosen for  small dimensions of the input. Reducing this value decreases the FHE inference time  considerably but also decreases the robustness and accuracy of model training.
- <b>`n_prune_neurons_percentage`</b> (float):  The percentage of neurons to prune in the hidden  layers. This can be used when setting `n_hidden_neurons_multiplier` with a high  number (3-4), once good accuracy is obtained, in order to speed up the model in FHE.
- <b>`activation_function`</b> (Type):  The activation function to use in the network  (e.g., torch.ReLU, torch.SELU, torch.Sigmoid, ...).
- <b>`quant_narrow`</b> (bool):  Whether this network should quantize the values using narrow range  (e.g a 2-bits signed quantization uses \[-1, 0, 1\] instead of \[-2, -1, 0, 1\]).
- <b>`quant_signed`</b> (bool):  Whether this network should quantize the values using signed  integers.
- <b>`power_of_two_scaling`</b> (bool):  Force quantization scales to be a power of two  to enable inference speed optimizations. Defaults to True

**Raises:**

- <b>`ValueError`</b>:  If the parameters have invalid values or the computed accumulator bit-width  is zero.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_pruning`

```python
enable_pruning() → None
```

Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

**Raises:**

- <b>`ValueError`</b>:  If the quantization parameters are invalid.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  network input

**Returns:**

- <b>`x`</b> (torch.Tensor):  network prediction

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_pruning_permanent`

```python
make_pruning_permanent() → None
```

Make the learned pruning permanent in the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max_active_neurons`

```python
max_active_neurons() → int
```

Compute the maximum number of active (non-zero weight) neurons.

The computation is done using the quantization parameters passed to the constructor. Warning: With the current quantization algorithm (asymmetric) the value returned by this function is not guaranteed to ensure FHE compatibility. For some weight distributions, weights that are 0 (which are pruned weights) will not be quantized to 0. Therefore the total number of active quantized neurons will not be equal to max_active_neurons.

**Returns:**

- <b>`int`</b>:  The maximum number of active neurons.

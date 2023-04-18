<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.qnn_module`

Sparse Quantized Neural Network torch module.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseQuantNeuralNetwork`

Sparse Quantized Neural Network.

This class implements an MLP that is compatible with FHE constraints. The weights and activations are quantized to low bit-width and pruning is used to ensure accumulators do not surpass an user-provided accumulator bit-width. The number of classes and number of layers are specified by the user, as well as the breadth of the network

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_dim,
    n_layers,
    n_outputs,
    n_hidden_neurons_multiplier=4,
    n_w_bits=3,
    n_a_bits=3,
    n_accum_bits=8,
    n_prune_neurons_percentage=0.0,
    activation_function=<class 'torch.nn.modules.activation.ReLU'>,
    quant_narrow=False,
    quant_signed=True
)
```

Sparse Quantized Neural Network constructor.

**Args:**

- <b>`input_dim`</b>:  Number of dimensions of the input data
- <b>`n_layers`</b>:  Number of linear layers for this network
- <b>`n_outputs`</b>:  Number of output classes or regression targets
- <b>`n_w_bits`</b>:  Number of weight bits
- <b>`n_a_bits`</b>:  Number of activation and input bits
- <b>`n_accum_bits`</b>:  Maximal allowed bit-width of intermediate accumulators
- <b>`n_hidden_neurons_multiplier`</b>:  The number of neurons on the hidden will be the number  of dimensions of the input multiplied by `n_hidden_neurons_multiplier`. Note that  pruning is used to adjust the accumulator size to attempt to  keep the maximum accumulator bit-width to  `n_accum_bits`, meaning that not all hidden layer neurons will be active.  The default value for `n_hidden_neurons_multiplier` is chosen for small dimensions  of the input. Reducing this value decreases the FHE inference time considerably  but also decreases the robustness and accuracy of model training.
- <b>`n_prune_neurons_percentage`</b>:  How many neurons to prune on the hidden layers. This  should be used mostly through the dedicated `.prune()` mechanism. This can  be used in when setting `n_hidden_neurons_multiplier` high (3-4), once good accuracy  is obtained, to speed up the model in FHE.
- <b>`activation_function`</b>:  a torch class that is used to construct activation functions in  the network (eg torch.ReLU, torch.SELU, torch.Sigmoid, etc)
- <b>`quant_narrow `</b>:  whether this network should use narrow range quantized integer values
- <b>`quant_signed `</b>:  whether to use signed quantized integer values

**Raises:**

- <b>`ValueError`</b>:  if the parameters have invalid values or the computed accumulator bit-width  is zero

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_pruning`

```python
enable_pruning() → None
```

Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

**Raises:**

- <b>`ValueError`</b>:  if the quantization parameters are invalid

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_pruning_permanent`

```python
make_pruning_permanent() → None
```

Make the learned pruning permanent in the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/qnn_module.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max_active_neurons`

```python
max_active_neurons() → int
```

Compute the maximum number of active (non-zero weight) neurons.

The computation is done using the quantization parameters passed to the constructor. Warning: With the current quantization algorithm (asymmetric) the value returned by this function is not guaranteed to ensure FHE compatibility. For some weight distributions, weights that are 0 (which are pruned weights) will not be quantized to 0. Therefore the total number of active quantized neurons will not be equal to max_active_neurons.

**Returns:**

- <b>`n`</b> (int):  maximum number of active neurons

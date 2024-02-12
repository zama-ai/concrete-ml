<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/pytest/torch_models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pytest.torch_models`

Torch modules for our pytests.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiOutputModel`

Multi-output model.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```

Torch Model.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input of the model.
- <b>`y`</b> (torch.Tensor):  The input of the model.

**Returns:**

- <b>`Tuple[torch.Tensor. torch.Tensor]`</b>:  Output of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimpleNet`

Fake torch model used to generate some onnx.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(inputs)
```

Forward function.

**Arguments:**

- <b>`inputs`</b>:  the inputs of the model.

**Returns:**

- <b>`torch.Tensor`</b>:  the result of the computation

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSmall`

Torch model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FC`

Torch model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output=3072)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNN`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNMaxPool`

Torch CNN model for the tests with a max pool.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNOther`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNInvalid`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNGrouped`

Torch CNN model with grouped convolution for compile torch tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithLoops`

Torch model, where we reuse some elements in a loop.

Torch model, where we reuse some elements in a loop in the forward and don't expect the user to define these elements in a particular order.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNN`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the first input of the NN
- <b>`y`</b>:  the second input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNConfigurable`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, input_output, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the first input of the NN
- <b>`y`</b>:  the second input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNDifferentSize`

Torch model to test multiple inputs with different shape in the forward pass.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_output,
    activation_function=None,
    is_brevitas_qat=False,
    n_bits=3
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b>:  The first input of the NN.
- <b>`y`</b>:  The second input of the NN.

**Returns:**
The output of the NN.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingGemmModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L458"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L476"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnivariateModule`

Torch model that calls univariate and shape functions of torch.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L485"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StepActivationModule`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L505"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass with a quantizer built into the computation graph.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConcatUnsqueeze`

Torch model to test the concat and unsqueeze operators.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiOpOnSingleInputConvNN`

Network that applies two quantized operations on a single input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L575"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(can_remove_input_tlu: bool)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L601"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeq`

Torch model that should generate MatMul->Add ONNX patterns.

This network generates additions with a constant scalar

<a href="../../../src/concrete/ml/pytest/torch_models.py#L607"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeqAddBiasVec`

Torch model that should generate MatMul->Add ONNX patterns.

This network tests the addition with a constant vector

<a href="../../../src/concrete/ml/pytest/torch_models.py#L645"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L663"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyCNN`

A very small CNN.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L680"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, act) → None
```

Create the tiny CNN with two conv layers.

**Args:**

- <b>`n_classes`</b>:  number of classes
- <b>`act`</b>:  the activation

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L695"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward the two layers with the chosen activation function.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyQATCNN`

A very small QAT CNN to classify the sklearn digits data-set.

This class also allows pruning to a maximum of 10 active neurons, which should help keep the accumulator bit-width low.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L716"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_classes,
    n_bits,
    n_active,
    signed,
    narrow,
    power_of_two_scaling
) → None
```

Construct the CNN with a configurable number of classes.

**Args:**

- <b>`n_classes`</b> (int):  number of outputs of the neural net
- <b>`n_bits`</b> (int):  number of weight and activation bits for quantization
- <b>`n_active`</b> (int):  number of active (non-zero weight) neurons to keep
- <b>`signed`</b> (bool):  whether quantized integer values are signed
- <b>`narrow`</b> (bool):  whether the range of quantized integer values is narrow/symmetric
- <b>`power_of_two_scaling`</b> (bool):  whether to use power-of-two scaling quantizers which  allows to test the round PBS optimization when the scales are power-of-two

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L827"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Run inference on the tiny CNN, apply the decision layer on the reshaped conv output.

**Args:**

- <b>`x`</b>:  the input to the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L795"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_pruning`

```python
toggle_pruning(enable)
```

Enable or remove pruning.

**Args:**

- <b>`enable`</b>:  if we enable the pruning or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L853"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimpleQAT`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L856"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, n_bits=2, disable_bit_check=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L892"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass with a quantizer built into the computation graph.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L936"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QATTestModule`

Torch model that implements a simple non-uniform quantizer.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L939"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L944"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass with a quantizer built into the computation graph.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L974"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SingleMixNet`

Torch model that with a single conv layer that produces the output, e.g., a blur filter.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L979"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, inp_size, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1011"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Execute the single convolution.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1024"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DoubleQuantQATMixNet`

Torch model that with two different quantizers on the input.

Used to test that it keeps the input TLU.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1030"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, inp_size, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1048"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Execute the single convolution.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1063"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchSum`

Torch model to test the ReduceSum ONNX operator in a leveled circuit.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1066"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim=(0,), keepdim=True, with_pbs=False)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not
- <b>`with_pbs`</b> (bool):  If the forward function should be forced to consider at least one PBS

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1084"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model

**Returns:**

- <b>`torch_sum`</b> (torch.tensor):  The sum of the input's tensor elements along the given axis

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConstantsFoldedBeforeOps`

Torch QAT model that does not quantize the inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    hparams: dict,
    bits: int,
    act_quant=<class 'brevitas.quant.scaled_int.Int8ActPerTensorFloat'>,
    weight_quant=<class 'brevitas.quant.scaled_int.Int8WeightPerTensorFloat'>
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ShapeOperationsNet`

Torch QAT model that reshapes the input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_qat)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PaddingNet`

Torch QAT model that applies various padding patterns.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantCustomModel`

A small quantized network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_shape: int,
    output_shape: int,
    hidden_shape: int = 100,
    n_bits: int = 5,
    act_quant=<class 'brevitas.quant.scaled_int.Int8ActPerTensorFloat'>,
    weight_quant=<class 'brevitas.quant.scaled_int.Int8WeightPerTensorFloat'>,
    bias_quant=None
)
```

Quantized Torch Model with Brevitas.

**Args:**

- <b>`input_shape`</b> (int):  Input size
- <b>`output_shape`</b> (int):  Output size
- <b>`hidden_shape`</b> (int):  Hidden size
- <b>`n_bits`</b> (int):  Bit of quantization
- <b>`weight_quant`</b> (brevitas.quant):  Quantization protocol of weights
- <b>`act_quant`</b> (brevitas.quant):  Quantization protocol of activations.
- <b>`bias_quant`</b> (brevitas.quant):  Quantizer for the linear layer bias

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model.

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchCustomModel`

A small network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_shape, hidden_shape, output_shape)
```

Torch Model.

**Args:**

- <b>`input_shape`</b> (int):  Input size
- <b>`output_shape`</b> (int):  Output size
- <b>`hidden_shape`</b> (int):  Hidden size

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model.

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcatFancyIndexing`

Concat with fancy indexing.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_shape,
    hidden_shape,
    output_shape,
    n_bits: int = 4,
    n_blocks: int = 3
) → None
```

Torch Model.

**Args:**

- <b>`input_shape`</b> (int):   Input size
- <b>`output_shape`</b> (int):  Output size
- <b>`hidden_shape`</b> (int):  Hidden size
- <b>`n_bits`</b> (int):        Number of bits
- <b>`n_blocks`</b> (int):      Number of blocks

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1381"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model.

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PartialQATModel`

A model with a QAT Module.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_shape: int, output_shape: int, n_bits: int)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model.

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EncryptedMatrixMultiplicationModel`

PyTorch module for performing matrix multiplication between two encrypted values.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1433"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(input1)
```

Forward function for matrix multiplication.

**Args:**

- <b>`input1`</b> (torch.Tensor):  The first input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The result of the matrix multiplication.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ManualLogisticRegressionTraining`

PyTorch module for performing SGD training.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(learning_rate=0.1)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1463"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y, weights, bias)
```

Forward function for matrix multiplication.

**Args:**

- <b>`x`</b> (torch.Tensor):  The training data tensor.
- <b>`y`</b> (torch.Tensor):  The target tensor.
- <b>`weights`</b> (torch.Tensor):  The weights to be learned.
- <b>`bias`</b> (torch.Tensor):  The bias to be learned.

**Returns:**

- <b>`torch.Tensor`</b>:  The updated weights after performing a training step.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1485"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(x, weights, bias)
```

Predicts based on weights and bias as inputs.

**Args:**

- <b>`x`</b> (torch.Tensor):  Input data tensor.
- <b>`weights`</b> (torch.Tensor):  Weights tensor.
- <b>`bias`</b> (torch.Tensor):  Bias tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The predicted outputs for the given inputs.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1508"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AddNet`

Torch model that performs a simple addition between two inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1511"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, input_output, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b>:  First input tensor.
- <b>`y`</b>:  Second input tensor.

**Returns:**
Result of adding x and y.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExpandModel`

Minimalist network that expands the input tensor to a larger size.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_qat)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Expand the input tensor to the target size.

**Args:**

- <b>`x`</b> (torch.Tensor):  Input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  Expanded tensor.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Conv1dModel`

Small model that uses a 1D convolution operator.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function) → None
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1564"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The model's input.

**Returns:**

- <b>`torch.Tensor`</b>:  The model's output.

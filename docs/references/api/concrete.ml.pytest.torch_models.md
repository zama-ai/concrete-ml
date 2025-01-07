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

- <b>`Tuple[torch.Tensor. torch.Tensor]`</b>:  Outputs of the network.

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
__init__(input_output, activation_function, hidden=None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FC`

Torch model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output=3072)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNN`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNMaxPool`

Torch CNN model for the tests with a max pool.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNOther`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNInvalid`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNGrouped`

Torch CNN model with grouped convolution for compile torch tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithLoops`

Torch model, where we reuse some elements in a loop.

Torch model, where we reuse some elements in a loop in the forward and don't expect the user to define these elements in a particular order.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNN`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L337"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNConfigurable`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, input_output, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L381"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNDifferentSize`

Torch model to test multiple inputs with different shape in the forward pass.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingGemmModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L466"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L478"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnivariateModule`

Torch model that calls univariate and shape functions of torch.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L503"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StepActivationModule`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L507"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L543"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConcatUnsqueeze`

Torch model to test the concat and unsqueeze operators.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L574"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiOpOnSingleInputConvNN`

Network that applies two quantized operations on a single input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L577"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(can_remove_input_tlu: bool)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L583"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L603"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeq`

Torch model that should generate MatMul->Add ONNX patterns.

This network generates additions with a constant scalar

<a href="../../../src/concrete/ml/pytest/torch_models.py#L609"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L641"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeqAddBiasVec`

Torch model that should generate MatMul->Add ONNX patterns.

This network tests the addition with a constant vector

<a href="../../../src/concrete/ml/pytest/torch_models.py#L647"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L679"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyCNN`

A very small CNN.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, act) → None
```

Create the tiny CNN with two conv layers.

**Args:**

- <b>`n_classes`</b>:  number of classes
- <b>`act`</b>:  the activation

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L697"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L711"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyQATCNN`

A very small QAT CNN to classify the sklearn digits data-set.

This class also allows pruning to a maximum of 10 active neurons, which should help keep the accumulator bit-width low.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L829"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_pruning`

```python
toggle_pruning(enable)
```

Enable or remove pruning.

**Args:**

- <b>`enable`</b>:  if we enable the pruning or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L855"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StepFunctionPTQ`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L858"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
__init__(dim=(0,), keepdim=True)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1082"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1096"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConstantsFoldedBeforeOps`

Torch QAT model that does not quantize the inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1099"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ShapeOperationsNet`

Torch QAT model that reshapes the input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_qat)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PaddingNet`

Torch QAT model that applies various padding patterns.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantCustomModel`

A small quantized network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchCustomModel`

A small network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1341"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcatFancyIndexing`

Concat with fancy indexing.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1344"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PartialQATModel`

A model with a QAT Module.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_shape: int, output_shape: int, n_bits: int)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EncryptedMatrixMultiplicationModel`

PyTorch module for performing matrix multiplication between two encrypted values.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1447"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ManualLogisticRegressionTraining`

PyTorch module for performing SGD training.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(learning_rate=0.1)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1454"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1476"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WhereNet`

Simple network with a where operation for testing.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1502"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_hidden)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1507"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The output tensor after applying the where operation.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AddNet`

Torch model that performs a simple addition between two inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1523"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, input_output, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1527"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExpandModel`

Minimalist network that expands the input tensor to a larger size.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_qat)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1550"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1565"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Conv1dModel`

Small model that uses a 1D convolution operator.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function) → None
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The model's input.

**Returns:**

- <b>`torch.Tensor`</b>:  The model's output.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1592"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IdentityExpandModel`

Model that only adds an empty dimension at axis 0.

This model is mostly useful for testing the composition feature.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1598"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input of the model.

**Returns:**

- <b>`Tuple[torch.Tensor. torch.Tensor]`</b>:  Outputs of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IdentityExpandMultiOutputModel`

Model that only adds an empty dimension at axis 0, and returns the initial input as well.

This model is mostly useful for testing the composition feature.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1616"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input of the model.

**Returns:**

- <b>`Tuple[torch.Tensor. torch.Tensor]`</b>:  Outputs of the network.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchDivide`

Torch model that performs a encrypted division between two inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1631"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1634"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The first input tensor.
- <b>`y`</b> (torch.Tensor):  The second input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The result of the division.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1648"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchMultiply`

Torch model that performs a encrypted multiplication between two inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1651"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1654"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, y)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The first input tensor.
- <b>`y`</b> (torch.Tensor):  The second input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The result of the multiplication.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EmbeddingModel`

A torch model with an embedding layer.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    num_embeddings,
    embedding_dim,
    activation_function=<class 'torch.nn.modules.activation.ReLU'>
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor containing indices.

**Returns:**

- <b>`torch.Tensor`</b>:  The output tensor after embedding.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1693"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllZeroCNN`

A CNN class that has all zero weights and biases.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1696"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b>:  the input of the NN

**Returns:**
the output of the NN

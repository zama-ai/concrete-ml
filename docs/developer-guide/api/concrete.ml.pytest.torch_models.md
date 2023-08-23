<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/pytest/torch_models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pytest.torch_models`

Torch modules for our pytests.

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimpleNet`

Fake torch model used to generate some onnx.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSmall`

Torch model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FC`

Torch model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output=3072)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNN`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNMaxPool`

Torch CNN model for the tests with a max pool.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNOther`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNInvalid`

Torch CNN model for the tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNGrouped`

Torch CNN model with grouped convolution for compile torch tests.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, groups)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithLoops`

Torch model, where we reuse some elements in a loop.

Torch model, where we reuse some elements in a loop in the forward and don't expect the user to define these elements in a particular order.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNN`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNConfigurable`

Torch model to test multiple inputs forward.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, input_output, n_bits)
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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNNDifferentSize`

Torch model to test multiple inputs with different shape in the forward pass.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L411"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingGemmModule`

Torch model with some branching and skip connections.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnivariateModule`

Torch model that calls univariate and shape functions of torch.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L458"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L474"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StepActivationModule`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L478"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L514"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConcatUnsqueeze`

Torch model to test the concat and unsqueeze operators.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L526"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiOpOnSingleInputConvNN`

Network that applies two quantized operations on a single input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(can_remove_input_tlu: bool)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

## <kbd>class</kbd> `FCSeq`

Torch model that should generate MatMul->Add ONNX patterns.

This network generates additions with a constant scalar

<a href="../../../src/concrete/ml/pytest/torch_models.py#L580"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L598"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L612"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeqAddBiasVec`

Torch model that should generate MatMul->Add ONNX patterns.

This network tests the addition with a constant vector

<a href="../../../src/concrete/ml/pytest/torch_models.py#L618"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyCNN`

A very small CNN.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L653"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, act) → None
```

Create the tiny CNN with two conv layers.

**Args:**

- <b>`n_classes`</b>:  number of classes
- <b>`act`</b>:  the activation

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyQATCNN`

A very small QAT CNN to classify the sklearn digits data-set.

This class also allows pruning to a maximum of 10 active neurons, which should help keep the accumulator bit-width low.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L689"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, n_bits, n_active, signed, narrow) → None
```

Construct the CNN with a configurable number of classes.

**Args:**

- <b>`n_classes`</b> (int):  number of outputs of the neural net
- <b>`n_bits`</b> (int):  number of weight and activation bits for quantization
- <b>`n_active`</b> (int):  number of active (non-zero weight) neurons to keep
- <b>`signed`</b> (bool):  whether quantized integer values are signed
- <b>`narrow`</b> (bool):  whether the range of quantized integer values is narrow/symmetric

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L759"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L784"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `test_torch`

```python
test_torch(test_loader)
```

Test the network: measure accuracy on the test set.

**Args:**

- <b>`test_loader`</b>:  the test loader

**Returns:**

- <b>`res`</b>:  the number of correctly classified test examples

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_pruning`

```python
toggle_pruning(enable)
```

Enable or remove pruning.

**Args:**

- <b>`enable`</b>:  if we enable the pruning or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L826"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimpleQAT`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L829"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, n_bits=2, disable_bit_check=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L865"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L909"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QATTestModule`

Torch model that implements a simple non-uniform quantizer.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L912"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L917"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L947"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SingleMixNet`

Torch model that with a single conv layer that produces the output, e.g., a blur filter.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L952"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, inp_size, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L984"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DoubleQuantQATMixNet`

Torch model that with two different quantizers on the input.

Used to test that it keeps the input TLU.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1003"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, inp_size, n_bits)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1021"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1036"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchSum`

Torch model to test the ReduceSum ONNX operator in a leveled circuit.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1039"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim=(0,), keepdim=True)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1055"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1068"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchSumMod`

Torch model to test the ReduceSum ONNX operator in a circuit containing a PBS.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1039"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim=(0,), keepdim=True)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1071"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1088"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConstantsFoldedBeforeOps`

Torch QAT model that does not quantize the inputs.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1091"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ShapeOperationsNet`

Torch QAT model that reshapes the input.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_qat)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PaddingNet`

Torch QAT model that applies various padding patterns.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantCustomModel`

A small quantized network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_shape: int,
    output_shape: int,
    hidden_shape: int = 100,
    n_bits: int = 5,
    act_quant=<class 'brevitas.quant.scaled_int.Int8ActPerTensorFloat'>,
    weight_quant=<class 'brevitas.quant.scaled_int.Int8WeightPerTensorFloat'>
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

______________________________________________________________________

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchCustomModel`

A small network with Brevitas, trained on make_classification.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcatFancyIndexing`

Concat with fancy indexing.

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/pytest/torch_models.py#L1360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model.

**Returns:**

- <b>`torch.tensor`</b>:  Output of the network.

<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pytest.torch_models`

Torch modules for our pytests.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSmall`

Torch model for the tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FC`

Torch model for the tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output=3072)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNN`

Torch CNN model for the tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNMaxPool`

Torch CNN model for the tests with a max pool.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNOther`

Torch CNN model for the tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNInvalid`

Torch CNN model for the tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, padding, groups, gather_slice)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNNGrouped`

Torch CNN model with grouped convolution for compile torch tests.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, groups)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithLoops`

Torch model, where we reuse some elements in a loop.

Torch model, where we reuse some elements in a loop in the forward and don't expect the user to define these elements in a particular order.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiInputNN`

Torch model to test multiple inputs forward.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingModule`

Torch model with some branching and skip connections.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BranchingGemmModule`

Torch model with some branching and skip connections.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnivariateModule`

Torch model that calls univariate and shape functions of torch.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StepActivationModule`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWithConcatUnsqueeze`

Torch model to test the concat and unsqueeze operators.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function, input_output, n_fc_layers)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L427"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiOpOnSingleInputConvNN`

Network that applies two quantized operations on a single input.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeq`

Torch model that should generate MatMul->Add ONNX patterns.

This network generates additions with a constant scalar

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FCSeqAddBiasVec`

Torch model that should generate MatMul->Add ONNX patterns.

This network tests the addition with a constant vector

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L493"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, act)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L511"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L525"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyCNN`

A very small CNN.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, act) → None
```

Create the tiny CNN with two conv layers.

**Args:**

- <b>`n_classes`</b>:  number of classes
- <b>`act`</b>:  the activation

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L543"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TinyQATCNN`

A very small QAT CNN to classify the sklearn digits dataset.

This class also allows pruning to a maximum of 10 active neurons, which should help keep the accumulator bit width low.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L564"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L634"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L659"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_pruning`

```python
toggle_pruning(enable)
```

Enable or remove pruning.

**Args:**

- <b>`enable`</b>:  if we enable the pruning or not

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L701"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimpleQAT`

Torch model implements a step function that needs Greater, Cast and Where.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L704"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(input_output, activation_function, n_bits=2, disable_bit_check=False)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L740"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L784"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QATTestModule`

Torch model that implements a simple non-uniform quantizer.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L787"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(activation_function)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L792"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L822"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SingleMixNet`

Torch model that with a single conv layer that produces the output, e.g. a blur filter.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(use_conv, use_qat, inp_size, n_bits)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L867"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchSum`

Torch model to test the ReduceSum ONNX operator in a leveled circuit.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L870"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim=(0,), keepdim=True)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L886"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L899"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchSumMod`

Torch model to test the ReduceSum ONNX operator in a circuit containing a PBS.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L870"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim=(0,), keepdim=True)
```

Initialize the module.

**Args:**

- <b>`dim`</b> (Tuple\[int\]):  The axis along which the sum should be executed
- <b>`keepdim`</b> (bool):  If the output should keep the same dimension as the input or not

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/pytest/torch_models.py#L902"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.tensor):  The input of the model

**Returns:**

- <b>`torch_sum`</b> (torch.tensor):  The sum of the input's tensor elements along the given axis

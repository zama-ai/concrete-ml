<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.linear_op_glwe_backend`

GLWE backend for some supported layers.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `has_glwe_backend`

```python
has_glwe_backend()
```

Check if the GLWE backend is installed.

**Returns:**

- <b>`bool`</b>:  True if the GLWE backend is installed, False otherwise.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GLWELinearLayerExecutor`

GLWE execution helper for pure linear layers.

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(private_key=None, compression_key=None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor, q_module: QuantizedModule, fhe: HybridFHEMode) → Tensor
```

Perform the inference of this linear layer.

**Args:**

- <b>`x`</b> (numpy.ndarray):  inputs or previous layer activations
- <b>`q_module`</b> (QuantizedModule):  quantized module that contains the layer which  is executed through this helper. Stores quantized weights and input  and output quantizers
- <b>`fhe`</b> (HybridFHEMode):  execution mode, can be 'disable' or 'execute'

**Returns:**

- <b>`numpy.ndarray`</b>:  result of applying the linear layer

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/linear_op_glwe_backend.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen()
```

Generate private and compression key.

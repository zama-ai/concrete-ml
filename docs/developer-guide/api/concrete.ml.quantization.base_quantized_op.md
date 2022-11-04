<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.base_quantized_op`

Base Quantized Op class that implements quantization for a float numpy op.

## **Global Variables**

- **ONNX_OPS_TO_NUMPY_IMPL**
- **ALL_QUANTIZED_OPS**
- **ONNX_OPS_TO_QUANTIZED_IMPL**
- **DEFAULT_MODEL_BITS**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOp`

Base class for quantized ONNX ops implemented in numpy.

**Args:**

- <b>`n_bits_output`</b> (int):  The number of bits to use for the quantization of the output
- <b>`int_input_names`</b> (Set\[str\]):  The set of names of integer tensors that are inputs to this op
- <b>`constant_inputs`</b> (Optional\[Union\[Dict\[str, Any\], Dict\[int, Any\]\]\]):  The constant tensors  that are inputs to this op
- <b>`input_quant_opts`</b> (QuantizationOptions):  Input quantizer options, determine the quantization  that is applied to input tensors (that are not constants)

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Optional[Set[str]] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> op_type

Get the type of this operation.

**Returns:**

- <b>`op_type`</b> (str):  The type of this operation, in the ONNX referential

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(*inputs: ndarray) → ndarray
```

Create corresponding QuantizedArray for the output of the activation function.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  Calibration sample inputs.

**Returns:**

- <b>`numpy.ndarray`</b>:  the output values for the provided calibration samples.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call_impl`

```python
call_impl(*inputs: Optional[ndarray, QuantizedArray], **attrs) → ndarray
```

Call self.impl to centralize mypy bug workaround.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  real valued inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`numpy.ndarray`</b>:  return value of self.impl

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if the operator impedes graph fusion.

This function shall be overloaded by inheriting classes to test self.\_int_input_names, to determine whether the operation can be fused to a TLU or not. For example an operation that takes inputs produced by a unique integer tensor can be fused to a TLU. Example: f(x) = x * (x + 1) can be fused. A function that does f(x) = x * (x @ w + 1) can't be fused.

**Returns:**

- <b>`bool`</b>:  whether this instance of the QuantizedOp produces Concrete Numpy code  that can be fused to TLUs

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `must_quantize_input`

```python
must_quantize_input(input_name_or_idx: int) → bool
```

Determine if an input must be quantized.

Quantized ops and numpy onnx ops take inputs and attributes. Inputs can be either constant or variable (encrypted). Note that this does not handle attributes, which are handled by QuantizedOp classes separately in their constructor.

**Args:**

- <b>`input_name_or_idx`</b> (int):  Index of the input to check.

**Returns:**

- <b>`result`</b> (bool):  Whether the input must be quantized (must be a `QuantizedArray`) or  if it stays as a raw `numpy.array` read from ONNX.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_output`

```python
prepare_output(qoutput_activation: ndarray) → QuantizedArray
```

Quantize the output of the activation function.

The calibrate method needs to be called with sample data before using this function.

**Args:**

- <b>`qoutput_activation`</b> (numpy.ndarray):  Output of the activation function.

**Returns:**

- <b>`QuantizedArray`</b>:  Quantized output.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Execute the quantized forward.

**Args:**

- <b>`*q_inputs (QuantizedArray)`</b>:  Quantized inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`QuantizedArray`</b>:  The returned quantized value.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOpUnivariateOfEncrypted`

An univariate operator of an encrypted value.

This operation is not really operating as a quantized operation. It is useful when the computations get fused into a TLU, as in e.g. Act(x) = x || (x + 42)).

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Optional[Set[str]] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> op_type

Get the type of this operation.

**Returns:**

- <b>`op_type`</b> (str):  The type of this operation, in the ONNX referential

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(*inputs: ndarray) → ndarray
```

Create corresponding QuantizedArray for the output of the activation function.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  Calibration sample inputs.

**Returns:**

- <b>`numpy.ndarray`</b>:  the output values for the provided calibration samples.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call_impl`

```python
call_impl(*inputs: Optional[ndarray, QuantizedArray], **attrs) → ndarray
```

Call self.impl to centralize mypy bug workaround.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  real valued inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`numpy.ndarray`</b>:  return value of self.impl

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x || (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `must_quantize_input`

```python
must_quantize_input(input_name_or_idx: int) → bool
```

Determine if an input must be quantized.

Quantized ops and numpy onnx ops take inputs and attributes. Inputs can be either constant or variable (encrypted). Note that this does not handle attributes, which are handled by QuantizedOp classes separately in their constructor.

**Args:**

- <b>`input_name_or_idx`</b> (int):  Index of the input to check.

**Returns:**

- <b>`result`</b> (bool):  Whether the input must be quantized (must be a `QuantizedArray`) or  if it stays as a raw `numpy.array` read from ONNX.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_output`

```python
prepare_output(qoutput_activation: ndarray) → QuantizedArray
```

Quantize the output of the activation function.

The calibrate method needs to be called with sample data before using this function.

**Args:**

- <b>`qoutput_activation`</b> (numpy.ndarray):  Output of the activation function.

**Returns:**

- <b>`QuantizedArray`</b>:  Quantized output.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/quantization/base_quantized_op.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Execute the quantized forward.

**Args:**

- <b>`*q_inputs (QuantizedArray)`</b>:  Quantized inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`QuantizedArray`</b>:  The returned quantized value.

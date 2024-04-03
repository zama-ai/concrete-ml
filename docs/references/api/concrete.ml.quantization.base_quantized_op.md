<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.base_quantized_op`

Base Quantized Op class that implements quantization for a float numpy op.

## **Global Variables**

- **ONNX_OPS_TO_NUMPY_IMPL**
- **ALL_QUANTIZED_OPS**
- **ONNX_OPS_TO_QUANTIZED_IMPL**
- **DEFAULT_MODEL_BITS**

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOp`

Base class for quantized ONNX ops implemented in numpy.

**Args:**

- <b>`n_bits_output`</b> (int):  The number of bits to use for the quantization of the output
- <b>`op_instance_name`</b> (str):  The name that should be assigned to this operation, used  to retrieve it later or get debugging information about this op (bit-width, value range,  integer intermediary values, op-specific error messages). Usually this name is the same  as the ONNX operation name for which this operation is constructed.
- <b>`int_input_names`</b> (Set\[str\]):  The set of names of integer tensors that are inputs to this op
- <b>`constant_inputs`</b> (Optional\[Union\[Dict\[str, Any\], Dict\[int, Any\]\]\]):  The constant tensors  that are inputs to this op
- <b>`input_quant_opts`</b> (QuantizationOptions):  Input quantizer options, determine the quantization  that is applied to input tensors (that are not constants)

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Optional[Set[str]] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L788"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if the operator impedes graph fusion.

This function shall be overloaded by inheriting classes to test self.\_int_input_names, to determine whether the operation can be fused to a TLU or not. For example an operation that takes inputs produced by a unique integer tensor can be fused to a TLU. Example: f(x) = x * (x + 1) can be fused. A function that does f(x) = x * (x @ w + 1) can't be fused.

**Returns:**

- <b>`bool`</b>:  whether this QuantizedOp instance produces Concrete code that can be fused to TLUs

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: <class 'TextIO'>) → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizedOp`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `op_type`

```python
op_type()
```

Get the type of this operation.

**Returns:**

- <b>`op_type`</b> (str):  The type of this operation, in the ONNX referential

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Execute the quantized forward.

**Args:**

- <b>`*q_inputs (ONNXOpInputOutputType)`</b>:  Quantized inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`ONNXOpInputOutputType`</b>:  The returned quantized value.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L818"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOpUnivariateOfEncrypted`

An univariate operator of an encrypted value.

This operation is not really operating as a quantized operation. It is useful when the computations get fused into a TLU, as in e.g., Act(x) = x || (x + 42)).

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Optional[Set[str]] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x || (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: <class 'TextIO'>) → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizedOp`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `op_type`

```python
op_type()
```

Get the type of this operation.

**Returns:**

- <b>`op_type`</b> (str):  The type of this operation, in the ONNX referential

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Execute the quantized forward.

**Args:**

- <b>`*q_inputs (ONNXOpInputOutputType)`</b>:  Quantized inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`ONNXOpInputOutputType`</b>:  The returned quantized value.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L871"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMixingOp`

An operator that mixes (adds or multiplies) together encrypted inputs.

Mixing operators cannot be fused to TLUs.

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    *args,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None,
    **kwargs
) → None
```

Initialize quantized ops parameters plus specific parameters.

**Args:**

- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  if not None,  every accumulators in the model are rounded down to the given bits of precision.  Can be an int or a dictionary with keys 'method' and 'n_bits', where 'method' is  either fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE, and 'n_bits' is either  'auto' or an int.
- <b>`*args`</b>:  positional argument to pass to the parent class.
- <b>`**kwargs`</b>:  named argument to pass to the parent class.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L900"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Mixing operations cannot be fused since it must be performed over integer tensors and it combines different encrypted elements of the input tensors. Mixing operations are Conv, MatMul, etc.

**Returns:**

- <b>`bool`</b>:  False, this operation cannot be fused as it adds different encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L953"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cnp_round`

```python
cnp_round(
    x: Union[ndarray, Tracer],
    calibrate_rounding: bool,
    rounding_operation_id: Optional[str] = 'single_rounding_op'
) → ndarray
```

Round the input array to the specified number of bits.

**Args:**

- <b>`x`</b> (Union\[numpy.ndarray, fhe.tracing.Tracer\]):  The input array to be rounded.
- <b>`calibrate_rounding`</b> (bool):  Whether to calibrate the rounding  (compute the lsbs_to_remove).
- <b>`rounding_operation_id`</b> (Optional\[str\]):  The identifier for a specific rounding  operation in a quantized operation. Used to create and access the  lsbs_to_remove value in the dictionary. Defaults to "single_rounding_op"  if not provided.

**Returns:**

- <b>`numpy.ndarray`</b>:  The rounded array.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: <class 'TextIO'>) → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizedOp`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L913"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_output_quant_parameters`

```python
make_output_quant_parameters(
    q_values: Union[ndarray, Any],
    scale: float64,
    zero_point: Union[int, float, ndarray]
) → QuantizedArray
```

Build a quantized array from quantized integer results of the op and quantization params.

**Args:**

- <b>`q_values`</b> (Union\[numpy.ndarray, Any\]):  the quantized integer values to wrap  in the QuantizedArray
- <b>`scale`</b> (float):  the pre-computed scale of the quantized values
- <b>`zero_point`</b> (Union\[int, float, numpy.ndarray\]):  the pre-computed zero_point of  the q_values

**Returns:**

- <b>`QuantizedArray`</b>:  the quantized array that will be passed to the QuantizedModule output.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `op_type`

```python
op_type()
```

Get the type of this operation.

**Returns:**

- <b>`op_type`</b> (str):  The type of this operation, in the ONNX referential

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/base_quantized_op.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Execute the quantized forward.

**Args:**

- <b>`*q_inputs (ONNXOpInputOutputType)`</b>:  Quantized inputs.
- <b>`**attrs`</b>:  the QuantizedOp attributes.

**Returns:**

- <b>`ONNXOpInputOutputType`</b>:  The returned quantized value.

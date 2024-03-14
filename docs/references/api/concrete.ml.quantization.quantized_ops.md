<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_ops`

Quantized versions of the ONNX operators for post training quantization.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSigmoid`

Quantized sigmoid op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSigmoid`

Quantized HardSigmoid op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRelu`

Quantized Relu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPRelu`

Quantized PRelu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLeakyRelu`

Quantized LeakyRelu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSwish`

Quantized Hardswish op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedElu`

Quantized Elu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSelu`

Quantized Selu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCelu`

Quantized Celu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedClip`

Quantized clip op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRound`

Quantized round op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPow`

Quantized pow op.

Only works for a float constant power. This operation will be fused to a (potentially larger) TLU.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGemm`

Quantized Gemm op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L495"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMatMul`

Quantized MatMul op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAdd`

Quantized Addition operator.

Can add either two variables (both encrypted) or a variable and a constant

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L613"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTanh`

Quantized Tanh op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L633"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSoftplus`

Quantized Softplus op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedExp`

Quantized Exp op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L645"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLog`

Quantized Log op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L651"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAbs`

Quantized Abs op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L657"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedIdentity`

Quantized Identity op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L662"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReshape`

Quantized Reshape op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L717"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Max Pooling operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Reshape the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L729"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedConv`

Quantized Conv op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L734"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

Construct the quantized convolution operator and retrieve parameters.

**Args:**

- <b>`n_bits_output`</b>:  number of bits for the quantization of the outputs of this operator
- <b>`op_instance_name`</b> (str):  The name that should be assigned to this operation, used  to retrieve it later or get debugging information about this op (bit-width, value  range, integer intermediary values, op-specific error messages). Usually this name  is the same as the ONNX operation name for which this operation is constructed.
- <b>`int_input_names`</b>:  names of integer tensors that are taken as input for this operation
- <b>`constant_inputs`</b>:  the weights and activations
- <b>`input_quant_opts`</b>:  options for the input quantizer
- <b>`attrs`</b>:  convolution options
- <b>`dilations`</b> (Tuple\[int\]):  dilation of the kernel. Default to 1 on all dimensions.
- <b>`group`</b> (int):  number of convolution groups. Default to 1.
- <b>`kernel_shape`</b> (Tuple\[int\]):  shape of the kernel. Should have 2 elements for 2d conv
- <b>`pads`</b> (Tuple\[int\]):  padding in ONNX format (begin, end) on each axis
- <b>`strides`</b> (Tuple\[int\]):  stride of the convolution on each axis

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L801"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Compute the quantized convolution between two quantized tensors.

Allows an optional quantized bias.

**Args:**

- <b>`q_inputs`</b>:  input tuple, contains
- <b>`x`</b> (numpy.ndarray):  input data. Shape is N x C x H x W for 2d
- <b>`w`</b> (numpy.ndarray):  weights tensor. Shape is (O x I x Kh x Kw) for 2d
- <b>`b`</b> (numpy.ndarray, Optional):  bias tensor, Shape is (O,)
- <b>`calibrate_rounding`</b> (bool):  Whether to calibrate rounding
- <b>`attrs`</b>:  convolution options handled in constructor

**Returns:**

- <b>`res`</b> (QuantizedArray):  result of the quantized integer convolution

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1020"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAvgPool`

Quantized Average Pooling op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1026"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1091"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMaxPool`

Quantized Max Pooling op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Max Pooling operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPad`

Quantized Padding op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Pad operation cannot be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation cannot be fused as it is manipulates integer tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1394"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedWhere`

Where operator on quantized arrays.

Supports only constants for the results produced on the True/False branches.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCast`

Cast the input to the required data type.

In FHE we only support a limited number of output types. Booleans are cast to integers.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreater`

Comparison operator >.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1475"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreaterOrEqual`

Comparison operator >=.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1484"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1507"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLess`

Comparison operator \<.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1516"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1539"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLessOrEqual`

Comparison operator \<=.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1571"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOr`

Or operator ||.

This operation is not really working as a quantized operation. It just works when things got fused, as in e.g., Act(x) = x || (x + 42))

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedDiv`

Div operator /.

This operation is not really working as a quantized operation. It just works when things got fused, as in e.g., Act(x) = 1000 / (x + 42))

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1591"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMul`

Multiplication operator.

Only multiplies an encrypted tensor with a float constant for now. This operation will be fused to a (potentially larger) TLU.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1601"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSub`

Subtraction operator.

This works the same as addition on both encrypted - encrypted and on encrypted - constant.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L613"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBatchNormalization`

Quantized Batch normalization with encrypted input and in-the-clear normalization params.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedFlatten`

Quantized flatten for encrypted inputs.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1664"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Flatten operation cannot be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation cannot be fused as it is manipulates integer tensors.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1623"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Flatten the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0
- <b>`attrs`</b>:  contains axis attribute

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReduceSum`

ReduceSum with encrypted input.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1681"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

Construct the quantized ReduceSum operator and retrieve parameters.

**Args:**

- <b>`n_bits_output`</b> (int):  Number of bits for the operator's quantization of outputs.
- <b>`op_instance_name`</b> (str):  The name that should be assigned to this operation, used  to retrieve it later or get debugging information about this op (bit-width, value  range, integer intermediary values, op-specific error messages). Usually this name  is the same as the ONNX operation name for which this operation is constructed.
- <b>`int_input_names`</b> (Optional\[Set\[str\]\]):  Names of input integer tensors. Default to None.
- <b>`constant_inputs`</b> (Optional\[Dict\]):  Input constant tensor.
- <b>`axes`</b> (Optional\[numpy.ndarray\]):  Array of integers along which to reduce.  The default is to reduce over all the dimensions of the input tensor if  'noop_with_empty_axes' is false, else act as an Identity op when  'noop_with_empty_axes' is true. Accepted range is \[-r, r-1\] where  r = rank(data). Default to None.
- <b>`input_quant_opts`</b> (Optional\[QuantizationOptions\]):  Options for the input quantizer.  Default to None.
- <b>`attrs`</b> (dict):  RecuseSum options.
- <b>`keepdims`</b> (int):  Keep the reduced dimension or not, 1 means keeping the  input dimension, 0 will reduce it along the given axis. Default to 1.
- <b>`noop_with_empty_axes`</b> (int):  Defines behavior if 'axes' is empty or set to None.  Default behavior with 0 is to reduce all axes. When axes is empty and this  attribute is set to true 1, input tensor will not be reduced, and the output  tensor would be equivalent to input tensor. Default to 0.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(*inputs: ndarray) → ndarray
```

Create corresponding QuantizedArray for the output of the activation function.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  Calibration sample inputs.

**Returns:**

- <b>`numpy.ndarray`</b>:  The output values for the provided calibration samples.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1756"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Sum the encrypted tensor's values along the given axes.

**Args:**

- <b>`q_inputs`</b> (QuantizedArray):  An encrypted integer tensor at index 0.
- <b>`calibrate_rounding`</b> (bool):  Whether to calibrate rounding or not.
- <b>`attrs`</b> (Dict):  Options are handled in constructor.

**Returns:**

- <b>`(QuantizedArray)`</b>:  The sum of all values along the given axes.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1849"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedErf`

Quantized erf op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1855"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedNot`

Quantized Not op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1861"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBrevitasQuant`

Brevitas uniform quantization with encrypted input.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1871"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

Construct the Brevitas quantization operator.

**Args:**

- <b>`n_bits_output`</b> (int):  Number of bits for the operator's quantization of outputs.  Not used, will be overridden by the bit_width in ONNX
- <b>`op_instance_name`</b> (str):  The name that should be assigned to this operation, used  to retrieve it later or get debugging information about this op (bit-width, value  range, integer intermediary values, op-specific error messages). Usually this name  is the same as the ONNX operation name for which this operation is constructed.
- <b>`int_input_names`</b> (Optional\[Set\[str\]\]):  Names of input integer tensors. Default to None.
- <b>`constant_inputs`</b> (Optional\[Dict\]):  Input constant tensor.
- <b>`scale`</b> (float):  Quantizer scale
- <b>`zero_point`</b> (float):  Quantizer zero-point
- <b>`bit_width`</b> (int):  Number of bits of the integer representation
- <b>`input_quant_opts`</b> (Optional\[QuantizationOptions\]):  Options for the input quantizer.  Default to None. attrs (dict):
- <b>`rounding_mode`</b> (str):  Rounding mode (default and only accepted option is "ROUND")
- <b>`signed`</b> (int):  Whether this op quantizes to signed integers (default 1),
- <b>`narrow`</b> (int):  Whether this op quantizes to a narrow range of integers  e.g., \[-2**n_bits-1 .. 2**n_bits-1\] (default 0),

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1973"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(*inputs: ndarray) → ndarray
```

Create corresponding QuantizedArray for the output of Quantization function.

**Args:**

- <b>`*inputs (numpy.ndarray)`</b>:  Calibration sample inputs.

**Returns:**

- <b>`numpy.ndarray`</b>:  the output values for the provided calibration samples.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1999"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Quantize values.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0,  scale, zero_point, n_bits at indices 1,2,3
- <b>`attrs`</b>:  additional optional attributes

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2064"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTranspose`

Transpose operator for quantized inputs.

This operator performs quantization and transposes the encrypted data. When the inputs are pre-computed QAT the input is only quantized if needed.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Transpose can not be fused since it must be performed over integer tensors as it moves around different elements of these input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it copies encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2074"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Transpose the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  transposed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedFloor`

Quantized Floor op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMax`

Quantized Max op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMin`

Quantized Min op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedNeg`

Quantized Neg op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSign`

Quantized Neg op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedUnsqueeze`

Unsqueeze operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Unsqueeze can not be fused since it must be performed over integer tensors as it reshapes an encrypted tensor.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it operates on encrypted tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Unsqueeze the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0, axes at index 1
- <b>`attrs`</b>:  additional optional unsqueeze options

**Returns:**

- <b>`result`</b> (QuantizedArray):  unsqueezed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedConcat`

Concatenate operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Concatenation can not be fused since it must be performed over integer tensors as it copies encrypted integers from one tensor to another.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it copies encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Concatenate the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor
- <b>`attrs`</b>:  additional optional concatenate options

**Returns:**

- <b>`result`</b> (QuantizedArray):  concatenated encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSqueeze`

Squeeze operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Squeeze can not be fused since it must be performed over integer tensors as it reshapes encrypted tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it reshapes encrypted tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Squeeze the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0, axes at index 1
- <b>`attrs`</b>:  additional optional squeeze options

**Returns:**

- <b>`result`</b> (QuantizedArray):  squeezed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXShape`

Shape operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns the shape of the tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXConstantOfShape`

ConstantOfShape operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns a new encrypted tensor and thus can not be fused.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXGather`

Gather operator.

Returns values at requested indices from the input tensor.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns values from a tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXSlice`

Slice operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns values from a tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2413"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedExpand`

Expand operator for quantized tensors.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Unsqueeze can not be fused since it must be performed over integer tensors as it reshapes an encrypted tensor.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it operates on encrypted tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

Expand the input tensor to a specified shape.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0, shape at index 1
- <b>`attrs`</b>:  additional optional expand options

**Returns:**

- <b>`result`</b> (QuantizedArray):  expanded encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedEqual`

Comparison operator ==.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2533"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedUnfold`

Quantized Unfold op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2539"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    op_instance_name: str,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`Set[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2584"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Union[ndarray, QuantizedArray, NoneType, bool, int, float],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType, bool, int, float]
```

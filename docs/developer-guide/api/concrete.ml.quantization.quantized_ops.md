<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_ops`

Quantized versions of the ONNX operators for post training quantization.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSigmoid`

Quantized sigmoid op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSigmoid`

Quantized HardSigmoid op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRelu`

Quantized Relu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPRelu`

Quantized PRelu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLeakyRelu`

Quantized LeakyRelu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSwish`

Quantized Hardswish op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedElu`

Quantized Elu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSelu`

Quantized Selu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCelu`

Quantized Celu op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedClip`

Quantized clip op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRound`

Quantized round op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPow`

Quantized pow op.

Only works for a float constant power. This operation will be fused to a (potentially larger) TLU.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGemm`

Quantized Gemm op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L300"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMatMul`

Quantized MatMul op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAdd`

Quantized Addition operator.

Can add either two variables (both encrypted) or a variable and a constant

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTanh`

Quantized Tanh op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSoftplus`

Quantized Softplus op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedExp`

Quantized Exp op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L444"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLog`

Quantized Log op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAbs`

Quantized Abs op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedIdentity`

Quantized Identity op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L461"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L475"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReshape`

Quantized Reshape op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L516"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Max Pooling operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L481"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Reshape the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedConv`

Quantized Conv op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L533"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L598"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
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

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L779"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAvgPool`

Quantized Average Pooling op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L785"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L831"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMaxPool`

Quantized Max Pooling op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L928"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1039"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Max Pooling operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L976"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1051"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPad`

Quantized Padding op.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1056"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Pad operation cannot be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation cannot be fused as it is manipulates integer tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1082"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    calibrate_rounding: bool = False,
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedWhere`

Where operator on quantized arrays.

Supports only constants for the results produced on the True/False branches.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCast`

Cast the input to the required data type.

In FHE we only support a limited number of output types. Booleans are cast to integers.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreater`

Comparison operator >.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreaterOrEqual`

Comparison operator >=.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLess`

Comparison operator \<.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLessOrEqual`

Comparison operator \<=.

Only supports comparison with a constant.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOr`

Or operator ||.

This operation is not really working as a quantized operation. It just works when things got fused, as in eg Act(x) = x || (x + 42))

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedDiv`

Div operator /.

This operation is not really working as a quantized operation. It just works when things got fused, as in eg Act(x) = 1000 / (x + 42))

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMul`

Multiplication operator.

Only multiplies an encrypted tensor with a float constant for now. This operation will be fused to a (potentially larger) TLU.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1341"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSub`

Subtraction operator.

This works the same as addition on both encrypted - encrypted and on encrypted - constant.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1351"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBatchNormalization`

Quantized Batch normalization with encrypted input and in-the-clear normalization params.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedFlatten`

Quantized flatten for encrypted inputs.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1404"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Flatten operation cannot be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation cannot be fused as it is manipulates integer tensors.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Flatten the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0
- <b>`attrs`</b>:  contains axis attribute

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReduceSum`

ReduceSum with encrypted input.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1495"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Sum the encrypted tensor's values along the given axes.

**Args:**

- <b>`q_inputs`</b> (QuantizedArray):  An encrypted integer tensor at index 0.
- <b>`attrs`</b> (Dict):  Options are handled in constructor.

**Returns:**

- <b>`(QuantizedArray)`</b>:  The sum of all values along the given axes.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1618"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedErf`

Quantized erf op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1624"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedNot`

Quantized Not op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1630"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBrevitasQuant`

Brevitas uniform quantization with encrypted input.

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1640"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
- <b>`narrow`</b> (int):  Whether this op quantizes to a narrow range of integers  eg \[-2**n_bits-1 .. 2**n_bits-1\] (default 0),

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1742"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1768"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Quantize values.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0,  scale, zero_point, n_bits at indices 1,2,3
- <b>`attrs`</b>:  additional optional attributes

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1833"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTranspose`

Transpose operator for quantized inputs.

This operator performs quantization and transposes the encrypted data. When the inputs are pre-computed QAT the input is only quantized if needed.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1875"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Transpose can not be fused since it must be performed over integer tensors as it moves around different elements of these input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it copies encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1843"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Transpose the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  transposed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1887"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedFloor`

Quantized Floor op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1893"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMax`

Quantized Max op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1899"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMin`

Quantized Min op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1905"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedNeg`

Quantized Neg op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1911"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSign`

Quantized Neg op.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1917"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedUnsqueeze`

Unsqueeze operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1957"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Unsqueeze can not be fused since it must be performed over integer tensors as it reshapes an encrypted tensor.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it operates on encrypted tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1923"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Unsqueeze the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0, axes at index 1
- <b>`attrs`</b>:  additional optional unsqueeze options

**Returns:**

- <b>`result`</b> (QuantizedArray):  unsqueezed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1969"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedConcat`

Concatenate operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Concatenation can not be fused since it must be performed over integer tensors as it copies encrypted integers from one tensor to another.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it copies encrypted integers

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L1975"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Concatenate the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor
- <b>`attrs`</b>:  additional optional concatenate options

**Returns:**

- <b>`result`</b> (QuantizedArray):  concatenated encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2037"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSqueeze`

Squeeze operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2077"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Squeeze can not be fused since it must be performed over integer tensors as it reshapes encrypted tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it reshapes encrypted tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2043"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

Squeeze the input tensors on a given axis.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0, axes at index 1
- <b>`attrs`</b>:  additional optional squeeze options

**Returns:**

- <b>`result`</b> (QuantizedArray):  squeezed encrypted integer tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2089"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXShape`

Shape operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns the shape of the tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2094"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXConstantOfShape`

ConstantOfShape operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns a new encrypted tensor and thus can not be fused.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXGather`

Gather operator.

Returns values at requested indices from the input tensor.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns values from a tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXSlice`

Slice operator.

______________________________________________________________________

#### <kbd>property</kbd> int_input_names

Get the names of encrypted integer tensors that are used by this op.

**Returns:**

- <b>`List[str]`</b>:  the names of the tensors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

This operation returns values from a tensor and thus can not be fused into a univariate TLU.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_ops.py#L2182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(
    *q_inputs: Optional[ndarray, QuantizedArray],
    **attrs
) → Union[ndarray, QuantizedArray, NoneType]
```

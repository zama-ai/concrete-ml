<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_ops`

Quantized versions of the ONNX operators for post training quantization.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSigmoid`

Quantized sigmoid op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSigmoid`

Quantized HardSigmoid op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRelu`

Quantized Relu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPRelu`

Quantized PRelu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLeakyRelu`

Quantized LeakyRelu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedHardSwish`

Quantized Hardswish op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedElu`

Quantized Elu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSelu`

Quantized Selu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCelu`

Quantized Celu op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedClip`

Quantized clip op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedRound`

Quantized round op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPow`

Quantized pow op.

Only works for a float constant power. This operation will be fused to a (potentially larger) TLU.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Power raising can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x \*\* (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGemm`

Quantized Gemm op.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse()
```

Determine if this op can be fused.

Gemm operation can not be fused since it must be performed over integer tensors and it combines different values of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMatMul`

Quantized MatMul op.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse()
```

Determine if this op can be fused.

Gemm operation can not be fused since it must be performed over integer tensors and it combines different values of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAdd`

Quantized Addition operator.

Can add either two variables (both encrypted) or a variable and a constant

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L385"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTanh`

Quantized Tanh op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSoftplus`

Quantized Softplus op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedExp`

Quantized Exp op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLog`

Quantized Log op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAbs`

Quantized Abs op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedIdentity`

Quantized Identity op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReshape`

Quantized Reshape op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Reshape the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedConv`

Quantized Conv op.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L470"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

Construct the quantized convolution operator and retrieve parameters.

**Args:**

- <b>`n_bits_output`</b>:  number of bits for the quantization of the outputs of this operator
- <b>`int_input_names`</b>:  names of integer tensors that are taken as input for this operation
- <b>`constant_inputs`</b>:  the weights and activations
- <b>`input_quant_opts`</b>:  options for the input quantizer
- <b>`attrs`</b>:  convolution options
- <b>`dilations`</b> (Tuple\[int\]):  dilation of the kernel, default 1 on all dimensions.
- <b>`group`</b> (int):  number of convolution groups, default 1
- <b>`kernel_shape`</b> (Tuple\[int\]):  shape of the kernel. Should have 2 elements for 2d conv
- <b>`pads`</b> (Tuple\[int\]):  padding in ONNX format (begin, end) on each axis
- <b>`strides`</b> (Tuple\[int\]):  stride of the convolution on each axis

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L649"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Conv operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L534"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Compute the quantized convolution between two quantized tensors.

Allows an optional quantized bias.

**Args:**

- <b>`q_inputs`</b>:  input tuple, contains
- <b>`x`</b> (numpy.ndarray):  input data. Shape is N x C x H x W for 2d
- <b>`w`</b> (numpy.ndarray):  weights tensor. Shape is (O x I x Kh x Kw) for 2d
- <b>`b`</b> (numpy.ndarray, Optional):  bias tensor, Shape is (O,)
- <b>`attrs`</b>:  convolution options handled in constructor

**Returns:**

- <b>`res`</b> (QuantizedArray):  result of the quantized integer convolution

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L662"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedAvgPool`

Quantized Average Pooling op.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Avg Pooling operation can not be fused since it must be performed over integer tensors and it combines different elements of the input tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it adds different encrypted integers

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L717"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L766"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedPad`

Quantized Padding op.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L771"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L793"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Pad operation can not be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it is manipulates integer tensors

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L804"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedWhere`

Where operator on quantized arrays.

Supports only constants for the results produced on the True/False branches.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L813"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L836"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedCast`

Cast the input to the required data type.

In FHE we only support a limited number of output types. Booleans are cast to integers.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L845"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreater`

Comparison operator >.

Only supports comparison with a constant.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L869"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedGreaterOrEqual`

Comparison operator >=.

Only supports comparison with a constant.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L878"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L893"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLess`

Comparison operator \<.

Only supports comparison with a constant.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L902"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L917"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedLessOrEqual`

Comparison operator \<=.

Only supports comparison with a constant.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L926"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L941"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedOr`

Or operator ||.

This operation is not really working as a quantized operation. It just works when things got fused, as in e.g. Act(x) = x || (x + 42))

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L950"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L969"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Or can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x || (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L982"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedDiv`

Div operator /.

This operation is not really working as a quantized operation. It just works when things got fused, as in e.g. Act(x) = 1000 / (x + 42))

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L991"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1010"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Div can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x / (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1023"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedMul`

Multiplication operator.

Only multiplies an encrypted tensor with a float constant for now. This operation will be fused to a (potentially larger) TLU.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1032"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: QuantizationOptions = None,
    **attrs
) → None
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1051"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Multiplication can be fused and computed in float when a single integer tensor generates both the operands. For example in the formula: f(x) = x * (x + 1)  where x is an integer tensor.

**Returns:**

- <b>`bool`</b>:  Can fuse

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1064"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSub`

Subtraction operator.

This works the same as addition on both encrypted - encrypted and on encrypted - constant.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Add operation can be computed in float and fused if it operates over inputs produced by a single integer tensor. For example the expression x + x * 1.75, where x is an encrypted tensor, can be computed with a single TLU.

**Returns:**

- <b>`bool`</b>:  Whether the number of integer input tensors allows computing this op as a TLU

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1074"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBatchNormalization`

Quantized Batch normalization with encrypted input and in-the-clear normalization params.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1080"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedFlatten`

Quantized flatten for encrypted inputs.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_fuse`

```python
can_fuse() → bool
```

Determine if this op can be fused.

Flatten operation can not be fused since it must be performed over integer tensors.

**Returns:**

- <b>`bool`</b>:  False, this operation can not be fused as it is manipulates integer tensors.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1086"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Flatten the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0
- <b>`attrs`</b>:  contains axis attribute

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedReduceSum`

ReduceSum with encrypted input.

This operator is currently an experimental feature.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

Construct the quantized ReduceSum operator and retrieve parameters.

**Args:**

- <b>`n_bits_output`</b> (int):  Number of bits for the operator's quantization of outputs.
- <b>`int_input_names`</b> (Optional\[Set\[str\]\]):  Names of input integer tensors. Default to None.
- <b>`constant_inputs`</b> (Optional\[Dict\]):  Input constant tensor.
- <b>`axes`</b> (Optional\[numpy.ndarray\]):  Array of integers along which to reduce.  The default is to reduce over all the dimensions of the input tensor if  'noop_with_empty_axes' is false, else act as an Identity op when  'noop_with_empty_axes' is true. Accepted range is \[-r, r-1\] where  r = rank(data). Default to None.
- <b>`input_quant_opts`</b> (Optional\[QuantizationOptions\]):  Options for the input quantizer.  Default to None.
- <b>`attrs`</b> (dict):  RecuseSum options.
- <b>`keepdims`</b> (int):  Keep the reduced dimension or not, 1 means keeping the  input dimension, 0 will reduce it along the given axis. Default to 1.
- <b>`noop_with_empty_axes`</b> (int):  Defines behavior if 'axes' is empty or set to None.  Default behavior with 0 is to reduce all axes. When axes is empty and this  attribute is set to true 1, input tensor will not be reduced, and the output  tensor would be equivalent to input tensor. Default to 0.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Sum the encrypted tensor's values over axis 1.

**Args:**

- <b>`q_inputs`</b> (QuantizedArray):  An encrypted integer tensor at index 0.
- <b>`attrs`</b> (Dict):  Contains axis attribute.

**Returns:**

- <b>`(QuantizedArray)`</b>:  The sum of all values along axis 1 as an encrypted integer tensor.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tree_sum`

```python
tree_sum(input_qarray, is_calibration=False)
```

Large sum without overflow (only MSB remains).

**Args:**

- <b>`input_qarray`</b>:  Enctyped integer tensor.
- <b>`is_calibration`</b>:  Whether we are calibrating the tree sum. If so, it will create all the  quantizers for the downscaling.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  The MSB (based on the precision self.n_bits) of the integers sum.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedErf`

Quantized erf op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedNot`

Quantized Not op.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedBrevitasQuant`

Brevitas uniform quantization with encrypted input.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits_output: int,
    int_input_names: Set[str] = None,
    constant_inputs: Optional[Dict[str, Any], Dict[int, Any]] = None,
    input_quant_opts: Optional[QuantizationOptions] = None,
    **attrs
) → None
```

Construct the Brevitas quantization operator.

**Args:**

- <b>`n_bits_output`</b> (int):  Number of bits for the operator's quantization of outputs.  Not used, will be overridden by the bit_width in ONNX
- <b>`int_input_names`</b> (Optional\[Set\[str\]\]):  Names of input integer tensors. Default to None.
- <b>`constant_inputs`</b> (Optional\[Dict\]):  Input constant tensor.
- <b>`scale`</b> (float):  Quantizer scale
- <b>`zero_point`</b> (float):  Quantizer zero-point
- <b>`bit_width`</b> (int):  Number of bits of the integer representation
- <b>`input_quant_opts`</b> (Optional\[QuantizationOptions\]):  Options for the input quantizer.  Default to None. attrs (dict):
- <b>`rounding_mode`</b> (str):  Rounding mode (default and only accepted option is "ROUND")
- <b>`signed`</b> (int):  Whether this op quantizes to signed integers (default 1),
- <b>`narrow`</b> (int):  Whether this op quantizes to a narrow range of integers  e.g. \[-2**n_bits-1 .. 2**n_bits-1\] (default 0),

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Quantize values.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTranspose`

Transpose operator for quantized inputs.

This operator performs quantization, transposes the encrypted data, then dequantizes again.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_ops.py#L1566"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `q_impl`

```python
q_impl(*q_inputs: QuantizedArray, **attrs) → QuantizedArray
```

Reshape the input integer encrypted tensor.

**Args:**

- <b>`q_inputs`</b>:  an encrypted integer tensor at index 0 and one constant shape at index 1
- <b>`attrs`</b>:  additional optional reshape options

**Returns:**

- <b>`result`</b> (QuantizedArray):  reshaped encrypted integer tensor

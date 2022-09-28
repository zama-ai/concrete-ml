<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantizers`

Quantization utilities for a numpy array/tensor.

## **Global Variables**

- **STABILITY_CONST**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fill_from_kwargs`

```python
fill_from_kwargs(obj, klass, **kwargs)
```

Fill a parameter set structure from kwargs parameters.

**Args:**

- <b>`obj`</b>:  an object of type klass, if None the object is created if any of the type's  members appear in the kwargs
- <b>`klass`</b>:  the type of object to fill
- <b>`kwargs`</b>:  parameter names and values to fill into an instance of the klass type

**Returns:**

- <b>`obj`</b>:  an object of type klass
- <b>`kwargs`</b>:  remaining parameter names and values that were not filled into obj

**Raises:**

- <b>`TypeError`</b>:  if the types of the parameters in kwargs could not be converted  to the corresponding types of members of klass

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizationOptions`

Options for quantization.

Determines the number of bits for quantization and the method of quantization of the values. Signed quantization allows negative quantized values. Symmetric quantization assumes the float values are distributed symmetrically around x=0 and assigns signed values around 0 to the float values. QAT (quantization aware training) quantization assumes the values are already quantized, taking a discrete set of values, and assigns these values to integers, computing only the scale.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits,
    is_signed: bool = False,
    is_symmetric: bool = False,
    is_qat: bool = False
) → None
```

______________________________________________________________________

#### <kbd>property</kbd> quant_options

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_opts`

```python
copy_opts(opts)
```

Copy the options from a different structure.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  structure to copy parameters from.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinMaxQuantizationStats`

Calibration set statistics.

This class stores the statistics for the calibration set or for a calibration data batch. Currently we only store min/max to determine the quantization range. The min/max are computed from the calibration set.

______________________________________________________________________

#### <kbd>property</kbd> quant_stats

Get a copy of the calibration set statistics.

**Returns:**

- <b>`MinMaxQuantizationStats`</b>:  a copy of the current quantization stats

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_stats`

```python
compute_quantization_stats(values: ndarray) → None
```

Compute the calibration set quantization statistics.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Calibration set on which to compute statistics.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_stats`

```python
copy_stats(stats) → None
```

Copy the statistics from a different structure.

**Args:**

- <b>`stats`</b> (MinMaxQuantizationStats):  structure to copy statistics from.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UniformQuantizationParameters`

Quantization parameters for uniform quantization.

This class stores the parameters used for quantizing real values to discrete integer values. The parameters are computed from quantization options and quantization statistics.

______________________________________________________________________

#### <kbd>property</kbd> quant_params

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_parameters`

```python
compute_quantization_parameters(
    options: QuantizationOptions,
    stats: MinMaxQuantizationStats
) → None
```

Compute the quantization parameters.

**Args:**

- <b>`options`</b> (QuantizationOptions):  quantization options set
- <b>`stats`</b> (MinMaxQuantizationStats):  calibrated statistics for quantization

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_params`

```python
copy_params(params) → None
```

Copy the parameters from a different structure.

**Args:**

- <b>`params`</b> (UniformQuantizationParameters):  parameter structure to copy

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UniformQuantizer`

Uniform quantizer.

Contains all information necessary for uniform quantization and provides quantization/dequantization functionality on numpy arrays.

**Args:**

- <b>`options`</b> (QuantizationOptions):  Quantization options set
- <b>`stats`</b> (Optional\[MinMaxQuantizationStats\]):  Quantization batch statistics set
- <b>`params`</b> (Optional\[UniformQuantizationParameters\]):  Quantization parameters set  (scale, zero-point)

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    options: QuantizationOptions = None,
    stats: Optional[MinMaxQuantizationStats] = None,
    params: Optional[UniformQuantizationParameters] = None,
    **kwargs
)
```

______________________________________________________________________

#### <kbd>property</kbd> quant_options

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

#### <kbd>property</kbd> quant_params

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

#### <kbd>property</kbd> quant_stats

Get a copy of the calibration set statistics.

**Returns:**

- <b>`MinMaxQuantizationStats`</b>:  a copy of the current quantization stats

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_parameters`

```python
compute_quantization_parameters(
    options: QuantizationOptions,
    stats: MinMaxQuantizationStats
) → None
```

Compute the quantization parameters.

**Args:**

- <b>`options`</b> (QuantizationOptions):  quantization options set
- <b>`stats`</b> (MinMaxQuantizationStats):  calibrated statistics for quantization

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_stats`

```python
compute_quantization_stats(values: ndarray) → None
```

Compute the calibration set quantization statistics.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Calibration set on which to compute statistics.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_opts`

```python
copy_opts(opts)
```

Copy the options from a different structure.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  structure to copy parameters from.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_params`

```python
copy_params(params) → None
```

Copy the parameters from a different structure.

**Args:**

- <b>`params`</b> (UniformQuantizationParameters):  parameter structure to copy

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_stats`

```python
copy_stats(stats) → None
```

Copy the statistics from a different structure.

**Args:**

- <b>`stats`</b> (MinMaxQuantizationStats):  structure to copy statistics from.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L375"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant(qvalues: ndarray) → ndarray
```

Dequantize values.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  integer values to dequantize

**Returns:**

- <b>`numpy.ndarray`</b>:  Dequantized float values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant(values: ndarray) → ndarray
```

Quantize values.

**Args:**

- <b>`values`</b> (numpy.ndarray):  float values to quantize

**Returns:**

- <b>`numpy.ndarray`</b>:  Integer quantized values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L392"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedArray`

Abstraction of quantized array.

Contains float values and their quantized integer counter-parts. Quantization is performed by the quantizer member object. Float and int values are kept in sync. Having both types of values is useful since quantized operators in Concrete ML graphs might need one or the other depending on how the operator works (in float or in int). Moreover, when the encrypted function needs to return a value, it must return integer values.

See https://arxiv.org/abs/1712.05877.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Values to be quantized.
- <b>`n_bits`</b> (int):  The number of bits to use for quantization.
- <b>`value_is_float`</b> (bool, optional):  Whether the passed values are real (float) values or not.  If False, the values will be quantized according to the passed scale and zero_point.  Defaults to True.
- <b>`options`</b> (QuantizationOptions):  Quantization options set
- <b>`stats`</b> (Optional\[MinMaxQuantizationStats\]):  Quantization batch statistics set
- <b>`params`</b> (Optional\[UniformQuantizationParameters\]):  Quantization parameters set  (scale, zero-point)
- <b>`kwargs`</b>:  Any member of the options, stats, params sets as a key-value pair. The parameter  sets need to be completely parametrized if their members appear in kwargs.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits,
    values: Optional[ndarray],
    value_is_float: bool = True,
    options: QuantizationOptions = None,
    stats: Optional[MinMaxQuantizationStats] = None,
    params: Optional[UniformQuantizationParameters] = None,
    **kwargs
)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L566"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant() → ndarray
```

Dequantize self.qvalues.

**Returns:**

- <b>`numpy.ndarray`</b>:  Dequantized values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant() → Optional[ndarray]
```

Quantize self.values.

**Returns:**

- <b>`numpy.ndarray`</b>:  Quantized values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_quantized_values`

```python
update_quantized_values(qvalues: ndarray) → ndarray
```

Update qvalues to get their corresponding values using the related quantized parameters.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  Values to replace self.qvalues

**Returns:**

- <b>`values`</b> (numpy.ndarray):  Corresponding values

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantizers.py#L526"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_values`

```python
update_values(values: ndarray) → ndarray
```

Update values to get their corresponding qvalues using the related quantized parameters.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Values to replace self.values

**Returns:**

- <b>`qvalues`</b> (numpy.ndarray):  Corresponding qvalues

<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/quantizers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantizers`

Quantization utilities for a numpy array/tensor.

## **Global Variables**

- **QUANT_ROUND_LIKE_ROUND_PBS**
- **STABILITY_CONST**

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fill_from_kwargs`

```python
fill_from_kwargs(obj, klass, accept_missing, **kwargs)
```

Fill a parameter set structure from kwargs parameters.

**Args:**

- <b>`obj`</b>:  an object of type klass, if None the object is created if any of the type's  members appear in the kwargs
- <b>`klass`</b>:  the type of object to fill
- <b>`accept_missing`</b>:  don't assert if the fields are None in the kwargs
- <b>`kwargs`</b>:  parameter names and values to fill into an instance of the klass type

**Returns:**

- <b>`obj`</b>:  an object of type klass
- <b>`kwargs`</b>:  remaining parameter names and values that were not filled into obj

**Raises:**

- <b>`TypeError`</b>:  if the types of the parameters in kwargs could not be converted  to the corresponding types of members of klass

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizationOptions`

Options for quantization.

Determines the number of bits for quantization and the method of quantization of the values. Signed quantization allows negative quantized values. Symmetric quantization assumes the float values are distributed symmetrically around x=0 and assigns signed values around 0 to the float values. QAT (quantization aware training) quantization assumes the values are already quantized, taking a discrete set of values, and assigns these values to integers, computing only the scale.

<a href="../../../src/concrete/ml/quantization/quantizers.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: 'int',
    is_signed: 'bool' = False,
    is_symmetric: 'bool' = False,
    is_qat: 'bool' = False
)
```

______________________________________________________________________

#### <kbd>property</kbd> quant_options

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_opts`

```python
copy_opts(opts)
```

Copy the options from a different structure.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  structure to copy parameters from.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_equal`

```python
is_equal(opts, ignore_sign_qat: 'bool' = False) → bool
```

Compare two quantization options sets.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  options to compare this instance to
- <b>`ignore_sign_qat`</b> (bool):  ignore sign comparison for QAT options

**Returns:**

- <b>`bool`</b>:  whether the two quantization options compared are equivalent

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: 'Dict')
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizationOptions`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinMaxQuantizationStats`

Calibration set statistics.

This class stores the statistics for the calibration set or for a calibration data batch. Currently we only store min/max to determine the quantization range. The min/max are computed from the calibration set.

<a href="../../../src/concrete/ml/quantization/quantizers.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(rmax: 'Optional[float]' = None, rmin: 'Optional[float]' = None)
```

______________________________________________________________________

#### <kbd>property</kbd> quant_stats

Get a copy of the calibration set statistics.

**Returns:**

- <b>`MinMaxQuantizationStats`</b>:  a copy of the current quantization stats

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_stats`

```python
compute_quantization_stats(values: 'ndarray') → None
```

Compute the calibration set quantization statistics.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Calibration set on which to compute statistics.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_stats`

```python
copy_stats(stats) → None
```

Copy the statistics from a different structure.

**Args:**

- <b>`stats`</b> (MinMaxQuantizationStats):  structure to copy statistics from.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: 'Dict')
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizationOptions`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UniformQuantizationParameters`

Quantization parameters for uniform quantization.

This class stores the parameters used for quantizing real values to discrete integer values. The parameters are computed from quantization options and quantization statistics.

<a href="../../../src/concrete/ml/quantization/quantizers.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    scale: 'Optional[float64]' = None,
    zero_point: 'Optional[Union[int, float, ndarray]]' = None,
    offset: 'Optional[int]' = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> quant_params

Get a copy of the quantization parameters.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  a copy of the current quantization parameters

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_parameters`

```python
compute_quantization_parameters(
    options: 'QuantizationOptions',
    stats: 'MinMaxQuantizationStats'
) → None
```

Compute the quantization parameters.

**Args:**

- <b>`options`</b> (QuantizationOptions):  quantization options set
- <b>`stats`</b> (MinMaxQuantizationStats):  calibrated statistics for quantization

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_params`

```python
copy_params(params) → None
```

Copy the parameters from a different structure.

**Args:**

- <b>`params`</b> (UniformQuantizationParameters):  parameter structure to copy

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: 'Dict') → UniformQuantizationParameters
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`UniformQuantizationParameters`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L530"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UniformQuantizer`

Uniform quantizer.

Contains all information necessary for uniform quantization and provides quantization/de-quantization functionality on numpy arrays.

**Args:**

- <b>`options`</b> (QuantizationOptions):  Quantization options set
- <b>`stats`</b> (Optional\[MinMaxQuantizationStats\]):  Quantization batch statistics set
- <b>`params`</b> (Optional\[UniformQuantizationParameters\]):  Quantization parameters set  (scale, zero-point)

<a href="../../../src/concrete/ml/quantization/quantizers.py#L544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    options: 'Optional[QuantizationOptions]' = None,
    stats: 'Optional[MinMaxQuantizationStats]' = None,
    params: 'Optional[UniformQuantizationParameters]' = None,
    no_clipping: 'bool' = False
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

<a href="../../../src/concrete/ml/quantization/quantizers.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_parameters`

```python
compute_quantization_parameters(
    options: 'QuantizationOptions',
    stats: 'MinMaxQuantizationStats'
) → None
```

Compute the quantization parameters.

**Args:**

- <b>`options`</b> (QuantizationOptions):  quantization options set
- <b>`stats`</b> (MinMaxQuantizationStats):  calibrated statistics for quantization

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_quantization_stats`

```python
compute_quantization_stats(values: 'ndarray') → None
```

Compute the calibration set quantization statistics.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Calibration set on which to compute statistics.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_opts`

```python
copy_opts(opts)
```

Copy the options from a different structure.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  structure to copy parameters from.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_params`

```python
copy_params(params) → None
```

Copy the parameters from a different structure.

**Args:**

- <b>`params`</b> (UniformQuantizationParameters):  parameter structure to copy

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy_stats`

```python
copy_stats(stats) → None
```

Copy the statistics from a different structure.

**Args:**

- <b>`stats`</b> (MinMaxQuantizationStats):  structure to copy statistics from.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L721"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant(qvalues: 'ndarray') → Union[float, ndarray, Tracer]
```

De-quantize values.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  integer values to de-quantize

**Returns:**

- <b>`Union[numpy.ndarray, Tracer]`</b>:  De-quantized float values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L600"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L660"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_equal`

```python
is_equal(opts, ignore_sign_qat: 'bool' = False) → bool
```

Compare two quantization options sets.

**Args:**

- <b>`opts`</b> (QuantizationOptions):  options to compare this instance to
- <b>`ignore_sign_qat`</b> (bool):  ignore sign comparison for QAT options

**Returns:**

- <b>`bool`</b>:  whether the two quantization options compared are equivalent

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: 'Dict') → UniformQuantizer
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`UniformQuantizer`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant(values: 'ndarray', dtype: 'DTypeLike' = <class 'numpy.int64'>) → ndarray
```

Quantize values.

**Args:**

- <b>`values`</b> (numpy.ndarray):  float values to quantize
- <b>`dtype`</b> (numpy.typing.DTypeLike):  optional user-specified datatype for the output

**Returns:**

- <b>`numpy.ndarray`</b>:  Integer quantized values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TorchUniformQuantizer`

Uniform quantizer with a PyTorch implementation.

Contains all information necessary for uniform quantization and provides quantization/de-quantization functionality on torch tensors.

**Args:**

- <b>`quantizer`</b> (UniformQuantizer):  Underlying numpy quantizer containing all parameters

<a href="../../../src/concrete/ml/quantization/quantizers.py#L761"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(quantizer: 'UniformQuantizer')
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L791"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant(qvalues: 'Tensor') → Tensor
```

De-quantize values.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  integer values to de-quantize

**Returns:**

- <b>`Union[numpy.ndarray, Tracer]`</b>:  De-quantized float values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L764"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant(values: 'Tensor', dtype: 'Optional[dtype]' = None) → Tensor
```

Quantize values.

**Args:**

- <b>`values`</b> (numpy.ndarray):  float values to quantize
- <b>`dtype`</b> (Optional\[torch.dtype\]):  optional user-specified datatype for the output

**Returns:**

- <b>`numpy.ndarray`</b>:  Integer quantized values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L806"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/quantization/quantizers.py#L835"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits,
    values: 'Union[None, float, int, ndarray]',
    value_is_float: 'bool' = True,
    options: 'Optional[QuantizationOptions]' = None,
    stats: 'Optional[MinMaxQuantizationStats]' = None,
    params: 'Optional[UniformQuantizationParameters]' = None,
    **kwargs
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L1051"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant() → Union[ndarray, Tracer]
```

De-quantize self.qvalues.

**Returns:**

- <b>`Union[numpy.ndarray, Tracer]`</b>:  De-quantized values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L957"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L989"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L971"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: 'Dict') → QuantizedArray
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizedArray`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L1041"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant() → Union[ndarray, Tracer]
```

Quantize self.values.

**Returns:**

- <b>`Union[numpy.ndarray, Tracer]`</b>:  Quantized values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L1022"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_quantized_values`

```python
update_quantized_values(
    qvalues: 'Union[ndarray, Tracer]'
) → Union[ndarray, Tracer]
```

Update qvalues to get their corresponding values using the related quantized parameters.

**Args:**

- <b>`qvalues`</b> (Union\[numpy.ndarray, Tracer\]):  Values to replace self.qvalues

**Returns:**

- <b>`values`</b> (Union\[numpy.ndarray, Tracer\]):  Corresponding values

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantizers.py#L1005"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_values`

```python
update_values(values: 'Union[ndarray, Tracer]') → Union[ndarray, Tracer]
```

Update values to get their corresponding qvalues using the related quantized parameters.

**Args:**

- <b>`values`</b> (Union\[numpy.ndarray, Tracer\]):  Values to replace self.values

**Returns:**

- <b>`qvalues`</b> (Union\[numpy.ndarray, Tracer\]):  Corresponding qvalues

<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/post_training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.post_training`

Post Training Quantization methods.

## **Global Variables**

- **ONNX_OPS_TO_NUMPY_IMPL**
- **DEFAULT_MODEL_BITS**
- **ONNX_OPS_TO_QUANTIZED_IMPL**

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_n_bits_dict`

```python
get_n_bits_dict(n_bits: Union[int, Dict[str, int]]) → Dict[str, int]
```

Convert the n_bits parameter into a proper dictionary.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  number of bits for quantization, can be a single value or  a dictionary with the following keys :
  \- "op_inputs" and "op_weights" (mandatory)
  \- "model_inputs" and "model_outputs" (optional, default to 5 bits).  When using a single integer for n_bits, its value is assigned to "op_inputs" and  "op_weights" bits. The maximum between this value and a default value (5) is then  assigned to the number of "model_inputs" "model_outputs". This default value is a  compromise between model accuracy and runtime performance in FHE. "model_outputs" gives  the precision of the final network's outputs, while "model_inputs" gives the precision  of the network's inputs. "op_inputs" and "op_weights" both control the quantization for  inputs and weights of all layers.

**Returns:**

- <b>`n_bits_dict`</b> (Dict\[str, int\]):  A dictionary properly representing the number of bits to use  for quantization.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXConverter`

Base ONNX to Concrete ML computation graph conversion class.

This class provides a method to parse an ONNX graph and apply several transformations. First, it creates QuantizedOps for each ONNX graph op. These quantized ops have calibrated quantizers that are useful when the operators work on integer data or when the output of the ops is the output of the encrypted program. For operators that compute in float and will be merged to TLUs, these quantizers are not used. Second, this converter creates quantized tensors for initializer and weights stored in the graph.

This class should be sub-classed to provide specific calibration and quantization options depending on the usage (Post-training quantization vs Quantization Aware training).

**Arguments:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  number of bits for quantization, can be a single value or  a dictionary with the following keys :
  \- "op_inputs" and "op_weights" (mandatory)
  \- "model_inputs" and "model_outputs" (optional, default to 5 bits).  When using a single integer for n_bits, its value is assigned to "op_inputs" and  "op_weights" bits. The maximum between this value and a default value (5) is then  assigned to the number of "model_inputs" "model_outputs". This default value is a  compromise between model accuracy and runtime performance in FHE. "model_outputs" gives  the precision of the final network's outputs, while "model_inputs" gives the precision  of the network's inputs. "op_inputs" and "op_weights" both control the quantization for  inputs and weights of all layers.
- <b>`numpy_model`</b> (NumpyModule):  Model in numpy.
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)

<a href="../../../src/concrete/ml/quantization/post_training.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: Union[int, Dict],
    numpy_model: NumpyModule,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_inputs

Get the number of bits to use for the quantization of the first layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for input quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_outputs

Get the number of bits to use for the quantization of the last layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for output quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_inputs

Get the number of bits to use for the quantization of any operators' inputs.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for the quantization of the operators' inputs

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_weights

Get the number of bits to use for the quantization of any constants (usually weights).

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for quantizing constants used by operators

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_module`

```python
quantize_module(*calibration_data: ndarray) → QuantizedModule
```

Quantize numpy module.

Following https://arxiv.org/abs/1712.05877 guidelines.

**Args:**

- <b>`*calibration_data (numpy.ndarray)`</b>:   Data that will be used to compute the bounds,  scales and zero point values for every quantized  object.

**Returns:**

- <b>`QuantizedModule`</b>:  Quantized numpy module

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L823"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PostTrainingAffineQuantization`

Post-training Affine Quantization.

Create the quantized version of the passed numpy module.

**Args:**

- <b>`n_bits`</b> (int, Dict):              Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for activation,  inputs and weights. If a dict is passed, then it should  contain  "model_inputs", "op_inputs", "op_weights" and  "model_outputs" keys with corresponding number of  quantization bits for:
  \- model_inputs : number of bits for model input
  \- op_inputs : number of bits to quantize layer input values
  \- op_weights: learned parameters or constants in the network
  \- model_outputs: final model output quantization bits
- <b>`numpy_model`</b> (NumpyModule):       Model in numpy.
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  if not None, every  accumulators in the model are rounded down to the given  bits of precision. Can be an int or a dictionary with keys  'method' and 'n_bits', where 'method' is either  fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE, and  'n_bits' is either 'auto' or an int.
- <b>`is_signed`</b>:                       Whether the weights of the layers can be signed.  Currently, only the weights can be signed.

**Returns:**

- <b>`QuantizedModule`</b>:  A quantized version of the numpy model.

<a href="../../../src/concrete/ml/quantization/post_training.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: Union[int, Dict],
    numpy_model: NumpyModule,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_inputs

Get the number of bits to use for the quantization of the first layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for input quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_outputs

Get the number of bits to use for the quantization of the last layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for output quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_inputs

Get the number of bits to use for the quantization of any operators' inputs.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for the quantization of the operators' inputs

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_weights

Get the number of bits to use for the quantization of any constants (usually weights).

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for quantizing constants used by operators

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_module`

```python
quantize_module(*calibration_data: ndarray) → QuantizedModule
```

Quantize numpy module.

Following https://arxiv.org/abs/1712.05877 guidelines.

**Args:**

- <b>`*calibration_data (numpy.ndarray)`</b>:   Data that will be used to compute the bounds,  scales and zero point values for every quantized  object.

**Returns:**

- <b>`QuantizedModule`</b>:  Quantized numpy module

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L974"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PostTrainingQATImporter`

Converter of Quantization Aware Training networks.

This class provides specific configuration for QAT networks during ONNX network conversion to Concrete ML computation graphs.

<a href="../../../src/concrete/ml/quantization/post_training.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: Union[int, Dict],
    numpy_model: NumpyModule,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_inputs

Get the number of bits to use for the quantization of the first layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for input quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_model_outputs

Get the number of bits to use for the quantization of the last layer's output.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for output quantization

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_inputs

Get the number of bits to use for the quantization of any operators' inputs.

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for the quantization of the operators' inputs

______________________________________________________________________

#### <kbd>property</kbd> n_bits_op_weights

Get the number of bits to use for the quantization of any constants (usually weights).

**Returns:**

- <b>`n_bits`</b> (int):  number of bits for quantizing constants used by operators

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/post_training.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_module`

```python
quantize_module(*calibration_data: ndarray) → QuantizedModule
```

Quantize numpy module.

Following https://arxiv.org/abs/1712.05877 guidelines.

**Args:**

- <b>`*calibration_data (numpy.ndarray)`</b>:   Data that will be used to compute the bounds,  scales and zero point values for every quantized  object.

**Returns:**

- <b>`QuantizedModule`</b>:  Quantized numpy module
